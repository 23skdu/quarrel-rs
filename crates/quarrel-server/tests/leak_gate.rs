//! Local leak gate for P8-04: 16 load/search/close cycles with < 5 MB heap growth.
//!
//! This is a dedicated integration-test binary so its process-wide counting allocator cannot
//! affect any other test. The allocator wraps `System`, prepends a small header to every block,
//! and records **live bytes** (`alloc` and `realloc` growth minus the matching `dealloc`) only for
//! allocations made while a thread-local flag is enabled on the test thread. A monotonic epoch in
//! the header means blocks that outlive a counter reset are ignored when they are eventually
//! freed, so warm-up residue cannot show up as negative growth during the measured cycles.
//!
//! What one measured cycle does: create a temporary directory with two dummy `.gguf` files, build
//! the real server `Config` and `AppState` (whose `InferenceService` uses quarrel-core's
//! `ModelResolver` for discovery and resolution), discover and validate both model ids, construct
//! the production axum router, serve one in-memory `/v1/models` lookup through
//! `tower::ServiceExt::oneshot`, and drop everything including the temporary directory. Two
//! warm-up cycles absorb one-time loader/router/allocator caches before the counters are reset.
//!
//! The 5 MiB threshold is the release-gate contract from `docs/gates.md`: a load/search/close cycle
//! that retains more than 5 MB is a leak whether or not RSS has caught up yet. Live heap bytes are
//! counted instead of RSS because RSS includes allocator caches and unrelated process noise; a
//! steady-state server must return to the same live-heap baseline after every cycle. The extra
//! first-half/second-half comparison catches a slow monotonic trend that would stay under the
//! absolute 5 MB ceiling. No sleeps, sockets, or global metric recorder are used: the router is
//! driven in memory and the metrics handle is detached.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    fs,
    path::{Path, PathBuf},
    ptr,
    sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering},
};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use quarrel_server::{
    api::{AppState, build_router},
    config::Config,
    metrics::Metrics,
};
use tokio::runtime::Runtime;
use tower::util::ServiceExt;

const WARMUP_CYCLES: usize = 2;
const MEASURED_CYCLES: usize = 16;
const MAX_GROWTH_BYTES: i64 = 5 * 1024 * 1024;
/// A trend larger than 1 MiB between the two halves is a leak even if the absolute gate passes.
const TREND_TOLERANCE_BYTES: i64 = 1024 * 1024;
const MODEL_IDS: [&str; 2] = ["alpha", "beta"];

#[repr(C)]
struct Header {
    size: usize,
    epoch: u64,
}

/// Returns the padded layout and the header size. The header is a power-of-two multiple of the
/// requested alignment, so the user pointer keeps its original alignment.
fn padded(layout: Layout) -> (Layout, usize) {
    let align = layout.align().max(align_of::<Header>());
    let header = align.max(size_of::<Header>());
    debug_assert_eq!(header % align, 0, "header must preserve alignment");
    let padded = Layout::from_size_align(layout.size() + header, align)
        .expect("allocation size plus header must be a valid Layout");
    (padded, header)
}

static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static EPOCH: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static COUNT_ON_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Epoch to stamp on an allocation made right now, or 0 when this thread is not being measured.
fn counted_epoch() -> u64 {
    if COUNT_ON_THIS_THREAD.with(Cell::get) {
        EPOCH.load(Ordering::Relaxed)
    } else {
        0
    }
}

fn add_live(epoch: u64, size: usize) {
    if epoch != 0 {
        LIVE_BYTES.fetch_add(size as i64, Ordering::Relaxed);
    }
}

fn remove_live(record: Header) {
    if record.epoch != 0 && record.epoch == EPOCH.load(Ordering::Relaxed) {
        LIVE_BYTES.fetch_sub(record.size as i64, Ordering::Relaxed);
    }
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (padded, header) = padded(layout);
        // SAFETY: `padded` is a valid non-zero-sized layout derived from `layout`.
        let base = unsafe { System.alloc(padded) };
        if base.is_null() {
            return ptr::null_mut();
        }
        let epoch = counted_epoch();
        add_live(epoch, layout.size());
        let user = unsafe { base.add(header) };
        // SAFETY: `base` is aligned for `Header` and at least `header` bytes precede `user`.
        unsafe {
            base.cast::<Header>().write(Header {
                size: layout.size(),
                epoch,
            });
        }
        user
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let (padded, header) = padded(layout);
        // SAFETY: `padded` is a valid non-zero-sized layout derived from `layout`.
        let base = unsafe { System.alloc_zeroed(padded) };
        if base.is_null() {
            return ptr::null_mut();
        }
        let epoch = counted_epoch();
        add_live(epoch, layout.size());
        let user = unsafe { base.add(header) };
        // SAFETY: `base` is aligned for `Header` and at least `header` bytes precede `user`.
        unsafe {
            base.cast::<Header>().write(Header {
                size: layout.size(),
                epoch,
            });
        }
        user
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (padded, header) = padded(layout);
        let base = unsafe { ptr.sub(header) };
        // SAFETY: `ptr` came from this allocator with the same layout; the header was written
        // immediately before it.
        let record = unsafe { base.cast::<Header>().read() };
        remove_live(record);
        // SAFETY: `base` and `padded` are the original allocation parameters.
        unsafe { System.dealloc(base, padded) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (old_padded, old_header) = padded(layout);
        let old_base = unsafe { ptr.sub(old_header) };
        // SAFETY: the pointer and layout originate from an allocation made by this allocator.
        let old_record = unsafe { old_base.cast::<Header>().read() };

        let new_layout = Layout::from_size_align(new_size, layout.align())
            .expect("realloc layout must be valid");
        let (new_padded, new_header) = padded(new_layout);
        // SAFETY: `new_padded` is a valid non-zero-sized layout.
        let new_base = unsafe { System.alloc(new_padded) };
        if new_base.is_null() {
            return ptr::null_mut();
        }
        let copy_len = layout.size().min(new_size);
        // SAFETY: both regions are valid for `copy_len` bytes and cannot overlap.
        unsafe { ptr::copy_nonoverlapping(ptr, new_base.add(new_header), copy_len) };

        let epoch = if COUNT_ON_THIS_THREAD.with(Cell::get) {
            let epoch = EPOCH.load(Ordering::Relaxed);
            remove_live(old_record);
            add_live(epoch, new_size);
            epoch
        } else {
            // Preserve the accounting lineage when a block moves between threads.
            old_record.epoch
        };
        // SAFETY: `new_base` is aligned for `Header` and has room for it before `user`.
        unsafe {
            new_base.cast::<Header>().write(Header {
                size: new_size,
                epoch,
            });
        }
        // SAFETY: `old_base` and `old_padded` are the original allocation parameters.
        unsafe { System.dealloc(old_base, old_padded) };
        // SAFETY: `new_base` points to a block with `new_header` bytes before the user region.
        unsafe { new_base.add(new_header) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn reset_counters() {
    LIVE_BYTES.store(0, Ordering::Relaxed);
    let _ = EPOCH.fetch_add(1, Ordering::Relaxed);
}

fn begin_counting() {
    COUNT_ON_THIS_THREAD.set(true);
}

fn end_counting() -> i64 {
    COUNT_ON_THIS_THREAD.set(false);
    LIVE_BYTES.load(Ordering::Relaxed)
}

struct TempModels {
    path: PathBuf,
}

impl TempModels {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "quarrel-leak-gate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create temporary models directory");
        for id in MODEL_IDS {
            fs::write(path.join(format!("{id}.gguf")), b"dummy gguf fixture")
                .expect("write dummy GGUF file");
        }
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempModels {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn cycle(runtime: &Runtime, metrics: &Metrics) {
    let models = TempModels::new();
    let config = Config {
        listen: "127.0.0.1:0".parse().expect("socket address"),
        models_dir: models.path().to_path_buf(),
        log_json: false,
        api_key: None,
        print_openapi: false,
    };

    let state = AppState::new(config, metrics.clone());
    let discovered = state.service.models().expect("discover dummy models");
    assert_eq!(discovered.data.len(), MODEL_IDS.len());
    for id in MODEL_IDS {
        assert!(
            state.service.validate_model(id).is_ok(),
            "model {id} must resolve"
        );
    }

    let app = build_router(state);
    let body = runtime.block_on(async {
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("build model lookup request"),
            )
            .await
            .expect("route model lookup request");
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read model lookup response")
    });
    let body = String::from_utf8_lossy(&body);
    for id in MODEL_IDS {
        assert!(body.contains(id), "model lookup lost {id}: {body}");
    }
}

#[test]
fn sixteen_cycles_keep_live_heap_growth_below_five_megabytes() {
    let runtime = Runtime::new().expect("tokio runtime");
    let metrics = Metrics::detached().expect("detached Prometheus handle");

    for _ in 0..WARMUP_CYCLES {
        cycle(&runtime, &metrics);
    }

    reset_counters();
    let mut samples = [0_i64; MEASURED_CYCLES];
    for sample in &mut samples {
        begin_counting();
        cycle(&runtime, &metrics);
        *sample = end_counting();
    }

    for (index, growth) in samples.iter().enumerate() {
        assert!(
            *growth < MAX_GROWTH_BYTES,
            "cycle {index} retained {growth} live bytes (limit {MAX_GROWTH_BYTES}); samples: {samples:?}"
        );
    }

    let half = MEASURED_CYCLES / 2;
    let first_avg = samples[..half].iter().sum::<i64>() as f64 / half as f64;
    let second_avg = samples[half..].iter().sum::<i64>() as f64 / (MEASURED_CYCLES - half) as f64;
    assert!(
        second_avg <= first_avg + TREND_TOLERANCE_BYTES as f64,
        "second-half average ({second_avg:.0} B) exceeds first-half average ({first_avg:.0} B) \
         by more than {TREND_TOLERANCE_BYTES} B; samples: {samples:?}"
    );
}
