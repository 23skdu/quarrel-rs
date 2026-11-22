//! Allocation guard for a small CPU batch forward pass.

use quarrel_backend_cpu::{CpuBackend, KernelMode};
use quarrel_engine::batch::{Backend, Batch, InMemoryKvCache};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let _ = PEAK_ALLOCATION.fetch_max(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed);
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            let _ = PEAK_ALLOCATION.fetch_max(new_size, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn small_forward_has_no_unexpected_large_allocation() {
    let backend = CpuBackend::with_mode(32, 16, 8, KernelMode::Scalar).expect("backend");
    let batch = Batch::from_rows([(1_u64, vec![1, 2, 3]), (2_u64, vec![4, 5, 6])]).expect("batch");
    let mut kv = InMemoryKvCache::new(backend.kv_geometry()).expect("cache");
    for seq in batch.seq_ids() {
        kv.ensure_sequence(*seq);
    }

    PEAK_ALLOCATION.store(0, Ordering::Relaxed);
    let before = LIVE_BYTES.load(Ordering::Relaxed);
    let logits = backend.forward(&batch, &mut kv).expect("logits");
    let peak = PEAK_ALLOCATION.load(Ordering::Relaxed);
    assert!(
        peak <= 64 * 1024,
        "largest single allocation was {peak} bytes"
    );
    drop(logits);
    drop(kv);
    drop(batch);
    let after = LIVE_BYTES.load(Ordering::Relaxed);
    assert!(
        after <= before + 4096,
        "live bytes drifted from {before} to {after}"
    );
}
