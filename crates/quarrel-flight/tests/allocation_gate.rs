//! Consolidated P8-03 allocation gate for `quarrel-flight`.
//!
//! # What "zero allocations" means here
//!
//! After a deterministic warm-up pass, each measured region must perform
//! **zero payload-sized allocations on the test thread**. An allocation is
//! payload-sized when the requested layout size is at least
//! [`PAYLOAD_ALLOCATION_BYTES`] (4 KiB). `alloc` and `realloc` calls are counted;
//! deallocations are not. A `realloc` counts once at its new requested size.
//!
//! Everything below that threshold is fixed-size bookkeeping: Arrow field
//! metadata, FlatBuffer scratch, small `Vec`/`HashMap` capacities, IPC message
//! headers and per-frame IPC buffers. Bookkeeping is allowed but bounded: each
//! phase also declares an explicit cap on the number of allocations and on the
//! summed requested bytes. All fixtures are sized so that one complete IPC
//! frame stays below 4 KiB (the largest observed single allocation is ~2.7 KiB
//! for the 512-token generate request), which lets the gate assert zero
//! payload-sized allocations outright, including for `ipc_to_batch`. A future
//! change that introduces a bulk copy, an unbounded scratch buffer or a
//! per-request payload allocation of 4 KiB or more fails this gate.
//!
//! The counting allocator exists only in this integration-test binary. It is
//! gated by a process-wide flag *and* a thread-local flag, so harness threads
//! and other tests in this binary cannot be counted, and no global recorder is
//! installed in library code. A shared mutex serialises the measured regions.
//! There are no sleeps and no sockets: the Flight encoder is driven in process
//! with the same `RecordBatchStream` + `FlightDataEncoderBuilder` pipeline that
//! `QuarrelFlightService::do_get` builds.
//!
//! The older, more granular checks in `tests/zero_copy_alloc.rs` are preserved;
//! this file is the consolidated gate over both request schemas, builder reuse
//! and both response schemas.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use arrow::{array::Int32Array, array::StringArray, datatypes::SchemaRef};
use arrow_flight::{
    FlightData, encode::FlightDataEncoderBuilder, error::FlightError as ArrowFlightError, utils,
};
use futures::{StreamExt, executor::block_on};
use quarrel_core::{GenerateRequest, ModelId, RequestId, SamplingParams, Token};
use quarrel_flight::{
    GenerateResponseBuilder, RecordBatch, ResponseBatchPool, batch_to_ipc,
    decode_embed_request_rows, decode_generate_request_view, decode_generate_response,
    embed_request_schema, embed_response_schema, encode_embedding, encode_generate_request,
    generate_request_schema, generate_response_schema, ipc_to_batch, record_batch_stream,
};

/// Requested size at or above which an allocation is considered payload-sized.
const PAYLOAD_ALLOCATION_BYTES: usize = 4096;
/// Fixed number of warm-up passes before any counter is read.
const WARMUP_PASSES: usize = 8;
/// Tokens in the generate request fixture (2 KiB of token ids).
const TOKEN_COUNT: usize = 512;
/// Rows in the embed request fixture.
const EMBED_ROWS: usize = 64;
/// Response batches fed through the Flight `DoGet` encoder per schema.
const RESPONSE_BATCHES: usize = 8;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static PAYLOAD_SIZED: AtomicUsize = AtomicUsize::new(0);
static LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static COUNT_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) && COUNT_THIS_THREAD.with(Cell::get) {
            record(layout.size());
        }
        // SAFETY: forwarded with the exact layout supplied by the caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the pointer and layout supplied by the caller.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) && COUNT_THIS_THREAD.with(Cell::get) {
            record(new_size);
        }
        // SAFETY: forwarded with the pointer and layouts supplied by the caller.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size, Ordering::Relaxed);
    if size >= PAYLOAD_ALLOCATION_BYTES {
        PAYLOAD_SIZED.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    allocations: usize,
    bytes: usize,
    payload_sized: usize,
}

fn start_counting() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    PAYLOAD_SIZED.store(0, Ordering::Relaxed);
    COUNT_THIS_THREAD.set(true);
    COUNTING.store(true, Ordering::Relaxed);
}

fn stop_counting() -> Counts {
    COUNTING.store(false, Ordering::Relaxed);
    COUNT_THIS_THREAD.set(false);
    Counts {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
        payload_sized: PAYLOAD_SIZED.load(Ordering::Relaxed),
    }
}

fn lock_test() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Run `body` with counters reset, then enforce the gate for that phase.
///
/// The budgets are roughly twice the measured steady-state bookkeeping and are
/// rounded to powers of two. They bound small allocations so that trading one
/// payload-sized buffer for many small ones is still visible.
fn measured<T>(
    phase: &str,
    max_allocations: usize,
    max_bytes: usize,
    body: impl FnOnce() -> T,
) -> T {
    start_counting();
    let value = body();
    let counts = stop_counting();
    assert_eq!(
        counts.payload_sized, 0,
        "{phase} made a payload-sized allocation (>= {PAYLOAD_ALLOCATION_BYTES} bytes): {counts:?}"
    );
    assert!(
        counts.allocations <= max_allocations,
        "{phase} exceeded its {max_allocations} allocation bookkeeping budget: {counts:?}"
    );
    assert!(
        counts.bytes <= max_bytes,
        "{phase} exceeded its {max_bytes} byte bookkeeping budget: {counts:?}"
    );
    value
}

fn generate_request() -> GenerateRequest {
    GenerateRequest::new(
        RequestId::new("allocation-gate").unwrap(),
        ModelId::new("reference").unwrap(),
        (0..TOKEN_COUNT as i32).collect(),
        SamplingParams::default(),
        true,
    )
    .unwrap()
}

fn embed_request_batch() -> RecordBatch {
    let ids = (0..EMBED_ROWS)
        .map(|row| format!("embed-{row}"))
        .collect::<Vec<_>>();
    RecordBatch::try_new(
        embed_request_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Int32Array::from((0..EMBED_ROWS as i32).collect::<Vec<_>>())),
        ],
    )
    .unwrap()
}

/// Encode response batches with the exact stream/encoder combination built by
/// `QuarrelFlightService::do_get`, without a gRPC transport.
fn do_get_frames(batches: Vec<RecordBatch>, schema: SchemaRef) -> Vec<FlightData> {
    let input = record_batch_stream(batches).map(Ok::<_, ArrowFlightError>);
    let stream = FlightDataEncoderBuilder::new()
        .with_schema(schema)
        .build(input);
    block_on(stream.collect::<Vec<_>>())
        .into_iter()
        .map(|result| result.unwrap())
        .collect()
}

#[test]
fn ipc_decode_into_borrowed_generate_and_embed_inputs_is_gated() {
    let _guard = lock_test();
    let generate_ipc =
        batch_to_ipc(&encode_generate_request(&generate_request()).unwrap()).unwrap();
    let embed_ipc = batch_to_ipc(&embed_request_batch()).unwrap();

    for _ in 0..WARMUP_PASSES {
        let batch = ipc_to_batch(&generate_ipc, generate_request_schema()).unwrap();
        assert_eq!(
            decode_generate_request_view(&batch)
                .unwrap()
                .token_ids
                .len(),
            TOKEN_COUNT
        );
        let batch = ipc_to_batch(&embed_ipc, embed_request_schema()).unwrap();
        assert_eq!(decode_embed_request_rows(&batch).unwrap().len(), EMBED_ROWS);
    }

    let decoded_generate = ipc_to_batch(&generate_ipc, generate_request_schema()).unwrap();
    let decoded_embed = ipc_to_batch(&embed_ipc, embed_request_schema()).unwrap();

    // Adapter conversions borrow the prompt and return only bounded metadata.
    let view = measured("borrowed generate view", 16, 512, || {
        decode_generate_request_view(&decoded_generate).unwrap()
    });
    let expected_sum: i64 = (0..TOKEN_COUNT as i64).sum();
    assert_eq!(
        view.token_ids.iter().map(|id| i64::from(*id)).sum::<i64>(),
        expected_sum
    );
    // The slice points into the decoded Arrow buffer, not into a fresh copy.
    let lists = decoded_generate
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .unwrap();
    let values = lists
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(view.token_ids.as_ptr(), values.values().as_ptr());

    let rows = measured("borrowed embed rows", 256, 8192, || {
        decode_embed_request_rows(&decoded_embed).unwrap()
    });
    assert_eq!(rows.len(), EMBED_ROWS);
    assert_eq!(rows[0].0.as_str(), "embed-0");
    assert_eq!(rows[EMBED_ROWS - 1].1, EMBED_ROWS as i32 - 1);

    // Full Arrow IPC decode: one bounded body buffer per message, no bulk
    // scratch or second payload copy (see module docs for the threshold).
    let decoded = measured("generate IPC decode", 256, 32 * 1024, || {
        ipc_to_batch(&generate_ipc, generate_request_schema()).unwrap()
    });
    assert_eq!(
        decode_generate_request_view(&decoded).unwrap().token_ids,
        (0..TOKEN_COUNT as i32).collect::<Vec<_>>()
    );
    let decoded = measured("embed IPC decode", 128, 8192, || {
        ipc_to_batch(&embed_ipc, embed_request_schema()).unwrap()
    });
    assert_eq!(
        decode_embed_request_rows(&decoded).unwrap().len(),
        EMBED_ROWS
    );
}

#[test]
fn response_builder_reuse_is_gated() {
    let _guard = lock_test();
    let id = RequestId::new("builder-gate").unwrap();
    let token = Token::new(9, -0.125);

    let mut builder = GenerateResponseBuilder::new().unwrap();
    for index in 0..64 {
        builder.encode(&id, index, &token, false).unwrap();
    }
    let warm_capacity = builder.scratch_capacity();
    let batch = measured("reused builder encode", 32, 4096, || {
        builder.encode(&id, 65, &token, true).unwrap()
    });
    assert_eq!(builder.scratch_capacity(), warm_capacity);
    assert_eq!(decode_generate_response(&batch).unwrap().len(), 1);

    let pool = ResponseBatchPool::new(1).unwrap();
    let mut first = pool.acquire().unwrap();
    for index in 0..64 {
        first.encode(&id, index, &token, false).unwrap();
    }
    let pool_capacity = first.scratch_capacity().unwrap();
    first.release().unwrap();

    let batch = measured("pooled builder reuse", 32, 4096, || {
        let mut reused = pool.acquire().unwrap();
        let batch = reused.encode(&id, 65, &token, true).unwrap();
        assert_eq!(reused.scratch_capacity().unwrap(), pool_capacity);
        reused.release().unwrap();
        batch
    });
    assert_eq!(decode_generate_response(&batch).unwrap().len(), 1);
    assert_eq!(pool.in_use(), 0);
    assert_eq!(pool.available(), 1);
}

#[test]
fn do_get_payload_encoding_is_gated_for_generate_and_embed() {
    let _guard = lock_test();
    let id = RequestId::new("do-get-gate").unwrap();
    let token = Token::new(271, -0.25);
    let mut builder = GenerateResponseBuilder::new().unwrap();
    for index in 0..32 {
        builder.encode(&id, index, &token, false).unwrap();
    }
    let generate_batches = (0..RESPONSE_BATCHES)
        .map(|index| {
            builder
                .encode(&id, index as u64 + 1, &token, index + 1 == RESPONSE_BATCHES)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let embed_id = RequestId::new("do-get-embed-gate").unwrap();
    let embedding = vec![0.25f32; quarrel_flight::DEFAULT_EMBEDDING_DIM as usize];
    let embed_batches = (0..RESPONSE_BATCHES)
        .map(|index| encode_embedding(&embed_id, index as i32, &embedding).unwrap())
        .collect::<Vec<_>>();

    for _ in 0..WARMUP_PASSES / 2 {
        let frames = do_get_frames(generate_batches.clone(), generate_response_schema());
        assert_eq!(
            utils::flight_data_to_batches(&frames).unwrap().len(),
            RESPONSE_BATCHES
        );
        let frames = do_get_frames(embed_batches.clone(), embed_response_schema());
        assert_eq!(
            utils::flight_data_to_batches(&frames).unwrap().len(),
            RESPONSE_BATCHES
        );
    }

    let frames = measured("generate DoGet encoding", 2048, 256 * 1024, || {
        do_get_frames(generate_batches.clone(), generate_response_schema())
    });
    let decoded = utils::flight_data_to_batches(&frames).unwrap();
    assert_eq!(decoded.len(), RESPONSE_BATCHES);
    // The encoder hydrates the dictionary `finish_reason` column, so the
    // client-visible schema is not byte-identical to the server schema.
    let tokens = decoded
        .iter()
        .flat_map(|batch| decode_generate_response(batch).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(tokens.len(), RESPONSE_BATCHES);
    assert!(tokens.iter().all(|token| token.id == 271));

    let frames = measured("embed DoGet encoding", 1024, 128 * 1024, || {
        do_get_frames(embed_batches.clone(), embed_response_schema())
    });
    let decoded = utils::flight_data_to_batches(&frames).unwrap();
    assert_eq!(decoded.len(), RESPONSE_BATCHES);
    assert!(
        decoded
            .iter()
            .all(|batch| batch.schema() == embed_response_schema())
    );
}
