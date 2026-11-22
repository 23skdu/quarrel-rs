//! Allocation-counting proof for the zero-copy Flight adapter path.
//!
//! This test binary installs a test-only counting global allocator. Counting is
//! opt-in per thread (a thread-local flag) and every test takes a shared mutex
//! around its measured region, so parallel tests cannot pollute each other.
//!
//! Exactly what is counted:
//!
//! * calls to `alloc` and `realloc` on the test thread while counting is on;
//! * the requested layout size of those calls (a `realloc` counts as one new
//!   allocation of `new_size`, not as a delta);
//! * "large" allocations whose requested size is at least 1 KiB, used as a
//!   proxy for payload buffers rather than small metadata.
//!
//! Deallocation is forwarded to `System` and is not counted. IPC decoding
//! necessarily materialises Arrow buffers, so the decode phase records those
//! payload copies and bounds them. The adapter read phase is then measured
//! with counters reset: borrowing token/KV slices must allocate **nothing**.
//! This proves the adapter performs no payload-copy allocations after the
//! one-time IPC materialisation; it does not claim that IPC decoding itself
//! is zero-copy.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use quarrel_core::{
    BlockId, GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams,
};
use quarrel_flight::{
    GenerateResponseBuilder, batch_to_ipc, encode_generate_request, encode_kv_block,
    generate_request_schema, ipc_to_batch, kv_block_payload, kv_blocks_schema_for,
    request_token_ids,
};

const LARGE_ALLOCATION_BYTES: usize = 1024;
const TOKEN_COUNT: usize = 1024;
const KV_BYTES: usize = 4096;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

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

fn record(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    if size >= LARGE_ALLOCATION_BYTES {
        LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    allocations: usize,
    bytes: usize,
    large: usize,
}

fn start_counting() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LARGE_ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNT_THIS_THREAD.set(true);
    COUNTING.store(true, Ordering::Relaxed);
}

fn stop_counting() -> Counts {
    COUNTING.store(false, Ordering::Relaxed);
    COUNT_THIS_THREAD.set(false);
    Counts {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        large: LARGE_ALLOCATIONS.load(Ordering::Relaxed),
    }
}

fn lock_test() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

fn request(ipc: bool) -> GenerateRequest {
    let token_ids = (0..TOKEN_COUNT as i32).collect::<Vec<_>>();
    GenerateRequest::new(
        RequestId::new("alloc-test").unwrap(),
        ModelId::new("reference").unwrap(),
        if ipc { token_ids } else { vec![token_ids[0]] },
        SamplingParams::default(),
        true,
    )
    .unwrap()
}

#[test]
fn ipc_decode_then_borrowed_token_and_kv_slices_copy_no_payload_after_warmup() {
    let _guard = lock_test();
    let request = request(true);
    let request_batch = encode_generate_request(&request).unwrap();
    let request_ipc = batch_to_ipc(&request_batch).unwrap();
    let block = KvBlock::new(
        RequestId::new("alloc-test").unwrap(),
        BlockId::new(1),
        0,
        KvDtype::F32,
        vec![3_u8; KV_BYTES],
    );
    let kv_ipc = batch_to_ipc(&encode_kv_block(&block).unwrap()).unwrap();

    for _ in 0..8 {
        let warm = ipc_to_batch(&request_ipc, generate_request_schema()).unwrap();
        assert_eq!(request_token_ids(&warm).unwrap().len(), TOKEN_COUNT);
        let warm_kv = ipc_to_batch(&kv_ipc, kv_blocks_schema_for(KV_BYTES)).unwrap();
        assert_eq!(kv_block_payload(&warm_kv).unwrap().len(), KV_BYTES);
    }

    // Phase 1: Arrow IPC materialisation. Metadata allocations are allowed;
    // the two payload buffers (4 KiB and 16 KiB) are the only large copies we
    // expect, with a small allowance for reader scratch buffers.
    start_counting();
    let decoded_request = ipc_to_batch(&request_ipc, generate_request_schema()).unwrap();
    let decoded_kv = ipc_to_batch(&kv_ipc, kv_blocks_schema_for(KV_BYTES)).unwrap();
    let decode = stop_counting();
    let payload_bytes = TOKEN_COUNT * std::mem::size_of::<i32>() + KV_BYTES;
    assert!(
        decode.large >= 2,
        "IPC decode should materialise both payload buffers, got {decode:?}"
    );
    assert!(
        decode.large <= 8,
        "IPC decode made unexpected large allocations: {decode:?}"
    );
    assert!(
        decode.bytes <= payload_bytes * 2 + 65_536,
        "IPC decode exceeded bounded-copy budget ({payload_bytes} payload bytes): {decode:?}"
    );

    // Phase 2: adapter reads only. Any allocation here would be a payload copy
    // or accidental materialisation of a token/byte Vec.
    start_counting();
    let ids = request_token_ids(&decoded_request).unwrap();
    let payload = kv_block_payload(&decoded_kv).unwrap();
    let id_sum: i64 = ids.iter().map(|value| i64::from(*value)).sum();
    let byte_sum: u64 = payload.iter().map(|value| u64::from(*value)).sum();
    let reads = stop_counting();
    let expected_id_sum: i64 = (0..TOKEN_COUNT as i64).sum();
    assert_eq!(id_sum, expected_id_sum);
    assert_eq!(byte_sum, 3 * KV_BYTES as u64);
    assert_eq!(
        reads,
        Counts {
            allocations: 0,
            bytes: 0,
            large: 0
        },
        "borrowed slice reads must not allocate"
    );

    // The slices point into the decoded Arrow buffers, not into fresh copies.
    let raw_ids = decoded_request
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .unwrap();
    let raw_ints = raw_ids
        .values()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    assert_eq!(ids.as_ptr(), raw_ints.values().as_ptr());
    let raw_list = decoded_kv
        .column(4)
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
        .unwrap();
    let raw_bytes = raw_list
        .values()
        .as_any()
        .downcast_ref::<arrow::array::UInt8Array>()
        .unwrap();
    assert_eq!(payload.as_ptr(), raw_bytes.values().as_ptr());
}

#[test]
fn response_builder_scratch_stops_growing_after_warmup() {
    let _guard = lock_test();
    let id = RequestId::new("builder-alloc-test").unwrap();
    let mut builder = GenerateResponseBuilder::new().unwrap();
    let token = quarrel_core::Token::new(9, -0.5);
    for index in 0..64 {
        builder.encode(&id, index, &token, false).unwrap();
    }
    let warm_capacity = builder.scratch_capacity();

    start_counting();
    let batch = builder.encode(&id, 65, &token, true).unwrap();
    let steady = stop_counting();

    assert_eq!(builder.scratch_capacity(), warm_capacity);
    assert_eq!(
        quarrel_flight::decode_generate_response(&batch)
            .unwrap()
            .len(),
        1
    );
    // Output Arrow buffers must still be allocated; the assertion is that no
    // additional scratch growth occurred (checked above via capacity).
    assert!(
        steady.allocations > 0,
        "owned output buffers must be allocated"
    );
    assert!(
        steady.bytes < 16_384,
        "one small response should not need large scratch allocations: {steady:?}"
    );
}
