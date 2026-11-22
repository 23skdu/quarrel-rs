//! Integration coverage for the response builder pool and batch streams.

use std::sync::Arc;

use arrow::{
    array::{Array, Int32Array},
    datatypes::{DataType, Field, Schema},
};
use futures::{StreamExt, executor::block_on};
use quarrel_core::{RequestId, Token};
use quarrel_flight::{
    FlightError, RecordBatch, ResponseBatchPool, decode_generate_response, record_batch_stream,
    record_batch_stream_from_iter,
};

fn int_batch(values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))])
        .expect("one-column int batch")
}

fn int_values(batch: &RecordBatch) -> Vec<i32> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int column")
        .values()
        .to_vec()
}

fn first_value_ptr(batch: &RecordBatch) -> *const i32 {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int column")
        .values()
        .as_ptr()
}

#[test]
fn response_pool_acquires_releases_and_reuses_builders() {
    let pool = ResponseBatchPool::new(1).expect("capacity one pool");
    assert_eq!(pool.capacity(), 1);
    assert_eq!(pool.available(), 0);
    assert_eq!(pool.in_use(), 0);

    let request_id = RequestId::new("pool-test").expect("valid request id");
    let token = Token::new(7, -0.25);

    let mut first = pool.acquire().expect("first builder");
    assert_eq!(pool.in_use(), 1);
    let batch = first
        .encode(&request_id, 1, &token, true)
        .expect("encode response");
    assert_eq!(
        decode_generate_response(&batch)
            .expect("decodable response")
            .len(),
        1
    );
    first.release().expect("release builder");
    assert_eq!(pool.in_use(), 0);
    assert_eq!(pool.available(), 1);

    let mut reused = pool.acquire().expect("reused builder");
    assert_eq!(pool.in_use(), 1);
    assert_eq!(pool.available(), 0);
    reused.scratch_capacity().expect("live builder");
    reused
        .encode(&request_id, 2, &token, false)
        .expect("encode with reused builder");
    // Dropping the guard also returns the builder to the pool.
    drop(reused);
    assert_eq!(pool.in_use(), 0);
    assert_eq!(pool.available(), 1);
}

#[test]
fn response_pool_rejects_zero_capacity_exhaustion_and_invalid_release() {
    assert!(matches!(
        ResponseBatchPool::new(0),
        Err(FlightError::InvalidArgument(_))
    ));

    let pool = ResponseBatchPool::new(1).expect("capacity one pool");
    let guard = pool.acquire().expect("first builder");
    match pool.acquire() {
        Err(FlightError::PoolExhausted { capacity }) => assert_eq!(capacity, 1),
        other => panic!("expected pool exhaustion, got {other:?}"),
    }
    drop(guard);

    let other_pool = ResponseBatchPool::new(1).expect("second pool");
    let foreign = other_pool.acquire().expect("foreign builder");
    match pool.release(foreign) {
        Err(FlightError::InvalidBuilderRelease(_)) => {}
        other => panic!("expected invalid release, got {other:?}"),
    }
    // The wrongly released guard is returned to its actual pool rather than
    // leaking capacity there.
    assert_eq!(other_pool.in_use(), 0);
    assert_eq!(other_pool.available(), 1);
    other_pool
        .acquire()
        .expect("builder still usable after invalid release")
        .release()
        .expect("release to owning pool");
}

#[test]
fn record_batch_stream_yields_batches_in_order_without_copying_payload() {
    let first = int_batch(&[1, 2, 3]);
    let second = int_batch(&[4, 5]);
    let first_ptr = first_value_ptr(&first);
    let second_ptr = first_value_ptr(&second);

    let stream = record_batch_stream(vec![first, second]);
    let yielded = block_on(stream.collect::<Vec<_>>());

    assert_eq!(yielded.len(), 2);
    assert_eq!(int_values(&yielded[0]), vec![1, 2, 3]);
    assert_eq!(int_values(&yielded[1]), vec![4, 5]);
    assert_eq!(first_value_ptr(&yielded[0]), first_ptr);
    assert_eq!(first_value_ptr(&yielded[1]), second_ptr);
}

#[test]
fn record_batch_stream_from_iter_preserves_iterator_order() {
    let batches = vec![int_batch(&[9]), int_batch(&[8, 7]), int_batch(&[6])];
    let expected = batches.iter().map(int_values).collect::<Vec<_>>();
    let stream = record_batch_stream_from_iter(batches);
    let yielded = block_on(stream.collect::<Vec<_>>());
    let actual = yielded.iter().map(int_values).collect::<Vec<_>>();
    assert_eq!(actual, expected);
}
