//! 250k x 384 f32 Arrow/IPC round-trip benchmark.
//!
//! The fixture is a canonical `FixedSizeList<Float32>` embedding response with
//! 250_000 rows and width 384 (96M f32 values, ~384 MiB). The benchmark times
//! IPC encoding and IPC decode plus a borrowed slice read; it never converts
//! the payload into a `Vec<Token>` or through JSON.

use std::{hint::black_box, sync::Arc, time::Duration};

use arrow::{
    array::{FixedSizeListArray, Float32Array, Int32Builder, RecordBatch, StringBuilder},
    datatypes::{DataType, Field},
};
use criterion::{Criterion, criterion_group, criterion_main};
use quarrel_flight::{
    batch_to_ipc, embedding_matrix_values, embedding_response_schema_for, ipc_to_batch,
};

const ROWS: usize = 250_000;
const DIM: usize = 384;

fn matrix_batch(rows: usize, dim: usize) -> RecordBatch {
    let values = Float32Array::from_iter_values((0..rows * dim).map(|index| {
        let column = index % dim;
        (column as f32) * 0.001
    }));
    let embeddings = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        i32::try_from(dim).expect("dimension fits in i32"),
        Arc::new(values),
        None,
    )
    .expect("valid fixed-size list");

    let mut request_ids = StringBuilder::with_capacity(rows, rows.saturating_mul(8));
    let mut token_ids = Int32Builder::with_capacity(rows);
    for row in 0..rows {
        request_ids.append_value("bench");
        token_ids.append_value(i32::try_from(row).unwrap_or(i32::MAX));
    }

    RecordBatch::try_new(
        embedding_response_schema_for(dim),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(token_ids.finish()),
            Arc::new(embeddings),
        ],
    )
    .expect("valid matrix batch")
}

fn arrow_roundtrip(criterion: &mut Criterion) {
    let batch = matrix_batch(ROWS, DIM);
    let ipc = batch_to_ipc(&batch).expect("encode benchmark fixture");
    let schema = embedding_response_schema_for(DIM);

    let mut group = criterion.benchmark_group("arrow_f32_matrix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("encode_ipc_250k_x_384", |bencher| {
        bencher.iter(|| {
            let encoded = batch_to_ipc(black_box(&batch)).expect("encode batch");
            black_box(encoded.len());
        });
    });
    group.bench_function("decode_ipc_and_borrow_slices_250k_x_384", |bencher| {
        bencher.iter(|| {
            let decoded = ipc_to_batch(black_box(&ipc), schema.clone()).expect("decode batch");
            let values = embedding_matrix_values(&decoded).expect("borrow values");
            black_box(values.len());
            black_box(values[0]);
        });
    });
    group.finish();
}

criterion_group!(benches, arrow_roundtrip);
criterion_main!(benches);
