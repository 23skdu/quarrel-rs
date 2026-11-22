//! Small-shape smoke test for the 250k x 384 Arrow/IPC benchmark path.

use std::sync::Arc;

use arrow::{
    array::{FixedSizeListArray, Float32Array, Int32Builder, StringBuilder},
    datatypes::{DataType, Field},
    record_batch::RecordBatch,
};
use quarrel_flight::{
    batch_to_ipc, embedding_matrix_values, embedding_response_schema_for, ipc_to_batch,
};

const ROWS: usize = 8;
const DIM: usize = 16;

fn matrix_batch(rows: usize, dim: usize) -> RecordBatch {
    let values = Float32Array::from_iter_values((0..rows * dim).map(|index| (index % dim) as f32));
    let embeddings = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        i32::try_from(dim).unwrap(),
        Arc::new(values),
        None,
    )
    .unwrap();
    let mut request_ids = StringBuilder::new();
    let mut token_ids = Int32Builder::new();
    for row in 0..rows {
        request_ids.append_value("smoke");
        token_ids.append_value(i32::try_from(row).unwrap());
    }
    RecordBatch::try_new(
        embedding_response_schema_for(dim),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(token_ids.finish()),
            Arc::new(embeddings),
        ],
    )
    .unwrap()
}

#[test]
fn small_float_matrix_ipc_round_trip() {
    let batch = matrix_batch(ROWS, DIM);
    let ipc = batch_to_ipc(&batch).unwrap();
    let decoded = ipc_to_batch(&ipc, embedding_response_schema_for(DIM)).unwrap();
    let values = embedding_matrix_values(&decoded).unwrap();
    assert_eq!(values.len(), ROWS * DIM);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[DIM - 1], (DIM - 1) as f32);
    assert_eq!(values[(ROWS - 1) * DIM], 0.0);
}
