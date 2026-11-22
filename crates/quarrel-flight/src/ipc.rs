//! Arrow IPC stream helpers.
//!
//! These helpers are deliberately defensive: `ipc_to_batch` accepts arbitrary
//! bytes without panicking, as required by the Flight fuzz target.

use std::{
    io::Cursor,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use arrow::{
    datatypes::Schema,
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};

use crate::error::FlightError;

/// Serialise a record batch as an Arrow IPC stream.
pub fn batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, FlightError> {
    let mut writer = StreamWriter::try_new(Vec::new(), batch.schema().as_ref())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(writer.into_inner()?)
}

/// Deserialise an Arrow IPC stream containing exactly one record batch.
///
/// The embedded schema must match `schema` (field metadata and dictionary ids
/// aside, as those are not part of the Flight contract). Arbitrary bytes yield
/// an [`Err`]; no input can make this function panic.
pub fn ipc_to_batch(bytes: &[u8], schema: Arc<Schema>) -> Result<RecordBatch, FlightError> {
    catch_unwind(AssertUnwindSafe(|| decode_one(bytes, schema)))
        .map_err(|_| FlightError::InvalidPayload("decoder panicked on malformed input".into()))?
}

fn decode_one(bytes: &[u8], schema: Arc<Schema>) -> Result<RecordBatch, FlightError> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|err| FlightError::InvalidPayload(err.to_string()))?;

    let actual = reader.schema();
    if actual.as_ref() != schema.as_ref() {
        return Err(FlightError::SchemaMismatch {
            expected: crate::schema_hash(&schema),
            actual: crate::schema_hash(&actual),
        });
    }

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|err| FlightError::InvalidPayload(err.to_string()))?);
    }
    match batches.len() {
        1 => Ok(batches.remove(0)),
        0 => Err(FlightError::InvalidPayload(
            "IPC stream has no batches".into(),
        )),
        n => Err(FlightError::InvalidPayload(format!(
            "IPC stream has {n} batches, expected exactly one"
        ))),
    }
}
