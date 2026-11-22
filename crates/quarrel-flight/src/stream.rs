//! Streaming helpers for Arrow record batches.

use std::pin::Pin;

use arrow::record_batch::RecordBatch;
use futures::{Stream, stream};

/// A boxed, sendable stream of owned Arrow [`RecordBatch`] values.
///
/// Batches are *moved* into the stream by the helpers in this module. Their
/// column `Buffer`s are reference-counted Arrow buffers, not copied payloads,
/// so streaming a batch does not duplicate token or embedding data.
pub type RecordBatchStream = Pin<Box<dyn Stream<Item = RecordBatch> + Send>>;

/// Stream a vector of record batches in order without cloning payload buffers.
///
/// The vector is consumed by the returned stream; each batch is yielded once.
pub fn record_batch_stream(batches: Vec<RecordBatch>) -> RecordBatchStream {
    Box::pin(stream::iter(batches))
}

/// Stream any owned iterator of record batches without cloning payload buffers.
///
/// This is the generic form of [`record_batch_stream`]. The iterator is moved
/// into the returned stream and polled lazily.
pub fn record_batch_stream_from_iter<I>(batches: I) -> RecordBatchStream
where
    I: IntoIterator<Item = RecordBatch> + Send + 'static,
    I::IntoIter: Send,
{
    Box::pin(stream::iter(batches))
}
