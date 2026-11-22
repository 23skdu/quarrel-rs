//! GGUF v1-v3 reader, deterministic writer and allocation-free quant kernels.
//!
//! [`writer`] builds canonical GGUF v1-v3 files from typed metadata and encoded
//! tensor payloads. Its [`writer::requantize_gguf`] helper converts F32/F16
//! tensors to F32, F16, Q8_0, Q4_K or Q6_K while preserving the input version,
//! metadata and tensor layout. The `quarrel-quantize` binary wraps that helper:
//! `quarrel-quantize INPUT OUTPUT DTYPE`.
//!
//! The parser is built entirely from bounds-checked little-endian slice reads;
//! it never panics on untrusted input. [`GgufFile`] parses bytes borrowed from
//! any source, while [`GgufMmap`] owns a read-only memory map of a file on disk.
//! [`matvec`] exposes allocation-free row-wise matrix-vector products for
//! F32, F16, BF16, Q8_0, Q4_0 and all five K-quant formats (Q2_K, Q3_K,
//! Q4_K, Q5_K, Q6_K), using the canonical ggml block layouts. Unknown dtype
//! ids still report a typed [`GgufError::UnsupportedDType`].
//!
//! ```
//! use quarrel_gguf::{DType, GgufFile, quant};
//!
//! # fn example(bytes: &[u8]) -> Result<(), quarrel_gguf::GgufError> {
//! let file = GgufFile::from_bytes(bytes)?;
//! let tensor = file.tensor("token_embeddings.weight")?;
//! if tensor.dtype() == DType::Q8_0 {
//!     let mut row = vec![0.0f32; 32];
//!     quant::embedding_lookup_q8_0(tensor.as_bytes(), 0, 32, &mut row)?;
//! }
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

mod dtype;
mod error;
mod file;
mod header;
pub mod matvec;
pub mod quant;
pub mod writer;

#[cfg(test)]
mod fixture;
#[cfg(test)]
mod tests;

pub use dtype::{DType, GgufVersion};
pub use error::GgufError;
pub use file::{GgufFile, GgufMmap, TensorView};
pub use header::{GgufHeader, MetadataValue, TensorDescriptor};
pub use writer::GgufWriter;
