//! GGUF v1-v3 reader with borrowed tensor views and allocation-free quant kernels.
//!
//! The parser is built entirely from bounds-checked little-endian slice reads;
//! it never panics on untrusted input. [`GgufFile`] parses bytes borrowed from
//! any source, while [`GgufMmap`] owns a read-only memory map of a file on disk.
//! [`matvec`] exposes allocation-free row-wise matrix-vector products for
//! F32, F16, BF16, Q8_0 and Q4_0 weights; K-quant formats report a typed
//! [`GgufError::UnsupportedDType`] rather than an invented layout.
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

mod dtype;
mod error;
mod file;
mod header;
pub mod matvec;
pub mod quant;

#[cfg(test)]
mod fixture;
#[cfg(test)]
mod tests;

pub use dtype::{DType, GgufVersion};
pub use error::GgufError;
pub use file::{GgufFile, GgufMmap, TensorView};
pub use header::{GgufHeader, MetadataValue, TensorDescriptor};
