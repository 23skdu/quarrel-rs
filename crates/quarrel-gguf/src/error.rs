//! Typed errors returned by the GGUF reader and quant kernels.

use crate::dtype::DType;
use thiserror::Error;

/// All fallible operations in this crate report errors through this enum.
///
/// The enum is `#[non_exhaustive]` so new error variants can be added without a
/// breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GgufError {
    /// The underlying file could not be opened, read or mapped.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The input ended before the requested number of bytes was available.
    #[error("truncated input at byte {offset}: {needed} more bytes required")]
    Truncated {
        /// Absolute offset in the input where the read failed.
        offset: u64,
        /// Number of bytes still required.
        needed: u64,
    },

    /// The file does not start with the ASCII bytes `GGUF`.
    #[error("invalid GGUF magic (expected b\"GGUF\")")]
    InvalidMagic,

    /// The version field is not 1, 2 or 3.
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),

    /// A metadata entry uses a value type id this reader does not know.
    #[error("unsupported GGUF metadata value type {0}")]
    UnsupportedValueType(u32),

    /// A quant kernel was asked to process a dtype it cannot dequantize.
    #[error("unsupported dtype for this operation: {0:?}")]
    UnsupportedDType(DType),

    /// A length-prefixed string was not valid UTF-8.
    #[error("invalid UTF-8 in GGUF string: {0}")]
    InvalidString(#[from] std::str::Utf8Error),

    /// A tensor's relative offset or byte length points outside the file.
    #[error(
        "tensor {name:?} has invalid data range: data_offset={data_offset}, \
         relative_offset={offset}, byte_len={byte_len}, file_len={file_len}"
    )]
    BadTensorOffset {
        /// Tensor name, for diagnostics.
        name: String,
        /// 32-byte-aligned start of the tensor data section.
        data_offset: u64,
        /// Tensor offset relative to `data_offset`.
        offset: u64,
        /// Computed byte length of the tensor.
        byte_len: u64,
        /// Total input length.
        file_len: u64,
    },

    /// A shape has more than four dimensions, a zero dimension, or overflows.
    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    /// A tensor name was not present in the header.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// Checked integer arithmetic overflowed while parsing a header.
    #[error("GGUF header arithmetic overflow")]
    Overflow,

    /// A metadata array claims more elements than the input could possibly hold.
    #[error("metadata array of {len} elements exceeds the {remaining} bytes available")]
    ArrayTooLong {
        /// Declared array length.
        len: u64,
        /// Number of bytes left in the input.
        remaining: u64,
    },

    /// GGUF arrays may not contain other arrays.
    #[error("nested metadata arrays are not supported")]
    NestedArray,

    /// A byte buffer is not a whole number of quant blocks for its dtype.
    #[error("invalid byte length {actual}: expected a multiple of {block_bytes}")]
    InvalidBlockLength {
        /// Block size in bytes for the requested dtype.
        block_bytes: usize,
        /// Actual buffer length.
        actual: usize,
    },

    /// A caller-provided output buffer is too small.
    #[error("output buffer too small: need {needed} f32 values, got {actual}")]
    OutputTooSmall {
        /// Required number of `f32` values.
        needed: usize,
        /// Number of values provided.
        actual: usize,
    },

    /// A dot-product input has a different number of elements than the blocks.
    #[error("length mismatch: quant data holds {quant} elements, lhs holds {lhs}")]
    LengthMismatch {
        /// Number of elements implied by the quantized bytes.
        quant: usize,
        /// Number of elements in the floating-point operand.
        lhs: usize,
    },

    /// A matvec weight buffer does not match the shape implied by its operands.
    #[error(
        "invalid matvec shape: {actual} weight bytes for {rows} rows x {cols} cols \
         (expected {expected})"
    )]
    InvalidMatvecShape {
        /// Byte length required by the shape.
        expected: usize,
        /// Actual byte length of the weight buffer.
        actual: usize,
        /// Number of rows (output length).
        rows: usize,
        /// Number of columns (input length).
        cols: usize,
    },

    /// A kernel needs finite values but found a NaN or infinity.
    #[error("non-finite {role} at index {index}: {value}")]
    NonFiniteValue {
        /// Operand the value came from (`"input"` or `"weights"`).
        role: &'static str,
        /// Logical element index within that operand.
        index: usize,
        /// The offending value.
        value: f32,
    },

    /// An embedding row index is outside the tensor.
    #[error("row index {index} out of range (tensor has {rows} rows)")]
    RowOutOfRange {
        /// Requested row.
        index: usize,
        /// Number of rows in the tensor.
        rows: usize,
    },

    /// A quant block or dtype-specific invariant was violated by the caller.
    #[error("invalid quant argument: {0}")]
    InvalidQuantArgument(String),
}
