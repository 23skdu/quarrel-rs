//! Typed errors for GGUF model loading, tokenization and generation.

use quarrel_gguf::GgufError;
use thiserror::Error;

/// Errors returned by the native runtime.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// The underlying GGUF reader rejected the file or a tensor view.
    #[error(transparent)]
    Gguf(#[from] GgufError),

    /// A required metadata key was absent or had an unexpected type.
    #[error("invalid metadata key {key:?}: {reason}")]
    Metadata {
        /// Metadata key that failed validation.
        key: String,
        /// Reason the value was rejected.
        reason: String,
    },

    /// The GGUF advertises an architecture this runtime cannot execute.
    #[error("unsupported GGUF architecture {0:?}")]
    UnsupportedArchitecture(String),

    /// A tensor shape did not match the model configuration.
    #[error("tensor {name:?} has shape {shape:?}, expected {expected:?}")]
    TensorShape {
        /// Tensor name from the GGUF descriptor.
        name: String,
        /// Shape found in the file.
        shape: Vec<u64>,
        /// Shape required by the runtime.
        expected: Vec<u64>,
    },

    /// A numeric kernel reported an invalid value.
    #[error("numeric kernel failed in {operation}: {reason}")]
    Numeric {
        /// Operation that failed.
        operation: &'static str,
        /// Failure description.
        reason: String,
    },

    /// A prompt or generated sequence exceeded the configured context.
    #[error("context length {actual} exceeds configured limit {limit}")]
    ContextLength {
        /// Requested number of tokens.
        actual: usize,
        /// Maximum supported number of tokens.
        limit: usize,
    },

    /// A token id was outside the model vocabulary.
    #[error("token id {id} is outside vocabulary of size {vocab_size}")]
    TokenId {
        /// Invalid token id.
        id: u32,
        /// Number of tokens in the vocabulary.
        vocab_size: usize,
    },

    /// The tokenizer metadata was malformed.
    #[error("invalid tokenizer metadata: {0}")]
    Tokenizer(String),

    /// Generation parameters were inconsistent.
    #[error("invalid generation parameters: {0}")]
    Generation(String),
}
