//! Execution engine for `quarrel-rs`.
//!
//! This crate provides the pieces needed to turn a prompt into tokens without
//! depending on a particular model implementation:
//!
//! * [`sampler`] — deterministic, parameter-driven token sampling.
//! * [`kv`] — bounded paged KV caches and an LRU radix prefix cache.
//! * [`health`] — lightweight logit sanity auditing.
//! * [`backend`] — the [`backend::Backend`] trait, a deterministic reference
//!   backend and a small `generate` driver.
//!
//! The reference backend is a test fixture, **not** a Go-parity model. Use a
//! real backend implementation when logits must match the Go engine.

#![deny(missing_docs)]

// These dependencies are reserved for the model-backed backend that will
// replace `ReferenceBackend`; naming them keeps the dependency contract
// explicit without pulling them into the current public API.
use quarrel_gguf as _;
use quarrel_simd as _;
use smallvec as _;

pub mod backend;
pub mod health;
pub mod kv;
pub mod sampler;

/// Converts a numeric field into `usize` without assuming a particular
/// integer width in `quarrel-core`.
pub(crate) fn to_usize<T: TryInto<usize>>(value: T) -> Result<usize, EngineError> {
    value.try_into().map_err(|_| EngineError::InvalidParams {
        reason: "numeric value is not representable as usize",
    })
}

/// Errors produced by the engine.
///
/// The enum is non-exhaustive so new typed failures can be added without a
/// breaking change. Callers should use pattern matching with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A sampling or softmax operation was given an empty logit slice.
    #[error("logits are empty")]
    EmptyLogits,
    /// A logit was not finite, or became non-finite during sampling.
    #[error("logit at index {index} is not finite")]
    InvalidLogits {
        /// Zero-based index of the offending logit.
        index: usize,
    },
    /// `quarrel_core` rejected the supplied sampling parameters.
    #[error("invalid sampling parameters: {reason}")]
    InvalidParams {
        /// Short, stable description of the rejected input.
        reason: &'static str,
    },
    /// Every available token was a stop token, so no token could be returned.
    #[error("stop token {token_id} was sampled and no alternative was available")]
    StopTokenSampled {
        /// Token id that would have been emitted.
        token_id: i32,
    },
    /// A paged cache was constructed with a zero block size.
    #[error("KV cache block size must be greater than zero")]
    InvalidBlockSize,
    /// A paged cache has no free block left for the requested token count.
    #[error("KV cache cannot hold {requested} tokens (capacity {capacity} tokens)")]
    CapacityExceeded {
        /// Total number of tokens the cache can hold.
        capacity: usize,
        /// Number of tokens that could not be accommodated.
        requested: usize,
    },
    /// The sequence id is unknown or has already been freed.
    #[error("unknown KV sequence {seq}")]
    UnknownSequence {
        /// Sequence id supplied by the caller.
        seq: u64,
    },
    /// A rollback requested more tokens than the sequence currently holds.
    #[error("cannot roll back {requested} tokens from a sequence holding {available}")]
    InvalidRollback {
        /// Number of tokens the caller asked to remove.
        requested: usize,
        /// Number of tokens actually available.
        available: usize,
    },
    /// A paged cache invariant was violated internally.
    #[error("KV cache invariant violated: {reason}")]
    CacheInvariant {
        /// Short, stable description of the violated invariant.
        reason: &'static str,
    },
    /// The backend returned logits the engine refuses to sample from.
    #[error("backend '{backend}' returned unhealthy logits: {reason}")]
    BackendLogits {
        /// Name reported by [`backend::Backend::name`].
        backend: String,
        /// Human-readable reason from the logit audit.
        reason: String,
    },
    /// A logit index cannot be represented as an `i32` token id.
    #[error("logit index {index} is outside the i32 token-id range")]
    TokenIdOverflow {
        /// Index that could not be converted.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::EngineError;

    #[test]
    fn errors_have_readable_messages() {
        assert_eq!(EngineError::EmptyLogits.to_string(), "logits are empty");
        assert!(
            EngineError::InvalidLogits { index: 7 }
                .to_string()
                .contains('7')
        );
        assert!(
            EngineError::UnknownSequence { seq: 3 }
                .to_string()
                .contains('3')
        );
    }
}
