//! Execution engine for `quarrel-rs`.
//!
//! This crate provides the pieces needed to turn a prompt into tokens without
//! depending on a particular model implementation:
//!
//! * [`sampler`] — deterministic, parameter-driven token sampling.
//! * [`grammar`] — deterministic pushdown-constrained decoding over token strings.
//! * [`kv`] — bounded paged KV caches, block tables and an LRU radix prefix tree.
//! * [`scheduler`] — deterministic iteration-level continuous batching.
//! * [`health`] — lightweight logit sanity auditing.
//! * [`backend`] — the token-context [`backend::Backend`] trait, a deterministic
//!   reference backend and a small `generate` driver.
//! * [`batch`] — a separate batch-oriented [`batch::Backend`] trait,
//!   [`batch::Batch`], [`batch::Logits`] and [`batch::KvCacheView`]. The old
//!   token-context trait remains unchanged; the new trait lives at a distinct
//!   path because batch execution also carries per-sequence KV state.
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
pub mod batch;
pub mod grammar;
pub mod health;
pub mod kv;
pub mod sampler;
pub mod scheduler;

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
    /// A batch was constructed without any sequence rows.
    #[error("batch must contain at least one sequence row")]
    EmptyBatch,
    /// A batch context row has zero tokens.
    #[error("batch context length must be greater than zero")]
    EmptyContext,
    /// Batch or logit rows do not have the declared width.
    #[error("batch shape mismatch: expected {expected}, got {actual}")]
    BatchShapeMismatch {
        /// Required number of entries, tokens or values.
        expected: usize,
        /// Number of entries actually supplied.
        actual: usize,
    },
    /// The same sequence id occurs more than once in a batch.
    #[error("duplicate sequence id {seq} in batch")]
    DuplicateSequence {
        /// Sequence id that appeared twice.
        seq: u64,
    },
    /// A row index is outside a batch or logits result.
    #[error("row {index} is outside a batch of {rows} rows")]
    RowOutOfRange {
        /// Row index supplied by the caller.
        index: usize,
        /// Number of available rows.
        rows: usize,
    },
    /// A KV cache geometry does not match the backend's expectation.
    #[error("KV geometry mismatch: expected {expected:?}, got {actual:?}")]
    KvGeometryMismatch {
        /// Geometry required by the backend.
        expected: crate::batch::KvGeometry,
        /// Geometry reported by the cache.
        actual: crate::batch::KvGeometry,
    },
    /// A KV layer index is outside the cache geometry.
    #[error("KV layer {layer} is outside a cache with {layers} layers")]
    UnknownKvLayer {
        /// Layer index supplied by the caller.
        layer: usize,
        /// Number of available layers.
        layers: usize,
    },
    /// A KV read starts beyond the stored length of a sequence.
    #[error("KV read {start}..{end} exceeds sequence {seq} holding {len} tokens")]
    KvRangeOutOfBounds {
        /// Sequence id being read.
        seq: u64,
        /// First requested token offset.
        start: usize,
        /// Exclusive end of the requested range.
        end: usize,
        /// Number of stored tokens.
        len: usize,
    },
    /// A KV write would leave an unwritten gap in a sequence.
    #[error("cannot write KV tokens at {start} for sequence {seq} holding {len} tokens")]
    KvWriteGap {
        /// Sequence id being written.
        seq: u64,
        /// Requested token offset.
        start: usize,
        /// Number of stored tokens.
        len: usize,
    },
    /// A KV block contains more tokens than the cache block size.
    #[error("KV block has {token_count} tokens (block size is {block_size})")]
    KvBlockTooLarge {
        /// Number of tokens in the supplied block.
        token_count: usize,
        /// Maximum number of tokens per block.
        block_size: usize,
    },
    /// KV key/value slices have the wrong length.
    #[error("KV block shape mismatch: expected {expected} values, got {actual}")]
    KvBlockShape {
        /// Required number of `f32` values in each slice.
        expected: usize,
        /// Larger of the two supplied slice lengths.
        actual: usize,
    },
    /// A bounded scheduler queue cannot accept another request.
    #[error("scheduler queue is full (depth {depth}, capacity {capacity})")]
    QueueFull {
        /// Current number of waiting requests.
        depth: usize,
        /// Maximum number of waiting requests.
        capacity: usize,
    },
    /// A shared-prefix request extended past the source sequence.
    #[error("cannot share {requested} prefix tokens from a sequence holding {available}")]
    InvalidPrefixLength {
        /// Number of prefix tokens the caller requested.
        requested: usize,
        /// Number of tokens available in the source sequence.
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
    /// A numeric kernel rejected an input the backend considered valid.
    #[error("numeric kernel in backend '{backend}' failed: {reason}")]
    KernelFailure {
        /// Backend that invoked the kernel.
        backend: &'static str,
        /// Human-readable kernel error.
        reason: String,
    },
    /// A logit index cannot be represented as an `i32` token id.
    #[error("logit index {index} is outside the i32 token-id range")]
    TokenIdOverflow {
        /// Index that could not be converted.
        index: usize,
    },
    /// A token id has no string in the active grammar alphabet.
    #[error("token id {token_id} has no grammar token string")]
    GrammarTokenString {
        /// Token id that could not be resolved.
        token_id: i32,
    },
    /// The grammar has already reached its accepting state.
    #[error("grammar is already accepting; token '{token}' cannot be consumed")]
    GrammarAlreadyAccepting {
        /// Token string that was offered after acceptance.
        token: String,
    },
    /// No token string in the alphabet is legal in the current grammar state.
    #[error("grammar state '{state}' has no allowed token in the supplied alphabet")]
    GrammarNoAllowedTokens {
        /// Grammar state that could not be satisfied.
        state: String,
    },
    /// A grammar references a state that was never declared.
    #[error("grammar references undeclared state '{state}'")]
    GrammarInvalidState {
        /// Name of the missing state.
        state: String,
    },
    /// Two transitions in one grammar state match the same token string.
    #[error("ambiguous grammar transition in state '{state}' for token '{token}'")]
    GrammarAmbiguous {
        /// Grammar state with overlapping transitions.
        state: String,
        /// Token string matched by more than one transition.
        token: String,
    },
    /// A flat object repeated a key the grammar already consumed.
    #[error("grammar rejected duplicate object key '{key}'")]
    GrammarDuplicateKey {
        /// Key that appeared twice.
        key: String,
    },
    /// The token string is not legal in the current grammar state.
    #[error("grammar state '{state}' rejected token '{token}'")]
    GrammarRejected {
        /// Grammar state that rejected the token.
        state: String,
        /// Offending token string.
        token: String,
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
