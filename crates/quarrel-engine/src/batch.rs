//! Batch-oriented, KV-aware backend abstraction.
//!
//! This module is deliberately separate from [`crate::backend`]. The original
//! [`crate::backend::Backend`] takes a single token context and returns one
//! logit row; it is a public API used by the existing reference fixture,
//! generation driver and scheduler, so it stays unchanged. A batch execution
//! backend needs a different contract: it consumes several context rows at
//! once and may read or update per-sequence KV state while computing them.
//! Exposing that contract as [`Backend`] under this distinct module path keeps
//! the old trait source-compatible while making the two execution models
//! explicit at the call site.
//!
//! The concrete cache is hidden behind [`KvCacheView`]. A backend only sees
//! geometry, sequence lengths and bounded block reads/writes, so CPU, CUDA or
//! test caches can be swapped without changing backend code.
//!
//! ```
//! use quarrel_engine::batch::{Batch, InMemoryKvCache, KvGeometry};
//!
//! let batch = Batch::from_rows([(7_u64, vec![1, 2, 3]), (8_u64, vec![4, 5, 6])])?;
//! let mut kv = InMemoryKvCache::new(KvGeometry::new(4, 1, 8))?;
//! for seq in batch.seq_ids() {
//!     kv.ensure_sequence(*seq);
//! }
//! assert_eq!(kv.len(7)?, 0);
//! # Ok::<(), quarrel_engine::EngineError>(())
//! ```

use crate::EngineError;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// Geometry of the KV blocks exposed by a [`KvCacheView`].
///
/// Keys and values are stored as `f32` values, laid out as
/// `layer -> token -> feature`. Both keys and values use `kv_dim` features per
/// token and layer. A cache with zero layers, zero tokens per block or zero
/// features per token is rejected by [`KvGeometry::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct KvGeometry {
    /// Maximum number of tokens in one block.
    pub block_size: usize,
    /// Number of model layers visible to the backend.
    pub layers: usize,
    /// Number of `f32` features in one key or value vector.
    pub kv_dim: usize,
}

impl KvGeometry {
    /// Creates a geometry.
    #[must_use]
    pub const fn new(block_size: usize, layers: usize, kv_dim: usize) -> Self {
        Self {
            block_size,
            layers,
            kv_dim,
        }
    }

    /// Returns the number of `f32` values in one full block of one layer.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] if the product overflows.
    pub fn block_values(&self) -> Result<usize, EngineError> {
        self.block_size
            .checked_mul(self.kv_dim)
            .ok_or(EngineError::InvalidParams {
                reason: "KV block size times kv_dim overflows usize",
            })
    }

    /// Validates that every dimension is non-zero and representable.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for a zero dimension or an
    /// overflowing geometry.
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.block_size == 0 || self.layers == 0 || self.kv_dim == 0 {
            return Err(EngineError::InvalidParams {
                reason: "KV geometry dimensions must be greater than zero",
            });
        }
        let per_layer = self.block_values()?;
        per_layer
            .checked_mul(self.layers)
            .ok_or(EngineError::InvalidParams {
                reason: "KV geometry overflows usize",
            })?;
        Ok(())
    }
}

/// One equal-width batch of context rows, with one sequence id per row.
///
/// A `Batch` is immutable after construction. Its dense row layout lets a
/// backend process all sequences with the same kernel shape, while the
/// sequence ids keep per-sequence KV state unambiguous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    seq_ids: Vec<u64>,
    tokens: Vec<i32>,
    context_len: usize,
}

impl Batch {
    /// Creates a batch from parallel sequence ids and row-major tokens.
    ///
    /// `tokens` must contain exactly `seq_ids.len() * context_len` entries.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyBatch`], [`EngineError::EmptyContext`],
    /// [`EngineError::BatchShapeMismatch`] or
    /// [`EngineError::DuplicateSequence`] for invalid input.
    pub fn new(
        seq_ids: Vec<u64>,
        tokens: Vec<i32>,
        context_len: usize,
    ) -> Result<Self, EngineError> {
        let batch = Self {
            seq_ids,
            tokens,
            context_len,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Creates a batch from `(sequence id, context tokens)` rows.
    ///
    /// Every row must have the same non-zero length.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Batch::new`].
    pub fn from_rows<I, R>(rows: I) -> Result<Self, EngineError>
    where
        I: IntoIterator<Item = (u64, R)>,
        R: AsRef<[i32]>,
    {
        let mut rows = rows.into_iter();
        let Some((first_seq, first_tokens)) = rows.next() else {
            return Err(EngineError::EmptyBatch);
        };
        let context_len = first_tokens.as_ref().len();
        let mut seq_ids = Vec::new();
        let mut tokens = Vec::new();
        for (seq, row) in std::iter::once((first_seq, first_tokens)).chain(rows) {
            let row = row.as_ref();
            if row.len() != context_len {
                return Err(EngineError::BatchShapeMismatch {
                    expected: context_len,
                    actual: row.len(),
                });
            }
            seq_ids.push(seq);
            tokens.extend_from_slice(row);
        }
        Self::new(seq_ids, tokens, context_len)
    }

    /// Returns the number of sequence rows in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_ids.len()
    }

    /// Returns whether the batch has no rows. A validated batch never has zero
    /// rows; the method exists for conventional container ergonomics.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    /// Returns the number of context tokens in each row.
    #[must_use]
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// Returns the sequence ids in row order.
    #[must_use]
    pub fn seq_ids(&self) -> &[u64] {
        &self.seq_ids
    }

    /// Returns the row-major context tokens.
    #[must_use]
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Returns the context row at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RowOutOfRange`] if `index` is outside the batch.
    pub fn row(&self, index: usize) -> Result<&[i32], EngineError> {
        if index >= self.seq_ids.len() {
            return Err(EngineError::RowOutOfRange {
                index,
                rows: self.seq_ids.len(),
            });
        }
        let start = index
            .checked_mul(self.context_len)
            .ok_or(EngineError::CacheInvariant {
                reason: "batch row offset overflow",
            })?;
        let end = start
            .checked_add(self.context_len)
            .ok_or(EngineError::CacheInvariant {
                reason: "batch row end overflow",
            })?;
        self.tokens
            .get(start..end)
            .ok_or(EngineError::CacheInvariant {
                reason: "batch token length disagrees with context_len",
            })
    }

    /// Finds the context row belonging to `seq`, if present.
    #[must_use]
    pub fn row_for_seq(&self, seq: u64) -> Option<&[i32]> {
        let index = self
            .seq_ids
            .iter()
            .position(|&candidate| candidate == seq)?;
        self.row(index).ok()
    }

    /// Revalidates the batch invariants.
    ///
    /// A `Batch` returned by a constructor is always valid, so this is mainly
    /// useful for diagnostics and for callers accepting deserialized data.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Batch::new`].
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.seq_ids.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        if self.context_len == 0 {
            return Err(EngineError::EmptyContext);
        }
        let expected =
            self.context_len
                .checked_mul(self.seq_ids.len())
                .ok_or(EngineError::InvalidParams {
                    reason: "batch shape overflows usize",
                })?;
        if self.tokens.len() != expected {
            return Err(EngineError::BatchShapeMismatch {
                expected,
                actual: self.tokens.len(),
            });
        }
        let mut seen = HashSet::with_capacity(self.seq_ids.len());
        for &seq in &self.seq_ids {
            if !seen.insert(seq) {
                return Err(EngineError::DuplicateSequence { seq });
            }
        }
        Ok(())
    }
}

/// One logit row per sequence in a [`Batch`].
///
/// Rows use the same sequence ids and order as the batch they answer. Every
/// logit must be finite; flat or otherwise unhealthy distributions are left to
/// the caller's health audit.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    seq_ids: Vec<u64>,
    values: Vec<f32>,
    vocab_size: usize,
}

impl Logits {
    /// Creates logits from parallel sequence ids and row-major values.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyBatch`], [`EngineError::InvalidParams`],
    /// [`EngineError::BatchShapeMismatch`], [`EngineError::DuplicateSequence`]
    /// or [`EngineError::InvalidLogits`] for invalid input.
    pub fn new(
        seq_ids: Vec<u64>,
        values: Vec<f32>,
        vocab_size: usize,
    ) -> Result<Self, EngineError> {
        if seq_ids.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        if vocab_size == 0 {
            return Err(EngineError::InvalidParams {
                reason: "vocab_size must be greater than zero",
            });
        }
        let expected = vocab_size
            .checked_mul(seq_ids.len())
            .ok_or(EngineError::InvalidParams {
                reason: "logits shape overflows usize",
            })?;
        if values.len() != expected {
            return Err(EngineError::BatchShapeMismatch {
                expected,
                actual: values.len(),
            });
        }
        let mut seen = HashSet::with_capacity(seq_ids.len());
        for &seq in &seq_ids {
            if !seen.insert(seq) {
                return Err(EngineError::DuplicateSequence { seq });
            }
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(EngineError::InvalidLogits { index });
            }
        }
        Ok(Self {
            seq_ids,
            values,
            vocab_size,
        })
    }

    /// Returns the number of logit rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_ids.len()
    }

    /// Returns whether the logits contain no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    /// Returns the number of logits per row.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the sequence ids in row order.
    #[must_use]
    pub fn seq_ids(&self) -> &[u64] {
        &self.seq_ids
    }

    /// Returns all logit values in row-major order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns the logit row at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RowOutOfRange`] if `index` is outside the logits.
    pub fn row(&self, index: usize) -> Result<&[f32], EngineError> {
        if index >= self.seq_ids.len() {
            return Err(EngineError::RowOutOfRange {
                index,
                rows: self.seq_ids.len(),
            });
        }
        let start = index
            .checked_mul(self.vocab_size)
            .ok_or(EngineError::CacheInvariant {
                reason: "logit row offset overflow",
            })?;
        let end = start
            .checked_add(self.vocab_size)
            .ok_or(EngineError::CacheInvariant {
                reason: "logit row end overflow",
            })?;
        self.values
            .get(start..end)
            .ok_or(EngineError::CacheInvariant {
                reason: "logit length disagrees with vocab_size",
            })
    }

    /// Finds the logit row belonging to `seq`, if present.
    #[must_use]
    pub fn row_for_seq(&self, seq: u64) -> Option<&[f32]> {
        let index = self
            .seq_ids
            .iter()
            .position(|&candidate| candidate == seq)?;
        self.row(index).ok()
    }
}

/// Borrowed key and value values for a bounded range of KV tokens.
///
/// Both slices use `token -> feature` layout and must contain exactly
/// `token_count * kv_dim` values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvBlock<'a> {
    /// Number of tokens represented by this block.
    pub token_count: usize,
    /// Key values for the tokens in the block.
    pub keys: &'a [f32],
    /// Value vectors for the tokens in the block.
    pub values: &'a [f32],
}

impl KvBlock<'_> {
    /// Checks the block against `geometry` and returns the expected slice
    /// length in `f32` values.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::KvBlockTooLarge`] or
    /// [`EngineError::KvBlockShape`] for a malformed block.
    pub fn validate(&self, geometry: &KvGeometry) -> Result<usize, EngineError> {
        validate_block_shape(
            geometry,
            self.token_count,
            self.keys.len(),
            self.values.len(),
        )
    }
}

/// Mutable key and value buffers for a bounded range of KV tokens.
///
/// `token_count` is the maximum number of tokens the caller can receive.
/// Unfilled trailing values are zeroed by a conforming [`KvCacheView`].
#[derive(Debug, PartialEq)]
pub struct KvBlockMut<'a> {
    /// Maximum number of tokens to read into the buffers.
    pub token_count: usize,
    /// Destination buffer for key values.
    pub keys: &'a mut [f32],
    /// Destination buffer for value vectors.
    pub values: &'a mut [f32],
}

impl KvBlockMut<'_> {
    /// Checks the buffers against `geometry` and returns their required length
    /// in `f32` values.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::KvBlockTooLarge`] or
    /// [`EngineError::KvBlockShape`] for malformed buffers.
    pub fn validate(&self, geometry: &KvGeometry) -> Result<usize, EngineError> {
        validate_block_shape(
            geometry,
            self.token_count,
            self.keys.len(),
            self.values.len(),
        )
    }
}

fn validate_block_shape(
    geometry: &KvGeometry,
    token_count: usize,
    key_len: usize,
    value_len: usize,
) -> Result<usize, EngineError> {
    if token_count > geometry.block_size {
        return Err(EngineError::KvBlockTooLarge {
            token_count,
            block_size: geometry.block_size,
        });
    }
    let expected = token_count
        .checked_mul(geometry.kv_dim)
        .ok_or(EngineError::InvalidParams {
            reason: "KV block shape overflows usize",
        })?;
    if key_len != expected || value_len != expected {
        return Err(EngineError::KvBlockShape {
            expected,
            actual: key_len.max(value_len),
        });
    }
    Ok(expected)
}

/// Read/write access to per-sequence KV blocks.
///
/// The trait is object-safe so a batch backend can accept `&mut dyn
/// KvCacheView`. Implementations own their storage; the backend only sees
/// geometry, sequence lengths and bounded token ranges. Writes may append at
/// `start_token == seq_len(seq)`, but gaps are rejected.
pub trait KvCacheView {
    /// Returns the fixed geometry of this cache.
    fn geometry(&self) -> KvGeometry;

    /// Returns the number of stored tokens for `seq`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] for an unknown sequence.
    fn seq_len(&self, seq: u64) -> Result<usize, EngineError>;

    /// Copies up to `out.token_count` tokens starting at `start_token` into
    /// `out` and returns the number of tokens actually copied.
    ///
    /// Reading past the end is allowed and zero-fills the unused tail. A start
    /// beyond `seq_len(seq)` is an error.
    ///
    /// # Errors
    ///
    /// Returns typed errors for unknown sequences, invalid layers, malformed
    /// blocks or an out-of-range start.
    fn read(
        &self,
        seq: u64,
        layer: usize,
        start_token: usize,
        out: KvBlockMut<'_>,
    ) -> Result<usize, EngineError>;

    /// Writes `block.token_count` tokens starting at `start_token` and returns
    /// the number of tokens written.
    ///
    /// The sequence grows when the write reaches past its current length;
    /// `start_token > seq_len(seq)` is rejected because it would create a gap.
    ///
    /// # Errors
    ///
    /// Returns typed errors for unknown sequences, invalid layers, malformed
    /// blocks, gaps or arithmetic overflow.
    fn write(
        &mut self,
        seq: u64,
        layer: usize,
        start_token: usize,
        block: KvBlock<'_>,
    ) -> Result<usize, EngineError>;
}

/// Batch counterpart of [`crate::backend::Backend`].
///
/// The old token-context trait stays available at `quarrel_engine::backend`
/// for existing callers. This trait is for execution paths that process a
/// dense batch and need access to per-sequence KV state.
pub trait Backend {
    /// Stable backend name, used in errors and diagnostics.
    fn name(&self) -> &'static str;

    /// Number of logits emitted for each sequence row.
    fn vocab_size(&self) -> usize;

    /// Computes one logit row per sequence in `batch`.
    ///
    /// Implementations may read and update KV state through `kv`, but must not
    /// panic on malformed input; validation failures are returned as
    /// [`EngineError`] values.
    ///
    /// # Errors
    ///
    /// Returns typed errors for shape, KV and backend failures.
    fn forward(&self, batch: &Batch, kv: &mut dyn KvCacheView) -> Result<Logits, EngineError>;
}

#[derive(Debug, Clone)]
struct SequenceKv {
    len: usize,
    capacity: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Simple in-memory [`KvCacheView`] for tests and backend consumers.
///
/// Storage is a flat `f32` buffer per sequence and per key/value kind, so no
/// concrete paged cache or model type is required to exercise a batch backend.
#[derive(Debug, Clone)]
pub struct InMemoryKvCache {
    geometry: KvGeometry,
    sequences: HashMap<u64, SequenceKv>,
}

impl InMemoryKvCache {
    /// Creates an empty cache with `geometry`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for an invalid geometry.
    pub fn new(geometry: KvGeometry) -> Result<Self, EngineError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            sequences: HashMap::new(),
        })
    }

    /// Returns the cache geometry.
    #[must_use]
    pub const fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    /// Registers `seq` as an empty sequence if it is not already present.
    pub fn ensure_sequence(&mut self, seq: u64) {
        self.sequences.entry(seq).or_insert_with(|| SequenceKv {
            len: 0,
            capacity: 0,
            keys: Vec::new(),
            values: Vec::new(),
        });
    }

    /// Returns whether `seq` is registered.
    #[must_use]
    pub fn contains(&self, seq: u64) -> bool {
        self.sequences.contains_key(&seq)
    }

    /// Removes `seq` and releases its storage.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if `seq` is not registered.
    pub fn remove(&mut self, seq: u64) -> Result<(), EngineError> {
        self.sequences
            .remove(&seq)
            .map(|_| ())
            .ok_or(EngineError::UnknownSequence { seq })
    }

    /// Returns the stored token count of `seq`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if `seq` is not registered.
    pub fn len(&self, seq: u64) -> Result<usize, EngineError> {
        self.sequences
            .get(&seq)
            .map(|entry| entry.len)
            .ok_or(EngineError::UnknownSequence { seq })
    }

    /// Returns whether `seq` is registered and holds no tokens.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownSequence`] if `seq` is not registered.
    pub fn is_empty(&self, seq: u64) -> Result<bool, EngineError> {
        Ok(self.len(seq)? == 0)
    }
}

impl KvCacheView for InMemoryKvCache {
    fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    fn seq_len(&self, seq: u64) -> Result<usize, EngineError> {
        self.len(seq)
    }

    fn read(
        &self,
        seq: u64,
        layer: usize,
        start_token: usize,
        out: KvBlockMut<'_>,
    ) -> Result<usize, EngineError> {
        let geometry = self.geometry;
        let expected_values = out.validate(&geometry)?;
        let entry = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        check_layer(layer, geometry.layers)?;
        if start_token > entry.len {
            return Err(EngineError::KvRangeOutOfBounds {
                seq,
                start: start_token,
                end: start_token.saturating_add(out.token_count),
                len: entry.len,
            });
        }
        let available = entry.len - start_token;
        let count = available.min(out.token_count);
        let end = start_token + count;
        let range = value_range(&geometry, layer, start_token, end, entry.capacity)?;
        out.keys[..count * geometry.kv_dim].copy_from_slice(&entry.keys[range.clone()]);
        out.values[..count * geometry.kv_dim].copy_from_slice(&entry.values[range]);
        out.keys[count * geometry.kv_dim..expected_values].fill(0.0);
        out.values[count * geometry.kv_dim..expected_values].fill(0.0);
        Ok(count)
    }

    fn write(
        &mut self,
        seq: u64,
        layer: usize,
        start_token: usize,
        block: KvBlock<'_>,
    ) -> Result<usize, EngineError> {
        let geometry = self.geometry;
        let expected_values = block.validate(&geometry)?;
        check_layer(layer, geometry.layers)?;
        let len = self
            .sequences
            .get(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?
            .len;
        if start_token > len {
            return Err(EngineError::KvWriteGap {
                seq,
                start: start_token,
                len,
            });
        }
        let end =
            start_token
                .checked_add(block.token_count)
                .ok_or(EngineError::CapacityExceeded {
                    capacity: usize::MAX,
                    requested: block.token_count,
                })?;
        let entry = self
            .sequences
            .get_mut(&seq)
            .ok_or(EngineError::UnknownSequence { seq })?;
        ensure_capacity(entry, end, &geometry)?;
        let range = value_range(&geometry, layer, start_token, end, entry.capacity)?;
        entry.keys[range.clone()].copy_from_slice(&block.keys[..expected_values]);
        entry.values[range].copy_from_slice(&block.values[..expected_values]);
        if end > entry.len {
            entry.len = end;
        }
        Ok(block.token_count)
    }
}

fn check_layer(layer: usize, layers: usize) -> Result<(), EngineError> {
    if layer >= layers {
        return Err(EngineError::UnknownKvLayer { layer, layers });
    }
    Ok(())
}

fn value_range(
    geometry: &KvGeometry,
    layer: usize,
    start: usize,
    end: usize,
    capacity: usize,
) -> Result<Range<usize>, EngineError> {
    if end > capacity {
        return Err(EngineError::CacheInvariant {
            reason: "KV range exceeds allocated block capacity",
        });
    }
    let layer_offset = layer
        .checked_mul(capacity)
        .ok_or(EngineError::CacheInvariant {
            reason: "KV layer offset overflow",
        })?;
    let first = layer_offset
        .checked_add(start)
        .and_then(|value| value.checked_mul(geometry.kv_dim))
        .ok_or(EngineError::CacheInvariant {
            reason: "KV start offset overflow",
        })?;
    let last = layer_offset
        .checked_add(end)
        .and_then(|value| value.checked_mul(geometry.kv_dim))
        .ok_or(EngineError::CacheInvariant {
            reason: "KV end offset overflow",
        })?;
    Ok(first..last)
}

fn ensure_capacity(
    entry: &mut SequenceKv,
    tokens: usize,
    geometry: &KvGeometry,
) -> Result<(), EngineError> {
    if tokens <= entry.capacity {
        return Ok(());
    }
    let blocks = tokens.div_ceil(geometry.block_size);
    let capacity =
        blocks
            .checked_mul(geometry.block_size)
            .ok_or(EngineError::CapacityExceeded {
                capacity: entry.capacity,
                requested: tokens,
            })?;
    let per_layer = capacity
        .checked_mul(geometry.kv_dim)
        .ok_or(EngineError::CapacityExceeded {
            capacity: entry.capacity,
            requested: tokens,
        })?;
    let total = per_layer
        .checked_mul(geometry.layers)
        .ok_or(EngineError::CapacityExceeded {
            capacity: entry.capacity,
            requested: tokens,
        })?;
    entry.keys.resize(total, 0.0);
    entry.values.resize(total, 0.0);
    entry.capacity = capacity;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Batch, InMemoryKvCache, KvBlock, KvBlockMut, KvCacheView, KvGeometry, Logits};
    use crate::EngineError;

    fn geometry() -> KvGeometry {
        KvGeometry::new(4, 2, 3)
    }

    fn batch() -> Batch {
        Batch::from_rows([(11_u64, vec![1, 2, 3]), (12_u64, vec![4, 5, 6])]).expect("batch")
    }

    #[test]
    fn batch_validates_shape_and_sequence_ids() {
        assert!(matches!(
            Batch::new(vec![], vec![], 2),
            Err(EngineError::EmptyBatch)
        ));
        assert!(matches!(
            Batch::new(vec![1], vec![9], 0),
            Err(EngineError::EmptyContext)
        ));
        assert!(matches!(
            Batch::new(vec![1, 2], vec![9, 8], 2),
            Err(EngineError::BatchShapeMismatch { .. })
        ));
        assert!(matches!(
            Batch::new(vec![7, 7], vec![1, 2, 3, 4], 2),
            Err(EngineError::DuplicateSequence { seq: 7 })
        ));
        assert!(matches!(
            Batch::from_rows([(1_u64, vec![1]), (2_u64, vec![2, 3])]),
            Err(EngineError::BatchShapeMismatch { .. })
        ));
        assert!(matches!(
            Batch::from_rows::<_, Vec<i32>>([]),
            Err(EngineError::EmptyBatch)
        ));

        let valid = batch();
        assert_eq!(valid.len(), 2);
        assert!(!valid.is_empty());
        assert_eq!(valid.context_len(), 3);
        assert_eq!(valid.row(0), Ok(&[1, 2, 3][..]));
        assert_eq!(valid.row(1), Ok(&[4, 5, 6][..]));
        assert_eq!(valid.row_for_seq(12), Some(&[4, 5, 6][..]));
        assert_eq!(valid.row_for_seq(99), None);
        assert!(matches!(
            valid.row(2),
            Err(EngineError::RowOutOfRange { .. })
        ));
    }

    #[test]
    fn logits_validate_shape_and_finiteness() {
        assert!(matches!(
            Logits::new(vec![], vec![], 4),
            Err(EngineError::EmptyBatch)
        ));
        assert!(matches!(
            Logits::new(vec![1], vec![1.0], 0),
            Err(EngineError::InvalidParams { .. })
        ));
        assert!(matches!(
            Logits::new(vec![1, 2], vec![1.0, 2.0], 2),
            Err(EngineError::BatchShapeMismatch { .. })
        ));
        assert!(matches!(
            Logits::new(vec![3, 3], vec![1.0, 2.0], 1),
            Err(EngineError::DuplicateSequence { seq: 3 })
        ));
        assert!(matches!(
            Logits::new(vec![1], vec![f32::NAN], 1),
            Err(EngineError::InvalidLogits { index: 0 })
        ));

        let logits = Logits::new(vec![11, 12], vec![0.5, 1.5, 2.5, 3.5], 2).expect("logits");
        assert_eq!(logits.vocab_size(), 2);
        assert_eq!(logits.row(0), Ok(&[0.5, 1.5][..]));
        assert_eq!(logits.row_for_seq(12), Some(&[2.5, 3.5][..]));
        assert!(matches!(
            logits.row(7),
            Err(EngineError::RowOutOfRange { .. })
        ));
    }

    #[test]
    fn geometry_rejects_zero_and_overflow() {
        assert!(matches!(
            KvGeometry::new(0, 1, 1).validate(),
            Err(EngineError::InvalidParams { .. })
        ));
        assert!(matches!(
            KvGeometry::new(usize::MAX, 1, 2).validate(),
            Err(EngineError::InvalidParams { .. })
        ));
        assert_eq!(geometry().block_values(), Ok(12));
    }

    #[test]
    fn kv_write_and_read_preserve_per_sequence_data() {
        let geometry = geometry();
        let mut kv = InMemoryKvCache::new(geometry).expect("cache");
        kv.ensure_sequence(1);
        kv.ensure_sequence(2);
        assert!(kv.contains(1));
        assert!(kv.is_empty(1).expect("len"));

        let first = KvBlock {
            token_count: 3,
            keys: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            values: &[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        };
        assert_eq!(kv.write(1, 0, 0, first).expect("write"), 3);
        let second = KvBlock {
            token_count: 1,
            keys: &[100.0, 101.0, 102.0],
            values: &[-100.0, -101.0, -102.0],
        };
        assert_eq!(kv.write(2, 0, 0, second).expect("write"), 1);
        assert_eq!(kv.len(1), Ok(3));
        assert_eq!(kv.len(2), Ok(1));

        let mut keys = [0.0_f32; 12];
        let mut values = [0.0_f32; 12];
        let read = kv
            .read(
                1,
                0,
                0,
                KvBlockMut {
                    token_count: 4,
                    keys: &mut keys,
                    values: &mut values,
                },
            )
            .expect("read");
        assert_eq!(read, 3);
        assert_eq!(&keys[..9], first.keys);
        assert_eq!(&values[..9], first.values);
        assert_eq!(&keys[9..], &[0.0, 0.0, 0.0]);
        assert_eq!(&values[9..], &[0.0, 0.0, 0.0]);

        let mut other_keys = [0.0_f32; 3];
        let mut other_values = [0.0_f32; 3];
        assert_eq!(
            kv.read(
                2,
                0,
                0,
                KvBlockMut {
                    token_count: 1,
                    keys: &mut other_keys,
                    values: &mut other_values,
                },
            )
            .expect("read"),
            1
        );
        assert_eq!(other_keys, second.keys);
        assert_eq!(other_values, second.values);

        assert_eq!(kv.remove(1), Ok(()));
        assert!(matches!(
            kv.len(1),
            Err(EngineError::UnknownSequence { .. })
        ));
    }

    #[test]
    fn kv_view_rejects_malformed_access() {
        let mut kv = InMemoryKvCache::new(geometry()).expect("cache");
        kv.ensure_sequence(1);
        assert!(matches!(
            kv.seq_len(9),
            Err(EngineError::UnknownSequence { seq: 9 })
        ));

        let mut keys = [0.0_f32; 3];
        let mut values = [0.0_f32; 3];
        assert!(matches!(
            kv.read(
                1,
                0,
                0,
                KvBlockMut {
                    token_count: 1,
                    keys: &mut [0.0],
                    values: &mut [0.0, 0.0, 0.0],
                },
            ),
            Err(EngineError::KvBlockShape { .. })
        ));
        assert!(matches!(
            kv.read(
                1,
                2,
                0,
                KvBlockMut {
                    token_count: 1,
                    keys: &mut keys,
                    values: &mut values,
                },
            ),
            Err(EngineError::UnknownKvLayer { .. })
        ));
        assert!(matches!(
            kv.read(
                1,
                0,
                1,
                KvBlockMut {
                    token_count: 1,
                    keys: &mut keys,
                    values: &mut values,
                },
            ),
            Err(EngineError::KvRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            kv.read(
                1,
                0,
                0,
                KvBlockMut {
                    token_count: 5,
                    keys: &mut keys,
                    values: &mut values,
                },
            ),
            Err(EngineError::KvBlockTooLarge { .. })
        ));
        assert!(matches!(
            kv.write(
                1,
                0,
                1,
                KvBlock {
                    token_count: 1,
                    keys: &[0.0, 0.0, 0.0],
                    values: &[0.0, 0.0, 0.0],
                },
            ),
            Err(EngineError::KvWriteGap { .. })
        ));
        assert!(matches!(
            kv.write(
                9,
                0,
                0,
                KvBlock {
                    token_count: 1,
                    keys: &[0.0, 0.0, 0.0],
                    values: &[0.0, 0.0, 0.0],
                },
            ),
            Err(EngineError::UnknownSequence { .. })
        ));
    }
}
