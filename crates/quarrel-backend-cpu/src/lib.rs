//! Batch-aware CPU execution backend for `quarrel-rs`.
//!
//! [`CpuBackend`] implements [`quarrel_engine::batch::Backend`]: it consumes a
//! [`Batch`], reads and updates per-sequence KV state through
//! [`quarrel_engine::batch::KvCacheView`] and returns one [`Logits`] row per
//! sequence. The hot dot-product and softmax operations use
//! [`quarrel_simd`], which dispatches once at runtime and works on machines
//! with no optional CPU features. A fully portable scalar path is also
//! available via [`KernelMode::Scalar`] for bit-stable tests.
//!
//! This is a deterministic, allocation-light surrogate for a future
//! model-backed execution backend. Logits come from integer hashes of token
//! ids, sequence ids and pseudo-random weight vectors; they are **not** model
//! inference and are explicitly **not a Go-engine parity oracle**. Use a real
//! weight-backed backend when matching the Go engine matters.
//!
//! ```
//! use quarrel_backend_cpu::CpuBackend;
//! use quarrel_engine::batch::{Backend, Batch, InMemoryKvCache};
//!
//! let backend = CpuBackend::new(32, 16)?;
//! let batch = Batch::from_rows([(1_u64, vec![5, 6, 7]), (2_u64, vec![8, 9, 10])])?;
//! let mut kv = InMemoryKvCache::new(backend.kv_geometry())?;
//! for seq in batch.seq_ids() {
//!     kv.ensure_sequence(*seq);
//! }
//! let logits = backend.forward(&batch, &mut kv)?;
//! assert_eq!(logits.len(), 2);
//! # Ok::<(), quarrel_engine::EngineError>(())
//! ```

#![deny(missing_docs)]

use quarrel_engine::EngineError;
use quarrel_engine::batch::{Backend, Batch, KvBlock, KvBlockMut, KvCacheView, KvGeometry, Logits};
use quarrel_simd::SimdError;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const EMBED_SEED: u64 = 0xD1B5_4A32_D192_ED03;
const WEIGHT_SEED: u64 = 0xA076_1D64_78BD_642F;

/// Selects which numeric kernels a [`CpuBackend`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelMode {
    /// Runtime-dispatched `quarrel-simd` kernels (SSE2/AVX2/NEON when
    /// available, portable scalar otherwise).
    Dispatched,
    /// Fixed-order scalar arithmetic implemented in this crate.
    ///
    /// This path needs no hardware features and produces identical values on
    /// every target. It is the deterministic fallback for tests and portable
    /// execution.
    Scalar,
}

/// CPU implementation of the batch-oriented [`Backend`] trait.
///
/// The backend owns only small configuration values. Per-call scratch buffers
/// are bounded by `block_size * hidden` and `vocab_size`; weights are generated
/// deterministically on the fly rather than stored as a large matrix.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    vocab_size: usize,
    hidden: usize,
    block_size: usize,
    block_values: usize,
    mode: KernelMode,
    inv_sqrt_hidden: f32,
}

impl CpuBackend {
    /// Default number of KV tokens exposed in one read window.
    pub const DEFAULT_BLOCK_SIZE: usize = 8;

    /// Creates a backend using runtime-dispatched kernels and the default KV
    /// block size.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for zero dimensions or a
    /// `block_size * hidden` product that overflows `usize`.
    pub fn new(vocab_size: usize, hidden: usize) -> Result<Self, EngineError> {
        Self::with_mode(
            vocab_size,
            hidden,
            Self::DEFAULT_BLOCK_SIZE,
            KernelMode::Dispatched,
        )
    }

    /// Creates a backend using the portable scalar kernels and the default KV
    /// block size.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for invalid geometry.
    pub fn scalar(vocab_size: usize, hidden: usize) -> Result<Self, EngineError> {
        Self::with_mode(
            vocab_size,
            hidden,
            Self::DEFAULT_BLOCK_SIZE,
            KernelMode::Scalar,
        )
    }

    /// Creates a dispatched backend with an explicit KV block size.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for invalid geometry.
    pub fn with_block_size(
        vocab_size: usize,
        hidden: usize,
        block_size: usize,
    ) -> Result<Self, EngineError> {
        Self::with_mode(vocab_size, hidden, block_size, KernelMode::Dispatched)
    }

    /// Creates a backend with explicit geometry and kernel mode.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for zero dimensions or an
    /// overflowing block size.
    pub fn with_mode(
        vocab_size: usize,
        hidden: usize,
        block_size: usize,
        mode: KernelMode,
    ) -> Result<Self, EngineError> {
        if vocab_size == 0 || hidden == 0 || block_size == 0 {
            return Err(EngineError::InvalidParams {
                reason: "backend dimensions must be greater than zero",
            });
        }
        let block_values = block_size
            .checked_mul(hidden)
            .ok_or(EngineError::InvalidParams {
                reason: "backend block size times hidden dimension overflows usize",
            })?;
        let inv_sqrt_hidden = (hidden as f64).sqrt().recip() as f32;
        Ok(Self {
            vocab_size,
            hidden,
            block_size,
            block_values,
            mode,
            inv_sqrt_hidden,
        })
    }

    /// Returns the number of logits emitted per sequence.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the surrogate hidden dimension.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    /// Returns the KV token window size used for attention.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the kernel mode selected for this backend.
    #[must_use]
    pub const fn mode(&self) -> KernelMode {
        self.mode
    }

    /// Returns the KV geometry required from a cache passed to [`Backend::forward`].
    #[must_use]
    pub const fn kv_geometry(&self) -> KvGeometry {
        KvGeometry::new(self.block_size, 1, self.hidden)
    }

    fn dot(&self, lhs: &[f32], rhs: &[f32]) -> Result<f32, EngineError> {
        match self.mode {
            KernelMode::Dispatched => kernel(quarrel_simd::dot_f32(lhs, rhs), self.name()),
            KernelMode::Scalar => scalar_dot(lhs, rhs),
        }
    }

    fn softmax(&self, input: &[f32], output: &mut [f32]) -> Result<(), EngineError> {
        match self.mode {
            KernelMode::Dispatched => {
                kernel(quarrel_simd::softmax_into(input, output), self.name())
            }
            KernelMode::Scalar => scalar_softmax(input, output),
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        match self.mode {
            KernelMode::Dispatched => "cpu",
            KernelMode::Scalar => "cpu-scalar",
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn forward(&self, batch: &Batch, kv: &mut dyn KvCacheView) -> Result<Logits, EngineError> {
        batch.validate()?;
        let expected = self.kv_geometry();
        let actual = kv.geometry();
        if actual != expected {
            return Err(EngineError::KvGeometryMismatch { expected, actual });
        }

        let rows = batch.len();
        let total = rows
            .checked_mul(self.vocab_size)
            .ok_or(EngineError::InvalidParams {
                reason: "batch times vocab_size overflows usize",
            })?;
        let mut seq_ids = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(total);

        let mut embedding = vec![0.0_f32; self.hidden];
        let mut weights = vec![0.0_f32; self.hidden];
        let mut combined = vec![0.0_f32; self.hidden];
        let mut scores = vec![0.0_f32; self.block_size];
        let mut probs = vec![0.0_f32; self.block_size];
        let mut keys = vec![0.0_f32; self.block_values];
        let mut vals = vec![0.0_f32; self.block_values];

        for index in 0..rows {
            let seq = batch.seq_ids()[index];
            let tokens = batch.row(index)?;
            fill_embedding(seq, tokens, &mut embedding);

            keys.fill(0.0);
            vals.fill(0.0);
            scores.fill(0.0);
            probs.fill(0.0);

            let stored = kv.seq_len(seq)?;
            let window_start = stored.saturating_sub(self.block_size);
            let window_capacity = stored - window_start;
            let window_values = window_capacity * self.hidden;
            let read = kv.read(
                seq,
                0,
                window_start,
                KvBlockMut {
                    token_count: window_capacity,
                    keys: &mut keys[..window_values],
                    values: &mut vals[..window_values],
                },
            )?;
            if read > window_capacity {
                return Err(EngineError::CacheInvariant {
                    reason: "KV view returned more tokens than requested",
                });
            }

            for token in 0..read {
                let key = &keys[token * self.hidden..(token + 1) * self.hidden];
                scores[token] = self.dot(&embedding, key)? * self.inv_sqrt_hidden;
            }
            self.softmax(&scores[..read], &mut probs[..read])?;

            for feature in 0..self.hidden {
                let mut context = 0.0;
                for token in 0..read {
                    context += probs[token] * vals[token * self.hidden + feature];
                }
                combined[feature] = embedding[feature] + context;
            }

            let logit_offset = values.len();
            values.resize(logit_offset + self.vocab_size, 0.0);
            for vocab in 0..self.vocab_size {
                fill_weights(vocab, self.vocab_size, &mut weights);
                let logit = self.dot(&combined, &weights)? * self.inv_sqrt_hidden;
                if !logit.is_finite() {
                    return Err(EngineError::InvalidLogits { index: vocab });
                }
                values[logit_offset + vocab] = logit;
            }

            let written = kv.write(
                seq,
                0,
                stored,
                KvBlock {
                    token_count: 1,
                    keys: &combined,
                    values: &combined,
                },
            )?;
            if written != 1 {
                return Err(EngineError::CacheInvariant {
                    reason: "KV view did not persist the backend's token",
                });
            }
            seq_ids.push(seq);
        }

        Logits::new(seq_ids, values, self.vocab_size)
    }
}

fn kernel<T>(result: Result<T, SimdError>, backend: &'static str) -> Result<T, EngineError> {
    result.map_err(|error| EngineError::KernelFailure {
        backend,
        reason: error.to_string(),
    })
}

fn scalar_dot(lhs: &[f32], rhs: &[f32]) -> Result<f32, EngineError> {
    if lhs.len() != rhs.len() {
        return Err(EngineError::InvalidParams {
            reason: "scalar dot received slices of different lengths",
        });
    }
    let mut sum = 0.0_f32;
    for (left, right) in lhs.iter().zip(rhs) {
        sum += left * right;
    }
    Ok(sum)
}

fn scalar_softmax(input: &[f32], output: &mut [f32]) -> Result<(), EngineError> {
    if input.len() != output.len() {
        return Err(EngineError::InvalidParams {
            reason: "scalar softmax received slices of different lengths",
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    let mut max = f32::NEG_INFINITY;
    for (index, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::InvalidLogits { index });
        }
        if value > max {
            max = value;
        }
    }
    let mut total = 0.0_f32;
    for (slot, &value) in output.iter_mut().zip(input) {
        let exponential = (value - max).exp();
        *slot = exponential;
        total += exponential;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::InvalidParams {
            reason: "scalar softmax denominator is not positive and finite",
        });
    }
    for slot in output.iter_mut() {
        *slot /= total;
    }
    Ok(())
}

fn fill_embedding(seq: u64, tokens: &[i32], out: &mut [f32]) {
    let mut state =
        EMBED_SEED ^ seq.wrapping_mul(GOLDEN) ^ (tokens.len() as u64).wrapping_mul(GOLDEN);
    for (position, &token) in tokens.iter().enumerate() {
        let bits = i64::from(token) as u64;
        state = mix64(state ^ bits ^ (position as u64).wrapping_mul(GOLDEN));
    }
    for (feature, slot) in out.iter_mut().enumerate() {
        state = mix64(state ^ (feature as u64).wrapping_mul(GOLDEN));
        *slot = signed_unit(state);
    }
}

fn fill_weights(vocab: usize, vocab_size: usize, out: &mut [f32]) {
    let mut state = WEIGHT_SEED
        ^ (vocab as u64).wrapping_mul(GOLDEN)
        ^ (vocab_size as u64).wrapping_mul(GOLDEN >> 1);
    for (feature, slot) in out.iter_mut().enumerate() {
        state = mix64(state ^ (feature as u64).wrapping_mul(GOLDEN));
        *slot = signed_unit(state);
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn signed_unit(bits: u64) -> f32 {
    ((bits >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::{CpuBackend, KernelMode};
    use quarrel_engine::EngineError;
    use quarrel_engine::batch::{
        Backend, Batch, InMemoryKvCache, KvBlockMut, KvCacheView, KvGeometry,
    };

    fn backend(mode: KernelMode) -> CpuBackend {
        CpuBackend::with_mode(16, 8, 4, mode).expect("valid backend")
    }

    fn batch() -> Batch {
        Batch::from_rows([(1_u64, vec![1, 2, 3]), (2_u64, vec![4, 5, 6])]).expect("valid batch")
    }

    fn kv_for(backend: &CpuBackend, batch: &Batch) -> InMemoryKvCache {
        let mut kv = InMemoryKvCache::new(backend.kv_geometry()).expect("valid cache");
        for seq in batch.seq_ids() {
            kv.ensure_sequence(*seq);
        }
        kv
    }

    #[test]
    fn constructors_validate_geometry() {
        assert!(matches!(
            CpuBackend::new(0, 8),
            Err(EngineError::InvalidParams { .. })
        ));
        assert!(matches!(
            CpuBackend::scalar(8, 0),
            Err(EngineError::InvalidParams { .. })
        ));
        assert!(matches!(
            CpuBackend::with_block_size(8, 8, 0),
            Err(EngineError::InvalidParams { .. })
        ));
        assert!(matches!(
            CpuBackend::with_block_size(8, usize::MAX, 2),
            Err(EngineError::InvalidParams { .. })
        ));
        let cpu = CpuBackend::new(32, 16).expect("valid backend");
        assert_eq!(cpu.vocab_size(), 32);
        assert_eq!(cpu.hidden(), 16);
        assert_eq!(cpu.block_size(), CpuBackend::DEFAULT_BLOCK_SIZE);
        assert_eq!(cpu.mode(), KernelMode::Dispatched);
        assert_eq!(
            cpu.kv_geometry(),
            KvGeometry::new(CpuBackend::DEFAULT_BLOCK_SIZE, 1, 16)
        );
    }

    #[test]
    fn batch_shape_and_cache_validation_are_typed() {
        let cpu = backend(KernelMode::Scalar);
        let mut kv = kv_for(&cpu, &batch());
        // A sequence that was never registered is reported rather than
        // silently created by the backend.
        let missing = Batch::from_rows([(99_u64, vec![7])]).expect("valid batch");
        assert!(matches!(
            cpu.forward(&missing, &mut kv),
            Err(EngineError::UnknownSequence { seq: 99 })
        ));
        let mut wrong_geometry = InMemoryKvCache::new(KvGeometry::new(4, 1, 7)).expect("cache");
        assert!(matches!(
            cpu.forward(&batch(), &mut wrong_geometry),
            Err(EngineError::KvGeometryMismatch { .. })
        ));
        assert!(matches!(
            Batch::new(vec![1, 2], vec![1, 2, 3], 2),
            Err(EngineError::BatchShapeMismatch { .. })
        ));
    }

    #[test]
    fn scalar_logits_are_deterministic_and_finite() {
        let cpu = backend(KernelMode::Scalar);
        let batch = batch();
        let first = cpu
            .forward(&batch, &mut kv_for(&cpu, &batch))
            .expect("logits");
        let second = cpu
            .forward(&batch, &mut kv_for(&cpu, &batch))
            .expect("logits");
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert_eq!(first.vocab_size(), 16);
        assert!(first.values().iter().all(|value| value.is_finite()));
        assert!(
            first
                .row_for_seq(1)
                .is_some_and(|row| row.iter().any(|v| *v != 0.0))
        );
    }

    #[test]
    fn dispatched_logits_are_repeatable_and_close_to_scalar() {
        let dispatched = backend(KernelMode::Dispatched);
        let scalar = backend(KernelMode::Scalar);
        let batch = batch();
        let first = dispatched
            .forward(&batch, &mut kv_for(&dispatched, &batch))
            .expect("logits");
        let second = dispatched
            .forward(&batch, &mut kv_for(&dispatched, &batch))
            .expect("logits");
        assert_eq!(first, second);

        let expected = scalar
            .forward(&batch, &mut kv_for(&scalar, &batch))
            .expect("logits");
        for (left, right) in first.values().iter().zip(expected.values()) {
            assert!(
                (left - right).abs() < 1e-3,
                "dispatched {left} and scalar {right} diverged"
            );
        }

        let trait_object: &dyn Backend = &dispatched;
        assert_eq!(
            trait_object
                .forward(&batch, &mut kv_for(&dispatched, &batch))
                .expect("logits"),
            first
        );
    }

    #[test]
    fn sequences_are_isolated_in_a_batch() {
        let cpu = backend(KernelMode::Scalar);
        let batch = batch();
        let combined = cpu
            .forward(&batch, &mut kv_for(&cpu, &batch))
            .expect("logits");
        assert_ne!(
            combined.row_for_seq(1).expect("row 1"),
            combined.row_for_seq(2).expect("row 2")
        );

        for (index, seq) in batch.seq_ids().iter().enumerate() {
            let tokens = batch.row(index).expect("row").to_vec();
            let single = Batch::from_rows([(*seq, tokens)]).expect("single batch");
            let lone = cpu
                .forward(&single, &mut kv_for(&cpu, &single))
                .expect("logits");
            assert_eq!(
                combined.row_for_seq(*seq).expect("combined row"),
                lone.row_for_seq(*seq).expect("single row")
            );
        }
    }

    #[test]
    fn forward_reads_and_writes_kv_through_the_view() {
        let cpu = backend(KernelMode::Scalar);
        let batch = Batch::from_rows([(5_u64, vec![1, 2, 3])]).expect("batch");
        let mut kv = kv_for(&cpu, &batch);
        assert_eq!(kv.seq_len(5), Ok(0));

        let first = cpu.forward(&batch, &mut kv).expect("first logits");
        assert_eq!(kv.seq_len(5), Ok(1));
        let mut keys = [0.0_f32; 32];
        let mut values = [0.0_f32; 32];
        assert_eq!(
            kv.read(
                5,
                0,
                0,
                KvBlockMut {
                    token_count: 4,
                    keys: &mut keys,
                    values: &mut values,
                },
            )
            .expect("read"),
            1
        );
        assert!(keys[..cpu.hidden()].iter().all(|value| value.is_finite()));
        assert!(keys[..cpu.hidden()].iter().any(|value| *value != 0.0));
        assert_eq!(&keys[..cpu.hidden()], &values[..cpu.hidden()]);

        let second = cpu.forward(&batch, &mut kv).expect("second logits");
        assert_eq!(kv.seq_len(5), Ok(2));
        assert_ne!(
            first.row_for_seq(5).expect("first row"),
            second.row_for_seq(5).expect("second row")
        );
    }
}
