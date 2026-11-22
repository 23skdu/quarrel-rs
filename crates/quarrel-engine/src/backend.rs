//! Backend trait, reference implementation and generation driver.
//!
//! [`ReferenceBackend`] is a deterministic test fixture. It does **not**
//! reproduce the Go engine's model weights and must not be used as a parity
//! oracle; its logits are produced by an integer hash purely so tests have a
//! stable, finite distribution to sample from.

use crate::EngineError;
use crate::health::{HealthStatus, audit_logits};
use crate::sampler::Sampler;
use crate::to_usize;
use quarrel_core::{SamplingParams, Token};

/// A provider of logits for a token context.
pub trait Backend {
    /// Stable backend name, used in errors and diagnostics.
    fn name(&self) -> &'static str;

    /// Returns one logit per vocabulary entry for the given token context.
    ///
    /// # Errors
    ///
    /// Implementations should return an error when logits cannot be produced.
    fn forward(&self, tokens: &[i32]) -> Result<Vec<f32>, EngineError>;
}

/// A deterministic, allocation-light reference backend for tests.
///
/// Logits come from a splitmix-style integer hash of the token ids, their
/// positions and the vocabulary index. The same token slice always yields the
/// same finite logits, and unknown or negative token ids are mixed into the
/// hash rather than indexed into a table. This backend is intentionally *not*
/// a Go-parity model.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    /// Creates a reference backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Number of logits emitted by [`ReferenceBackend`].
const REFERENCE_VOCAB_SIZE: usize = 32;

impl Backend for ReferenceBackend {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn forward(&self, tokens: &[i32]) -> Result<Vec<f32>, EngineError> {
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut state = GOLDEN ^ (tokens.len() as u64).wrapping_mul(GOLDEN);
        for (position, &token) in tokens.iter().enumerate() {
            let id_bits = (i64::from(token)) as u64;
            state = mix64(state ^ id_bits ^ (position as u64).wrapping_mul(GOLDEN));
        }

        let mut logits = Vec::with_capacity(REFERENCE_VOCAB_SIZE);
        for index in 0..REFERENCE_VOCAB_SIZE {
            let hashed = mix64(state ^ (index as u64).wrapping_mul(GOLDEN));
            // The top 24 bits map exactly onto an f32 mantissa, so this is
            // deterministic across platforms and always finite.
            let unit = (hashed >> 40) as f32 / 16_777_216.0;
            logits.push(unit.mul_add(4.0, -2.0));
        }
        Ok(logits)
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Generates at most `max_tokens` tokens with `backend` and `params`.
///
/// The caller's seed fully determines the sampled token ids. Generation stops
/// when the length cap (the smaller of `max_tokens` and `params.max_tokens`) is
/// reached, when only stop tokens remain, or when the backend produces
/// unhealthy logits. In the last case an error is returned rather than
/// emitting tokens sampled from unusable values.
///
/// # Errors
///
/// Propagates parameter, logit and backend errors from [`Sampler::sample`] and
/// [`Backend::forward`].
pub fn generate(
    backend: &dyn Backend,
    prompt: &[i32],
    params: &SamplingParams,
    seed: u64,
    max_tokens: usize,
) -> Result<Vec<Token>, EngineError> {
    params.validate().map_err(|_| EngineError::InvalidParams {
        reason: "quarrel-core rejected the sampling parameters",
    })?;
    let param_limit = to_usize(params.max_tokens)?;
    let limit = max_tokens.min(param_limit);

    let mut sampler = Sampler::new(seed);
    let mut context = Vec::with_capacity(prompt.len().saturating_add(limit));
    context.extend_from_slice(prompt);
    let mut generated = Vec::with_capacity(limit);

    for _ in 0..limit {
        let logits = backend.forward(&context)?;
        if let HealthStatus::Unhealthy(reason) = audit_logits(&logits) {
            return Err(EngineError::BackendLogits {
                backend: backend.name().to_owned(),
                reason,
            });
        }
        match sampler.sample(&logits, params, &context) {
            Ok(token) => {
                context.push(token.id);
                generated.push(token);
            }
            Err(EngineError::StopTokenSampled { .. }) => break,
            Err(error) => return Err(error),
        }
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::{Backend, ReferenceBackend, generate};
    use crate::EngineError;
    use crate::health::{HealthStatus, audit_logits};
    use quarrel_core::SamplingParams;

    fn params() -> SamplingParams {
        let mut params = SamplingParams::new();
        params.temperature = 0.8;
        params.top_p = 0.9;
        params.top_k = 8;
        params.repetition_penalty = 1.0;
        params.max_tokens = 16;
        params.stop_token_ids.clear();
        params
    }

    struct BrokenBackend;

    impl Backend for BrokenBackend {
        fn name(&self) -> &'static str {
            "broken"
        }

        fn forward(&self, _tokens: &[i32]) -> Result<Vec<f32>, EngineError> {
            Ok(vec![0.0, f32::NAN, 1.0])
        }
    }

    #[test]
    fn reference_backend_is_deterministic_and_finite() {
        let backend = ReferenceBackend::new();
        assert_eq!(backend.name(), "reference");
        let first = backend.forward(&[1, 2, 3]).expect("logits");
        let second = backend.forward(&[1, 2, 3]).expect("logits");
        assert_eq!(first, second);
        assert_eq!(first.len(), 32);
        assert!(first.iter().all(|value| value.is_finite()));
        assert!(matches!(audit_logits(&first), HealthStatus::Healthy));

        let different = backend.forward(&[1, 2, 4]).expect("logits");
        assert_ne!(first, different);
        assert!(backend.forward(&[-7, i32::MIN, i32::MAX]).is_ok());
    }

    #[test]
    fn reference_backend_logits_snapshot() {
        let logits = ReferenceBackend::new()
            .forward(&[1, 2, 3, 5, 8])
            .expect("logits");
        insta::assert_debug_snapshot!(logits);
    }

    #[test]
    fn generate_is_deterministic_and_respects_length() {
        let backend = ReferenceBackend::new();
        let first = generate(&backend, &[1, 2, 3], &params(), 42, 6).expect("generation");
        let second = generate(&backend, &[1, 2, 3], &params(), 42, 6).expect("generation");
        assert_eq!(first.len(), 6);
        assert_eq!(first, second);
        assert!(first.iter().all(|token| token.logprob.is_finite()));
    }

    #[test]
    fn generate_uses_smaller_length_cap() {
        let mut limited = params();
        limited.max_tokens = 2;
        let tokens = generate(&ReferenceBackend::new(), &[1], &limited, 3, 9).expect("generation");
        assert_eq!(tokens.len(), 2);
        let tokens = generate(&ReferenceBackend::new(), &[1], &params(), 3, 0).expect("generation");
        assert!(tokens.is_empty());
    }

    #[test]
    fn generate_stops_when_all_tokens_are_stops() {
        let mut params = params();
        for id in 0..32 {
            params.stop_token_ids.push(id);
        }
        let tokens = generate(&ReferenceBackend::new(), &[1], &params, 1, 5).expect("generation");
        assert!(tokens.is_empty());
    }

    #[test]
    fn generate_rejects_unhealthy_backend_logits() {
        assert!(matches!(
            generate(&BrokenBackend, &[1], &params(), 1, 3),
            Err(EngineError::BackendLogits { .. })
        ));
    }

    #[test]
    fn generate_rejects_invalid_params() {
        let mut params = params();
        params.temperature = -1.0;
        assert!(matches!(
            generate(&ReferenceBackend::new(), &[1], &params, 1, 3),
            Err(EngineError::InvalidParams { .. })
        ));
    }

    #[test]
    fn generated_ids_snapshot() {
        let tokens =
            generate(&ReferenceBackend::new(), &[4, 5, 6], &params(), 2024, 8).expect("generation");
        let ids: Vec<i32> = tokens.iter().map(|token| token.id).collect();
        insta::assert_debug_snapshot!(ids);
    }
}
