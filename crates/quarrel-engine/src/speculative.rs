//! Deterministic speculative decoding over a target and a draft backend.
//!
//! A draft backend proposes [`SpeculativeConfig::draft_len`] tokens; the target
//! backend verifies them one at a time. A drafted token is accepted with
//! probability `min(1, p_target / p_draft)`, where both probabilities come from
//! the (possibly temperature-scaled) distributions at the same context. On the
//! first rejection the engine emits a token from the residual distribution
//! `max(p_target - p_draft, 0)`, renormalised. If every drafted token is
//! accepted, one bonus token is sampled from the target distribution after the
//! whole draft.
//!
//! Greedy decoding (`temperature == 0`) is handled by comparing one-hot
//! distributions: acceptance becomes a deterministic argmax comparison, the
//! residual collapses onto the target argmax and the seeded RNG is not
//! advanced. All stochastic draws come from a `StdRng` seeded from the config,
//! so identical inputs always produce identical decisions.
//!
//! This module does not change the existing [`crate::backend::Backend`] trait or
//! [`crate::backend::ReferenceBackend`]. It is a driver layered on top of them.
//!
//! ```
//! use quarrel_engine::backend::ReferenceBackend;
//! use quarrel_engine::speculative::{SpeculativeConfig, speculative_step};
//!
//! let config = SpeculativeConfig {
//!     temperature: 0.0,
//!     draft_len: 3,
//!     ..SpeculativeConfig::default()
//! };
//! let target = ReferenceBackend::new();
//! let draft = ReferenceBackend::new();
//! let decision = speculative_step(&target, &draft, &[1, 2, 3], &config)?;
//! assert_eq!(decision.outcomes.len(), 3);
//! assert_eq!(decision.tokens().len(), decision.emitted_len());
//! # Ok::<(), quarrel_engine::EngineError>(())
//! ```

use crate::{EngineError, backend::Backend};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Largest draft length the driver will verify in one step.
pub const MAX_DRAFT_LEN: usize = 8;

/// Parameters of one speculative step or generation run.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeConfig {
    /// Sampling temperature. Zero selects deterministic greedy decoding.
    pub temperature: f64,
    /// Number of tokens the draft backend proposes per step.
    pub draft_len: usize,
    /// Seed for every stochastic draw.
    pub seed: u64,
    /// Token cap used by [`speculative_generate`]; ignored by a single step.
    pub max_tokens: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            draft_len: 4,
            seed: 0,
            max_tokens: 16,
        }
    }
}

impl SpeculativeConfig {
    /// Validates temperature, draft length and shape before a step runs.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidParams`] for a non-finite or negative
    /// temperature and [`EngineError::InvalidDraftLength`] for a zero or
    /// oversized draft.
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::InvalidParams {
                reason: "speculative temperature must be finite and non-negative",
            });
        }
        if self.draft_len == 0 || self.draft_len > MAX_DRAFT_LEN {
            return Err(EngineError::InvalidDraftLength {
                length: self.draft_len,
                maximum: MAX_DRAFT_LEN,
            });
        }
        Ok(())
    }
}

/// Outcome of verifying one drafted token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Acceptance {
    /// The drafted token was accepted.
    Accepted {
        /// The clipped acceptance ratio `min(1, p_target / p_draft)`.
        ratio: f64,
    },
    /// The drafted token was rejected in favour of a target-distributed token.
    Rejected {
        /// The clipped acceptance ratio that failed the draw.
        ratio: f64,
    },
}

impl Acceptance {
    /// Returns whether the drafted token was accepted.
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted { .. })
    }

    /// Returns the clipped acceptance ratio.
    #[must_use]
    pub const fn ratio(self) -> f64 {
        match self {
            Self::Accepted { ratio } | Self::Rejected { ratio } => ratio,
        }
    }
}

/// Result of checking one drafted token against the target distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCheck {
    /// Vocabulary index of the drafted token.
    pub token_index: usize,
    /// Whether the token was accepted.
    pub accepted: bool,
    /// The clipped ratio `min(1, p_target / p_draft)` used for the draw.
    pub ratio: f64,
}

/// Result of one full speculative step.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDecision {
    /// Drafted tokens that were accepted, in draft order.
    pub accepted: Vec<i32>,
    /// Target token emitted on rejection, or the bonus token after a full
    /// acceptance. Always `Some` for a successful step.
    pub target_token: Option<i32>,
    /// One outcome per verified draft token, in draft order.
    pub outcomes: Vec<Acceptance>,
    /// Whether every drafted token was accepted.
    pub all_accepted: bool,
}

impl SpeculativeDecision {
    /// Returns the emitted token ids: accepted tokens followed by the target token.
    #[must_use]
    pub fn tokens(&self) -> Vec<i32> {
        let mut tokens = self.accepted.clone();
        if let Some(token) = self.target_token {
            tokens.push(token);
        }
        tokens
    }

    /// Returns the number of emitted token ids.
    #[must_use]
    pub fn emitted_len(&self) -> usize {
        self.accepted.len() + usize::from(self.target_token.is_some())
    }
}

/// Checks that a probability vector is finite, non-negative and bounded by one.
///
/// # Errors
///
/// Returns [`EngineError::EmptyDistribution`] for an empty slice and
/// [`EngineError::InvalidProbability`] for an offending entry.
pub fn validate_distribution(distribution: &[f64]) -> Result<(), EngineError> {
    if distribution.is_empty() {
        return Err(EngineError::EmptyDistribution);
    }
    for (index, &value) in distribution.iter().enumerate() {
        check_probability(value, index)?;
    }
    Ok(())
}

/// Builds a sampling distribution from logits.
///
/// For `temperature == 0` the result is one-hot at the argmax (ties resolve to
/// the lowest index); otherwise it is a numerically stable softmax of
/// `logits / temperature`. No sampling parameters other than temperature are
/// applied here.
///
/// # Errors
///
/// Returns [`EngineError::EmptyDistribution`] for empty logits,
/// [`EngineError::InvalidLogits`] for non-finite logits,
/// [`EngineError::InvalidParams`] for a bad temperature and
/// [`EngineError::InvalidProbability`] if the result cannot be normalised.
pub fn probabilities_from_logits(
    logits: &[f32],
    temperature: f64,
) -> Result<Vec<f64>, EngineError> {
    if logits.is_empty() {
        return Err(EngineError::EmptyDistribution);
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(EngineError::InvalidParams {
            reason: "speculative temperature must be finite and non-negative",
        });
    }
    let finite: Vec<f64> = logits
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let wide = f64::from(value);
            if wide.is_finite() {
                Ok(wide)
            } else {
                Err(EngineError::InvalidLogits { index })
            }
        })
        .collect::<Result<_, _>>()?;

    if temperature == 0.0 {
        let mut best = 0usize;
        for (index, &value) in finite.iter().enumerate().skip(1) {
            if value > finite[best] {
                best = index;
            }
        }
        let mut probabilities = vec![0.0; finite.len()];
        probabilities[best] = 1.0;
        return Ok(probabilities);
    }

    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = finite
        .iter()
        .map(|&value| ((value - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::InvalidProbability {
            index: 0,
            reason: "logits do not form a finite distribution",
        });
    }
    let probabilities: Vec<f64> = weights.iter().map(|&weight| weight / total).collect();
    validate_distribution(&probabilities)?;
    Ok(probabilities)
}

/// Computes the clipped acceptance ratio `min(1, p_target / p_draft)`.
///
/// A zero draft probability is treated as automatic acceptance: it cannot
/// argue against a non-negative target probability. Both inputs must be finite
/// probabilities in `[0, 1]`.
///
/// # Errors
///
/// Returns [`EngineError::InvalidProbability`] for an invalid input.
pub fn acceptance_ratio(target_p: f64, draft_p: f64) -> Result<f64, EngineError> {
    check_probability(target_p, 0)?;
    check_probability(draft_p, 0)?;
    if draft_p == 0.0 {
        return Ok(1.0);
    }
    Ok((target_p / draft_p).min(1.0))
}

/// Checks one drafted token against target and draft distributions.
///
/// Both slices must have the same length and must already be temperature-scaled
/// (see [`probabilities_from_logits`]). `draw` must be in `[0, 1)`. With one-hot
/// greedy distributions the result is deterministic and independent of `draw`.
///
/// # Errors
///
/// Returns [`EngineError::EmptyDistribution`],
/// [`EngineError::SpeculativeShapeMismatch`],
/// [`EngineError::InvalidProbability`],
/// [`EngineError::SpeculativeTokenOutOfRange`],
/// [`EngineError::InvalidDraw`] or [`EngineError::InvalidParams`].
pub fn check_token(
    target: &[f64],
    draft: &[f64],
    token_index: usize,
    draw: f64,
    temperature: f64,
) -> Result<TokenCheck, EngineError> {
    validate_distribution(target)?;
    validate_distribution(draft)?;
    if target.len() != draft.len() {
        return Err(EngineError::SpeculativeShapeMismatch {
            expected: target.len(),
            actual: draft.len(),
        });
    }
    if token_index >= target.len() {
        return Err(EngineError::SpeculativeTokenOutOfRange {
            index: token_index,
            vocab_size: target.len(),
        });
    }
    check_draw(draw)?;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(EngineError::InvalidParams {
            reason: "speculative temperature must be finite and non-negative",
        });
    }
    let ratio = acceptance_ratio(target[token_index], draft[token_index])?;
    Ok(TokenCheck {
        token_index,
        accepted: draw < ratio,
        ratio,
    })
}

/// Builds the normalised residual correction `max(p_target - p_draft, 0)`.
///
/// # Errors
///
/// Returns [`EngineError::EmptyDistribution`],
/// [`EngineError::SpeculativeShapeMismatch`],
/// [`EngineError::InvalidProbability`] or
/// [`EngineError::ZeroDistributionMass`] when every residual is zero.
pub fn residual_distribution(target: &[f64], draft: &[f64]) -> Result<Vec<f64>, EngineError> {
    validate_distribution(target)?;
    validate_distribution(draft)?;
    if target.len() != draft.len() {
        return Err(EngineError::SpeculativeShapeMismatch {
            expected: target.len(),
            actual: draft.len(),
        });
    }
    let residual: Vec<f64> = target
        .iter()
        .zip(draft)
        .map(|(&target_p, &draft_p)| (target_p - draft_p).max(0.0))
        .collect();
    let total: f64 = residual.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::ZeroDistributionMass);
    }
    let normalized: Vec<f64> = residual.iter().map(|&value| value / total).collect();
    validate_distribution(&normalized)?;
    Ok(normalized)
}

/// Picks the index whose cumulative weight contains `draw`.
///
/// The weights need not sum to one, but their total must be finite and
/// positive. A draw of zero selects the highest-weight entry (ties resolve to
/// the lowest index), which is how greedy steps use this helper.
///
/// # Errors
///
/// Returns [`EngineError::EmptyDistribution`],
/// [`EngineError::InvalidProbability`], [`EngineError::InvalidDraw`] or
/// [`EngineError::ZeroDistributionMass`].
pub fn sample_index(weights: &[f64], draw: f64) -> Result<usize, EngineError> {
    validate_distribution(weights)?;
    check_draw(draw)?;
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::ZeroDistributionMass);
    }
    let mut threshold = draw * total;
    for (index, &weight) in weights.iter().enumerate() {
        threshold -= weight;
        if threshold < 0.0 {
            return Ok(index);
        }
    }
    Ok(weights.len() - 1)
}

/// Runs one speculative step with `target` and `draft`.
///
/// The draft backend proposes `config.draft_len` tokens with the seeded RNG.
/// The target backend then verifies them in order; the first rejection stops
/// verification and emits a residual-correction token. If all tokens pass, a
/// bonus token is sampled from the target distribution after the full draft.
/// Greedy steps do not advance the RNG.
///
/// # Errors
///
/// Propagates validation, distribution and backend errors, including
/// [`EngineError::SpeculativeShapeMismatch`] when the backends disagree on
/// vocabulary size.
pub fn speculative_step(
    target: &dyn Backend,
    draft: &dyn Backend,
    context: &[i32],
    config: &SpeculativeConfig,
) -> Result<SpeculativeDecision, EngineError> {
    step_with_seed(target, draft, context, config, config.seed)
}

/// Repeatedly runs speculative steps until `config.max_tokens` are emitted.
///
/// Each step deterministically derives its RNG seed from `config.seed` and the
/// step index, so two runs with the same config and backends emit the same ids.
///
/// # Errors
///
/// Returns the same errors as [`speculative_step`].
pub fn speculative_generate(
    target: &dyn Backend,
    draft: &dyn Backend,
    prompt: &[i32],
    config: &SpeculativeConfig,
) -> Result<Vec<i32>, EngineError> {
    config.validate()?;
    let mut context = prompt.to_vec();
    let mut generated = Vec::new();
    let mut step = 0usize;
    while generated.len() < config.max_tokens {
        let step_seed = config
            .seed
            .wrapping_add(u64::try_from(step).unwrap_or(u64::MAX));
        let decision = step_with_seed(target, draft, &context, config, step_seed)?;
        let tokens = decision.tokens();
        if tokens.is_empty() {
            break;
        }
        for token in tokens {
            context.push(token);
            generated.push(token);
        }
        step = step.saturating_add(1);
    }
    generated.truncate(config.max_tokens);
    Ok(generated)
}

fn step_with_seed(
    target: &dyn Backend,
    draft: &dyn Backend,
    context: &[i32],
    config: &SpeculativeConfig,
    seed: u64,
) -> Result<SpeculativeDecision, EngineError> {
    config.validate()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut draft_context = context.to_vec();
    let mut drafted: Vec<(i32, Vec<f64>)> = Vec::with_capacity(config.draft_len);

    for _ in 0..config.draft_len {
        let logits = draft.forward(&draft_context)?;
        let probabilities = probabilities_from_logits(&logits, config.temperature)?;
        let index = sample_index(&probabilities, draw(&mut rng, config.temperature))?;
        let token = token_id(index)?;
        drafted.push((token, probabilities));
        draft_context.push(token);
    }

    let mut verification_context = context.to_vec();
    let mut accepted = Vec::with_capacity(config.draft_len);
    let mut outcomes = Vec::with_capacity(config.draft_len);

    for &(token, ref draft_probabilities) in &drafted {
        let target_logits = target.forward(&verification_context)?;
        let target_probabilities = probabilities_from_logits(&target_logits, config.temperature)?;
        if target_probabilities.len() != draft_probabilities.len() {
            return Err(EngineError::SpeculativeShapeMismatch {
                expected: target_probabilities.len(),
                actual: draft_probabilities.len(),
            });
        }
        let index = usize::try_from(token).map_err(|_| EngineError::InvalidParams {
            reason: "drafted token id is not a valid vocabulary index",
        })?;

        let check = check_token(
            &target_probabilities,
            draft_probabilities,
            index,
            draw(&mut rng, config.temperature),
            config.temperature,
        )?;
        outcomes.push(if check.accepted {
            Acceptance::Accepted { ratio: check.ratio }
        } else {
            Acceptance::Rejected { ratio: check.ratio }
        });

        if check.accepted {
            accepted.push(token);
            verification_context.push(token);
            continue;
        }

        let replacement = match residual_distribution(&target_probabilities, draft_probabilities) {
            Ok(residual) => residual,
            Err(EngineError::ZeroDistributionMass) => target_probabilities.clone(),
            Err(error) => return Err(error),
        };
        let replacement_index = sample_index(&replacement, draw(&mut rng, config.temperature))?;
        return Ok(SpeculativeDecision {
            accepted,
            target_token: Some(token_id(replacement_index)?),
            outcomes,
            all_accepted: false,
        });
    }

    let bonus_logits = target.forward(&verification_context)?;
    let bonus_probabilities = probabilities_from_logits(&bonus_logits, config.temperature)?;
    let bonus_index = sample_index(&bonus_probabilities, draw(&mut rng, config.temperature))?;
    Ok(SpeculativeDecision {
        accepted,
        target_token: Some(token_id(bonus_index)?),
        outcomes,
        all_accepted: true,
    })
}

fn draw(rng: &mut StdRng, temperature: f64) -> f64 {
    if temperature == 0.0 {
        0.0
    } else {
        rng.random::<f64>()
    }
}

fn token_id(index: usize) -> Result<i32, EngineError> {
    i32::try_from(index).map_err(|_| EngineError::TokenIdOverflow { index })
}

fn check_probability(value: f64, index: usize) -> Result<(), EngineError> {
    if !value.is_finite() {
        return Err(EngineError::InvalidProbability {
            index,
            reason: "not finite",
        });
    }
    if value < 0.0 {
        return Err(EngineError::InvalidProbability {
            index,
            reason: "negative",
        });
    }
    if value > 1.0 {
        return Err(EngineError::InvalidProbability {
            index,
            reason: "greater than one",
        });
    }
    Ok(())
}

fn check_draw(draw: f64) -> Result<(), EngineError> {
    if draw.is_finite() && (0.0..1.0).contains(&draw) {
        Ok(())
    } else {
        Err(EngineError::InvalidDraw {
            bits: draw.to_bits(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ReferenceBackend;
    use proptest::prelude::*;

    #[derive(Debug)]
    struct FixedBackend {
        name: &'static str,
        logits: Vec<f32>,
    }

    impl FixedBackend {
        fn new(name: &'static str, logits: Vec<f32>) -> Self {
            Self { name, logits }
        }
    }

    impl Backend for FixedBackend {
        fn name(&self) -> &'static str {
            self.name
        }

        fn forward(&self, _tokens: &[i32]) -> Result<Vec<f32>, EngineError> {
            Ok(self.logits.clone())
        }
    }

    fn config(temperature: f64, draft_len: usize, seed: u64) -> SpeculativeConfig {
        SpeculativeConfig {
            temperature,
            draft_len,
            seed,
            max_tokens: 12,
        }
    }

    fn normalize(values: &[f64]) -> Vec<f64> {
        let total: f64 = values.iter().sum();
        values.iter().map(|&value| value / total).collect()
    }

    #[test]
    fn all_accepted_path_emits_draft_plus_bonus() {
        let target = FixedBackend::new("target", vec![-1.0, 0.0, 1.0]);
        let draft = FixedBackend::new("draft", vec![-1.0, 0.0, 1.0]);
        let settings = config(0.8, 3, 7);
        let decision = speculative_step(&target, &draft, &[4, 5], &settings).expect("step");

        assert!(decision.all_accepted);
        assert_eq!(decision.outcomes.len(), 3);
        assert!(
            decision
                .outcomes
                .iter()
                .all(|outcome| outcome.is_accepted())
        );
        assert!(
            decision
                .outcomes
                .iter()
                .all(|outcome| (outcome.ratio() - 1.0).abs() < 1e-12)
        );
        assert_eq!(decision.accepted.len(), 3);
        assert_eq!(decision.emitted_len(), 4);
        assert_eq!(decision.tokens().len(), 4);
        assert_eq!(decision.target_token, decision.tokens().last().copied());
    }

    #[test]
    fn first_token_rejection_uses_residual_correction() {
        let target = [0.2_f64, 0.8];
        let draft = [0.7_f64, 0.3];
        let check = check_token(&target, &draft, 0, 0.9, 0.8).expect("check");
        assert!(!check.accepted);
        assert!((check.ratio - 0.2 / 0.7).abs() < 1e-12);

        let residual = residual_distribution(&target, &draft).expect("residual");
        assert!(residual[0].abs() < 1e-12);
        assert!((residual[1] - 1.0).abs() < 1e-12);
        assert_eq!(sample_index(&residual, 0.5), Ok(1));

        let greedy = FixedBackend::new("target", vec![0.0, 1.0, 5.0]);
        let drafted = FixedBackend::new("draft", vec![0.0, 5.0, 1.0]);
        let decision = speculative_step(&greedy, &drafted, &[1], &config(0.0, 3, 1))
            .expect("greedy rejection");
        assert!(!decision.all_accepted);
        assert!(decision.accepted.is_empty());
        assert_eq!(decision.outcomes.len(), 1);
        assert!(!decision.outcomes[0].is_accepted());
        assert_eq!(decision.target_token, Some(2));
        assert_eq!(decision.tokens(), vec![2]);
    }

    #[test]
    fn greedy_acceptance_and_rejection_are_deterministic() {
        let target = FixedBackend::new("target", vec![0.0, 5.0, 1.0]);
        let draft = FixedBackend::new("draft", vec![0.0, 5.0, 1.0]);
        let settings = config(0.0, 4, 99);
        let accepted = speculative_step(&target, &draft, &[2], &settings).expect("step");
        assert!(accepted.all_accepted);
        assert_eq!(accepted.target_token, Some(1));
        assert!(
            accepted
                .outcomes
                .iter()
                .all(|outcome| outcome.is_accepted())
        );

        let other = FixedBackend::new("other", vec![5.0, 0.0, 1.0]);
        let rejected = speculative_step(&target, &other, &[2], &settings).expect("step");
        assert!(!rejected.all_accepted);
        assert_eq!(rejected.target_token, Some(1));
        assert_eq!(rejected.outcomes.len(), 1);
        assert!(!rejected.outcomes[0].is_accepted());
        assert!((rejected.outcomes[0].ratio() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn zero_draft_probability_is_automatic_acceptance() {
        let target = [0.2_f64, 0.8];
        let draft = [0.0_f64, 1.0];
        assert_eq!(acceptance_ratio(0.2, 0.0), Ok(1.0));
        let check = check_token(&target, &draft, 0, 0.5, 0.8).expect("check");
        assert!(check.accepted);
        assert_eq!(check.ratio, 1.0);

        let both_zero = check_token(&[0.0, 1.0], &[0.0, 1.0], 0, 0.25, 0.8).expect("check");
        assert!(both_zero.accepted);
    }

    #[test]
    fn mismatched_vocabulary_and_shape_are_rejected() {
        assert!(matches!(
            check_token(&[0.5, 0.5], &[0.5, 0.5, 0.0], 0, 0.2, 0.8),
            Err(EngineError::SpeculativeShapeMismatch {
                expected: 2,
                actual: 3
            })
        ));
        assert!(matches!(
            residual_distribution(&[0.5, 0.5], &[0.5, 0.5, 0.0]),
            Err(EngineError::SpeculativeShapeMismatch { .. })
        ));

        let target = FixedBackend::new("target", vec![0.0, 1.0, 2.0]);
        let draft = FixedBackend::new("draft", vec![0.0, 1.0]);
        assert!(matches!(
            speculative_step(&target, &draft, &[1], &config(0.0, 2, 3)),
            Err(EngineError::SpeculativeShapeMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_and_invalid_probabilities_are_rejected() {
        let valid = [0.5_f64, 0.5];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            assert!(matches!(
                check_token(&[bad, 1.0 - bad], &valid, 0, 0.2, 0.8),
                Err(EngineError::InvalidProbability { .. })
            ));
        }
        assert!(matches!(
            acceptance_ratio(f64::NAN, 0.5),
            Err(EngineError::InvalidProbability { .. })
        ));
        assert!(matches!(
            validate_distribution(&[0.5, -0.5]),
            Err(EngineError::InvalidProbability { index: 1, .. })
        ));

        let broken = FixedBackend::new("broken", vec![0.0, f32::NAN]);
        assert!(matches!(
            speculative_step(&broken, &broken, &[1], &config(0.8, 1, 3)),
            Err(EngineError::InvalidLogits { .. })
        ));
    }

    #[test]
    fn empty_vocabulary_is_rejected() {
        assert!(matches!(
            validate_distribution(&[]),
            Err(EngineError::EmptyDistribution)
        ));
        assert!(matches!(
            probabilities_from_logits(&[], 0.8),
            Err(EngineError::EmptyDistribution)
        ));
        let empty = FixedBackend::new("empty", Vec::new());
        assert!(matches!(
            speculative_step(&empty, &empty, &[1], &config(0.0, 1, 3)),
            Err(EngineError::EmptyDistribution)
        ));
    }

    #[test]
    fn invalid_draft_lengths_and_draws_are_rejected() {
        let backend = ReferenceBackend::new();
        for length in [0, MAX_DRAFT_LEN + 1] {
            let settings = config(0.7, length, 1);
            assert!(matches!(
                speculative_step(&backend, &backend, &[1], &settings),
                Err(EngineError::InvalidDraftLength { .. })
            ));
            assert!(matches!(
                speculative_generate(&backend, &backend, &[1], &settings),
                Err(EngineError::InvalidDraftLength { .. })
            ));
        }
        assert!(matches!(
            check_token(&[0.5, 0.5], &[0.5, 0.5], 0, 1.0, 0.8),
            Err(EngineError::InvalidDraw { .. })
        ));
        assert!(matches!(
            check_token(&[0.5, 0.5], &[0.5, 0.5], 2, 0.5, 0.8),
            Err(EngineError::SpeculativeTokenOutOfRange { .. })
        ));
        assert!(matches!(
            check_token(&[0.5, 0.5], &[0.5, 0.5], 0, 0.5, -0.1),
            Err(EngineError::InvalidParams { .. })
        ));
    }

    #[test]
    fn residual_without_positive_mass_is_reported() {
        let distribution = [0.5_f64, 0.5];
        assert!(matches!(
            residual_distribution(&distribution, &distribution),
            Err(EngineError::ZeroDistributionMass)
        ));
    }

    #[test]
    fn probabilities_are_stable_and_greedy_is_one_hot() {
        let probabilities =
            probabilities_from_logits(&[1000.0, 1001.0, 999.0], 0.7).expect("finite probabilities");
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(probabilities[1] > probabilities[0]);
        assert_eq!(
            probabilities_from_logits(&[2.0, 3.0, 1.0], 0.0).expect("greedy"),
            vec![0.0, 1.0, 0.0]
        );
        assert_eq!(
            probabilities_from_logits(&[2.0, 2.0, 1.0], 0.0).expect("greedy tie"),
            vec![1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn fixed_seeds_are_deterministic() {
        let target = ReferenceBackend::new();
        let draft = ReferenceBackend::new();
        let settings = config(0.7, 4, 1234);
        let first = speculative_step(&target, &draft, &[1, 2, 3], &settings).expect("step");
        let second = speculative_step(&target, &draft, &[1, 2, 3], &settings).expect("step");
        assert_eq!(first, second);

        let first_run = speculative_generate(&target, &draft, &[5, 6], &settings).expect("run");
        let second_run = speculative_generate(&target, &draft, &[5, 6], &settings).expect("run");
        assert_eq!(first_run, second_run);
        assert!(first_run.len() <= settings.max_tokens);
        assert_eq!(first_run.len(), settings.max_tokens);

        let other = config(0.7, 4, 5678);
        let different = speculative_generate(&target, &draft, &[5, 6], &other).expect("run");
        assert_ne!(first_run, different);
    }

    proptest! {
        #[test]
        fn arbitrary_finite_distributions_never_panic(
            target_raw in prop::collection::vec(0.001f64..1.0, 1..16),
            draft_raw in prop::collection::vec(0.001f64..1.0, 1..16),
            draw in 0.0f64..1.0,
            temperature in prop_oneof![Just(0.0f64), 0.1f64..2.0],
        ) {
            let length = target_raw.len().min(draft_raw.len());
            let target = normalize(&target_raw[..length]);
            let draft = normalize(&draft_raw[..length]);
            let token_index = (draw * length as f64) as usize % length;

            let check = check_token(&target, &draft, token_index, draw, temperature)
                .expect("valid distributions");
            prop_assert_eq!(check.token_index, token_index);
            prop_assert!(check.ratio.is_finite());
            prop_assert!((0.0..=1.0).contains(&check.ratio));

            match residual_distribution(&target, &draft) {
                Ok(residual) => {
                    prop_assert_eq!(residual.len(), length);
                    prop_assert!(residual.iter().all(|&value| value.is_finite() && value >= 0.0));
                    let sum: f64 = residual.iter().sum();
                    prop_assert!((sum - 1.0).abs() < 1e-9);
                }
                Err(EngineError::ZeroDistributionMass) => {}
                Err(error) => return Err(TestCaseError::fail(error.to_string())),
            }

            let index = sample_index(&target, draw).expect("sampling");
            prop_assert!(index < length);
        }

        #[test]
        fn acceptance_ratio_matches_its_definition(
            target in 0.0f64..=1.0,
            draft in 0.001f64..=1.0,
        ) {
            let ratio = acceptance_ratio(target, draft).expect("valid probabilities");
            let expected = (target / draft).min(1.0);
            prop_assert!((ratio - expected).abs() < 1e-12);
            prop_assert!(ratio >= target - 1e-12);
        }
    }
}
