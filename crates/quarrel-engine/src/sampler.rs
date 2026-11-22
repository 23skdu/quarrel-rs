//! Deterministic sampling over a vector of logits.
//!
//! [`Sampler`] owns a seeded RNG, so identical seeds and inputs produce
//! identical tokens. [`argmax`] and [`softmax`] are also available as standalone
//! helpers for callers that only need one step of the pipeline.

use crate::EngineError;
use crate::grammar::{GrammarState, TokenAlphabet};
use crate::to_usize;
use quarrel_core::{SamplingParams, Token};
use rand::Rng;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;

/// Lets the sampler accept either `f32` or `f64` numeric fields from core.
trait AsF64 {
    fn to_f64(self) -> f64;
}

impl AsF64 for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl AsF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

/// A deterministic sampler.
///
/// The RNG is seeded with [`Sampler::new`] and advanced once per weighted draw,
/// so sampling a sequence twice from the same seed yields the same ids. Greedy
/// draws (`temperature == 0`) do not use the RNG.
#[derive(Debug, Clone)]
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    /// Creates a sampler whose weighted draws are determined by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Samples one token from `logits` using `params`.
    ///
    /// The pipeline is: validate, repetition penalty, stop-token masking,
    /// temperature scaling (or greedy argmax), top-k truncation, nucleus
    /// truncation, softmax and a deterministic weighted draw. The returned
    /// [`Token::logprob`] is the log-softmax probability of the selected id
    /// under the *original* logits.
    ///
    /// Stop tokens are removed from the candidate set whenever another token
    /// remains. If every candidate is a stop token, [`EngineError::StopTokenSampled`]
    /// is returned instead of emitting one.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyLogits`] for an empty slice,
    /// [`EngineError::InvalidLogits`] for non-finite values,
    /// [`EngineError::InvalidParams`] for rejected sampling parameters and
    /// [`EngineError::StopTokenSampled`] when no non-stop token exists.
    pub fn sample(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        previous_tokens: &[i32],
    ) -> Result<Token, EngineError> {
        self.sample_inner(logits, logits, params, previous_tokens, false)
    }

    /// Samples one token constrained by a grammar state and advances that state.
    ///
    /// The active [`TokenAlphabet`] maps token ids to strings; every id that the
    /// state would reject is masked to `-inf` before the normal sampling
    /// pipeline runs, and the returned state has consumed the sampled token.
    /// The returned token's log-probability is computed from the *unmasked*
    /// logits, which must be finite just as for [`Self::sample`]. Only ids in
    /// `0..logits.len()` can participate.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::GrammarNoAllowedTokens`] when no alphabet id is
    /// legal in the current state, grammar errors from advancing the state, and
    /// the same parameter/logit errors as [`Self::sample`]. The state is left
    /// unchanged on any error.
    pub fn sample_with_grammar<'g>(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        previous_tokens: &[i32],
        alphabet: &TokenAlphabet,
        state: &GrammarState<'g>,
    ) -> Result<(Token, GrammarState<'g>), EngineError> {
        require_finite(logits)?;
        let mut masked = logits.to_vec();
        state.mask_logits(alphabet, &mut masked)?;
        let token = self.sample_inner(&masked, logits, params, previous_tokens, true)?;
        let mut next = state.clone();
        next.advance(alphabet, token.id)?;
        Ok((token, next))
    }

    fn sample_inner(
        &mut self,
        scores_input: &[f32],
        logprob_logits: &[f32],
        params: &SamplingParams,
        previous_tokens: &[i32],
        masked: bool,
    ) -> Result<Token, EngineError> {
        check_params(params)?;
        if masked {
            require_finite_or_neg_inf(scores_input)?;
        } else {
            require_finite(scores_input)?;
        }

        let mut scores = scores_input.to_vec();
        apply_repetition_penalty(&mut scores, params, previous_tokens, masked)?;

        let stop_mask = stop_mask(params, scores.len());
        let has_alternative = stop_mask.iter().any(|&stopped| !stopped);
        if has_alternative {
            for (index, &stopped) in stop_mask.iter().enumerate() {
                if stopped {
                    scores[index] = f32::NEG_INFINITY;
                }
            }
        }

        let temperature = params.temperature.to_f64();
        if temperature == 0.0 {
            let index = argmax_with_neg_inf(&scores)?;
            return token_at(logprob_logits, index, &stop_mask);
        }

        scale_by_temperature(&mut scores, temperature)?;

        let top_k = to_usize(params.top_k)?;
        truncate_top_k(&mut scores, top_k);

        let top_p = params.top_p.to_f64();
        if top_p > 0.0 && top_p < 1.0 {
            truncate_top_p(&mut scores, top_p)?;
        }

        let weights = softmax_weights(&scores)?;
        let distribution =
            WeightedIndex::new(&weights).map_err(|_| EngineError::InvalidLogits { index: 0 })?;
        let index = self.rng.sample(distribution);

        token_at(logprob_logits, index, &stop_mask)
    }
}

/// Returns the index of the largest finite logit.
///
/// Ties resolve to the lowest index. An empty slice yields
/// [`EngineError::EmptyLogits`] and a non-finite value yields
/// [`EngineError::InvalidLogits`].
pub fn argmax(logits: &[f32]) -> Result<usize, EngineError> {
    require_finite(logits)?;
    let mut best_index = 0usize;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best_index] {
            best_index = index;
        }
    }
    Ok(best_index)
}

/// Computes a numerically stable softmax of `logits`.
///
/// The subtraction of the maximum logit keeps `exp` in range for large
/// negative or positive inputs. The result has the same length as the input
/// and (up to floating-point rounding) sums to one.
///
/// # Errors
///
/// Returns [`EngineError::EmptyLogits`] for an empty slice,
/// [`EngineError::InvalidLogits`] for non-finite values or when the
/// exponential sum underflows to zero.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, EngineError> {
    require_finite(logits)?;
    let weights = softmax_weights(logits)?;
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(EngineError::InvalidLogits { index: 0 });
    }
    Ok(weights.iter().map(|&weight| weight / sum).collect())
}

/// Internal argmax that accepts the `-inf` sentinels used for masked tokens.
fn argmax_with_neg_inf(scores: &[f32]) -> Result<usize, EngineError> {
    if scores.is_empty() {
        return Err(EngineError::EmptyLogits);
    }
    let mut best_index = 0usize;
    for (index, &value) in scores.iter().enumerate().skip(1) {
        if value.is_nan() || value == f32::INFINITY {
            return Err(EngineError::InvalidLogits { index });
        }
        if value > scores[best_index] {
            best_index = index;
        }
    }
    let best = scores[best_index];
    if best.is_nan() || best == f32::INFINITY || best == f32::NEG_INFINITY {
        return Err(EngineError::InvalidLogits { index: best_index });
    }
    Ok(best_index)
}

fn require_finite(logits: &[f32]) -> Result<(), EngineError> {
    if logits.is_empty() {
        return Err(EngineError::EmptyLogits);
    }
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::InvalidLogits { index });
        }
    }
    Ok(())
}

/// Grammar masking legitimately uses `-inf`; all other non-finite values remain invalid.
fn require_finite_or_neg_inf(logits: &[f32]) -> Result<(), EngineError> {
    if logits.is_empty() {
        return Err(EngineError::EmptyLogits);
    }
    for (index, &value) in logits.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return Err(EngineError::InvalidLogits { index });
        }
    }
    Ok(())
}

fn check_params(params: &SamplingParams) -> Result<(), EngineError> {
    let invalid = || EngineError::InvalidParams {
        reason: "quarrel-core rejected the sampling parameters",
    };
    params.validate().map_err(|_| invalid())?;

    let temperature = params.temperature.to_f64();
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(invalid());
    }
    let top_p = params.top_p.to_f64();
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(invalid());
    }
    let repetition_penalty = params.repetition_penalty.to_f64();
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return Err(invalid());
    }
    Ok(())
}

fn apply_repetition_penalty(
    scores: &mut [f32],
    params: &SamplingParams,
    previous_tokens: &[i32],
    masked: bool,
) -> Result<(), EngineError> {
    let penalty = params.repetition_penalty.to_f64();
    if penalty == 1.0 || previous_tokens.is_empty() {
        return Ok(());
    }
    for &token in previous_tokens {
        if token < 0 {
            continue;
        }
        let index = token as usize;
        if let Some(score) = scores.get_mut(index) {
            if masked && *score == f32::NEG_INFINITY {
                continue;
            }
            // The usual GPT2-style penalty: penalise positive logits by
            // division and negative ones by multiplication.
            *score = if *score > 0.0 {
                (*score as f64 / penalty) as f32
            } else {
                (*score as f64 * penalty) as f32
            };
            if !score.is_finite() {
                return Err(EngineError::InvalidLogits { index });
            }
        }
    }
    Ok(())
}

fn stop_mask(params: &SamplingParams, length: usize) -> Vec<bool> {
    let mut mask = vec![false; length];
    for &token in &params.stop_token_ids {
        if token >= 0 && (token as usize) < length {
            mask[token as usize] = true;
        }
    }
    mask
}

fn scale_by_temperature(scores: &mut [f32], temperature: f64) -> Result<(), EngineError> {
    if temperature < 0.0 || !temperature.is_finite() {
        return Err(EngineError::InvalidParams {
            reason: "temperature must be finite and non-negative",
        });
    }
    let inverse = 1.0 / temperature;
    for (index, score) in scores.iter_mut().enumerate() {
        if *score == f32::NEG_INFINITY {
            continue;
        }
        *score = ((*score as f64) * inverse) as f32;
        if !score.is_finite() {
            return Err(EngineError::InvalidLogits { index });
        }
    }
    Ok(())
}

fn truncate_top_k(scores: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= scores.len() {
        return;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });
    for &index in order.iter().skip(top_k) {
        scores[index] = f32::NEG_INFINITY;
    }
}

fn truncate_top_p(scores: &mut [f32], top_p: f64) -> Result<(), EngineError> {
    let mut survivors: Vec<usize> = (0..scores.len())
        .filter(|&index| scores[index].is_finite())
        .collect();
    if survivors.is_empty() {
        return Err(EngineError::InvalidLogits { index: 0 });
    }
    survivors.sort_unstable_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });

    let max_score = survivors
        .iter()
        .map(|&index| scores[index])
        .fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = survivors
        .iter()
        .map(|&index| f64::from((scores[index] - max_score).exp()))
        .sum();
    let target = total * top_p;
    let mut cumulative = 0.0f64;
    let mut keep = 0usize;
    for (rank, &index) in survivors.iter().enumerate() {
        cumulative += f64::from((scores[index] - max_score).exp());
        keep = rank + 1;
        if cumulative >= target {
            break;
        }
    }
    for &index in survivors.iter().skip(keep) {
        scores[index] = f32::NEG_INFINITY;
    }
    Ok(())
}

fn softmax_weights(scores: &[f32]) -> Result<Vec<f32>, EngineError> {
    let max_score = scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if max_score == f32::NEG_INFINITY {
        return Err(EngineError::InvalidLogits { index: 0 });
    }
    Ok(scores
        .iter()
        .map(|&score| {
            if score.is_finite() {
                (score - max_score).exp()
            } else {
                0.0
            }
        })
        .collect())
}

fn token_at(logits: &[f32], index: usize, stop_mask: &[bool]) -> Result<Token, EngineError> {
    if stop_mask.get(index).copied().unwrap_or(false) {
        let token_id = i32::try_from(index).map_err(|_| EngineError::TokenIdOverflow { index })?;
        return Err(EngineError::StopTokenSampled { token_id });
    }
    let id = i32::try_from(index).map_err(|_| EngineError::TokenIdOverflow { index })?;
    let logprob = logprob(logits, index)?;
    Ok(Token { id, logprob })
}

fn logprob(logits: &[f32], index: usize) -> Result<f32, EngineError> {
    let maximum = logits[index];
    let sum: f64 = logits
        .iter()
        .map(|&value| f64::from((value - maximum).exp()))
        .sum();
    let value = -(sum.ln());
    Ok(value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32)
}

#[cfg(test)]
mod tests {
    use super::{Sampler, argmax, softmax};
    use crate::EngineError;
    use proptest::prelude::*;
    use quarrel_core::{SamplingParams, Token};

    fn params() -> SamplingParams {
        let mut params = SamplingParams::new();
        params.temperature = 0.7;
        params.top_p = 1.0;
        params.top_k = 0;
        params.repetition_penalty = 1.0;
        params.max_tokens = 8;
        params.stop_token_ids.clear();
        params
    }

    fn sample_many(seed: u64, logits: &[f32], params: &SamplingParams, count: usize) -> Vec<i32> {
        let mut sampler = Sampler::new(seed);
        (0..count)
            .map(|_| sampler.sample(logits, params, &[]).map(|token| token.id))
            .collect::<Result<Vec<_>, _>>()
            .expect("sampling should succeed")
    }

    #[test]
    fn argmax_picks_largest_and_resolves_ties_low() {
        assert_eq!(argmax(&[-1.0, 3.0, 3.0, 0.0]), Ok(1));
        assert_eq!(argmax(&[5.0]), Ok(0));
    }

    #[test]
    fn argmax_rejects_bad_input() {
        assert!(matches!(argmax(&[]), Err(EngineError::EmptyLogits)));
        assert!(matches!(
            argmax(&[0.0, f32::NAN]),
            Err(EngineError::InvalidLogits { index: 1 })
        ));
        assert!(matches!(
            argmax(&[f32::INFINITY]),
            Err(EngineError::InvalidLogits { index: 0 })
        ));
    }

    #[test]
    fn softmax_is_normalised_and_stable() {
        let probabilities = softmax(&[1000.0, 1001.0, 999.0]).expect("finite logits");
        assert_eq!(probabilities.len(), 3);
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
        assert!(
            probabilities
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(probabilities[1] > probabilities[0]);
        assert!(probabilities[0] > probabilities[2]);
    }

    #[test]
    fn softmax_rejects_bad_input() {
        assert!(matches!(softmax(&[]), Err(EngineError::EmptyLogits)));
        assert!(matches!(
            softmax(&[0.0, f32::NEG_INFINITY]),
            Err(EngineError::InvalidLogits { index: 1 })
        ));
    }

    #[test]
    fn greedy_sampling_ignores_temperature_noise() {
        let mut sampling = params();
        sampling.temperature = 0.0;
        let mut sampler = Sampler::new(123);
        let token = sampler
            .sample(&[0.0, 4.0, 1.0], &sampling, &[])
            .expect("greedy sample");
        assert_eq!(token.id, 1);
        assert!(token.logprob.is_finite());
        assert!(token.logprob < 0.0);
    }

    #[test]
    fn greedy_sampling_resolves_ties_by_index() {
        let mut sampling = params();
        sampling.temperature = 0.0;
        let mut sampler = Sampler::new(9);
        let token = sampler
            .sample(&[2.0, 2.0, 2.0], &sampling, &[])
            .expect("greedy sample");
        assert_eq!(token.id, 0);
    }

    #[test]
    fn same_seed_is_deterministic() {
        let logits = [0.2, 1.3, -0.4, 0.9];
        let left = sample_many(77, &logits, &params(), 32);
        let right = sample_many(77, &logits, &params(), 32);
        assert_eq!(left, right);
    }

    #[test]
    fn top_k_keeps_only_the_best_tokens() {
        let mut sampling = params();
        sampling.top_k = 1;
        let ids = sample_many(5, &[-5.0, 20.0, -1.0], &sampling, 16);
        assert!(ids.iter().all(|&id| id == 1), "unexpected ids: {ids:?}");
    }

    #[test]
    fn nucleus_truncation_keeps_only_the_best_token() {
        let mut sampling = params();
        sampling.top_p = 0.0001;
        let ids = sample_many(11, &[0.0, 30.0, 0.0], &sampling, 16);
        assert!(ids.iter().all(|&id| id == 1), "unexpected ids: {ids:?}");
    }

    #[test]
    fn repetition_penalty_can_change_greedy_choice() {
        let mut sampling = params();
        sampling.temperature = 0.0;
        sampling.repetition_penalty = 2.0;
        let mut sampler = Sampler::new(1);
        let token = sampler
            .sample(&[1.5, 1.0, 0.0], &sampling, &[0])
            .expect("penalised sample");
        assert_eq!(token.id, 1);
    }

    #[test]
    fn stop_tokens_are_masked_when_alternatives_exist() {
        let mut sampling = params();
        sampling.temperature = 0.0;
        sampling.stop_token_ids.push(1);
        let mut sampler = Sampler::new(2);
        let token = sampler
            .sample(&[0.0, 10.0, 0.5], &sampling, &[])
            .expect("masked sample");
        assert_ne!(token.id, 1);
    }

    #[test]
    fn all_stop_tokens_are_rejected() {
        let mut sampling = params();
        sampling.temperature = 0.0;
        sampling.stop_token_ids.push(0);
        sampling.stop_token_ids.push(1);
        let mut sampler = Sampler::new(3);
        assert!(matches!(
            sampler.sample(&[1.0, 2.0], &sampling, &[]),
            Err(EngineError::StopTokenSampled { token_id: 1 })
        ));
    }

    #[test]
    fn invalid_logits_and_params_are_rejected() {
        let mut sampler = Sampler::new(4);
        assert!(matches!(
            sampler.sample(&[], &params(), &[]),
            Err(EngineError::EmptyLogits)
        ));
        assert!(matches!(
            sampler.sample(&[0.0, f32::NAN], &params(), &[]),
            Err(EngineError::InvalidLogits { index: 1 })
        ));

        let mut bad = params();
        bad.temperature = -1.0;
        assert!(matches!(
            sampler.sample(&[0.0, 1.0], &bad, &[]),
            Err(EngineError::InvalidParams { .. })
        ));
    }

    #[test]
    fn logprob_matches_log_softmax() {
        let logits = [1.0_f64, 2.0_f64, 3.0_f64];
        let logits: Vec<f32> = logits.iter().map(|&value| value as f32).collect();
        let mut sampling = params();
        sampling.temperature = 0.0;
        let token: Token = Sampler::new(1)
            .sample(&logits, &sampling, &[])
            .expect("greedy sample");
        let probabilities = softmax(&logits).expect("finite logits");
        let expected = f64::from(probabilities[token.id as usize]).ln() as f32;
        assert!((token.logprob - expected).abs() < 1e-5);
    }

    proptest! {
        #[test]
        fn arbitrary_finite_logits_never_panic(
            logits in proptest::collection::vec(-30.0f32..30.0, 1..17),
            seed in any::<u64>(),
        ) {
            let mut sampler = Sampler::new(seed);
            let mut sampling = params();
            let _ = sampler.sample(&logits, &sampling, &[]);
            sampling.temperature = 0.0;
            let _ = sampler.sample(&logits, &sampling, &[0, 1, 99, -3]);
        }

        #[test]
        fn arbitrary_invalid_params_never_panic(
            logits in proptest::collection::vec(-30.0f32..30.0, 1..17),
            mode in 0u8..5,
            seed in any::<u64>(),
        ) {
            let mut sampling = params();
            match mode {
                0 => sampling.temperature = -1.0,
                1 => sampling.top_p = 2.0,
                2 => sampling.top_p = 1.5,
                3 => sampling.repetition_penalty = 0.0,
                _ => sampling.top_k = 0,
            }
            let mut sampler = Sampler::new(seed);
            let _ = sampler.sample(&logits, &sampling, &[7, -1, 1000]);
        }
    }
}
