//! Deterministic and bounded sampling over the native model session.

use crate::error::RuntimeError;
use crate::forward::InferenceSession;
/// Controls token generation after the prompt has been consumed.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    /// Maximum number of tokens to emit.
    pub max_new_tokens: usize,
    /// Sampling temperature; values `<= 0.0` select deterministic argmax.
    pub temperature: f32,
    /// Keep only the `top_k` highest logits when sampling.
    pub top_k: usize,
    /// Keep the smallest top logit set whose cumulative probability reaches `top_p`.
    pub top_p: f32,
    /// Stop before emitting the model's EOS token.
    pub stop_on_eos: bool,
    /// Seed for temperature sampling.
    pub seed: u64,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            stop_on_eos: true,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl GenerationOptions {
    /// Validate sampling parameters.
    ///
    /// # Errors
    ///
    /// Returns an error for negative temperature, negative `top_p`, or a
    /// `top_p` greater than one.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RuntimeError::Generation(
                "temperature must be finite and non-negative".to_owned(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(RuntimeError::Generation(
                "top_p must be between zero and one".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Generate tokens from an already-encoded prompt.
///
/// The prompt is fed through the model in order, then at most
/// [`GenerationOptions::max_new_tokens`] tokens are sampled. The returned slice
/// contains generated tokens only; the prompt tokens are not repeated.
///
/// # Errors
///
/// Propagates forward-pass, parameter and context-length errors.
pub fn generate(
    session: &mut InferenceSession<'_>,
    prompt: &[u32],
    options: &GenerationOptions,
) -> Result<Vec<u32>, RuntimeError> {
    options.validate()?;
    if prompt.is_empty() {
        return Err(RuntimeError::Generation(
            "prompt must contain at least one token".to_owned(),
        ));
    }
    let config = session.model().config();
    if prompt.len() > config.context_length {
        return Err(RuntimeError::ContextLength {
            actual: prompt.len(),
            limit: config.context_length,
        });
    }
    session.reset();

    let mut position = 0;
    let mut logits = Vec::new();
    for &token in prompt {
        logits = session.forward(token, position)?;
        position += 1;
    }

    let mut rng = Rng::new(options.seed);
    let mut generated = Vec::with_capacity(options.max_new_tokens);
    for _ in 0..options.max_new_tokens {
        let next = sample(&logits, options, &mut rng)?;
        if options.stop_on_eos && next == config.eos_token {
            break;
        }
        generated.push(next);
        if position >= config.context_length {
            break;
        }
        logits = session.forward(next, position)?;
        position += 1;
    }
    Ok(generated)
}

fn sample(logits: &[f32], options: &GenerationOptions, rng: &mut Rng) -> Result<u32, RuntimeError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Generation(
            "model returned empty or non-finite logits".to_owned(),
        ));
    }
    if options.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| RuntimeError::Generation("no logits to sample".to_owned()));
    }

    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if options.top_k > 0 {
        candidates.truncate(options.top_k.min(candidates.len()));
    }
    let tempered: Vec<f32> = candidates
        .iter()
        .map(|(_, value)| *value / options.temperature)
        .collect();
    let max = tempered.first().copied().unwrap_or(0.0);
    let mut weights: Vec<(usize, f32)> = candidates
        .iter()
        .zip(tempered)
        .map(|((index, _), value)| (*index, (value - max).exp()))
        .collect();
    let total: f64 = weights
        .iter()
        .fold(0.0, |acc, (_, weight)| acc + f64::from(*weight));
    if total <= 0.0 || !total.is_finite() {
        return Err(RuntimeError::Generation(
            "temperature sampling produced an empty distribution".to_owned(),
        ));
    }
    if options.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for (index, (_, weight)) in weights.iter().enumerate() {
            cumulative += f64::from(*weight) / total;
            keep = index + 1;
            if cumulative >= f64::from(options.top_p) {
                break;
            }
        }
        weights.truncate(keep);
    }
    let total: f64 = weights
        .iter()
        .fold(0.0, |acc, (_, weight)| acc + f64::from(*weight));
    let mut threshold = rng.next_f64() * total;
    let mut chosen = 0usize;
    for (index, weight) in weights {
        chosen = index;
        threshold -= f64::from(weight);
        if threshold <= 0.0 {
            break;
        }
    }
    Ok(chosen as u32)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn next_f64(&mut self) -> f64 {
        f64::from((self.next_u64() >> 11) as u32) / 2_097_152.0
    }
}
