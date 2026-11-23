//! Transformer forward pass and incremental KV cache.

use crate::error::RuntimeError;
use crate::model::{Architecture, Model, ModelConfig, RopeType};
use quarrel_gguf::matvec::matvec;
use quarrel_simd::{dot_f32, softmax_into};

#[derive(Debug, Clone)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Per-layer key/value state for one autoregressive generation session.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerKv>,
    len: usize,
}

impl KvCache {
    /// Allocate an empty cache for `model`.
    #[must_use]
    pub fn new(model: &Model) -> Self {
        let config = model.config();
        let layers = (0..config.block_count)
            .map(|_| LayerKv {
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect();
        Self { layers, len: 0 }
    }

    /// Number of tokens currently stored per layer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Discard all cached keys and values.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
        self.len = 0;
    }

    fn append(
        &mut self,
        layer: usize,
        key: &[f32],
        value: &[f32],
        position: usize,
    ) -> Result<(), RuntimeError> {
        if position != self.len {
            return Err(RuntimeError::Generation(format!(
                "KV cache has {} tokens, cannot append at position {position}",
                self.len
            )));
        }
        let entry = self.layers.get_mut(layer).ok_or_else(|| {
            RuntimeError::Generation(format!("layer {layer} is outside the KV cache"))
        })?;
        entry.keys.extend_from_slice(key);
        entry.values.extend_from_slice(value);
        if layer + 1 == self.layers.len() {
            self.len += 1;
        }
        Ok(())
    }
}

/// A reusable model session with its own KV cache.
pub struct InferenceSession<'a> {
    model: &'a Model,
    kv: KvCache,
}

impl<'a> InferenceSession<'a> {
    /// Create an empty session for `model`.
    #[must_use]
    pub fn new(model: &'a Model) -> Self {
        let kv = KvCache::new(model);
        Self { model, kv }
    }

    /// Borrow the session's model.
    #[must_use]
    pub const fn model(&self) -> &'a Model {
        self.model
    }

    /// Borrow the session's KV cache.
    #[must_use]
    pub const fn kv(&self) -> &KvCache {
        &self.kv
    }

    /// Reset the KV cache so the next call uses position zero.
    pub fn reset(&mut self) {
        self.kv.clear();
    }

    /// Run one token through the model and return unnormalized logits.
    ///
    /// `position` must equal the number of tokens already passed to this
    /// session, starting at zero.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid tokens/positions, unsupported tensor
    /// shapes, or non-finite intermediate values.
    pub fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>, RuntimeError> {
        let config = self.model.config();
        if token as usize >= config.vocab_size {
            return Err(RuntimeError::TokenId {
                id: token,
                vocab_size: config.vocab_size,
            });
        }
        if position >= config.context_length {
            return Err(RuntimeError::ContextLength {
                actual: position + 1,
                limit: config.context_length,
            });
        }
        if position != self.kv.len() {
            return Err(RuntimeError::Generation(format!(
                "session is at token {}, cannot run position {position}",
                self.kv.len()
            )));
        }

        let mut x = vec![0.0f32; config.hidden_size];
        self.embedding(token, &mut x)?;
        if config.architecture == Architecture::Gemma3 {
            let scale = (config.hidden_size as f32).sqrt();
            for value in &mut x {
                *value *= scale;
            }
        }

        for layer in 0..config.block_count {
            self.transformer_layer(layer, position, &mut x)?;
        }

        let mut normalized = vec![0.0f32; config.hidden_size];
        rmsnorm(
            &x,
            self.model.output_norm(),
            config.rms_epsilon,
            &mut normalized,
        )?;
        self.logits(&normalized)
    }

    fn embedding(&self, token: u32, out: &mut [f32]) -> Result<(), RuntimeError> {
        self.model.embedding(token, out)
    }

    fn project(&self, name: &str, rows: usize, input: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let tensor = self.model.tensor(name)?;
        // GGUF stores 2-D weights with the innermost (input) dimension first.
        let expected = vec![input.len() as u64, rows as u64];
        if tensor.shape() != expected {
            return Err(RuntimeError::TensorShape {
                name: name.to_owned(),
                shape: tensor.shape().to_vec(),
                expected,
            });
        }
        let mut out = vec![0.0f32; rows];
        matvec(tensor.dtype(), tensor.as_bytes(), input, &mut out)?;
        Ok(out)
    }

    fn transformer_layer(
        &mut self,
        layer: usize,
        position: usize,
        x: &mut [f32],
    ) -> Result<(), RuntimeError> {
        let config = self.model.config();
        let norms = self.model.layer_norms(layer).clone();
        let prefix = format!("blk.{layer}");
        let gemma = config.architecture == Architecture::Gemma3;

        let mut residual = x.to_vec();
        let mut normalized = vec![0.0f32; config.hidden_size];
        rmsnorm(x, &norms.attn, config.rms_epsilon, &mut normalized)?;

        let mut q = self.project(
            &format!("{prefix}.attn_q.weight"),
            config.query_dim(),
            &normalized,
        )?;
        let mut k = self.project(
            &format!("{prefix}.attn_k.weight"),
            config.key_dim(),
            &normalized,
        )?;
        let v = self.project(
            &format!("{prefix}.attn_v.weight"),
            config.value_dim(),
            &normalized,
        )?;

        if gemma {
            let q_weight = norms.q.as_deref().ok_or_else(|| RuntimeError::Metadata {
                key: format!("{prefix}.attn_q_norm.weight"),
                reason: "missing Gemma 3 q norm".to_owned(),
            })?;
            let k_weight = norms.k.as_deref().ok_or_else(|| RuntimeError::Metadata {
                key: format!("{prefix}.attn_k_norm.weight"),
                reason: "missing Gemma 3 k norm".to_owned(),
            })?;
            head_rmsnorm(
                &mut q,
                config.head_count,
                config.key_length,
                q_weight,
                config.rms_epsilon,
            )?;
            head_rmsnorm(
                &mut k,
                config.head_count_kv,
                config.key_length,
                k_weight,
                config.rms_epsilon,
            )?;
        }

        let theta = config.rope_theta_for_layer(layer);
        let rope_type = config.rope_type();
        rope(
            &mut q,
            config.head_count,
            config.key_length,
            config.rope_dimension,
            position,
            theta,
            rope_type,
        );
        rope(
            &mut k,
            config.head_count_kv,
            config.key_length,
            config.rope_dimension,
            position,
            theta,
            rope_type,
        );
        if gemma {
            // Gemma 3 folds the 1/sqrt(head_dim) attention scale into Q and
            // passes a score scale of one into softmax.
            let scale = 1.0 / (config.key_length as f32).sqrt();
            for value in &mut q {
                *value *= scale;
            }
        }
        self.kv.append(layer, &k, &v, position)?;

        let mut attended = vec![0.0f32; config.query_dim()];
        attention(
            &self.kv.layers[layer],
            &q,
            config,
            layer,
            position,
            &mut attended,
        )?;
        let attn_out = self.project(
            &format!("{prefix}.attn_output.weight"),
            config.hidden_size,
            &attended,
        )?;
        let attn_out = if gemma {
            let post = norms
                .post_attn
                .as_deref()
                .ok_or_else(|| RuntimeError::Metadata {
                    key: format!("{prefix}.post_attention_norm.weight"),
                    reason: "missing Gemma 3 post-attention norm".to_owned(),
                })?;
            let mut post_normed = vec![0.0f32; config.hidden_size];
            rmsnorm(&attn_out, post, config.rms_epsilon, &mut post_normed)?;
            post_normed
        } else {
            attn_out
        };
        for ((value, skip), attention) in x.iter_mut().zip(residual.iter()).zip(attn_out.iter()) {
            *value = *skip + *attention;
        }

        residual.copy_from_slice(x);
        rmsnorm(x, &norms.ffn, config.rms_epsilon, &mut normalized)?;
        let gate = self.project(
            &format!("{prefix}.ffn_gate.weight"),
            config.intermediate_size,
            &normalized,
        )?;
        let up = self.project(
            &format!("{prefix}.ffn_up.weight"),
            config.intermediate_size,
            &normalized,
        )?;
        let mut activated = Vec::with_capacity(config.intermediate_size);
        for (gate_value, up_value) in gate.iter().zip(up.iter()) {
            activated.push(if gemma {
                gelu_tanh(*gate_value) * *up_value
            } else {
                silu(*gate_value) * *up_value
            });
        }
        let ffn_out = self.project(
            &format!("{prefix}.ffn_down.weight"),
            config.hidden_size,
            &activated,
        )?;
        let ffn_out = if gemma {
            let post = norms
                .post_ffw
                .as_deref()
                .ok_or_else(|| RuntimeError::Metadata {
                    key: format!("{prefix}.post_ffw_norm.weight"),
                    reason: "missing Gemma 3 post-FFN norm".to_owned(),
                })?;
            let mut post_normed = vec![0.0f32; config.hidden_size];
            rmsnorm(&ffn_out, post, config.rms_epsilon, &mut post_normed)?;
            post_normed
        } else {
            ffn_out
        };
        for ((value, skip), feed_forward) in x.iter_mut().zip(residual.iter()).zip(ffn_out.iter()) {
            *value = *skip + *feed_forward;
        }
        Ok(())
    }

    fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let tensor = self.model.tensor("token_embd.weight")?;
        let config = self.model.config();
        let mut logits = vec![0.0f32; config.vocab_size];
        matvec(tensor.dtype(), tensor.as_bytes(), hidden, &mut logits)?;
        Ok(logits)
    }
}

fn rmsnorm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<(), RuntimeError> {
    if input.is_empty() || input.len() != weight.len() || input.len() != out.len() {
        return Err(RuntimeError::Numeric {
            operation: "rmsnorm",
            reason: format!(
                "input {} weight {} output {} lengths do not match",
                input.len(),
                weight.len(),
                out.len()
            ),
        });
    }
    let sum_squares = input.iter().fold(0.0f64, |acc, &value| {
        acc + f64::from(value) * f64::from(value)
    });
    let scale = 1.0 / (sum_squares / input.len() as f64 + f64::from(epsilon)).sqrt();
    // GGUF Gemma converters store the `1 + weight` shift in the tensor itself,
    // so every architecture uses the stored weight directly here.
    for ((value, &weight), dst) in input.iter().zip(weight.iter()).zip(out) {
        *dst = (*value * scale as f32) * weight;
    }
    Ok(())
}

fn head_rmsnorm(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<(), RuntimeError> {
    if heads == 0 || head_dim == 0 || values.len() != heads * head_dim || weight.len() != head_dim {
        return Err(RuntimeError::Numeric {
            operation: "qk rmsnorm",
            reason: "head geometry does not match weight length".to_owned(),
        });
    }
    let mut normalized = vec![0.0f32; head_dim];
    for head in 0..heads {
        let start = head * head_dim;
        rmsnorm(
            &values[start..start + head_dim],
            weight,
            epsilon,
            &mut normalized,
        )?;
        values[start..start + head_dim].copy_from_slice(&normalized);
    }
    Ok(())
}

fn rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
    rope_type: RopeType,
) {
    if rope_dim == 0 || heads == 0 || head_dim == 0 {
        return;
    }
    let base = f64::from(theta);
    for head in 0..heads {
        let start = head * head_dim;
        if rope_type == RopeType::Interleaved {
            for pair in 0..rope_dim / 2 {
                let index = pair * 2;
                let exponent = -(index as f64) / rope_dim as f64;
                let angle = position as f64 * base.powf(exponent);
                let (cos, sin) = (angle.cos() as f32, angle.sin() as f32);
                let first = values[start + index];
                let second = values[start + index + 1];
                values[start + index] = first.mul_add(cos, -second * sin);
                values[start + index + 1] = second.mul_add(cos, first * sin);
            }
        } else {
            let half = rope_dim / 2;
            for pair in 0..half {
                let index = pair * 2;
                let exponent = -(index as f64) / rope_dim as f64;
                let angle = position as f64 * base.powf(exponent);
                let (cos, sin) = (angle.cos() as f32, angle.sin() as f32);
                let first = values[start + pair];
                let second = values[start + half + pair];
                values[start + pair] = first.mul_add(cos, -second * sin);
                values[start + half + pair] = second.mul_add(cos, first * sin);
            }
        }
    }
}

fn attention(
    cache: &LayerKv,
    query: &[f32],
    config: &ModelConfig,
    layer: usize,
    position: usize,
    out: &mut [f32],
) -> Result<(), RuntimeError> {
    let window = config.attention_window(layer);
    let start = position.saturating_sub(window.saturating_sub(1));
    let count = position - start + 1;
    let key_total = config.key_dim();
    let value_total = config.value_dim();
    if cache.keys.len() != (position + 1) * key_total
        || cache.values.len() != (position + 1) * value_total
        || query.len() != config.query_dim()
        || out.len() != config.query_dim()
    {
        return Err(RuntimeError::Numeric {
            operation: "attention",
            reason: "KV cache or projection geometry is inconsistent".to_owned(),
        });
    }
    let scale = config.attention_scale();
    let kv_group = config.head_count / config.head_count_kv;
    let mut scores = vec![0.0f32; count];
    let mut probs = vec![0.0f32; count];

    for head in 0..config.head_count {
        let q = &query[head * config.key_length..(head + 1) * config.key_length];
        let kv_head = head / kv_group;
        for (slot, token) in (start..=position).enumerate() {
            let offset = token * key_total + kv_head * config.key_length;
            let key = &cache.keys[offset..offset + config.key_length];
            scores[slot] = dot_f32(q, key).map_err(|error| RuntimeError::Numeric {
                operation: "attention score",
                reason: error.to_string(),
            })? * scale;
        }
        softmax_into(&scores, &mut probs).map_err(|error| RuntimeError::Numeric {
            operation: "attention softmax",
            reason: error.to_string(),
        })?;
        let dst = &mut out[head * config.value_length..(head + 1) * config.value_length];
        dst.fill(0.0);
        for (slot, token) in (start..=position).enumerate() {
            let offset = token * value_total + kv_head * config.value_length;
            let value = &cache.values[offset..offset + config.value_length];
            for (dst, &value) in dst.iter_mut().zip(value) {
                *dst = probs[slot].mul_add(value, *dst);
            }
        }
    }
    Ok(())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn gelu_tanh(value: f32) -> f32 {
    // ggml's tanh-GELU uses sqrt(2/pi), not sqrt(2*pi). Keep the constant at f64
    // precision so the f32 cast happens once, at the tanh result.
    const SQRT_TWO_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let inner = SQRT_TWO_OVER_PI * f64::from(value + 0.044_715 * value * value * value);
    0.5 * value * (1.0 + inner.tanh() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_uses_gguf_scale_then_gain() {
        let input = [3.0f32, 4.0];
        let weight = [2.0f32, 2.0];
        let scale = 1.0f64 / 12.5f64.sqrt();
        let mut out = [0.0f32; 2];
        rmsnorm(&input, &weight, 0.0, &mut out).expect("valid rmsnorm");
        assert!((out[0] - 3.0 * scale as f32 * 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0 * scale as f32 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_rejects_mismatched_lengths() {
        let mut out = [0.0f32; 2];
        let error =
            rmsnorm(&[1.0, 2.0], &[1.0], 1e-5, &mut out).expect_err("length mismatch must fail");
        assert!(error.to_string().contains("lengths do not match"));
    }

    #[test]
    fn rope_interleaved_rotates_adjacent_pairs() {
        let mut values = [1.0f32, 0.0, 0.0, 1.0];
        rope(&mut values, 1, 4, 4, 1, 1.0, RopeType::Interleaved);
        let (cos, sin) = (1.0f64.cos() as f32, 1.0f64.sin() as f32);
        assert!((values[0] - cos).abs() < 1e-6);
        assert!((values[1] - sin).abs() < 1e-6);
        assert!((values[2] + sin).abs() < 1e-6);
        assert!((values[3] - cos).abs() < 1e-6);
    }

    #[test]
    fn rope_neox_rotates_split_halves() {
        let mut values = [1.0f32, 0.0, 0.0, 1.0];
        rope(&mut values, 1, 4, 4, 1, 1.0, RopeType::NeoX);
        let (cos, sin) = (1.0f64.cos() as f32, 1.0f64.sin() as f32);
        assert!((values[0] - cos).abs() < 1e-6);
        assert!((values[1] + sin).abs() < 1e-6);
        assert!((values[2] - sin).abs() < 1e-6);
        assert!((values[3] - cos).abs() < 1e-6);
    }

    #[test]
    fn head_rmsnorm_normalizes_each_head_independently() {
        let mut values = [3.0f32, 4.0, 5.0, 12.0];
        head_rmsnorm(&mut values, 2, 2, &[1.0, 1.0], 0.0).expect("valid qk norm");
        let first_scale = 1.0f64 / 12.5f64.sqrt();
        let second_scale = 1.0f64 / 84.5f64.sqrt();
        assert!((values[0] - 3.0 * first_scale as f32).abs() < 1e-6);
        assert!((values[1] - 4.0 * first_scale as f32).abs() < 1e-6);
        assert!((values[2] - 5.0 * second_scale as f32).abs() < 1e-6);
        assert!((values[3] - 12.0 * second_scale as f32).abs() < 1e-6);
    }

    #[test]
    fn ffn_activations_have_expected_shapes() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(4.0) - 4.0).abs() < 0.1);
        assert!(silu(-12.0).abs() < 1e-4);
        assert_eq!(gelu_tanh(0.0), 0.0);
        assert!((0.5..1.0).contains(&gelu_tanh(1.0)));
        assert!((gelu_tanh(4.0) - 4.0).abs() < 1e-2);
        assert!(gelu_tanh(-12.0).abs() < 1e-4);
    }
}
