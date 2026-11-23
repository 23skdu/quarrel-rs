//! Model configuration and mmap-backed tensor access for supported GGUFs.

use crate::error::RuntimeError;
use quarrel_gguf::quant::{dequantize_bf16, dequantize_f16, dequantize_f32, embedding_lookup};
use quarrel_gguf::{DType, GgufMmap};
use quarrel_gguf::{GgufHeader, MetadataValue};
use std::path::Path;

/// RoPE coordinate convention used by a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Rotate adjacent coordinate pairs `(0,1), (2,3), …` (llama.cpp `NORM`).
    Interleaved,
    /// Rotate the first half against the second half (llama.cpp `NEOX`).
    NeoX,
}

/// Transformer family encoded by `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Llama-family models, including SmolLM2.
    Llama,
    /// Gemma 3 text models with q/k normalization and per-layer attention windows.
    Gemma3,
}

impl Architecture {
    /// GGUF metadata prefix used for this architecture's hyperparameters.
    #[must_use]
    pub const fn metadata_prefix(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Gemma3 => "gemma3",
        }
    }

    /// RoPE convention used by this family's GGUF weights.
    #[must_use]
    pub const fn rope_type(self) -> RopeType {
        match self {
            Self::Llama => RopeType::Interleaved,
            Self::Gemma3 => RopeType::NeoX,
        }
    }
}

/// Hyperparameters required to execute a transformer forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Model family.
    pub architecture: Architecture,
    /// Number of tokenizer entries and logits.
    pub vocab_size: usize,
    /// Number of transformer blocks.
    pub block_count: usize,
    /// Residual stream width.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Number of query heads.
    pub head_count: usize,
    /// Number of shared key/value heads.
    pub head_count_kv: usize,
    /// Width of one key head.
    pub key_length: usize,
    /// Width of one value head.
    pub value_length: usize,
    /// Number of dimensions rotated by RoPE.
    pub rope_dimension: usize,
    /// Rotary frequency base for full-attention layers.
    pub rope_theta: f32,
    /// Rotary frequency base for Gemma 3 sliding-window layers.
    pub sliding_rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Maximum supported context length.
    pub context_length: usize,
    /// Local attention window, when the architecture uses one.
    pub sliding_window: Option<usize>,
    /// Period of full-attention layers in Gemma 3.
    pub sliding_window_pattern: usize,
    /// Beginning-of-sequence token id.
    pub bos_token: u32,
    /// End-of-sequence token id.
    pub eos_token: u32,
    /// Whether plain text prompts receive a BOS token automatically.
    pub add_bos: bool,
}

impl ModelConfig {
    /// Width of the concatenated query projection.
    #[must_use]
    pub const fn query_dim(&self) -> usize {
        self.head_count * self.key_length
    }

    /// Width of the concatenated key projection.
    #[must_use]
    pub const fn key_dim(&self) -> usize {
        self.head_count_kv * self.key_length
    }

    /// Width of the concatenated value projection.
    #[must_use]
    pub const fn value_dim(&self) -> usize {
        self.head_count_kv * self.value_length
    }

    /// Whether layer `index` uses full attention rather than a sliding window.
    #[must_use]
    pub const fn is_global_layer(&self, index: usize) -> bool {
        match self.architecture {
            Architecture::Llama => true,
            Architecture::Gemma3 => {
                self.sliding_window_pattern > 0
                    && (index + 1).is_multiple_of(self.sliding_window_pattern)
            }
        }
    }

    /// Number of previous tokens visible to layer `index`.
    #[must_use]
    pub fn attention_window(&self, index: usize) -> usize {
        if self.is_global_layer(index) {
            self.context_length
        } else {
            self.sliding_window.unwrap_or(self.context_length)
        }
    }

    /// RoPE convention used by this model.
    #[must_use]
    pub const fn rope_type(&self) -> RopeType {
        self.architecture.rope_type()
    }

    /// RoPE frequency base for `index`.
    ///
    /// Gemma 3 uses a smaller base for local sliding-window layers and the
    /// global base only on its periodic full-attention layers.
    #[must_use]
    pub fn rope_theta_for_layer(&self, index: usize) -> f32 {
        if self.architecture == Architecture::Gemma3 && !self.is_global_layer(index) {
            self.sliding_rope_theta
        } else {
            self.rope_theta
        }
    }

    /// Score scale passed to softmax.
    ///
    /// Gemma 3 scales its query vectors before attention and uses a score scale
    /// of one; Llama-family models scale the scores directly.
    #[must_use]
    pub fn attention_scale(&self) -> f32 {
        match self.architecture {
            Architecture::Llama => 1.0 / (self.key_length as f32).sqrt(),
            Architecture::Gemma3 => 1.0,
        }
    }

    fn from_header(header: &GgufHeader, vocab_size: usize) -> Result<Self, RuntimeError> {
        let name = metadata_string(header, "general.architecture")?;
        let architecture = match name.as_str() {
            "llama" => Architecture::Llama,
            "gemma3" => Architecture::Gemma3,
            other => return Err(RuntimeError::UnsupportedArchitecture(other.to_owned())),
        };
        let prefix = architecture.metadata_prefix();
        let hidden_size = metadata_usize(header, &format!("{prefix}.embedding_length"))?;
        let key_length = optional_usize(header, &format!("{prefix}.attention.key_length"))?
            .unwrap_or_else(|| {
                hidden_size
                    / metadata_usize(header, &format!("{prefix}.attention.head_count")).unwrap_or(1)
            });
        let value_length = optional_usize(header, &format!("{prefix}.attention.value_length"))?
            .unwrap_or(key_length);
        let head_count_kv = metadata_usize(header, &format!("{prefix}.attention.head_count_kv"))?;
        let rope_dimension = optional_usize(header, &format!("{prefix}.rope.dimension_count"))?
            .unwrap_or(key_length);
        let rope_theta = metadata_f32(header, &format!("{prefix}.rope.freq_base"))
            .or_else(|_| metadata_f32(header, "rope.freq_base"))?;
        let sliding_rope_theta = optional_f32(header, "gemma3.rope.freq_base_swa")
            .or_else(|_| optional_f32(header, "rope.freq_base_swa"))?
            .unwrap_or(10_000.0);
        let sliding_window = optional_usize(header, &format!("{prefix}.attention.sliding_window"))?;
        let config = Self {
            architecture,
            vocab_size,
            block_count: metadata_usize(header, &format!("{prefix}.block_count"))?,
            hidden_size,
            intermediate_size: metadata_usize(header, &format!("{prefix}.feed_forward_length"))?,
            head_count: metadata_usize(header, &format!("{prefix}.attention.head_count"))?,
            head_count_kv,
            key_length,
            value_length,
            rope_dimension,
            rope_theta,
            sliding_rope_theta,
            rms_epsilon: metadata_f32(
                header,
                &format!("{prefix}.attention.layer_norm_rms_epsilon"),
            )?,
            context_length: metadata_usize(header, &format!("{prefix}.context_length"))?,
            sliding_window,
            sliding_window_pattern: optional_usize(header, "gemma3.sliding_window_pattern")?
                .unwrap_or(6),
            bos_token: optional_u32(header, "tokenizer.ggml.bos_token_id")?.unwrap_or(0),
            eos_token: optional_u32(header, "tokenizer.ggml.eos_token_id")?.unwrap_or(0),
            add_bos: optional_bool(header, "tokenizer.ggml.add_bos_token")?.unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), RuntimeError> {
        if self.vocab_size == 0
            || self.block_count == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.head_count == 0
            || self.head_count_kv == 0
            || self.key_length == 0
            || self.value_length == 0
            || self.rope_dimension == 0
            || self.context_length == 0
        {
            return Err(RuntimeError::Metadata {
                key: "model geometry".to_owned(),
                reason: "all model dimensions must be non-zero".to_owned(),
            });
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.sliding_rope_theta.is_finite()
            || self.sliding_rope_theta <= 0.0
        {
            return Err(RuntimeError::Metadata {
                key: "rope.freq_base".to_owned(),
                reason: "RoPE frequency bases must be finite and positive".to_owned(),
            });
        }
        if !self.head_count.is_multiple_of(self.head_count_kv) {
            return Err(RuntimeError::Metadata {
                key: "attention.head_count_kv".to_owned(),
                reason: "query head count must be divisible by KV head count".to_owned(),
            });
        }
        if self.rope_dimension > self.key_length || !self.rope_dimension.is_multiple_of(2) {
            return Err(RuntimeError::Metadata {
                key: "rope.dimension_count".to_owned(),
                reason: "RoPE dimension must be even and no larger than key width".to_owned(),
            });
        }
        if self.architecture == Architecture::Llama && self.key_length != self.value_length {
            return Err(RuntimeError::Metadata {
                key: "attention.value_length".to_owned(),
                reason: "llama models require equal key and value widths".to_owned(),
            });
        }
        Ok(())
    }
}

/// Preloaded F32 normalization vectors for every layer.
#[derive(Debug, Clone)]
pub(crate) struct LayerNorms {
    pub attn: Vec<f32>,
    pub ffn: Vec<f32>,
    pub post_attn: Option<Vec<f32>>,
    pub post_ffw: Option<Vec<f32>>,
    pub q: Option<Vec<f32>>,
    pub k: Option<Vec<f32>>,
}

/// A memory-mapped GGUF model with preloaded normalization weights.
///
/// Quantized transformer weights stay in the read-only mapping; only small
/// F32/F16 normalization vectors and tokenizer metadata are copied to the heap.
pub struct Model {
    mmap: GgufMmap,
    config: ModelConfig,
    output_norm: Vec<f32>,
    layer_norms: Vec<LayerNorms>,
}

impl Model {
    /// Open and validate a GGUF model.
    ///
    /// # Errors
    ///
    /// Returns an error when the file is not a supported GGUF, when required
    /// metadata is missing, or when normalization tensors have wrong shapes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let mmap = GgufMmap::open(path)?;
        let header = mmap.header();
        let config = ModelConfig::from_header(header, metadata_vocab_size(&mmap, header)?)?;
        let output_norm = read_f32(&mmap, "output_norm.weight", config.hidden_size)?;
        let mut layer_norms = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            let q = if config.architecture == Architecture::Gemma3 {
                Some(read_f32(
                    &mmap,
                    &format!("{prefix}.attn_q_norm.weight"),
                    config.key_length,
                )?)
            } else {
                None
            };
            let k = if config.architecture == Architecture::Gemma3 {
                Some(read_f32(
                    &mmap,
                    &format!("{prefix}.attn_k_norm.weight"),
                    config.key_length,
                )?)
            } else {
                None
            };
            let (post_attn, post_ffw) = if config.architecture == Architecture::Gemma3 {
                (
                    Some(read_f32(
                        &mmap,
                        &format!("{prefix}.post_attention_norm.weight"),
                        config.hidden_size,
                    )?),
                    Some(read_f32(
                        &mmap,
                        &format!("{prefix}.post_ffw_norm.weight"),
                        config.hidden_size,
                    )?),
                )
            } else {
                (None, None)
            };
            layer_norms.push(LayerNorms {
                attn: read_f32(
                    &mmap,
                    &format!("{prefix}.attn_norm.weight"),
                    config.hidden_size,
                )?,
                ffn: read_f32(
                    &mmap,
                    &format!("{prefix}.ffn_norm.weight"),
                    config.hidden_size,
                )?,
                post_attn,
                post_ffw,
                q,
                k,
            });
        }
        Ok(Self {
            mmap,
            config,
            output_norm,
            layer_norms,
        })
    }

    /// Model hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Parsed GGUF header, including tokenizer metadata arrays.
    #[must_use]
    pub fn header(&self) -> &GgufHeader {
        self.mmap.header()
    }

    /// Dequantize one token embedding row into `out`.
    ///
    /// The row width must be the model's hidden size. Quantized weights remain
    /// mmap-backed; only the requested row is decoded.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid token, wrong output length, or a tensor
    /// shape that does not match the configuration.
    pub fn embedding(&self, token: u32, out: &mut [f32]) -> Result<(), RuntimeError> {
        let tensor = self.tensor("token_embd.weight")?;
        let expected = vec![
            self.config.hidden_size as u64,
            self.config.vocab_size as u64,
        ];
        if tensor.shape() != expected {
            return Err(RuntimeError::TensorShape {
                name: "token_embd.weight".to_owned(),
                shape: tensor.shape().to_vec(),
                expected,
            });
        }
        embedding_lookup(
            tensor.as_bytes(),
            tensor.dtype(),
            token as usize,
            self.config.hidden_size,
            out,
        )?;
        Ok(())
    }

    /// Borrow a tensor by name.
    pub(crate) fn tensor(&self, name: &str) -> Result<quarrel_gguf::TensorView<'_>, RuntimeError> {
        Ok(self.mmap.tensor(name)?)
    }

    pub(crate) fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }

    pub(crate) fn layer_norms(&self, layer: usize) -> &LayerNorms {
        &self.layer_norms[layer]
    }
}

fn read_f32(mmap: &GgufMmap, name: &str, expected: usize) -> Result<Vec<f32>, RuntimeError> {
    let tensor = mmap.tensor(name)?;
    let expected_u64 = u64::try_from(expected).map_err(|_| RuntimeError::Metadata {
        key: name.to_owned(),
        reason: "expected length does not fit u64".to_owned(),
    })?;
    if tensor.shape() != [expected_u64] || tensor.element_count() != expected_u64 {
        return Err(RuntimeError::TensorShape {
            name: name.to_owned(),
            shape: tensor.shape().to_vec(),
            expected: vec![expected_u64],
        });
    }
    let mut out = vec![0.0; expected];
    match tensor.dtype() {
        DType::F32 => dequantize_f32(tensor.as_bytes(), &mut out)?,
        DType::F16 => dequantize_f16(tensor.as_bytes(), &mut out)?,
        DType::BF16 => dequantize_bf16(tensor.as_bytes(), &mut out)?,
        other => {
            return Err(RuntimeError::Metadata {
                key: name.to_owned(),
                reason: format!("expected a floating-point norm tensor, found {other:?}"),
            });
        }
    }
    Ok(out)
}

fn metadata_vocab_size(mmap: &GgufMmap, header: &GgufHeader) -> Result<usize, RuntimeError> {
    if let Some(size) = optional_usize(header, "llama.vocab_size")? {
        return Ok(size);
    }
    let architecture = metadata_string(header, "general.architecture")?;
    let from_metadata = optional_usize(header, &format!("{architecture}.vocab_size"))?;
    // Gemma 3 GGUFs commonly omit the vocabulary size. The token embedding
    // shape is authoritative in that case: `[hidden_size, vocab_size]`.
    let embedding = mmap.tensor("token_embd.weight")?;
    let shape = embedding.shape();
    let from_tensor = if shape.len() == 2 {
        usize::try_from(shape[1]).map_err(|_| RuntimeError::TensorShape {
            name: "token_embd.weight".to_owned(),
            shape: shape.to_vec(),
            expected: vec![shape[0], u64::MAX],
        })?
    } else {
        0
    };
    let from_tokens = match header.metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(values)) => values.len(),
        _ => 0,
    };
    match (from_metadata, from_tensor, from_tokens) {
        (Some(metadata), tensor, tokens) => {
            if tensor != 0 && tensor != metadata {
                return Err(RuntimeError::Metadata {
                    key: "vocab_size".to_owned(),
                    reason: format!(
                        "metadata says {metadata} but token_embd.weight has {tensor} rows"
                    ),
                });
            }
            if tokens != 0 && tokens != metadata {
                return Err(RuntimeError::Metadata {
                    key: "vocab_size".to_owned(),
                    reason: format!("metadata says {metadata} but tokenizer has {tokens} tokens"),
                });
            }
            Ok(metadata)
        }
        (None, tensor, tokens) => {
            if tensor != 0 && tokens != 0 && tensor != tokens {
                return Err(RuntimeError::Metadata {
                    key: "vocab_size".to_owned(),
                    reason: format!(
                        "token_embd.weight has {tensor} rows but tokenizer has {tokens} tokens"
                    ),
                });
            }
            let size = tensor.max(tokens);
            if size == 0 {
                return Err(RuntimeError::Metadata {
                    key: "vocab_size".to_owned(),
                    reason: "no vocabulary size metadata found".to_owned(),
                });
            }
            Ok(size)
        }
    }
}

fn metadata_string(header: &GgufHeader, key: &str) -> Result<String, RuntimeError> {
    match header.metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(value) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected string, found {value:?}"),
        }),
        None => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: "missing key".to_owned(),
        }),
    }
}

fn optional_string(header: &GgufHeader, key: &str) -> Result<Option<String>, RuntimeError> {
    if !header.metadata.contains_key(key) {
        return Ok(None);
    }
    metadata_string(header, key).map(Some)
}

fn metadata_u64(header: &GgufHeader, key: &str) -> Result<u64, RuntimeError> {
    match header.metadata.get(key) {
        Some(MetadataValue::Uint64(value)) => Ok(*value),
        Some(MetadataValue::Uint32(value)) => Ok(u64::from(*value)),
        Some(value) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected unsigned integer, found {value:?}"),
        }),
        None => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: "missing key".to_owned(),
        }),
    }
}

fn optional_u64(header: &GgufHeader, key: &str) -> Result<Option<u64>, RuntimeError> {
    if !header.metadata.contains_key(key) {
        return Ok(None);
    }
    metadata_u64(header, key).map(Some)
}

fn metadata_usize(header: &GgufHeader, key: &str) -> Result<usize, RuntimeError> {
    let value = metadata_u64(header, key)?;
    usize::try_from(value).map_err(|_| RuntimeError::Metadata {
        key: key.to_owned(),
        reason: format!("value {value} does not fit usize"),
    })
}

fn optional_usize(header: &GgufHeader, key: &str) -> Result<Option<usize>, RuntimeError> {
    optional_u64(header, key)?
        .map(|value| {
            usize::try_from(value).map_err(|_| RuntimeError::Metadata {
                key: key.to_owned(),
                reason: format!("value {value} does not fit usize"),
            })
        })
        .transpose()
}

fn optional_u32(header: &GgufHeader, key: &str) -> Result<Option<u32>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Uint32(value)) => Ok(Some(*value)),
        Some(MetadataValue::Uint64(value)) => {
            Some(u32::try_from(*value).map_err(|_| RuntimeError::Metadata {
                key: key.to_owned(),
                reason: format!("value {value} does not fit u32"),
            }))
            .transpose()
        }
        Some(value) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected unsigned 32-bit integer, found {value:?}"),
        }),
    }
}

fn metadata_f32(header: &GgufHeader, key: &str) -> Result<f32, RuntimeError> {
    match header.metadata.get(key) {
        Some(MetadataValue::Float32(value)) => Ok(*value),
        Some(MetadataValue::Float64(value)) => Ok(*value as f32),
        Some(value) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected float, found {value:?}"),
        }),
        None => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: "missing key".to_owned(),
        }),
    }
}

fn optional_f32(header: &GgufHeader, key: &str) -> Result<Option<f32>, RuntimeError> {
    if !header.metadata.contains_key(key) {
        return Ok(None);
    }
    metadata_f32(header, key).map(Some)
}

fn optional_bool(header: &GgufHeader, key: &str) -> Result<Option<bool>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected bool, found {value:?}"),
        }),
    }
}

pub(crate) fn optional_metadata_string(
    header: &GgufHeader,
    key: &str,
) -> Result<Option<String>, RuntimeError> {
    optional_string(header, key)
}
