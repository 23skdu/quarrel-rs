//! Deterministic routing from model/config identifiers to engine profiles.
//!
//! A GGUF file (or an inference request) usually identifies a model family with
//! a short string such as `llama` or `qwen35`. This module maps those strings to
//! a concrete [`ArchitectureProfile`]: attention/hybrid flags, a supported
//! layer range, a KV geometry and the hybrid SSM + full-attention marker used
//! by Qwen 3.5. The router is a pure function of its input: the same config
//! always yields the same profile or the same typed error.
//!
//! # Scope
//!
//! This is a **routing/config layer only**. The profiles encode defaults that
//! let the engine pick shapes and KV layouts; they are deliberately not a claim
//! of real model-weight parity, exact layer counts or bit-identical behaviour
//! with any upstream implementation. Values declared in a [`ModelArchitecture`]
//! config override the routing defaults after validation.
//!
//! ```
//! use quarrel_engine::architecture::{route, route_config, ModelArchitecture};
//!
//! let profile = route("qwen35")?;
//! assert!(profile.hybrid);
//! assert_eq!(profile.hybrid_marker.as_str(), "ssm+full_attention");
//!
//! let config = ModelArchitecture {
//!     architecture: "llama".to_owned(),
//!     layers: Some(32),
//!     ..ModelArchitecture::new("llama")
//! };
//! assert_eq!(route_config(&config)?.kv.layers, 32);
//! # Ok::<(), quarrel_engine::EngineError>(())
//! ```

use crate::{EngineError, batch::KvGeometry};
use serde::{Deserialize, Serialize};

/// Default tokens per KV block used by routing profiles.
pub const DEFAULT_KV_BLOCK_SIZE: usize = 16;

/// Full-attention layer cadence of the Qwen 3.5 hybrid SSM stack.
pub const QWEN35_ATTENTION_STRIDE: usize = 4;

/// Stable identifier for a model family understood by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArchitectureId {
    /// Llama-style dense decoder stacks.
    Llama,
    /// Qwen2 dense decoder stacks.
    Qwen2,
    /// Qwen 3.5 hybrid SSM + full-attention stacks.
    Qwen35,
    /// Mistral dense decoder stacks.
    Mistral,
    /// Gemma 4 dense decoder stacks.
    Gemma4,
    /// Phi-3 dense decoder stacks.
    Phi3,
    /// SmolLM2 dense decoder stacks.
    Smollm2,
    /// Granite dense decoder stacks.
    Granite,
}

impl ArchitectureId {
    /// Every architecture identifier supported by the router.
    pub const SUPPORTED: &'static [Self] = &[
        Self::Llama,
        Self::Qwen2,
        Self::Qwen35,
        Self::Mistral,
        Self::Gemma4,
        Self::Phi3,
        Self::Smollm2,
        Self::Granite,
    ];

    /// Parses an identifier, accepting the small set of documented aliases.
    ///
    /// Matching trims whitespace and is ASCII case-insensitive. `qwen3_5` and
    /// `qwen3-5` alias `qwen35`; `smol_lm2` aliases `smollm2`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::UnknownArchitecture`] for any other string.
    pub fn from_identifier(identifier: &str) -> Result<Self, EngineError> {
        let normalized = identifier.trim().to_ascii_lowercase();
        let id = match normalized.as_str() {
            "llama" => Self::Llama,
            "qwen2" => Self::Qwen2,
            "qwen35" | "qwen3_5" | "qwen3-5" => Self::Qwen35,
            "mistral" => Self::Mistral,
            "gemma4" => Self::Gemma4,
            "phi3" => Self::Phi3,
            "smollm2" | "smol_lm2" => Self::Smollm2,
            "granite" => Self::Granite,
            _ => {
                return Err(EngineError::UnknownArchitecture {
                    identifier: identifier.to_owned(),
                });
            }
        };
        Ok(id)
    }

    /// Returns the canonical identifier string used by configuration files.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
            Self::Mistral => "mistral",
            Self::Gemma4 => "gemma4",
            Self::Phi3 => "phi3",
            Self::Smollm2 => "smollm2",
            Self::Granite => "granite",
        }
    }
}

/// Hybrid layout marker recorded in an architecture profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridMarker {
    /// Every layer is a full-attention layer; no SSM state is needed.
    None,
    /// The stack mixes SSM layers with periodic full-attention layers.
    SsmPlusFullAttention,
}

impl HybridMarker {
    /// Parses a raw config marker (`none` or `ssm+full_attention`).
    ///
    /// Whitespace is trimmed, matching is ASCII case-insensitive and the
    /// hyphenated spelling `ssm+full-attention` is accepted as an alias.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidHybridMarker`] for any other marker.
    pub fn parse(raw: &str) -> Result<Self, EngineError> {
        let invalid = || EngineError::InvalidHybridMarker {
            marker: raw.to_owned(),
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "ssm+full_attention" | "ssm+full-attention" => Ok(Self::SsmPlusFullAttention),
            _ => Err(invalid()),
        }
    }

    /// Returns the canonical marker string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SsmPlusFullAttention => "ssm+full_attention",
        }
    }
}

/// Small typed model/config description consumed by the router.
///
/// The type derives serde so callers can deserialize it from GGUF metadata or a
/// JSON config. Only `architecture` is required; absent optional fields use the
/// routed profile's defaults. Qwen 3.5 configs must supply `hybrid_marker`
/// explicitly so an ambiguous config is rejected rather than silently routed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Model/config identifier, e.g. `llama` or `qwen35`.
    pub architecture: String,
    /// Number of layers declared by the config, when known.
    #[serde(default)]
    pub layers: Option<usize>,
    /// Number of `f32` KV features per token, when known.
    #[serde(default)]
    pub kv_dim: Option<usize>,
    /// Preferred tokens per KV block, when known.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Raw hybrid marker, required for Qwen 3.5 configs.
    #[serde(default)]
    pub hybrid_marker: Option<String>,
}

impl ModelArchitecture {
    /// Creates a config for `identifier` with every optional field unset.
    #[must_use]
    pub fn new(identifier: impl Into<String>) -> Self {
        Self {
            architecture: identifier.into(),
            layers: None,
            kv_dim: None,
            block_size: None,
            hybrid_marker: None,
        }
    }
}

/// Resolved routing profile for one architecture family.
///
/// All fields are derived deterministically by [`route`] or [`route_config`].
/// Layer counts are routing ranges, not assertions about any particular model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureProfile {
    /// Canonical family identifier.
    pub id: ArchitectureId,
    /// Canonical identifier string (same as [`ArchitectureId::as_str`]).
    pub identifier: &'static str,
    /// Whether every layer uses full self-attention.
    pub full_attention: bool,
    /// Whether the stack mixes SSM layers with full-attention layers.
    pub hybrid: bool,
    /// Smallest layer count this profile routes.
    pub min_layers: usize,
    /// Largest layer count this profile routes.
    pub max_layers: usize,
    /// Layer count used when a config does not declare one.
    pub default_layers: usize,
    /// KV geometry used by the engine, including config overrides.
    pub kv: KvGeometry,
    /// Canonical hybrid marker for this profile.
    pub hybrid_marker: HybridMarker,
    /// Full-attention cadence in hybrid stacks; `None` for pure attention.
    pub attention_stride: Option<usize>,
}

impl ArchitectureProfile {
    /// Returns whether `layers` is inside this profile's routing range.
    #[must_use]
    pub fn supports_layers(&self, layers: usize) -> bool {
        layers >= self.min_layers && layers <= self.max_layers
    }
}

/// Routes a bare identifier to its default profile.
///
/// This is the convenient path when only a GGUF `general.architecture` string
/// is available. Use [`route_config`] when the config also declares layers, KV
/// geometry or a hybrid marker that must be validated.
///
/// # Errors
///
/// Returns [`EngineError::UnknownArchitecture`] for unsupported identifiers.
pub fn route(identifier: &str) -> Result<ArchitectureProfile, EngineError> {
    Ok(base_profile(ArchitectureId::from_identifier(identifier)?))
}

/// Routes a typed config to a profile, validating every declared field.
///
/// Declared layer counts must be inside the profile's routing range and non-zero.
/// Declared `kv_dim` and `block_size` values must also be non-zero. The hybrid
/// marker must match the routed family: Qwen 3.5 requires
/// `ssm+full_attention`, while the other families require `none`. A Qwen 3.5
/// config without a marker is rejected as incomplete rather than guessed.
///
/// # Errors
///
/// Returns [`EngineError::UnknownArchitecture`],
/// [`EngineError::IncompleteArchitecture`],
/// [`EngineError::InvalidHybridMarker`],
/// [`EngineError::ArchitectureLayerRange`] or geometry validation errors.
pub fn route_config(config: &ModelArchitecture) -> Result<ArchitectureProfile, EngineError> {
    let id = ArchitectureId::from_identifier(&config.architecture)?;
    let identifier = id.as_str();
    let mut profile = base_profile(id);

    match config.hybrid_marker.as_deref() {
        Some(raw) => {
            let marker = HybridMarker::parse(raw)?;
            if marker != profile.hybrid_marker {
                return Err(EngineError::InvalidHybridMarker {
                    marker: raw.to_owned(),
                });
            }
        }
        None if profile.hybrid_marker == HybridMarker::SsmPlusFullAttention => {
            return Err(EngineError::IncompleteArchitecture {
                identifier: identifier.to_owned(),
                missing: "hybrid_marker",
            });
        }
        None => {}
    }

    let layers = match config.layers {
        Some(0) => {
            return Err(EngineError::IncompleteArchitecture {
                identifier: identifier.to_owned(),
                missing: "layers",
            });
        }
        Some(declared) if !profile.supports_layers(declared) => {
            return Err(EngineError::ArchitectureLayerRange {
                identifier: identifier.to_owned(),
                min: profile.min_layers,
                max: profile.max_layers,
                layers: declared,
            });
        }
        Some(declared) => declared,
        None => profile.default_layers,
    };

    let kv_dim = config.kv_dim.unwrap_or(profile.kv.kv_dim);
    if kv_dim == 0 {
        return Err(EngineError::IncompleteArchitecture {
            identifier: identifier.to_owned(),
            missing: "kv_dim",
        });
    }
    let block_size = config.block_size.unwrap_or(profile.kv.block_size);
    if block_size == 0 {
        return Err(EngineError::IncompleteArchitecture {
            identifier: identifier.to_owned(),
            missing: "block_size",
        });
    }

    profile.kv = KvGeometry::new(block_size, layers, kv_dim);
    profile.kv.validate()?;
    Ok(profile)
}

fn base_profile(id: ArchitectureId) -> ArchitectureProfile {
    let (min_layers, max_layers, default_layers, kv_dim, attention_stride) = match id {
        ArchitectureId::Llama => (1, 80, 32, 128, None),
        ArchitectureId::Qwen2 => (1, 80, 28, 128, None),
        ArchitectureId::Qwen35 => (1, 80, 36, 128, Some(QWEN35_ATTENTION_STRIDE)),
        ArchitectureId::Mistral => (1, 80, 32, 128, None),
        ArchitectureId::Gemma4 => (1, 64, 32, 256, None),
        ArchitectureId::Phi3 => (1, 48, 32, 128, None),
        ArchitectureId::Smollm2 => (1, 32, 16, 64, None),
        ArchitectureId::Granite => (1, 48, 24, 64, None),
    };
    let hybrid = id == ArchitectureId::Qwen35;
    ArchitectureProfile {
        id,
        identifier: id.as_str(),
        full_attention: !hybrid,
        hybrid,
        min_layers,
        max_layers,
        default_layers,
        kv: KvGeometry::new(DEFAULT_KV_BLOCK_SIZE, default_layers, kv_dim),
        hybrid_marker: if hybrid {
            HybridMarker::SsmPlusFullAttention
        } else {
            HybridMarker::None
        },
        attention_stride,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{Batch, InMemoryKvCache, KvBlock, KvBlockMut, KvCacheView};

    fn assert_profile(
        identifier: &str,
        id: ArchitectureId,
        hybrid: bool,
        min_layers: usize,
        max_layers: usize,
        kv_dim: usize,
    ) -> ArchitectureProfile {
        let profile = route(identifier).expect("routed profile");
        assert_eq!(profile.id, id);
        assert_eq!(profile.identifier, id.as_str());
        assert_eq!(profile.hybrid, hybrid);
        assert_eq!(profile.full_attention, !hybrid);
        assert_eq!(profile.min_layers, min_layers);
        assert_eq!(profile.max_layers, max_layers);
        assert!(profile.supports_layers(profile.default_layers));
        assert!(!profile.supports_layers(0));
        assert!(!profile.supports_layers(max_layers + 1));
        assert_eq!(profile.kv.kv_dim, kv_dim);
        assert_eq!(profile.kv.block_size, DEFAULT_KV_BLOCK_SIZE);
        assert_eq!(profile.kv.layers, profile.default_layers);
        assert_eq!(
            profile.kv,
            KvGeometry::new(DEFAULT_KV_BLOCK_SIZE, profile.default_layers, kv_dim)
        );
        profile
    }

    #[test]
    fn routes_llama() {
        let profile = assert_profile("llama", ArchitectureId::Llama, false, 1, 80, 128);
        assert_eq!(profile.hybrid_marker, HybridMarker::None);
        assert_eq!(profile.attention_stride, None);
    }

    #[test]
    fn routes_qwen2() {
        let profile = assert_profile("qwen2", ArchitectureId::Qwen2, false, 1, 80, 128);
        assert_eq!(profile.hybrid_marker, HybridMarker::None);
    }

    #[test]
    fn routes_qwen35_with_hybrid_ssm_marker() {
        let profile = assert_profile("qwen35", ArchitectureId::Qwen35, true, 1, 80, 128);
        assert_eq!(profile.hybrid_marker, HybridMarker::SsmPlusFullAttention);
        assert_eq!(profile.hybrid_marker.as_str(), "ssm+full_attention");
        assert_eq!(profile.attention_stride, Some(QWEN35_ATTENTION_STRIDE));
        assert_eq!(route("qwen3-5").expect("alias"), profile);
    }

    #[test]
    fn routes_mistral() {
        assert_profile("mistral", ArchitectureId::Mistral, false, 1, 80, 128);
    }

    #[test]
    fn routes_gemma4() {
        assert_profile("gemma4", ArchitectureId::Gemma4, false, 1, 64, 256);
    }

    #[test]
    fn routes_phi3() {
        assert_profile("phi3", ArchitectureId::Phi3, false, 1, 48, 128);
    }

    #[test]
    fn routes_smollm2() {
        let profile = assert_profile("smollm2", ArchitectureId::Smollm2, false, 1, 32, 64);
        assert_eq!(route(" SmolLM2 ").expect("case-insensitive"), profile);
    }

    #[test]
    fn routes_granite() {
        assert_profile("granite", ArchitectureId::Granite, false, 1, 48, 64);
    }

    #[test]
    fn unknown_architecture_is_rejected() {
        assert!(matches!(
            route("wizardlm"),
            Err(EngineError::UnknownArchitecture { .. })
        ));
        assert!(matches!(
            route_config(&ModelArchitecture::new("")),
            Err(EngineError::UnknownArchitecture { .. })
        ));
        assert_eq!(ArchitectureId::SUPPORTED.len(), 8);
    }

    #[test]
    fn incomplete_config_is_rejected() {
        let mut qwen = ModelArchitecture::new("qwen35");
        assert!(matches!(
            route_config(&qwen),
            Err(EngineError::IncompleteArchitecture {
                missing: "hybrid_marker",
                ..
            })
        ));
        qwen.hybrid_marker = Some("ssm+full_attention".to_owned());

        let mut llama = ModelArchitecture::new("llama");
        llama.layers = Some(0);
        assert!(matches!(
            route_config(&llama),
            Err(EngineError::IncompleteArchitecture {
                missing: "layers",
                ..
            })
        ));
        llama.layers = None;
        llama.kv_dim = Some(0);
        assert!(matches!(
            route_config(&llama),
            Err(EngineError::IncompleteArchitecture {
                missing: "kv_dim",
                ..
            })
        ));
        llama.kv_dim = None;
        llama.block_size = Some(0);
        assert!(matches!(
            route_config(&llama),
            Err(EngineError::IncompleteArchitecture {
                missing: "block_size",
                ..
            })
        ));
    }

    #[test]
    fn malformed_hybrid_marker_is_rejected() {
        assert!(matches!(
            HybridMarker::parse("ssm-only"),
            Err(EngineError::InvalidHybridMarker { .. })
        ));
        let mut config = ModelArchitecture::new("qwen35");
        config.hybrid_marker = Some("ssm-only".to_owned());
        assert!(matches!(
            route_config(&config),
            Err(EngineError::InvalidHybridMarker { .. })
        ));

        let mut dense = ModelArchitecture::new("llama");
        dense.hybrid_marker = Some("ssm+full_attention".to_owned());
        assert!(matches!(
            route_config(&dense),
            Err(EngineError::InvalidHybridMarker { .. })
        ));

        assert_eq!(
            HybridMarker::parse(" SSM+Full-Attention ").expect("marker"),
            HybridMarker::SsmPlusFullAttention
        );
    }

    #[test]
    fn config_overrides_are_validated_and_applied() {
        let mut config = ModelArchitecture::new("gemma4");
        config.layers = Some(48);
        config.kv_dim = Some(64);
        config.block_size = Some(8);
        config.hybrid_marker = Some("none".to_owned());
        let profile = route_config(&config).expect("override profile");
        assert_eq!(profile.kv, KvGeometry::new(8, 48, 64));
        assert_eq!(profile.kv, route_config(&config).expect("same").kv);

        config.layers = Some(65);
        assert!(matches!(
            route_config(&config),
            Err(EngineError::ArchitectureLayerRange {
                min: 1,
                max: 64,
                layers: 65,
                ..
            })
        ));
    }

    #[test]
    fn config_deserializes_with_serde() {
        use serde::Deserialize;
        use serde::de::value::{Error as ValueError, MapDeserializer, StrDeserializer};

        let entries = [("architecture", "qwen35")];
        let map = MapDeserializer::<_, ValueError>::new(entries.into_iter().map(|(key, value)| {
            (
                StrDeserializer::<'static, ValueError>::new(key),
                StrDeserializer::<'static, ValueError>::new(value),
            )
        }));
        let config = ModelArchitecture::deserialize(map).expect("deserialized config");
        assert_eq!(config.architecture, "qwen35");
        assert_eq!(config.layers, None);
        assert_eq!(config.kv_dim, None);
        assert_eq!(config.block_size, None);
        assert_eq!(config.hybrid_marker, None);

        let id = ArchitectureId::deserialize(StrDeserializer::<'static, ValueError>::new("llama"))
            .expect("architecture id");
        assert_eq!(id, ArchitectureId::Llama);
        let marker = HybridMarker::deserialize(StrDeserializer::<'static, ValueError>::new(
            "ssm_plus_full_attention",
        ))
        .expect("hybrid marker");
        assert_eq!(marker, HybridMarker::SsmPlusFullAttention);

        let mut complete = config;
        complete.hybrid_marker = Some(marker.as_str().to_owned());
        assert_eq!(
            route_config(&complete).expect("routed"),
            route("qwen35").expect("default")
        );
    }

    #[test]
    fn profile_geometry_integrates_with_batch_and_kv() {
        let profile = route("llama").expect("profile");
        let batch = Batch::from_rows([(7_u64, vec![1, 2, 3])]).expect("batch");
        let mut kv = InMemoryKvCache::new(profile.kv).expect("kv cache");
        assert_eq!(kv.geometry(), profile.kv);
        for &seq in batch.seq_ids() {
            kv.ensure_sequence(seq);
            assert_eq!(kv.seq_len(seq), Ok(0));
        }

        let dim = profile.kv.kv_dim;
        let keys = vec![1.5_f32; dim * 2];
        let values = vec![2.5_f32; dim * 2];
        assert_eq!(
            kv.write(
                7,
                0,
                0,
                KvBlock {
                    token_count: 2,
                    keys: &keys,
                    values: &values,
                },
            ),
            Ok(2)
        );
        let mut read_keys = vec![0.0_f32; dim * 2];
        let mut read_values = vec![0.0_f32; dim * 2];
        assert_eq!(
            kv.read(
                7,
                0,
                0,
                KvBlockMut {
                    token_count: 2,
                    keys: &mut read_keys,
                    values: &mut read_values,
                },
            ),
            Ok(2)
        );
        assert_eq!(read_keys, keys);
        assert_eq!(read_values, values);
        assert_eq!(kv.len(7), Ok(2));
    }
}
