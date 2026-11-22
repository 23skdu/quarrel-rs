//! Transport-independent domain model, configuration and errors for `quarrel-rs`.
//!
//! This crate deliberately has no dependency on a transport (HTTP, Flight), an inference
//! backend, or an async runtime. It owns the small value types exchanged between crates:
//! typed identifiers, sampling parameters, generation requests and KV-cache metadata.
//!
//! The library is split into three modules:
//!
//! * [`config`] — command-line/environment/JSON configuration.
//! * [`chat_template`] — a small, dependency-free Jinja subset for rendering prompts.
//! * [`resolver`] — filesystem discovery and resolution of GGUF models.

#![deny(missing_docs)]

pub mod chat_template;
pub mod config;
pub mod resolver;

use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

/// Every fallible operation in `quarrel-core` reports errors through this enum.
///
/// The enum is non-exhaustive so new error categories can be added without a breaking change.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// A request identifier was rejected.
    #[error("invalid request id: {0}")]
    InvalidRequestId(String),

    /// A model identifier was rejected.
    #[error("invalid model id: {0}")]
    InvalidModelId(String),

    /// A parameter was outside its documented range or otherwise malformed.
    #[error("invalid parameter: {0}")]
    InvalidParams(String),

    /// A generation request contained no prompt tokens.
    #[error("prompt must contain at least one token")]
    EmptyPrompt,

    /// Configuration could not be parsed or was internally inconsistent.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Model discovery or name resolution failed.
    #[error("model resolver: {0}")]
    Resolver(String),

    /// A chat template could not be parsed or rendered.
    #[error("chat template: {0}")]
    Template(String),
}

/// An opaque identifier that correlates a generation request with its streamed responses.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
#[serde(transparent)]
pub struct RequestId(String);

impl RequestId {
    /// Creates a request id, rejecting the empty string.
    #[must_use = "returns an error for an empty request id"]
    pub fn new(value: &str) -> Result<Self, CoreError> {
        if value.is_empty() {
            return Err(CoreError::InvalidRequestId(
                "request id must not be empty".to_owned(),
            ));
        }
        Ok(Self(value.to_owned()))
    }

    /// Returns the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for RequestId {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for RequestId {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(&value).map_err(serde::de::Error::custom)
    }
}

/// An identifier for a model, normally the file stem of a discovered `.gguf` file.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a model id, rejecting the empty string.
    #[must_use = "returns an error for an empty model id"]
    pub fn new(value: &str) -> Result<Self, CoreError> {
        if value.is_empty() {
            return Err(CoreError::InvalidModelId(
                "model id must not be empty".to_owned(),
            ));
        }
        Ok(Self(value.to_owned()))
    }

    /// Returns the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelId {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<DeserializerType>(
        deserializer: DeserializerType,
    ) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(&value).map_err(serde::de::Error::custom)
    }
}

/// A monotonically increasing index into a paged KV-cache block table.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BlockId(u32);

impl BlockId {
    /// Wraps a raw block index.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw block index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Why the engine stopped producing tokens for a generation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A stop token or stop sequence was reached.
    Stop,
    /// The token or context-length budget was exhausted.
    Length,
    /// The model requested a tool call.
    ToolCalls,
    /// The client disconnected or the request was cancelled.
    Cancelled,
}

impl FinishReason {
    /// Returns the stable wire/serialization representation of this reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A single sampled token and its log-probability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// Tokenizer vocabulary index.
    pub id: i32,
    /// Natural log of the token probability under the sampling distribution.
    pub logprob: f32,
}

impl Token {
    /// Creates a token from its vocabulary id and log-probability.
    #[must_use]
    pub const fn new(id: i32, logprob: f32) -> Self {
        Self { id, logprob }
    }
}

/// A dense embedding vector returned by an embedding request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding components, in model output order.
    pub values: Vec<f32>,
}

impl Embedding {
    /// Creates an embedding from its components.
    #[must_use]
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Returns the number of components in the embedding.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// On-disk element type of a raw KV-cache block payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvDtype {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 half precision.
    F16,
    /// TurboQuant-compressed values.
    #[serde(rename = "turboquant")]
    TurboQuant,
}

impl KvDtype {
    /// Returns the canonical lowercase string used by the Flight schema.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::TurboQuant => "turboquant",
        }
    }

    /// Parses a canonical dtype string (case-sensitive).
    ///
    /// This inherent helper mirrors [`std::str::FromStr`], which is also implemented.
    #[allow(clippy::should_implement_trait)]
    #[must_use = "returns an error for an unknown dtype"]
    pub fn from_str(value: &str) -> Result<Self, CoreError> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "turboquant" => Ok(Self::TurboQuant),
            other => Err(CoreError::InvalidParams(format!(
                "unknown KV cache dtype: {other}"
            ))),
        }
    }
}

impl FromStr for KvDtype {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_str(value)
    }
}

/// Raw bytes for one layer of one KV-cache block of one request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KvBlock {
    /// Request that owns this block.
    pub request_id: RequestId,
    /// Index of the block in the request's block table.
    pub block_id: BlockId,
    /// Transformer layer that produced the cached keys and values.
    pub layer_id: u16,
    /// Element type of the payload.
    pub dtype: KvDtype,
    /// Opaque tensor bytes; the engine interprets them according to `dtype`.
    pub bytes: Vec<u8>,
}

impl KvBlock {
    /// Assembles a KV-cache block from its components.
    #[must_use]
    pub fn new(
        request_id: RequestId,
        block_id: BlockId,
        layer_id: u16,
        dtype: KvDtype,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            request_id,
            block_id,
            layer_id,
            dtype,
            bytes,
        }
    }
}

/// Sampling controls for a generation request.
///
/// [`SamplingParams::new`] returns permissive defaults; call [`SamplingParams::validate`]
/// before handing parameters to an engine. [`GenerateRequest::new`] does this automatically.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` selects greedy decoding; valid range: `0.0..=2.0`.
    pub temperature: f32,
    /// Nucleus-sampling mass. Valid range: `0.0..=1.0`.
    pub top_p: f32,
    /// Top-k truncation; `0` disables it. Valid range: `0..=100_000`.
    pub top_k: u32,
    /// Penalty applied to already-generated tokens. Valid range: `0.5..=2.0`.
    pub repetition_penalty: f32,
    /// Maximum number of tokens to generate. Valid range: `1..=1_000_000`.
    pub max_tokens: u32,
    /// Vocabulary ids that terminate generation; at most 256 entries.
    pub stop_token_ids: Vec<i32>,
}

impl SamplingParams {
    /// The default temperature.
    pub const DEFAULT_TEMPERATURE: f32 = 0.8;
    /// The default nucleus-sampling mass.
    pub const DEFAULT_TOP_P: f32 = 0.95;
    /// The default top-k value (`0` disables top-k).
    pub const DEFAULT_TOP_K: u32 = 0;
    /// The default repetition penalty.
    pub const DEFAULT_REPETITION_PENALTY: f32 = 1.0;
    /// The default maximum number of generated tokens.
    pub const DEFAULT_MAX_TOKENS: u32 = 256;
    /// The maximum number of accepted stop-token ids.
    pub const MAX_STOP_TOKEN_IDS: usize = 256;
    /// The largest accepted top-k value.
    pub const MAX_TOP_K: u32 = 100_000;
    /// The largest accepted token budget.
    pub const MAX_MAX_TOKENS: u32 = 1_000_000;

    /// Creates sampling parameters with the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks every parameter against its documented range.
    #[must_use = "validation errors must be handled before starting generation"]
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(CoreError::InvalidParams(format!(
                "temperature must be finite and in 0.0..=2.0, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(CoreError::InvalidParams(format!(
                "top_p must be finite and in 0.0..=1.0, got {}",
                self.top_p
            )));
        }
        if !self.repetition_penalty.is_finite() || !(0.5..=2.0).contains(&self.repetition_penalty) {
            return Err(CoreError::InvalidParams(format!(
                "repetition_penalty must be finite and in 0.5..=2.0, got {}",
                self.repetition_penalty
            )));
        }
        if !(1..=Self::MAX_MAX_TOKENS).contains(&self.max_tokens) {
            return Err(CoreError::InvalidParams(format!(
                "max_tokens must be in 1..={}, got {}",
                Self::MAX_MAX_TOKENS,
                self.max_tokens
            )));
        }
        if self.top_k > Self::MAX_TOP_K {
            return Err(CoreError::InvalidParams(format!(
                "top_k must be at most {}, got {}",
                Self::MAX_TOP_K,
                self.top_k
            )));
        }
        if self.stop_token_ids.len() > Self::MAX_STOP_TOKEN_IDS {
            return Err(CoreError::InvalidParams(format!(
                "stop_token_ids must contain at most {} entries, got {}",
                Self::MAX_STOP_TOKEN_IDS,
                self.stop_token_ids.len()
            )));
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Self::DEFAULT_TEMPERATURE,
            top_p: Self::DEFAULT_TOP_P,
            top_k: Self::DEFAULT_TOP_K,
            repetition_penalty: Self::DEFAULT_REPETITION_PENALTY,
            max_tokens: Self::DEFAULT_MAX_TOKENS,
            stop_token_ids: Vec::new(),
        }
    }
}

/// A validated request to generate tokens for a prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Identifier echoed on every response for this request.
    pub request_id: RequestId,
    /// Model to generate with.
    pub model: ModelId,
    /// Pre-tokenized prompt; must contain at least one token id.
    pub token_ids: Vec<i32>,
    /// Sampling controls, already validated by [`GenerateRequest::new`].
    pub params: SamplingParams,
    /// Whether the engine should emit incremental token messages.
    pub stream: bool,
}

impl GenerateRequest {
    /// Creates a request, validating `params` and rejecting an empty prompt.
    #[must_use = "returns an error for invalid params or an empty prompt"]
    pub fn new(
        request_id: RequestId,
        model: ModelId,
        token_ids: Vec<i32>,
        params: SamplingParams,
        stream: bool,
    ) -> Result<Self, CoreError> {
        params.validate()?;
        if token_ids.is_empty() {
            return Err(CoreError::EmptyPrompt);
        }
        Ok(Self {
            request_id,
            model,
            token_ids,
            params,
            stream,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use proptest::prelude::*;
    use serde::{Serialize, de::DeserializeOwned};

    use super::*;

    fn hash_of(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn serde_round_trip<T>(value: &T) -> T
    where
        T: Serialize + DeserializeOwned,
    {
        let json = serde_json::to_string(value).expect("serialize value");
        serde_json::from_str(&json).expect("deserialize value")
    }

    #[test]
    fn request_id_valid_invalid_and_display() {
        let id = RequestId::new("req-42").expect("valid request id");
        assert_eq!(id.as_str(), "req-42");
        assert_eq!(id.to_string(), "req-42");
        assert_eq!("req-42".parse::<RequestId>().expect("parse id"), id);
        assert_eq!(serde_round_trip(&id), id);
        assert_eq!(
            serde_json::to_string(&id).expect("serialize id"),
            "\"req-42\""
        );

        assert!(matches!(
            RequestId::new(""),
            Err(CoreError::InvalidRequestId(_))
        ));
        assert!(matches!(
            "".parse::<RequestId>(),
            Err(CoreError::InvalidRequestId(_))
        ));
        assert!(serde_json::from_str::<RequestId>("\"\"").is_err());
        assert_eq!(
            hash_of(&id),
            hash_of(&RequestId::new("req-42").expect("id"))
        );
        assert_ne!(hash_of(&id), hash_of(&RequestId::new("other").expect("id")));
    }

    #[test]
    fn model_id_valid_invalid_and_display() {
        let id = ModelId::new("llama-3.2").expect("valid model id");
        assert_eq!(id.as_str(), "llama-3.2");
        assert_eq!(id.to_string(), "llama-3.2");
        assert_eq!("llama-3.2".parse::<ModelId>().expect("parse id"), id);
        assert_eq!(serde_round_trip(&id), id);
        assert!(matches!(
            ModelId::new(""),
            Err(CoreError::InvalidModelId(_))
        ));
        assert!(serde_json::from_str::<ModelId>("\"\"").is_err());
    }

    #[test]
    fn block_id_new_get_display_and_serde() {
        let block = BlockId::new(7);
        assert_eq!(block.get(), 7);
        assert_eq!(block.to_string(), "7");
        assert_eq!(serde_round_trip(&block), block);
        assert_eq!(serde_json::to_string(&block).expect("serialize block"), "7");
        assert!(BlockId::new(1) < BlockId::new(2));
    }

    #[test]
    fn finish_reason_strings_and_serde() {
        let cases = [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCalls, "tool_calls"),
            (FinishReason::Cancelled, "cancelled"),
        ];
        for (reason, expected) in cases {
            assert_eq!(reason.as_str(), expected);
            assert_eq!(serde_round_trip(&reason), reason);
            assert_eq!(
                serde_json::to_string(&reason).expect("serialize reason"),
                format!("\"{expected}\"")
            );
        }
    }

    #[test]
    fn token_and_embedding_constructors() {
        let token = Token::new(11, -0.25);
        assert_eq!(token.id, 11);
        assert_eq!(token.logprob, -0.25);
        assert_eq!(serde_round_trip(&token), token);

        let empty = Embedding::new(Vec::new());
        assert_eq!(empty.dim(), 0);
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(embedding.dim(), 3);
        assert_eq!(embedding.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(serde_round_trip(&embedding), embedding);
    }

    #[test]
    fn kv_dtype_strings_and_parsing() {
        let cases = [
            (KvDtype::F32, "f32"),
            (KvDtype::F16, "f16"),
            (KvDtype::TurboQuant, "turboquant"),
        ];
        for (dtype, expected) in cases {
            assert_eq!(dtype.as_str(), expected);
            assert_eq!(KvDtype::from_str(expected).expect("parse dtype"), dtype);
            assert_eq!(expected.parse::<KvDtype>().expect("FromStr dtype"), dtype);
            assert_eq!(serde_round_trip(&dtype), dtype);
            assert_eq!(
                serde_json::to_string(&dtype).expect("serialize dtype"),
                format!("\"{expected}\"")
            );
        }
        assert!(matches!(
            KvDtype::from_str("F32"),
            Err(CoreError::InvalidParams(_))
        ));
    }

    #[test]
    fn kv_block_fields_and_serde() {
        let request_id = RequestId::new("req-1").expect("request id");
        let block = KvBlock::new(
            request_id.clone(),
            BlockId::new(3),
            12,
            KvDtype::F16,
            vec![0_u8, 1, 2, 3],
        );
        assert_eq!(block.request_id, request_id);
        assert_eq!(block.block_id.get(), 3);
        assert_eq!(block.layer_id, 12);
        assert_eq!(block.dtype, KvDtype::F16);
        assert_eq!(block.bytes, vec![0_u8, 1, 2, 3]);
        assert_eq!(serde_round_trip(&block), block);
    }

    #[test]
    fn sampling_params_defaults_are_valid() {
        let params = SamplingParams::new();
        assert_eq!(params, SamplingParams::default());
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.max_tokens, 256);
        assert!(params.stop_token_ids.is_empty());
        params.validate().expect("defaults are valid");
        assert_eq!(serde_round_trip(&params), params);
    }

    #[test]
    fn sampling_params_validate_accepts_boundaries() {
        let mut params = SamplingParams::new();
        params.temperature = 0.0;
        params.top_p = 0.0;
        params.repetition_penalty = 0.5;
        params.max_tokens = 1;
        params.top_k = 0;
        params.stop_token_ids = vec![0; 256];
        params.validate().expect("lower boundaries are valid");

        params.temperature = 2.0;
        params.top_p = 1.0;
        params.repetition_penalty = 2.0;
        params.max_tokens = 1_000_000;
        params.top_k = 100_000;
        params.validate().expect("upper boundaries are valid");
    }

    #[test]
    fn sampling_params_validate_rejects_each_invalid_field() {
        let mut params = SamplingParams::new();
        params.temperature = f32::NAN;
        assert!(params.validate().is_err());
        params.temperature = f32::INFINITY;
        assert!(params.validate().is_err());
        params.temperature = 2.1;
        assert!(params.validate().is_err());

        params = SamplingParams::new();
        params.top_p = -0.1;
        assert!(params.validate().is_err());
        params.top_p = f32::NEG_INFINITY;
        assert!(params.validate().is_err());

        params = SamplingParams::new();
        params.repetition_penalty = 0.49;
        assert!(params.validate().is_err());
        params.repetition_penalty = 2.01;
        assert!(params.validate().is_err());

        params = SamplingParams::new();
        params.max_tokens = 0;
        assert!(params.validate().is_err());
        params.max_tokens = 1_000_001;
        assert!(params.validate().is_err());

        params = SamplingParams::new();
        params.top_k = 100_001;
        assert!(params.validate().is_err());

        params = SamplingParams::new();
        params.stop_token_ids = vec![1; 257];
        assert!(matches!(
            params.validate(),
            Err(CoreError::InvalidParams(_))
        ));
    }

    #[test]
    fn generate_request_constructor_validates_inputs() {
        let id = RequestId::new("req-1").expect("request id");
        let model = ModelId::new("model-a").expect("model id");
        let params = SamplingParams::new();

        let request = GenerateRequest::new(
            id.clone(),
            model.clone(),
            vec![1, 2, 3],
            params.clone(),
            true,
        )
        .expect("valid request");
        assert_eq!(request.request_id, id);
        assert_eq!(request.model, model);
        assert_eq!(request.token_ids, vec![1, 2, 3]);
        assert_eq!(request.params, params);
        assert!(request.stream);
        assert_eq!(serde_round_trip(&request), request);

        assert!(matches!(
            GenerateRequest::new(id.clone(), model.clone(), Vec::new(), params.clone(), false),
            Err(CoreError::EmptyPrompt)
        ));

        let mut bad_params = params;
        bad_params.temperature = -1.0;
        assert!(matches!(
            GenerateRequest::new(id, model, vec![1], bad_params, false),
            Err(CoreError::InvalidParams(_))
        ));
    }

    #[test]
    fn core_error_display_and_equality() {
        let cases = [
            (
                CoreError::InvalidRequestId("x".to_owned()),
                "invalid request id: x",
            ),
            (
                CoreError::InvalidModelId("m".to_owned()),
                "invalid model id: m",
            ),
            (
                CoreError::InvalidParams("range".to_owned()),
                "invalid parameter: range",
            ),
            (
                CoreError::EmptyPrompt,
                "prompt must contain at least one token",
            ),
            (
                CoreError::Config("bad".to_owned()),
                "invalid configuration: bad",
            ),
            (
                CoreError::Resolver("missing".to_owned()),
                "model resolver: missing",
            ),
            (
                CoreError::Template("bad tag".to_owned()),
                "chat template: bad tag",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
            assert_eq!(&error, &error);
        }
        assert_ne!(
            CoreError::Config("a".to_owned()),
            CoreError::Config("b".to_owned())
        );
    }

    proptest! {
        #[test]
        fn sampling_params_validate_never_panics(
            temperature in any::<f32>(),
            top_p in any::<f32>(),
            top_k in any::<u32>(),
            repetition_penalty in any::<f32>(),
            max_tokens in any::<u32>(),
            stop_token_ids in prop::collection::vec(any::<i32>(), 0..300),
        ) {
            let stop_count = stop_token_ids.len();
            let params = SamplingParams {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_tokens,
                stop_token_ids,
            };
            if params.validate().is_ok() {
                prop_assert!(temperature.is_finite() && (0.0..=2.0).contains(&temperature));
                prop_assert!(top_p.is_finite() && (0.0..=1.0).contains(&top_p));
                prop_assert!(repetition_penalty.is_finite() && (0.5..=2.0).contains(&repetition_penalty));
                prop_assert!((1..=SamplingParams::MAX_MAX_TOKENS).contains(&max_tokens));
                prop_assert!(top_k <= SamplingParams::MAX_TOP_K);
                prop_assert!(stop_count <= SamplingParams::MAX_STOP_TOKEN_IDS);
            }
        }
    }
}
