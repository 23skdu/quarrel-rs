//! OpenAPI 3.1 contract and shared HTTP DTOs.

use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};

use crate::error::{ErrorDetail, ErrorEnvelope};

/// A model as reported by `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct Model {
    /// Model identifier used in requests.
    pub id: String,
    /// Always `model` for OpenAI compatibility.
    pub object: String,
    /// Owning organisation, reserved for future multi-tenant support.
    pub owned_by: String,
}

/// OpenAI model list response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ModelList {
    /// Always `list`.
    pub object: String,
    /// Available models; empty until Phase 1 lands the resolver.
    pub data: Vec<Model>,
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatMessage {
    /// One of `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Message content.
    pub content: String,
}

/// OpenAI chat completion request (subset supported by the eventual implementation).
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Stream server-sent events when true.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of generated tokens.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Legacy completion request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Prompt text.
    pub prompt: String,
    /// Maximum number of generated tokens.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: String,
    /// Input strings to embed.
    pub input: Vec<String>,
}

/// Liveness probe response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct HealthResponse {
    /// Always `ok` when the process is alive.
    pub status: String,
}

/// Readiness probe response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ReadyResponse {
    /// `ready` or `unavailable`.
    pub status: String,
    /// Whether a model is loaded (false until Phase 1).
    pub model_loaded: bool,
}

/// Generated OpenAPI document for the whole control plane.
#[derive(Debug, OpenApi)]
#[openapi(
    info(
        title = "quarrel-rs API",
        version = "0.1.0",
        description = "OpenAI-compatible inference API and operations endpoints. Inference handlers return 501 until Phase 5 of docs/roadmap.md."
    ),
    paths(
        crate::api::root,
        crate::api::healthz,
        crate::api::readyz,
        crate::api::models,
        crate::api::chat_completions,
        crate::api::completions,
        crate::api::embeddings,
        crate::api::metrics,
        crate::api::openapi_json,
        crate::api::docs,
    ),
    components(schemas(
        ErrorEnvelope,
        ErrorDetail,
        Model,
        ModelList,
        ChatMessage,
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        HealthResponse,
        ReadyResponse,
    )),
    tags(
        (name = "system", description = "Liveness, readiness and metrics"),
        (name = "openai", description = "OpenAI-compatible inference endpoints"),
    )
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::ApiDoc;
    use utoipa::OpenApi;

    #[test]
    fn document_is_valid_and_covers_all_endpoints() {
        let doc = ApiDoc::openapi();
        let json = serde_json::to_value(&doc).unwrap();
        assert_eq!(json["openapi"], "3.1.0");
        let paths = json["paths"].as_object().unwrap();
        for path in [
            "/",
            "/healthz",
            "/readyz",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/metrics",
            "/openapi.json",
            "/docs",
        ] {
            assert!(paths.contains_key(path), "missing OpenAPI path {path}");
        }
    }
}
