//! OpenAPI 3.1 contract generated from the real axum handlers.
//!
//! The DTOs themselves live in `quarrel-api`; they are re-exported here for backwards compatibility
//! with earlier versions of this crate.

use utoipa::OpenApi;

pub use quarrel_api::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ChatMessage, ChatStreamChoice, CompletionChoice, CompletionRequest, CompletionResponse,
    EmbeddingList, EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingUsage,
    EncodingFormat, ErrorDetail, ErrorEnvelope, HealthResponse, Model, ModelList, ReadyResponse,
    StringOrList, Usage,
};

/// Description embedded in the generated document; also asserted by the drift test.
pub const API_DESCRIPTION: &str = "OpenAI-compatible inference API and operations endpoints. \
Generation and embeddings use quarrel-engine's deterministic ReferenceBackend fixture \
(not Go parity); streaming responses emit chat.completion.chunk events.";

/// Generated OpenAPI document for the whole control plane.
#[derive(Debug, OpenApi)]
#[openapi(
    info(
        title = "quarrel-rs API",
        version = "0.1.0",
        description = API_DESCRIPTION
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
        ChatChoice,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatDelta,
        ChatStreamChoice,
        ChatCompletionChunk,
        CompletionRequest,
        CompletionChoice,
        CompletionResponse,
        Usage,
        StringOrList,
        EmbeddingRequest,
        EmbeddingObject,
        EmbeddingPayload,
        EmbeddingList,
        EmbeddingUsage,
        EncodingFormat,
        HealthResponse,
        ReadyResponse,
    )),
    tags(
        (name = "system", description = "Liveness, readiness and metrics"),
        (name = "openai", description = "OpenAI-compatible inference endpoints backed by the reference backend"),
    )
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::{API_DESCRIPTION, ApiDoc};
    use utoipa::OpenApi;

    #[test]
    fn document_is_valid_and_covers_all_endpoints() {
        let doc = ApiDoc::openapi();
        let json = serde_json::to_value(&doc).unwrap();
        assert_eq!(json["openapi"], "3.1.0");
        assert_eq!(json["info"]["description"], API_DESCRIPTION);
        assert!(
            json["info"]["description"]
                .as_str()
                .unwrap()
                .contains("ReferenceBackend")
        );
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
        assert!(!json.to_string().contains("501"));
    }

    #[test]
    fn all_dtos_are_registered_as_schemas() {
        let doc = serde_json::to_value(ApiDoc::openapi()).unwrap();
        let schemas = doc["components"]["schemas"].as_object().unwrap();
        for name in [
            "ChatCompletionRequest",
            "ChatCompletionResponse",
            "ChatCompletionChunk",
            "CompletionRequest",
            "CompletionResponse",
            "EmbeddingList",
            "ModelList",
            "ErrorEnvelope",
        ] {
            assert!(schemas.contains_key(name), "missing schema {name}");
        }
    }
}
