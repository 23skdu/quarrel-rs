//! axum router, handlers and middleware for the OpenAI-compatible control plane.
//!
//! Inference is delegated to `quarrel-api`'s [`InferenceService`], which uses the deterministic
//! `ReferenceBackend` fixture. It is not a Go-parity model.

use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::Body,
    extract::rejection::JsonRejection,
    extract::{Request, State},
    http::header::{AUTHORIZATION, CONTENT_TYPE},
    middleware::{Next, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use quarrel_api::{
    ChatCompletionRequest, ChatCompletionResponse, ChatEventSse, ChatMessage, CompletionRequest,
    CompletionResponse, EmbeddingList, EmbeddingRequest, ErrorEnvelope, HealthResponse,
    InferenceService, ModelList, OllamaChatRequest, OllamaChatResponse, OllamaGenerateRequest,
    OllamaGenerateResponse, RateLimiter, ReadyResponse, WEBUI_HTML,
};
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, trace::TraceLayer};
use tracing::warn;
use utoipa::OpenApi;

use crate::{config::Config, error::ApiError, metrics::Metrics, openapi::ApiDoc};

/// Maximum accepted request body (1 MiB). Larger requests are rejected with HTTP 413.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

/// State shared by every handler. Immutable after construction apart from the bounded limiter.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Parsed CLI/environment configuration.
    pub config: Arc<Config>,
    /// Prometheus recorder handle.
    pub metrics: Metrics,
    /// Model discovery and deterministic reference inference.
    pub service: Arc<InferenceService>,
    /// Bounded per-key request limiter shared by all `/v1/*` handlers.
    pub limiter: Arc<RateLimiter>,
}

impl AppState {
    /// Creates application state with the production limits.
    #[must_use]
    pub fn new(config: Config, metrics: Metrics) -> Self {
        let service = Arc::new(InferenceService::new(config.models_dir.clone()));
        let limiter = Arc::new(RateLimiter::default());
        Self {
            config: Arc::new(config),
            metrics,
            service,
            limiter,
        }
    }
}

/// Build the complete router, including middleware, limits and Swagger UI.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(docs))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .fallback(fallback)
        .layer(from_fn_with_state(state.clone(), observe_requests))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        .with_state(state)
}

/// Spawn the periodic exporter maintenance task.
#[must_use]
pub fn spawn_upkeep(metrics: Metrics) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(30));
        loop {
            ticker.tick().await;
            metrics.run_upkeep();
        }
    })
}

/// Middleware that authenticates and rate-limits `/v1/*`, records metrics and starts a span.
async fn observe_requests(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_owned();
    let endpoint = endpoint_label(&path);
    let started = std::time::Instant::now();

    let response = if path.starts_with("/v1/") || is_ollama_inference_path(&path) {
        if !is_authorized(&request, state.config.api_key.as_deref()) {
            warn!(%path, "rejected unauthenticated inference request");
            ApiError::Unauthorized.into_response()
        } else {
            let key = rate_limit_key(&request);
            match state.limiter.check(&key) {
                Ok(()) => next.run(request).await,
                Err(error) => {
                    warn!(%path, %key, "rate limit exceeded");
                    error.into_response()
                }
            }
        }
    } else {
        next.run(request).await
    };

    let status = response.status().as_u16();
    if status >= 500 {
        state.metrics.record_engine_error("http_server_error");
    }
    state
        .metrics
        .record_request(endpoint, &method, status, started.elapsed());
    response
}

/// Ollama shims carry inference traffic, so they share the `/v1/*` auth and rate-limit policy.
fn is_ollama_inference_path(path: &str) -> bool {
    matches!(path, "/api/generate" | "/api/chat")
}

fn is_authorized(request: &Request, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    bearer_token(request).is_some_and(|token| token == expected)
}

/// Uses the bearer token as the limiter key when available, otherwise a dedicated API-key header,
/// otherwise a single anonymous bucket. Tokens are never logged in full except by `warn!` above.
fn rate_limit_key(request: &Request) -> String {
    bearer_token(request)
        .or_else(|| {
            request
                .headers()
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .unwrap_or("anonymous")
        .to_owned()
}

fn bearer_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

/// Collapse concrete URLs into a bounded label set to keep Prometheus cardinality low.
fn endpoint_label(path: &str) -> &'static str {
    match path {
        "/" => "root",
        "/healthz" => "healthz",
        "/readyz" => "readyz",
        "/metrics" => "metrics",
        "/v1/models" => "v1_models",
        "/v1/chat/completions" => "v1_chat_completions",
        "/v1/completions" => "v1_completions",
        "/v1/embeddings" => "v1_embeddings",
        "/api/generate" => "api_generate",
        "/api/chat" => "api_chat",
        _ if path.starts_with("/docs") || path == "/openapi.json" => "docs",
        _ => "other",
    }
}

/// Generated OpenAPI 3.1 document.
#[utoipa::path(
    get,
    path = "/openapi.json",
    tag = "system",
    responses((status = 200, description = "OpenAPI document", body = String))
)]
async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

/// Swagger UI page for the generated contract. Assets are pinned from unpkg.
#[utoipa::path(
    get,
    path = "/docs",
    tag = "system",
    responses((status = 200, description = "Swagger UI", body = String))
)]
async fn docs() -> HtmlPage {
    HtmlPage(
        r##"<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>quarrel-rs API</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.31.0/swagger-ui.css">
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5.31.0/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url: "/openapi.json", dom_id: "#swagger-ui"});</script>
</body></html>"##
            .to_owned(),
    )
}

/// Self-contained WebUI with model discovery, chat (including SSE streaming) and embeddings.
#[utoipa::path(
    get,
    path = "/",
    tag = "system",
    responses((status = 200, description = "Embedded WebUI", body = String))
)]
async fn root() -> HtmlPage {
    HtmlPage(WEBUI_HTML.to_owned())
}

/// Newtype so the root handler can return HTML without another dependency.
struct HtmlPage(String);

impl IntoResponse for HtmlPage {
    fn into_response(self) -> Response {
        ([(CONTENT_TYPE, "text/html; charset=utf-8")], self.0).into_response()
    }
}

/// Liveness probe.
#[utoipa::path(
    get,
    path = "/healthz",
    tag = "system",
    responses((status = 200, description = "Process is alive", body = HealthResponse))
)]
async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse::ok())
}

/// Readiness probe. Reports whether GGUF models were discovered.
#[utoipa::path(
    get,
    path = "/readyz",
    tag = "system",
    responses(
        (status = 200, description = "Ready", body = ReadyResponse),
        (status = 503, description = "Not ready", body = ErrorEnvelope)
    )
)]
async fn readyz(State(state): State<AppState>) -> Result<Json<ReadyResponse>, ApiError> {
    if !state.config.models_dir.is_dir() {
        return Err(ApiError::Unavailable(format!(
            "models directory {:?} does not exist",
            state.config.models_dir
        )));
    }
    Ok(Json(ReadyResponse::new(
        !state.service.models()?.data.is_empty(),
    )))
}

/// Unknown routes return an OpenAI-shaped 404 instead of axum's plain-text default.
async fn fallback(uri: axum::http::Uri) -> Result<Json<()>, ApiError> {
    Err(ApiError::NotFound(uri.path().to_owned()))
}

/// Prometheus exposition endpoint.
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "system",
    responses((status = 200, description = "Prometheus text exposition", body = String))
)]
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        state.metrics.render(),
    )
}

/// OpenAI model list, discovered from `config.models_dir` by `quarrel_core::resolver`.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "openai",
    responses(
        (status = 200, description = "Model list", body = ModelList),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn models(State(state): State<AppState>) -> Result<Json<ModelList>, ApiError> {
    Ok(Json(state.service.models()?))
}

/// OpenAI chat completions. `stream: true` returns `text/event-stream` chunks terminated by
/// `data: [DONE]`; heartbeats keep idle streams alive.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "openai",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion JSON, or chat.completion.chunk SSE when stream=true", body = ChatCompletionResponse),
        (status = 400, description = "Invalid request or unknown model", body = ErrorEnvelope),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 413, description = "Request body too large", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(rejection_error)?;
    if request.stream.unwrap_or(false) {
        Ok(state.service.chat_stream(&request)?.into_response())
    } else {
        Ok(Json(state.service.chat(&request)?).into_response())
    }
}

/// OpenAI legacy completions. Multiple prompts produce one choice per prompt.
/// `stream: true` uses the same `chat.completion.chunk` SSE framing as chat completions.
#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "openai",
    request_body = CompletionRequest,
    responses(
        (status = 200, description = "Text completion, or chunked SSE when stream=true", body = CompletionResponse),
        (status = 400, description = "Invalid request or unknown model", body = ErrorEnvelope),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn completions(
    State(state): State<AppState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(rejection_error)?;
    if request.stream.unwrap_or(false) {
        let prompt = request.prompt.as_slice().join("\n");
        let chat_request = ChatCompletionRequest {
            model: request.model,
            messages: vec![ChatMessage::new("user", prompt)],
            stream: Some(true),
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stop: None,
        };
        let sse: ChatEventSse = state.service.chat_stream(&chat_request)?;
        Ok(sse.into_response())
    } else {
        Ok(Json(state.service.completion(&request)?).into_response())
    }
}

/// Deterministic pseudo-embeddings from reference-backend logits, as float arrays or base64.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "openai",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Embedding list", body = EmbeddingList),
        (status = 400, description = "Invalid request or unknown model", body = ErrorEnvelope),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn embeddings(
    State(state): State<AppState>,
    payload: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> Result<Json<EmbeddingList>, ApiError> {
    let Json(request) = payload.map_err(rejection_error)?;
    Ok(Json(state.service.embedding(&request)?))
}

/// Ollama-compatible text generation, backed by the same reference backend as `/v1/*`.
///
/// `stream: true` emits Ollama's newline-delimited JSON framing (`application/x-ndjson`), reusing
/// the SSE producer for pacing. This is a compatibility shim, not a Go-parity model.
#[utoipa::path(
    post,
    path = "/api/generate",
    tag = "ollama",
    request_body = OllamaGenerateRequest,
    responses(
        (status = 200, description = "Ollama generate response, or newline-delimited JSON chunks when stream=true", body = OllamaGenerateResponse),
        (status = 400, description = "Invalid request or unknown model", body = ErrorEnvelope),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 413, description = "Request body too large", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn ollama_generate(
    State(state): State<AppState>,
    payload: Result<Json<OllamaGenerateRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(rejection_error)?;
    if request.stream.unwrap_or(false) {
        let stream = state.service.ollama_generate_stream(&request)?;
        Ok((
            [(CONTENT_TYPE, "application/x-ndjson")],
            Body::from_stream(stream),
        )
            .into_response())
    } else {
        Ok(Json(state.service.ollama_generate(&request)?).into_response())
    }
}

/// Ollama-compatible chat, backed by the same reference backend as `/v1/chat/completions`.
///
/// `stream: true` emits Ollama's newline-delimited JSON framing (`application/x-ndjson`).
#[utoipa::path(
    post,
    path = "/api/chat",
    tag = "ollama",
    request_body = OllamaChatRequest,
    responses(
        (status = 200, description = "Ollama chat response, or newline-delimited JSON chunks when stream=true", body = OllamaChatResponse),
        (status = 400, description = "Invalid request or unknown model", body = ErrorEnvelope),
        (status = 401, description = "Missing or invalid API key", body = ErrorEnvelope),
        (status = 413, description = "Request body too large", body = ErrorEnvelope),
        (status = 429, description = "Rate limit exceeded", body = ErrorEnvelope)
    )
)]
async fn ollama_chat(
    State(state): State<AppState>,
    payload: Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(rejection_error)?;
    if request.stream.unwrap_or(false) {
        let stream = state.service.ollama_chat_stream(&request)?;
        Ok((
            [(CONTENT_TYPE, "application/x-ndjson")],
            Body::from_stream(stream),
        )
            .into_response())
    } else {
        Ok(Json(state.service.ollama_chat(&request)?).into_response())
    }
}

fn rejection_error(rejection: JsonRejection) -> ApiError {
    ApiError::BadRequest(rejection.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        AppState, CONTENT_TYPE, MAX_BODY_BYTES, build_router, endpoint_label, is_authorized,
        is_ollama_inference_path,
    };
    use crate::{config::Config, metrics::test_metrics};
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use clap::Parser;
    use quarrel_api::{
        ApiError, ChatCompletionRequest, ChatMessage, ErrorEnvelope, InferenceService, RateLimiter,
    };
    use std::{path::PathBuf, sync::Arc, time::Duration};
    use tower::util::ServiceExt;

    fn fixture_models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../quarrel-api/tests/fixtures/models")
    }

    fn test_state(api_key: Option<String>) -> AppState {
        let config = Config {
            api_key,
            models_dir: fixture_models_dir(),
            ..Config::parse_from(["quarrel"])
        };
        AppState::new(config, test_metrics())
    }

    async fn post_json(
        state: AppState,
        uri: &str,
        value: serde_json::Value,
    ) -> axum::response::Response {
        build_router(state)
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&value).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn error_envelope(response: axum::response::Response) -> ErrorEnvelope {
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn chat_request(model: &str) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "max_tokens": 3,
            "seed": 42
        })
    }

    #[test]
    fn endpoint_labels_are_bounded() {
        assert_eq!(
            endpoint_label("/v1/chat/completions"),
            "v1_chat_completions"
        );
        assert_eq!(endpoint_label("/api/generate"), "api_generate");
        assert_eq!(endpoint_label("/api/chat"), "api_chat");
        assert_eq!(endpoint_label("/docs/swagger-ui.js"), "docs");
        assert_eq!(endpoint_label("/unknown"), "other");
        assert!(is_ollama_inference_path("/api/chat"));
        assert!(!is_ollama_inference_path("/api/unknown"));
    }

    #[test]
    fn bearer_auth_is_constant_time_free_and_explicit() {
        let mut request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        assert!(is_authorized(&request, None));
        assert!(!is_authorized(&request, Some("secret")));

        request = Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .unwrap();
        assert!(is_authorized(&request, Some("secret")));
        assert!(!is_authorized(&request, Some("other")));
    }

    #[tokio::test]
    async fn root_serves_the_embedded_webui() {
        let response = build_router(test_state(None))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "text/html; charset=utf-8"
        );
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        for needle in [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/docs",
            "/metrics",
        ] {
            assert!(html.contains(needle), "missing WebUI control: {needle}");
        }
        assert!(!html.contains("http://") && !html.contains("https://"));
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body, r#"{"status":"ok"}"#);
    }

    #[tokio::test]
    async fn models_discover_gguf_files() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            body,
            r#"{"object":"list","data":[{"id":"reference-tiny","object":"model","owned_by":"quarrel-rs"}]}"#
        );
    }

    #[tokio::test]
    async fn models_from_empty_directory_are_empty() {
        let config = Config {
            models_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../quarrel-api/tests/fixtures/empty_models"),
            ..Config::parse_from(["quarrel"])
        };
        let state = AppState::new(config, test_metrics());
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body, r#"{"object":"list","data":[]}"#);
    }

    #[tokio::test]
    async fn readiness_reports_discovered_models() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body, r#"{"status":"ready","model_loaded":true}"#);
    }

    #[tokio::test]
    async fn api_key_is_required_for_v1_routes() {
        let response = build_router(test_state(Some("secret".to_owned())))
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            error_envelope(response).await.error.code.as_deref(),
            Some("invalid_api_key")
        );
    }

    #[tokio::test]
    async fn bearer_authentication_allows_v1_routes() {
        let response = build_router(test_state(Some("secret".to_owned())))
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_completion_returns_openai_shape() {
        let response = post_json(
            test_state(None),
            "/v1/chat/completions",
            chat_request("reference-tiny"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "reference-tiny");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completion_returns_openai_shape() {
        let response = post_json(
            test_state(None),
            "/v1/completions",
            serde_json::json!({"model": "reference-tiny", "prompt": ["a", "b"], "max_tokens": 2}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["choices"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn embeddings_return_float_or_base64_payloads() {
        let response = post_json(
            test_state(None),
            "/v1/embeddings",
            serde_json::json!({"model": "reference-tiny", "input": ["hi", "there"]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["embedding"].as_array().unwrap().len(), 32);

        let response = post_json(
            test_state(None),
            "/v1/embeddings",
            serde_json::json!({"model": "reference-tiny", "input": "hi", "encoding_format": "base64"}),
        )
        .await;
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["data"][0]["embedding"].as_str().unwrap().len() > 100);
    }

    fn ollama_generate_request(model: &str) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "prompt": "Hello!",
            "options": {"temperature": 0.7, "num_predict": 3, "seed": 42}
        })
    }

    fn ollama_chat_request(model: &str) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello!"}],
            "options": {"temperature": 0.7, "num_predict": 3, "seed": 42}
        })
    }

    async fn json_body(response: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn ollama_generate_returns_ollama_shape() {
        let response = post_json(
            test_state(None),
            "/api/generate",
            ollama_generate_request("reference-tiny"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
        let json = json_body(response).await;
        assert_eq!(json["model"], "reference-tiny");
        assert_eq!(json["response"].as_str().unwrap().chars().count(), 3);
        assert!(json["done"].as_bool().unwrap());
        assert_eq!(json["done_reason"], "length");
        assert!(json["created_at"].as_str().unwrap().ends_with('Z'));
        assert_eq!(json["eval_count"], 3);
        assert!(json.get("stats").is_none());
    }

    #[tokio::test]
    async fn ollama_chat_returns_ollama_shape() {
        let response = post_json(
            test_state(None),
            "/api/chat",
            ollama_chat_request("reference-tiny"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["message"]["role"], "assistant");
        assert!(json["done"].as_bool().unwrap());
        assert_eq!(json["done_reason"], "length");
        assert!(json["prompt_eval_count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn ollama_unknown_models_and_empty_messages_are_400() {
        for (uri, payload) in [
            ("/api/generate", ollama_generate_request("missing")),
            ("/api/chat", ollama_chat_request("missing")),
        ] {
            let response = post_json(test_state(None), uri, payload).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let envelope = error_envelope(response).await;
            assert_eq!(envelope.error.code.as_deref(), Some("invalid_model"));
            assert_eq!(envelope.error.param.as_deref(), Some("model"));
        }

        let response = post_json(
            test_state(None),
            "/api/chat",
            serde_json::json!({"model": "reference-tiny", "messages": []}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error_envelope(response).await.error.code.as_deref(),
            Some("invalid_request")
        );
    }

    #[tokio::test]
    async fn ollama_streams_newline_delimited_json() {
        let mut state = test_state(None);
        state.service = Arc::new(
            InferenceService::new(fixture_models_dir())
                .with_created_at(0)
                .with_timing(Duration::from_millis(2), Duration::from_millis(1)),
        );
        for (uri, mut payload, expected_role) in [
            (
                "/api/generate",
                ollama_generate_request("reference-tiny"),
                None,
            ),
            (
                "/api/chat",
                ollama_chat_request("reference-tiny"),
                Some("assistant"),
            ),
        ] {
            payload["stream"] = serde_json::json!(true);
            let response = post_json(state.clone(), uri, payload).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers().get(CONTENT_TYPE).unwrap(),
                "application/x-ndjson"
            );
            let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            let chunks: Vec<serde_json::Value> = text
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert!(chunks.len() > 2, "{text}");
            assert!(
                chunks[..chunks.len() - 1]
                    .iter()
                    .all(|chunk| !chunk["done"].as_bool().unwrap())
            );
            assert!(chunks.last().unwrap()["done"].as_bool().unwrap());
            if let Some(role) = expected_role {
                assert_eq!(chunks[0]["message"]["role"], role);
            }
        }
    }

    #[tokio::test]
    async fn ollama_routes_share_v1_auth_policy() {
        let response = post_json(
            test_state(Some("secret".to_owned())),
            "/api/generate",
            ollama_generate_request("reference-tiny"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            error_envelope(response).await.error.code.as_deref(),
            Some("invalid_api_key")
        );
    }

    #[tokio::test]
    async fn handler_errors_are_openai_shaped() {
        let cases = [
            (chat_request("unknown-model"), "invalid_model"),
            (
                serde_json::json!({"model": "reference-tiny", "messages": []}),
                "invalid_request",
            ),
            (
                serde_json::json!({"model": "reference-tiny", "messages": [{"role": "user", "content": "hi"}], "temperature": 5.0}),
                "invalid_request",
            ),
        ];
        for (payload, expected_code) in cases {
            let response = post_json(test_state(None), "/v1/chat/completions", payload).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let envelope = error_envelope(response).await;
            assert_eq!(envelope.error.code.as_deref(), Some(expected_code));
            assert_eq!(envelope.error.r#type, "invalid_request_error");
        }

        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from("{not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error_envelope(response).await.error.code.as_deref(),
            Some("invalid_request")
        );
    }

    #[tokio::test]
    async fn unknown_routes_are_openai_shaped() {
        let response = build_router(test_state(None))
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            error_envelope(response).await.error.code.as_deref(),
            Some("not_found")
        );
    }

    #[tokio::test]
    async fn request_size_limit_returns_413() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .method("POST")
                    .header("content-type", "application/json")
                    .header("content-length", (MAX_BODY_BYTES + 1).to_string())
                    .body(Body::from("x".repeat(MAX_BODY_BYTES + 1)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn streamed_chat_returns_sse_chunks_heartbeats_and_done() {
        let mut state = test_state(None);
        state.service = Arc::new(
            InferenceService::new(fixture_models_dir())
                .with_created_at(0)
                .with_timing(Duration::from_millis(2), Duration::from_millis(1)),
        );
        let mut request = chat_request("reference-tiny");
        request["stream"] = serde_json::json!(true);
        let response = post_json(state, "/v1/chat/completions", request).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
        let sse = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(sse.contains("chat.completion.chunk"), "{sse}");
        assert!(sse.contains("data: [DONE]"), "{sse}");
        assert!(sse.contains(":\n\n"), "missing heartbeat comment: {sse}");
        assert!(sse.contains("\"role\":\"assistant\""), "{sse}");
    }

    #[tokio::test]
    async fn rate_limiter_rejects_excess_requests_per_key() {
        let mut state = test_state(Some("secret".to_owned()));
        state.limiter = Arc::new(RateLimiter::new(2, Duration::from_secs(60)));
        for expected in [
            StatusCode::OK,
            StatusCode::OK,
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            let response = build_router(state.clone())
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .header("authorization", "Bearer secret")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            if expected == StatusCode::TOO_MANY_REQUESTS {
                assert_eq!(
                    error_envelope(response).await.error.r#type,
                    "rate_limit_error"
                );
            }
        }
        // Authentication is checked before the quota, so a bad key still yields 401.
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn openapi_document_is_served() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(doc["info"]["title"], "quarrel-rs API");
        assert!(
            doc["info"]["description"]
                .as_str()
                .unwrap()
                .contains("ReferenceBackend")
        );
    }

    #[test]
    fn api_error_reexport_matches_the_http_error_type() {
        let error = ApiError::InvalidModel("missing".to_owned());
        assert_eq!(error.error_type(), "invalid_request_error");
        assert_eq!(
            ChatMessage::new("user", "hi").content,
            ChatCompletionRequest {
                model: "m".to_owned(),
                messages: vec![ChatMessage::new("user", "hi")],
                stream: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                max_tokens: None,
                seed: None,
                stop: None,
            }
            .messages[0]
                .content
        );
    }
}
