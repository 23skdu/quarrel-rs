//! axum router, handlers and middleware for the OpenAI-compatible control plane.

use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{Request, State},
    http::header::CONTENT_TYPE,
    middleware::{Next, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::warn;
use utoipa::OpenApi;

use crate::{
    config::Config,
    error::ApiError,
    metrics::Metrics,
    openapi::{
        ApiDoc, ChatCompletionRequest, CompletionRequest, EmbeddingRequest, HealthResponse, Model,
        ModelList, ReadyResponse,
    },
};

/// State shared by every handler. Immutable after construction; no locks needed.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Parsed CLI/environment configuration.
    pub config: Arc<Config>,
    /// Prometheus recorder handle.
    pub metrics: Metrics,
}

/// Build the complete router, including middleware and Swagger UI.
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
        .fallback(fallback)
        .layer(from_fn_with_state(state.clone(), observe_requests))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
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

/// Middleware that authenticates `/v1/*`, records request metrics and starts a tracing span.
async fn observe_requests(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_owned();
    let endpoint = endpoint_label(&path);
    let started = std::time::Instant::now();

    let response =
        if path.starts_with("/v1/") && !is_authorized(&request, state.config.api_key.as_deref()) {
            warn!(%path, "rejected unauthenticated /v1 request");
            ApiError::Unauthorized.into_response()
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

fn is_authorized(request: &Request, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| token == expected)
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

/// Swagger UI page. Assets are pinned from unpkg; Phase 5 replaces this with an embedded bundle.
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

/// Minimal embedded WebUI placeholder (replaced by the embedded bundle in Phase 5).
#[utoipa::path(
    get,
    path = "/",
    tag = "system",
    responses((status = 200, description = "Landing page", body = String))
)]
async fn root() -> HtmlPage {
    HtmlPage(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><title>quarrel-rs</title></head>
<body><h1>quarrel-rs</h1><p>Bootstrap control plane. Inference lands in Phase 5.</p>
<ul><li><a href="/docs">OpenAPI UI</a></li><li><a href="/metrics">Prometheus metrics</a></li>
<li><a href="/healthz">Health</a></li></ul></body></html>"#
            .to_owned(),
    )
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
    Json(HealthResponse {
        status: "ok".to_owned(),
    })
}

/// Readiness probe. Reports that no model is loaded until Phase 1 is implemented.
#[utoipa::path(
    get,
    path = "/readyz",
    tag = "system",
    responses(
        (status = 200, description = "Ready", body = ReadyResponse),
        (status = 503, description = "Not ready", body = crate::error::ErrorEnvelope)
    )
)]
async fn readyz(State(state): State<AppState>) -> Result<Json<ReadyResponse>, ApiError> {
    if !state.config.models_dir.is_dir() {
        return Err(ApiError::Unavailable(format!(
            "models directory {:?} does not exist",
            state.config.models_dir
        )));
    }
    Ok(Json(ReadyResponse {
        status: "ready".to_owned(),
        model_loaded: false,
    }))
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

/// OpenAI model list. Empty until the model resolver lands in Phase 1.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "openai",
    responses((status = 200, description = "Model list", body = ModelList))
)]
async fn models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_owned(),
        data: Vec::<Model>::new(),
    })
}

/// OpenAI chat completions. Reserved for Phase 5.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "openai",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion"),
        (status = 501, description = "Not implemented until Phase 5")
    )
)]
async fn chat_completions(
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<()>, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "messages must not be empty".to_owned(),
        ));
    }
    Err(ApiError::NotImplemented("POST /v1/chat/completions"))
}

/// OpenAI completions. Reserved for Phase 5.
#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "openai",
    request_body = CompletionRequest,
    responses(
        (status = 200, description = "Completion"),
        (status = 501, description = "Not implemented until Phase 5")
    )
)]
async fn completions(Json(_request): Json<CompletionRequest>) -> Result<Json<()>, ApiError> {
    Err(ApiError::NotImplemented("POST /v1/completions"))
}

/// OpenAI embeddings. Reserved for Phase 5.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "openai",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Embeddings"),
        (status = 501, description = "Not implemented until Phase 5")
    )
)]
async fn embeddings(Json(_request): Json<EmbeddingRequest>) -> Result<Json<()>, ApiError> {
    Err(ApiError::NotImplemented("POST /v1/embeddings"))
}

#[cfg(test)]
mod tests {
    use super::{AppState, build_router, endpoint_label, is_authorized};
    use crate::{config::Config, metrics::test_metrics};
    use axum::{
        body::to_bytes,
        http::{Request, StatusCode},
    };
    use clap::Parser;
    use std::sync::Arc;
    use tower::util::ServiceExt;

    fn test_state(api_key: Option<String>) -> AppState {
        AppState {
            config: Arc::new(Config {
                api_key,
                ..Config::parse_from(["quarrel"])
            }),
            metrics: test_metrics(),
        }
    }

    #[test]
    fn endpoint_labels_are_bounded() {
        assert_eq!(
            endpoint_label("/v1/chat/completions"),
            "v1_chat_completions"
        );
        assert_eq!(endpoint_label("/docs/swagger-ui.js"), "docs");
        assert_eq!(endpoint_label("/unknown"), "other");
    }

    #[test]
    fn bearer_auth_is_constant_time_free_and_explicit() {
        let mut request = Request::builder()
            .uri("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(is_authorized(&request, None));
        assert!(!is_authorized(&request, Some("secret")));

        request = Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer secret")
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(is_authorized(&request, Some("secret")));
        assert!(!is_authorized(&request, Some("other")));
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body, r#"{"status":"ok"}"#);
    }

    #[tokio::test]
    async fn models_returns_openai_shape() {
        let response = build_router(test_state(None))
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body, r#"{"object":"list","data":[]}"#);
    }

    #[tokio::test]
    async fn api_key_is_required_for_v1_routes() {
        let response = build_router(test_state(Some("secret".to_owned())))
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
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
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(doc["info"]["title"], "quarrel-rs API");
    }
}
