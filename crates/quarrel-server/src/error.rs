//! OpenAI-shaped error responses shared by all HTTP handlers.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

/// Errors that can be returned by the control plane.
#[derive(Debug, Error)]
pub enum ApiError {
    /// The requested route does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// The request was malformed.
    #[error("invalid request: {0}")]
    BadRequest(String),

    /// The endpoint is reserved for a later roadmap phase.
    #[error("not implemented yet: {0} (see docs/roadmap.md)")]
    NotImplemented(&'static str),

    /// Authentication failed.
    #[error("unauthorized")]
    Unauthorized,

    /// The service is not ready to serve requests.
    #[error("service unavailable: {0}")]
    Unavailable(String),
}

impl ApiError {
    /// OpenAI error type string, kept stable for SDK error handling.
    #[must_use]
    pub const fn error_type(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "invalid_request_error",
            Self::BadRequest(_) => "invalid_request_error",
            Self::NotImplemented(_) => "server_error",
            Self::Unauthorized => "authentication_error",
            Self::Unavailable(_) => "server_error",
        }
    }

    /// Short machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::BadRequest(_) => "invalid_request",
            Self::NotImplemented(_) => "not_implemented",
            Self::Unauthorized => "invalid_api_key",
            Self::Unavailable(_) => "service_unavailable",
        }
    }

    const fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = ErrorEnvelope {
            error: ErrorDetail {
                message: self.to_string(),
                r#type: self.error_type().to_owned(),
                param: None,
                code: Some(self.code().to_owned()),
            },
        };
        (status, Json(body)).into_response()
    }
}

/// Top-level OpenAI error envelope.
#[derive(Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ErrorEnvelope {
    /// Error details.
    pub error: ErrorDetail,
}

/// OpenAI error detail object.
#[derive(Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ErrorDetail {
    /// Human-readable message.
    pub message: String,
    /// Error category.
    pub r#type: String,
    /// Offending parameter, when known.
    pub param: Option<String>,
    /// Machine-readable code.
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{ApiError, ErrorEnvelope};
    use axum::{body::to_bytes, response::IntoResponse};

    #[tokio::test]
    async fn not_implemented_matches_openai_shape() {
        let response = ApiError::NotImplemented("chat completions").into_response();
        assert_eq!(response.status(), 501);
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let envelope: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope.error.code.as_deref(), Some("not_implemented"));
        assert!(envelope.error.message.contains("docs/roadmap.md"));
    }

    #[tokio::test]
    async fn unauthorized_returns_401() {
        let response = ApiError::Unauthorized.into_response();
        assert_eq!(response.status(), 401);
    }
}
