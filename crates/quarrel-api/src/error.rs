//! Typed HTTP errors rendered as OpenAI-shaped JSON envelopes.

use axum::{
    Json,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use thiserror::Error;

use crate::dto::error::{ErrorDetail, ErrorEnvelope};

/// Errors that can be returned by the HTTP plane.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ApiError {
    /// The requested route does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// A request field was malformed or out of range.
    #[error("invalid request: {0}")]
    BadRequest(String),
    /// The requested model is not present in the models directory.
    #[error("unknown model `{0}`; use GET /v1/models to list available models")]
    InvalidModel(String),
    /// Authentication failed.
    #[error("invalid or missing API key")]
    Unauthorized,
    /// The per-key request quota has been exhausted.
    #[error("too many requests; retry after 60 seconds")]
    RateLimited,
    /// The request body exceeded the configured limit.
    #[error("request body is too large")]
    PayloadTooLarge,
    /// The service cannot currently serve the request.
    #[error("service unavailable: {0}")]
    Unavailable(String),
    /// Inference failed inside the engine.
    #[error("inference failed: {0}")]
    Engine(String),
}

impl ApiError {
    /// OpenAI error type string, kept stable for SDK error handling.
    #[must_use]
    pub const fn error_type(&self) -> &'static str {
        match self {
            Self::NotFound(_)
            | Self::BadRequest(_)
            | Self::InvalidModel(_)
            | Self::PayloadTooLarge => "invalid_request_error",
            Self::Unauthorized => "authentication_error",
            Self::RateLimited => "rate_limit_error",
            Self::Unavailable(_) | Self::Engine(_) => "server_error",
        }
    }

    /// Short machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::BadRequest(_) => "invalid_request",
            Self::InvalidModel(_) => "invalid_model",
            Self::Unauthorized => "invalid_api_key",
            Self::RateLimited => "rate_limit_exceeded",
            Self::PayloadTooLarge => "payload_too_large",
            Self::Unavailable(_) => "service_unavailable",
            Self::Engine(_) => "engine_error",
        }
    }

    /// Offending request parameter, when one can be named.
    #[must_use]
    pub const fn param(&self) -> Option<&'static str> {
        match self {
            Self::InvalidModel(_) => Some("model"),
            _ => None,
        }
    }

    const fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) | Self::InvalidModel(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut detail = ErrorDetail::new(self.to_string(), self.error_type(), self.code());
        if let Some(param) = self.param() {
            detail = detail.with_param(param);
        }
        let status = self.status();
        let mut response = (status, Json(ErrorEnvelope::new(detail))).into_response();
        if status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("60"));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::ApiError;
    use crate::dto::error::ErrorEnvelope;
    use axum::{body::to_bytes, http::StatusCode, response::IntoResponse};

    async fn envelope(error: ApiError) -> (StatusCode, ErrorEnvelope, axum::http::HeaderMap) {
        let response = error.into_response();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap(), headers)
    }

    #[tokio::test]
    async fn invalid_model_is_a_400_invalid_request_error() {
        let (status, envelope, _) = envelope(ApiError::InvalidModel("missing".to_owned())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(envelope.error.r#type, "invalid_request_error");
        assert_eq!(envelope.error.code.as_deref(), Some("invalid_model"));
        assert_eq!(envelope.error.param.as_deref(), Some("model"));
        assert!(envelope.error.message.contains("missing"));
    }

    #[tokio::test]
    async fn every_error_variant_has_a_stable_status_and_code() {
        let cases = [
            (ApiError::NotFound("/x".to_owned()), 404, "not_found"),
            (ApiError::BadRequest("x".to_owned()), 400, "invalid_request"),
            (ApiError::Unauthorized, 401, "invalid_api_key"),
            (ApiError::RateLimited, 429, "rate_limit_exceeded"),
            (ApiError::PayloadTooLarge, 413, "payload_too_large"),
            (
                ApiError::Unavailable("x".to_owned()),
                503,
                "service_unavailable",
            ),
            (ApiError::Engine("x".to_owned()), 500, "engine_error"),
        ];
        for (error, expected_status, expected_code) in cases {
            let (status, envelope, headers) = envelope(error).await;
            assert_eq!(status.as_u16(), expected_status);
            assert_eq!(envelope.error.code.as_deref(), Some(expected_code));
            if expected_status == 429 {
                assert_eq!(headers.get("retry-after").unwrap(), "60");
            }
        }
    }
}
