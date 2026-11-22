//! OpenAI-shaped error responses shared by all HTTP handlers.
//!
//! The concrete error type and envelope DTOs live in `quarrel-api`; they are re-exported here so
//! existing users of `quarrel_server::error` keep working.

pub use quarrel_api::{ApiError, ErrorDetail, ErrorEnvelope};

#[cfg(test)]
mod tests {
    use super::{ApiError, ErrorEnvelope};
    use axum::{body::to_bytes, response::IntoResponse};

    #[tokio::test]
    async fn bad_request_matches_openai_shape() {
        let response = ApiError::BadRequest("bad input".to_owned()).into_response();
        assert_eq!(response.status(), 400);
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let envelope: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope.error.r#type, "invalid_request_error");
        assert_eq!(envelope.error.code.as_deref(), Some("invalid_request"));
        assert_eq!(envelope.error.message, "invalid request: bad input");
    }

    #[tokio::test]
    async fn unauthorized_returns_401() {
        let response = ApiError::Unauthorized.into_response();
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn unknown_model_returns_a_typed_400() {
        let error = ApiError::InvalidModel("missing".to_owned());
        assert_eq!(error.error_type(), "invalid_request_error");
        assert_eq!(error.param(), Some("model"));
        assert_eq!(error.into_response().status(), 400);
    }
}
