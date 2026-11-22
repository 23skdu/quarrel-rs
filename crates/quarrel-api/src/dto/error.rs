//! OpenAI-shaped error envelope.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Top-level error object returned by every failing HTTP handler.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ErrorEnvelope {
    /// Error details.
    pub error: ErrorDetail,
}

impl ErrorEnvelope {
    /// Creates an envelope from its details.
    #[must_use]
    pub const fn new(error: ErrorDetail) -> Self {
        Self { error }
    }
}

/// The `error` object inside an [`ErrorEnvelope`].
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ErrorDetail {
    /// Human-readable message.
    pub message: String,
    /// Error category, for example `invalid_request_error`.
    pub r#type: String,
    /// Offending request parameter when known.
    pub param: Option<String>,
    /// Short machine-readable code.
    pub code: Option<String>,
}

impl ErrorDetail {
    /// Creates an error detail with no offending parameter.
    #[must_use]
    pub fn new(
        message: impl Into<String>,
        r#type: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            r#type: r#type.into(),
            param: None,
            code: Some(code.into()),
        }
    }

    /// Attaches the name of the offending parameter.
    #[must_use]
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorDetail, ErrorEnvelope};

    #[test]
    fn envelope_matches_openai_shape() {
        let envelope = ErrorEnvelope::new(
            ErrorDetail::new("bad model", "invalid_request_error", "invalid_model")
                .with_param("model"),
        );
        let json = serde_json::to_value(&envelope).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "model");
        assert_eq!(json["error"]["code"], "invalid_model");
        assert_eq!(
            serde_json::from_value::<ErrorEnvelope>(json).unwrap(),
            envelope
        );
    }
}
