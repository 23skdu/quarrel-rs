//! Operational probe responses.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /healthz` liveness response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct HealthResponse {
    /// Always `ok` while the process is alive.
    pub status: String,
}

impl HealthResponse {
    /// Creates the canonical healthy response.
    #[must_use]
    pub fn ok() -> Self {
        Self {
            status: "ok".to_owned(),
        }
    }
}

/// `GET /readyz` readiness response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ReadyResponse {
    /// `ready` when the configured models directory exists.
    pub status: String,
    /// Whether at least one GGUF model was discovered.
    pub model_loaded: bool,
}

impl ReadyResponse {
    /// Creates a readiness response; the service is ready whenever the models directory is usable,
    /// even if it contains no models.
    #[must_use]
    pub fn new(model_loaded: bool) -> Self {
        Self {
            status: "ready".to_owned(),
            model_loaded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthResponse, ReadyResponse};

    #[test]
    fn probe_shapes_are_stable() {
        assert_eq!(
            serde_json::to_string(&HealthResponse::ok()).unwrap(),
            r#"{"status":"ok"}"#
        );
        let ready = ReadyResponse::new(true);
        assert_eq!(ready.status, "ready");
        assert!(ready.model_loaded);
        let idle = ReadyResponse::new(false);
        assert_eq!(idle.status, "ready");
        assert!(!idle.model_loaded);
    }
}
