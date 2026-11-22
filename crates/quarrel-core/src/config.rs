//! Runtime configuration shared by every `quarrel-rs` binary.
//!
//! A single [`Config`] struct is populated from the command line, `QUARREL_`-prefixed
//! environment variables, or a JSON file. It intentionally contains only cheap, owned values
//! so it can be passed across crate boundaries without borrowing from a CLI parser.

use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use serde::{Deserialize, Serialize};

use crate::CoreError;

const DEFAULT_LISTEN: &str = "0.0.0.0:8080";
const DEFAULT_FLIGHT_LISTEN: &str = "0.0.0.0:50051";
const DEFAULT_MODELS_DIR: &str = "./models";
const DEFAULT_QUEUE_DEPTH: usize = 64;

/// Runtime configuration for a quarrel node.
///
/// Every CLI flag has a matching `QUARREL_`-prefixed environment variable and a serde default,
/// so an omitted field always has the same value regardless of the configuration source.
#[derive(Debug, Clone, PartialEq, Eq, Parser, Serialize, Deserialize)]
#[command(name = "quarrel", version, about = "quarrel-rs runtime configuration")]
#[serde(default)]
pub struct Config {
    /// Address for the HTTP API, OpenAPI UI and Prometheus endpoint.
    #[arg(long, env = "QUARREL_LISTEN", default_value = DEFAULT_LISTEN)]
    pub listen: String,

    /// Directory searched recursively for GGUF model files.
    #[arg(long, env = "QUARREL_MODELS_DIR", default_value = DEFAULT_MODELS_DIR)]
    pub models_dir: PathBuf,

    /// Emit newline-delimited JSON logs instead of human-readable text.
    #[arg(long, env = "QUARREL_LOG_JSON", default_value_t = false)]
    pub log_json: bool,

    /// Optional API key; when set, `/v1/*` routes require `Authorization: Bearer <key>`.
    #[arg(long, env = "QUARREL_API_KEY")]
    pub api_key: Option<String>,

    /// Address on which the Flight service serves token, embedding and KV streams.
    #[arg(long, env = "QUARREL_FLIGHT_LISTEN", default_value = DEFAULT_FLIGHT_LISTEN)]
    pub flight_listen: String,

    /// Capacity of the bounded queue between the API agent and the inference engine.
    #[arg(long, env = "QUARREL_QUEUE_DEPTH", default_value_t = DEFAULT_QUEUE_DEPTH)]
    pub queue_depth: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: DEFAULT_LISTEN.to_owned(),
            models_dir: PathBuf::from(DEFAULT_MODELS_DIR),
            log_json: false,
            api_key: None,
            flight_listen: DEFAULT_FLIGHT_LISTEN.to_owned(),
            queue_depth: DEFAULT_QUEUE_DEPTH,
        }
    }
}

impl Config {
    /// Parses configuration from process arguments and `QUARREL_*` environment variables.
    ///
    /// As with any clap [`Parser`], invalid input terminates the process with a usage message.
    /// Use `<Config as Parser>::try_parse` when a recoverable error is required.
    #[must_use]
    pub fn parse_cli() -> Self {
        Self::parse()
    }

    /// Deserializes configuration from a JSON document.
    ///
    /// Missing fields are filled from [`Config::default`].
    #[must_use = "returns an error for malformed JSON"]
    pub fn from_json_str(input: &str) -> Result<Self, CoreError> {
        serde_json::from_str(input)
            .map_err(|error| CoreError::Config(format!("invalid config JSON: {error}")))
    }

    /// Serializes configuration to a stable, pretty-printed JSON document.
    #[must_use = "returns an error if serialization fails"]
    pub fn to_json_string(&self) -> Result<String, CoreError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| CoreError::Config(format!("cannot serialize config: {error}")))
    }

    /// Checks that both listen addresses are socket addresses and that queue/paths are usable.
    #[must_use = "validation errors must be handled before starting the service"]
    pub fn validate(&self) -> Result<(), CoreError> {
        parse_socket_address("listen", &self.listen)?;
        parse_socket_address("flight_listen", &self.flight_listen)?;
        if self.models_dir.as_os_str().is_empty() {
            return Err(CoreError::Config("models_dir must not be empty".to_owned()));
        }
        if self.queue_depth == 0 {
            return Err(CoreError::Config(
                "queue_depth must be at least 1".to_owned(),
            ));
        }
        if self.api_key.as_deref() == Some("") {
            return Err(CoreError::Config(
                "api_key must be omitted or non-empty".to_owned(),
            ));
        }
        Ok(())
    }
}

fn parse_socket_address(field: &str, value: &str) -> Result<SocketAddr, CoreError> {
    value
        .parse::<SocketAddr>()
        .map_err(|error| CoreError::Config(format!("{field} must be a socket address ({error})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_match_struct_defaults() {
        let parsed = Config::parse_from(["quarrel"]);
        assert_eq!(parsed, Config::default());
        parsed.validate().expect("defaults are valid");
    }

    #[test]
    fn cli_flags_override_defaults() {
        let parsed = Config::parse_from([
            "quarrel",
            "--listen",
            "127.0.0.1:9090",
            "--models-dir",
            "/models",
            "--log-json",
            "--api-key",
            "secret",
            "--flight-listen",
            "127.0.0.1:50052",
            "--queue-depth",
            "128",
        ]);
        assert_eq!(parsed.listen, "127.0.0.1:9090");
        assert_eq!(parsed.models_dir, PathBuf::from("/models"));
        assert!(parsed.log_json);
        assert_eq!(parsed.api_key.as_deref(), Some("secret"));
        assert_eq!(parsed.flight_listen, "127.0.0.1:50052");
        assert_eq!(parsed.queue_depth, 128);
        parsed.validate().expect("explicit config is valid");
    }

    #[test]
    fn json_round_trip_preserves_all_fields() {
        let config = Config {
            listen: "127.0.0.1:8081".to_owned(),
            models_dir: PathBuf::from("/var/models"),
            log_json: true,
            api_key: Some("token".to_owned()),
            flight_listen: "[::1]:50051".to_owned(),
            queue_depth: 17,
        };
        let json = config.to_json_string().expect("serialize config");
        assert_eq!(
            Config::from_json_str(&json).expect("deserialize config"),
            config
        );
    }

    #[test]
    fn json_deserialize_fills_missing_fields_and_reports_bad_json() {
        let config = Config::from_json_str(r#"{"listen":"127.0.0.1:9000"}"#)
            .expect("partial config is accepted");
        assert_eq!(config.listen, "127.0.0.1:9000");
        assert_eq!(config.models_dir, PathBuf::from("./models"));
        assert_eq!(config.queue_depth, 64);
        assert!(config.api_key.is_none());

        assert!(matches!(
            Config::from_json_str("{not json"),
            Err(CoreError::Config(_))
        ));
    }

    #[test]
    fn validate_rejects_invalid_values() {
        let config = Config {
            listen: String::new(),
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            listen: "localhost:8080".to_owned(),
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            flight_listen: "bad".to_owned(),
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            models_dir: PathBuf::new(),
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            queue_depth: 0,
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            api_key: Some(String::new()),
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn default_config_snapshot() {
        let config = Config::default();
        insta::assert_snapshot!(config.to_json_string().expect("serialize default config"));
    }
}
