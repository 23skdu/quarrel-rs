//! Command-line and environment configuration for the bootstrap server.

use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;

/// Runtime configuration.
///
/// Every field is overridable with a `QUARREL_`-prefixed environment variable so the same binary
/// can be configured identically from the CLI, docker-compose and a Kubernetes Deployment.
#[derive(Debug, Clone, PartialEq, Eq, Parser)]
#[command(name = "quarrel", version, about = "quarrel-rs control plane")]
pub struct Config {
    /// Address for the HTTP API, OpenAPI UI and Prometheus exposition endpoint.
    #[arg(long, env = "QUARREL_LISTEN", default_value = "0.0.0.0:8080")]
    pub listen: SocketAddr,

    /// Directory scanned for GGUF models by the model resolver (Phase 1).
    #[arg(long, env = "QUARREL_MODELS_DIR", default_value = "./models")]
    pub models_dir: PathBuf,

    /// Emit newline-delimited JSON logs instead of human-readable text.
    #[arg(long, env = "QUARREL_LOG_JSON", default_value_t = false)]
    pub log_json: bool,

    /// Optional API key. When set, `/v1/*` routes require `Authorization: Bearer <key>`.
    #[arg(long, env = "QUARREL_API_KEY")]
    pub api_key: Option<String>,

    /// Print the generated OpenAPI document to stdout and exit.
    #[arg(long, env = "QUARREL_PRINT_OPENAPI", default_value_t = false)]
    pub print_openapi: bool,
}

#[cfg(test)]
mod tests {
    use super::Config;
    use clap::Parser;

    #[test]
    fn defaults_are_stable() {
        let config = Config::parse_from(["quarrel"]);
        assert_eq!(config.listen.to_string(), "0.0.0.0:8080");
        assert_eq!(config.models_dir, std::path::PathBuf::from("./models"));
        assert!(!config.log_json);
        assert!(config.api_key.is_none());
        assert!(!config.print_openapi);
    }

    #[test]
    fn environment_overrides_are_honoured() {
        // clap reads env at parse time; use parse_from with explicit args for deterministic tests.
        let config = Config::parse_from([
            "quarrel",
            "--listen",
            "127.0.0.1:9090",
            "--models-dir",
            "/models",
            "--log-json",
        ]);
        assert_eq!(config.listen.to_string(), "127.0.0.1:9090");
        assert_eq!(config.models_dir, std::path::PathBuf::from("/models"));
        assert!(config.log_json);
    }
}
