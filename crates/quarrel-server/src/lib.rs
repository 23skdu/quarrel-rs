//! Library facade for the quarrel-rs control plane.
//!
//! Keeping the application logic in a library makes it reusable from integration tests and from a
//! future single multi-purpose `quarrel` binary; `main.rs` stays a thin process entry point.

pub mod api;
pub mod config;
pub mod error;
pub mod metrics;
pub mod openapi;

use clap::Parser;
use config::Config;
use tracing::info;
use tracing_subscriber::{EnvFilter, prelude::*};
use utoipa::OpenApi;

/// Initialise logging, metrics, routing and the HTTP server, then run until SIGINT.
///
/// # Errors
///
/// Returns an error if tracing, the metrics recorder, the TCP listener or the HTTP server cannot
/// be initialised. All are fatal for the process and are reported by `main`.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();

    if config.print_openapi {
        println!(
            "{}",
            serde_json::to_string_pretty(&openapi::ApiDoc::openapi())?
        );
        return Ok(());
    }

    init_tracing(config.log_json);
    let metrics = metrics::Metrics::install()?;
    let upkeep = api::spawn_upkeep(metrics.clone());
    let state = api::AppState::new(config.clone(), metrics);
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    let router = api::build_router(state);

    info!(address = %config.listen, "quarrel control plane listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    upkeep.abort();
    info!("shutdown complete");
    Ok(())
}

fn init_tracing(json: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if json {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }
}

async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => info!("SIGINT received, draining connections"),
        Err(error) => tracing::error!(%error, "failed to install SIGINT handler"),
    }
}
