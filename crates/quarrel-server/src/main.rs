//! Thin process entry point for the quarrel-rs control plane.
//!
//! All behaviour lives in the `quarrel_server` library so integration tests and future binaries
//! can reuse it without starting a process.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    quarrel_server::run().await
}
