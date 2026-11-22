//! Minimal Flight server for local SDK and interoperability testing.
//!
//! Run with `cargo run -p quarrel-flight --example flight_server`; the Python SDK in
//! `pythonsdk/` can then connect to `grpc://127.0.0.1:50051`.

use quarrel_flight::{QuarrelFlightService, flight_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address = "0.0.0.0:50051".parse()?;
    let service = QuarrelFlightService::default();

    println!("quarrel-flight listening on {address}");
    tonic::transport::Server::builder()
        .add_service(flight_server(service))
        .serve(address)
        .await?;
    Ok(())
}
