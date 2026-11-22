//! Regenerate the golden Arrow IPC fixtures used by the drift tests.
//!
//! Run from the workspace root with:
//! `cargo run -p quarrel-flight --example write_fixtures`

use std::{fs, path::PathBuf};

use quarrel_core::{ModelId, RequestId, SamplingParams, Token};
use quarrel_flight::{
    batch_to_ipc, encode_generate_request, encode_generate_response, generate_request_schema,
    generate_response_schema,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    fs::create_dir_all(&directory)?;

    let request_id = RequestId::new("fixture-request-0001")?;
    let request = quarrel_core::GenerateRequest::new(
        request_id.clone(),
        ModelId::new("reference")?,
        vec![128_000, 271],
        SamplingParams::new(),
        true,
    )?;
    let request_bytes = batch_to_ipc(&encode_generate_request(&request)?)?;
    fs::write(directory.join("generate_request.arrow"), &request_bytes)?;

    let response_bytes = batch_to_ipc(&encode_generate_response(
        &request_id,
        1,
        &Token::new(271, -0.125),
        true,
    )?)?;
    fs::write(directory.join("generate_response.arrow"), &response_bytes)?;

    println!(
        "wrote {} ({} bytes) and generate_response.arrow ({} bytes)",
        directory.display(),
        request_bytes.len(),
        response_bytes.len()
    );

    // Touch the schema helpers so the generator also validates that the
    // contract schemas are constructible.
    let _ = (generate_request_schema(), generate_response_schema());
    Ok(())
}
