//! Golden IPC fixture drift tests.
//!
//! Regenerate with `cargo run -p quarrel-flight --example write_fixtures`.

use quarrel_flight::{
    batch_to_ipc, decode_generate_request, decode_generate_response, generate_request_schema,
    generate_response_schema, ipc_to_batch,
};

static GENERATE_REQUEST_IPC: &[u8] = include_bytes!("fixtures/generate_request.arrow");
static GENERATE_RESPONSE_IPC: &[u8] = include_bytes!("fixtures/generate_response.arrow");

#[test]
fn generate_request_fixture_does_not_drift() {
    let batch = ipc_to_batch(GENERATE_REQUEST_IPC, generate_request_schema()).unwrap();
    let request = decode_generate_request(&batch).unwrap();
    assert_eq!(request.request_id.as_str(), "fixture-request-0001");
    assert_eq!(request.token_ids, vec![128_000, 271]);
    assert_eq!(request.params.max_tokens, 256);
    assert_eq!(batch_to_ipc(&batch).unwrap(), GENERATE_REQUEST_IPC);
}

#[test]
fn generate_response_fixture_does_not_drift() {
    let batch = ipc_to_batch(GENERATE_RESPONSE_IPC, generate_response_schema()).unwrap();
    let tokens = decode_generate_response(&batch).unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].id, 271);
    assert_eq!(tokens[0].logprob, -0.125);
    assert_eq!(batch_to_ipc(&batch).unwrap(), GENERATE_RESPONSE_IPC);
}
