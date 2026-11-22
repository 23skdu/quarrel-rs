//! Contract tests replaying recorded OpenAI SDK-shaped request/response pairs.
//!
//! Requests in `fixtures/` deliberately carry extra SDK fields (`n`, `user`, `dimensions`,
//! `logprobs`) that this DTO set ignores. Responses are byte-for-byte deterministic because the
//! service is pinned to a fixed creation timestamp and the reference backend is seed-driven.

use std::path::PathBuf;

use quarrel_api::{
    ApiError, ChatCompletionRequest, CompletionRequest, EmbeddingRequest, ErrorDetail,
    ErrorEnvelope, InferenceService, ModelList,
};
use serde::Serialize;
use serde_json::Value;

const CHAT_REQUEST: &str = include_str!("fixtures/chat_request.json");
const CHAT_RESPONSE: &str = include_str!("fixtures/chat_response.json");
const CHAT_STREAM: &str = include_str!("fixtures/chat_stream_events.json");
const COMPLETION_REQUEST: &str = include_str!("fixtures/completion_request.json");
const COMPLETION_RESPONSE: &str = include_str!("fixtures/completion_response.json");
const EMBEDDING_REQUEST: &str = include_str!("fixtures/embedding_request.json");
const EMBEDDING_RESPONSE: &str = include_str!("fixtures/embedding_response.json");
const EMBEDDING_BASE64_REQUEST: &str = include_str!("fixtures/embedding_base64_request.json");
const EMBEDDING_BASE64_RESPONSE: &str = include_str!("fixtures/embedding_base64_response.json");
const MODELS_RESPONSE: &str = include_str!("fixtures/models_response.json");
const ERROR_RESPONSE: &str = include_str!("fixtures/error_response.json");

fn service() -> InferenceService {
    InferenceService::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models"))
        .with_created_at(1_700_000_000)
}

fn assert_matches_fixture(actual: &impl Serialize, expected: &str) {
    // Round-trip through text so f32 values are compared using serde_json's shortest
    // round-trippable representation rather than their widened f64 expansion.
    let wire = serde_json::to_string(actual).expect("serialize actual response");
    let actual: Value = serde_json::from_str(&wire).expect("reparse actual response");
    let expected: Value = serde_json::from_str(expected).expect("parse recorded response");
    assert_eq!(actual, expected, "contract drift in recorded response");
}

#[test]
fn chat_completion_contract_replays() {
    let request: ChatCompletionRequest = serde_json::from_str(CHAT_REQUEST).unwrap();
    assert_eq!(request.model, "reference-tiny");
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages[1].content, "Hello!");
    assert_eq!(request.seed, Some(42));
    assert_matches_fixture(&service().chat(&request).unwrap(), CHAT_RESPONSE);
}

#[test]
fn chat_stream_contract_replays() {
    let request: ChatCompletionRequest = serde_json::from_str(CHAT_REQUEST).unwrap();
    let events = service().chat_stream_events(&request).unwrap();
    let streamed: Vec<Value> = events
        .iter()
        .map(|event| {
            event.strip_prefix("data: ").map_or_else(
                || Value::String(event.clone()),
                |data| {
                    if data == "[DONE]" {
                        Value::String(data.to_owned())
                    } else {
                        serde_json::from_str(data).unwrap()
                    }
                },
            )
        })
        .collect();
    let expected: Value = serde_json::from_str(CHAT_STREAM).unwrap();
    let expected_events = expected.as_array().unwrap();
    assert_eq!(streamed.len(), expected_events.len());
    for (actual, expected) in streamed.iter().zip(expected_events) {
        if let (Value::Object(actual), Value::Object(expected)) = (actual, expected) {
            for (key, value) in expected {
                assert_eq!(actual.get(key), Some(value), "chunk field {key}");
            }
        } else {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn completion_contract_replays() {
    let request: CompletionRequest = serde_json::from_str(COMPLETION_REQUEST).unwrap();
    assert_eq!(request.prompt.as_slice(), ["Once upon a time"]);
    assert_eq!(request.max_tokens, Some(4));
    assert_matches_fixture(
        &service().completion(&request).unwrap(),
        COMPLETION_RESPONSE,
    );
}

#[test]
fn embedding_contract_replays_in_both_encodings() {
    let request: EmbeddingRequest = serde_json::from_str(EMBEDDING_REQUEST).unwrap();
    assert_eq!(request.input.as_slice(), ["hello", "world"]);
    assert_matches_fixture(&service().embedding(&request).unwrap(), EMBEDDING_RESPONSE);

    let request: EmbeddingRequest = serde_json::from_str(EMBEDDING_BASE64_REQUEST).unwrap();
    assert_matches_fixture(
        &service().embedding(&request).unwrap(),
        EMBEDDING_BASE64_RESPONSE,
    );
}

#[test]
fn model_list_contract_replays() {
    assert_matches_fixture(&service().models().unwrap(), MODELS_RESPONSE);
    assert_matches_fixture(&ModelList::empty(), r#"{"object":"list","data":[]}"#);
}

#[test]
fn error_envelope_contract_replays() {
    let error = ApiError::InvalidModel("does-not-exist".to_owned());
    let envelope = ErrorEnvelope::new(
        ErrorDetail::new(error.to_string(), error.error_type(), error.code())
            .with_param(error.param().unwrap()),
    );
    assert_matches_fixture(&envelope, ERROR_RESPONSE);
}
