//! Ollama-compatible request and response types for the `/api/generate` and `/api/chat` shims.
//!
//! These are deliberately small typed subsets of the Ollama wire contract. Unknown Ollama fields
//! (`images`, `tools`, `keep_alive`, and so on) are ignored on input so unmodified Ollama clients
//! can send their defaults. Responses use Ollama's field names, including the RFC 3339 `created_at`
//! timestamp and the final `done` / `done_reason` markers.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::common::StringOrList;

/// Sampling options accepted inside an Ollama request's `options` object.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct OllamaOptions {
    /// Softmax temperature; `0.0` selects greedy decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k truncation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Ollama's name for the maximum number of generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    /// Seed for deterministic sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Stop strings, either one string or a list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrList>,
    /// Repetition penalty applied to already-generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
}

/// `POST /api/generate` request in Ollama's shape.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct OllamaGenerateRequest {
    /// Model id to generate with; must appear in `GET /v1/models`.
    pub model: String,
    /// Generation prompt.
    pub prompt: String,
    /// Optional system instruction prepended to the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Ollama suffix field; accepted for client compatibility but not rendered by the fixture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// Sampling options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    /// Return newline-delimited JSON chunks when `true`; defaults to `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// One chat message in Ollama's shape.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct OllamaChatMessage {
    /// Message role, for example `user` or `assistant`.
    pub role: String,
    /// Message content.
    pub content: String,
}

impl OllamaChatMessage {
    /// Creates a message from a role and its content.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// `POST /api/chat` request in Ollama's shape.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct OllamaChatRequest {
    /// Model id to generate with; must appear in `GET /v1/models`.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OllamaChatMessage>,
    /// Sampling options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    /// Return newline-delimited JSON chunks when `true`; defaults to `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Non-streaming `POST /api/generate` response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct OllamaGenerateResponse {
    /// Model used for generation.
    pub model: String,
    /// RFC 3339 UTC timestamp.
    pub created_at: String,
    /// Generated text; empty on the final chunk of a stream.
    pub response: String,
    /// `true` on the final chunk of a stream.
    pub done: bool,
    /// Ollama finish reason, present when `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Number of prompt tokens, present on non-streaming responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Number of generated tokens, present on non-streaming responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// Non-streaming `POST /api/chat` response and each streamed chunk.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct OllamaChatResponse {
    /// Model used for generation.
    pub model: String,
    /// RFC 3339 UTC timestamp.
    pub created_at: String,
    /// Generated message; the role is always `assistant`.
    pub message: OllamaChatMessage,
    /// `true` on the final chunk of a stream.
    pub done: bool,
    /// Ollama finish reason, present when `done` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Number of prompt tokens, present on non-streaming responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Number of generated tokens, present on non-streaming responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// Formats a Unix timestamp as an RFC 3339 UTC string without another date-time dependency.
pub(crate) fn format_timestamp(epoch_secs: u64) -> String {
    let seconds_per_day = 86_400_i64;
    let days = (epoch_secs / seconds_per_day as u64) as i64;
    let remainder = epoch_secs % seconds_per_day as u64;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        remainder / 3600,
        remainder / 60 % 60,
        remainder % 60
    )
}

/// Howard Hinnant's civil-from-days algorithm, valid for the full `u64` Unix epoch range.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    (year + i64::from(month <= 2), month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_request_accepts_ollama_fields_and_ignores_extras() {
        let request: OllamaGenerateRequest = serde_json::from_str(
            r#"{"model":"reference-tiny","prompt":"hi","system":"be terse","suffix":"!","stream":true,"images":["abc"],"keep_alive":"5m","options":{"temperature":0.2,"num_predict":7,"seed":3,"stop":["END"],"repeat_penalty":1.1}}"#,
        )
        .unwrap();
        assert_eq!(request.model, "reference-tiny");
        assert_eq!(request.system.as_deref(), Some("be terse"));
        assert_eq!(request.stream, Some(true));
        let options = request.options.unwrap();
        assert_eq!(options.temperature, Some(0.2));
        assert_eq!(options.num_predict, Some(7));
        assert_eq!(options.seed, Some(3));
        assert_eq!(
            options.stop,
            Some(StringOrList::List(vec!["END".to_owned()]))
        );
        assert_eq!(options.repeat_penalty, Some(1.1));

        let minimal: OllamaGenerateRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi"}"#).unwrap();
        assert_eq!(minimal.stream, None);
        assert!(minimal.options.is_none());
    }

    #[test]
    fn chat_request_round_trips_with_ollama_message_shape() {
        let request: OllamaChatRequest = serde_json::from_str(
            r#"{"model":"reference-tiny","messages":[{"role":"user","content":"hi","images":[]}],"stream":false,"think":false,"options":{"top_p":0.9}}"#,
        )
        .unwrap();
        assert_eq!(request.messages, vec![OllamaChatMessage::new("user", "hi")]);
        assert_eq!(request.stream, Some(false));
        assert_eq!(request.options.as_ref().unwrap().top_p, Some(0.9));

        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serde_json::from_str::<OllamaChatRequest>(&json).unwrap(),
            request
        );
    }

    #[test]
    fn responses_serialize_with_ollama_field_names() {
        let response = OllamaGenerateResponse {
            model: "reference-tiny".to_owned(),
            created_at: "2023-11-14T22:13:20Z".to_owned(),
            response: "answer".to_owned(),
            done: true,
            done_reason: Some("stop".to_owned()),
            prompt_eval_count: Some(3),
            eval_count: Some(1),
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["created_at"], "2023-11-14T22:13:20Z");
        assert_eq!(json["done_reason"], "stop");
        assert_eq!(json["eval_count"], 1);
        assert_eq!(
            serde_json::from_value::<OllamaGenerateResponse>(json).unwrap(),
            response
        );

        let chunk = OllamaChatResponse {
            model: "reference-tiny".to_owned(),
            created_at: "2023-11-14T22:13:20Z".to_owned(),
            message: OllamaChatMessage::new("assistant", "a"),
            done: false,
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"message\":{\"role\":\"assistant\",\"content\":\"a\"}"));
        assert!(!json.contains("done_reason"));
        assert_eq!(
            serde_json::from_str::<OllamaChatResponse>(&json).unwrap(),
            chunk
        );
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        assert_eq!(format_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_timestamp(1_700_000_000), "2023-11-14T22:13:20Z");
        assert!(format_timestamp(u64::MAX).ends_with('Z'));
    }
}
