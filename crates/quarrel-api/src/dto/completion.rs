//! Legacy text-completion request and response types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::common::StringOrList;

/// `POST /v1/completions` request, restricted to text prompts.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct CompletionRequest {
    /// Model id to generate with; must appear in `GET /v1/models`.
    pub model: String,
    /// Prompt text, either one string or a list of strings.
    pub prompt: StringOrList,
    /// Return `text/event-stream` chunks when `true`; defaults to `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Softmax temperature. `0.0` selects greedy decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k truncation; `0` disables it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Repetition penalty applied to already-generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Maximum number of generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Seed for deterministic sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Token accounting shared by chat and completion responses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct Usage {
    /// Number of prompt tokens.
    pub prompt_tokens: u32,
    /// Number of generated tokens.
    pub completion_tokens: u32,
    /// Sum of prompt and completion tokens.
    pub total_tokens: u32,
}

impl Usage {
    /// Computes the total from the two counts.
    #[must_use]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// One choice in a completion response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct CompletionChoice {
    /// Zero-based choice index.
    pub index: u32,
    /// Generated text.
    pub text: String,
    /// `stop` or `length`, matching OpenAI's finish-reason vocabulary.
    pub finish_reason: Option<String>,
}

/// Non-streaming `POST /v1/completions` response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct CompletionResponse {
    /// Stable request identifier prefixed with `cmpl-`.
    pub id: String,
    /// Always `text_completion`.
    pub object: String,
    /// Unix creation timestamp.
    pub created: u64,
    /// Model used for generation.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token accounting for the request.
    pub usage: Usage,
}

impl CompletionResponse {
    /// Creates a response with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(
        id: String,
        created: u64,
        model: String,
        choices: Vec<CompletionChoice>,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object: "text_completion".to_owned(),
            created,
            model,
            choices,
            usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_total_is_the_sum() {
        let usage = Usage::new(7, 5);
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn request_accepts_single_or_multiple_prompts() {
        let single: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi","max_tokens":4}"#).unwrap();
        assert_eq!(single.prompt.as_slice(), ["hi"]);
        assert_eq!(single.max_tokens, Some(4));
        assert!(single.stream.is_none());

        let multiple: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":["a","b"],"n":1}"#).unwrap();
        assert_eq!(multiple.prompt.as_slice(), ["a", "b"]);
        assert!(multiple.temperature.is_none());
    }

    #[test]
    fn response_round_trips() {
        let response = CompletionResponse::new(
            "cmpl-1".to_owned(),
            1,
            "m".to_owned(),
            vec![CompletionChoice {
                index: 0,
                text: "x".to_owned(),
                finish_reason: Some("length".to_owned()),
            }],
            Usage::new(2, 1),
        );
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("text_completion"));
        assert_eq!(
            serde_json::from_str::<CompletionResponse>(&json).unwrap(),
            response
        );
    }
}
