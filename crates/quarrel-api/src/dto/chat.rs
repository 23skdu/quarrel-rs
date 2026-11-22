//! Chat-completion requests, responses and SSE chunk types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::common::StringOrList;
use super::completion::Usage;

/// A chat message in OpenAI role/content form.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatMessage {
    /// One of `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ChatMessage {
    /// Creates a message from a role and its content.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Subset of the OpenAI chat-completion request understood by the reference backend.
///
/// Fields not listed here are ignored on input so unmodified OpenAI SDKs can send their defaults.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct ChatCompletionRequest {
    /// Model id to generate with; must appear in `GET /v1/models`.
    pub model: String,
    /// Conversation messages, rendered as a `role: content` prompt.
    pub messages: Vec<ChatMessage>,
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
    /// Stop strings, either one string or a list of strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrList>,
}

/// One choice in a non-streaming chat response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatChoice {
    /// Zero-based choice index.
    pub index: u32,
    /// Generated message; the role is always `assistant`.
    pub message: ChatMessage,
    /// `stop` or `length`, matching OpenAI's finish-reason vocabulary.
    pub finish_reason: Option<String>,
}

/// Non-streaming `POST /v1/chat/completions` response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct ChatCompletionResponse {
    /// Stable request identifier prefixed with `chatcmpl-`.
    pub id: String,
    /// Always `chat.completion`.
    pub object: String,
    /// Unix creation timestamp.
    pub created: u64,
    /// Model used for generation.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Token accounting for the request.
    pub usage: Usage,
}

impl ChatCompletionResponse {
    /// Creates a response with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(
        id: String,
        created: u64,
        model: String,
        choices: Vec<ChatChoice>,
        usage: Usage,
    ) -> Self {
        Self {
            id,
            object: "chat.completion".to_owned(),
            created,
            model,
            choices,
            usage,
        }
    }
}

/// Incremental message content in an SSE chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatDelta {
    /// Role, present only in the first chunk of a stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Generated text fragment; absent on the empty role and finish chunks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatDelta {
    /// Creates a content-only delta.
    #[must_use]
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            role: None,
            content: Some(content.into()),
        }
    }
}

/// One choice in an SSE chunk.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatStreamChoice {
    /// Zero-based choice index.
    pub index: u32,
    /// Incremental message fragment.
    pub delta: ChatDelta,
    /// Finish reason, present only on the last chunk before `[DONE]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// One `chat.completion.chunk` emitted on the SSE stream.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatCompletionChunk {
    /// Identifier shared by every chunk of one stream.
    pub id: String,
    /// Always `chat.completion.chunk`.
    pub object: String,
    /// Unix creation timestamp shared by every chunk of one stream.
    pub created: u64,
    /// Model used for generation.
    pub model: String,
    /// Incremental choices.
    pub choices: Vec<ChatStreamChoice>,
}

impl ChatCompletionChunk {
    /// Creates a chunk with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(id: String, created: u64, model: String, choices: Vec<ChatStreamChoice>) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_owned(),
            created,
            model,
            choices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_json() -> &'static str {
        r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"stream":false,"n":1}"#
    }

    #[test]
    fn request_deserialises_with_defaults_and_ignores_unknown_fields() {
        let request: ChatCompletionRequest = serde_json::from_str(request_json()).unwrap();
        assert_eq!(request.model, "tiny");
        assert_eq!(request.messages, vec![ChatMessage::new("user", "hi")]);
        assert_eq!(request.stream, Some(false));
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.top_p, None);
        assert!(request.stop.is_none());

        let minimal: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
        assert_eq!(minimal.stream, None);
        assert!(minimal.messages.is_empty());
    }

    #[test]
    fn response_shapes_round_trip() {
        let response = ChatCompletionResponse::new(
            "chatcmpl-1".to_owned(),
            42,
            "tiny".to_owned(),
            vec![ChatChoice {
                index: 0,
                message: ChatMessage::new("assistant", "answer"),
                finish_reason: Some("stop".to_owned()),
            }],
            Usage::new(3, 1),
        );
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["usage"]["total_tokens"], 4);
        assert_eq!(
            serde_json::from_value::<ChatCompletionResponse>(json).unwrap(),
            response
        );

        let chunk = ChatCompletionChunk::new(
            "chatcmpl-1".to_owned(),
            42,
            "tiny".to_owned(),
            vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta::content("a"),
                finish_reason: None,
            }],
        );
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(!json.contains("finish_reason"));
        assert_eq!(
            serde_json::from_str::<ChatCompletionChunk>(&json).unwrap(),
            chunk
        );
    }
}
