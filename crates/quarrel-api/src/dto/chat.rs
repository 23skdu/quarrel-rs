//! Chat-completion requests, responses and SSE chunk types.

use serde::{Deserialize, Deserializer, Serialize};
use utoipa::ToSchema;

use super::common::StringOrList;
use super::completion::Usage;

/// A chat message in OpenAI role/content form.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatMessage {
    /// One of `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Message content. A JSON `null` from OpenAI SDKs is accepted as empty text.
    #[serde(default, deserialize_with = "content_or_empty")]
    pub content: String,
    /// Tool calls emitted by an assistant message, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Identifier of the tool call this message answers, for `role: "tool"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Creates a message from a role and its content.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

fn content_or_empty<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

/// An OpenAI tool description. Only `type: "function"` is supported.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct Tool {
    /// Discriminator; must be `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function name, description and JSON-schema parameters.
    pub function: FunctionDefinition,
}

impl Tool {
    /// Creates a function tool.
    #[must_use]
    pub fn function(function: FunctionDefinition) -> Self {
        Self {
            kind: "function".to_owned(),
            function,
        }
    }
}

/// An OpenAI function definition carried by a [`Tool`].
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FunctionDefinition {
    /// Function name used by `tool_choice` and emitted in tool calls.
    pub name: String,
    /// Optional human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON-schema parameters; only flat object properties are used for generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

impl FunctionDefinition {
    /// Creates a function definition with no description or parameters.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
        }
    }
}

/// OpenAI `tool_choice`: a mode string, a named function object, or an unsupported string.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `none`, `auto` or `required`.
    Mode(ToolChoiceMode),
    /// Force the named function.
    Function(NamedToolChoice),
    /// Any other string, preserved so the service can return a typed 400.
    Unsupported(String),
}

/// String forms of OpenAI's `tool_choice`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    /// Never request a tool call.
    None,
    /// Let the model decide; the reference backend declines.
    Auto,
    /// Request a tool call.
    Required,
}

/// Object form of OpenAI's `tool_choice`.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct NamedToolChoice {
    /// Discriminator; must be `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function to call.
    pub function: FunctionReference,
}

/// Function reference inside [`NamedToolChoice`].
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct FunctionReference {
    /// Name of a function declared in the request's `tools`.
    pub name: String,
}

/// A complete tool call emitted in an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ToolCall {
    /// Stable call identifier.
    pub id: String,
    /// Discriminator; always `function` for generated calls.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function name and JSON-encoded arguments string.
    pub function: ToolCallFunction,
}

/// Function payload of a [`ToolCall`].
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ToolCallFunction {
    /// Function name.
    pub name: String,
    /// Arguments encoded as a JSON object string.
    pub arguments: String,
}

/// One incremental tool-call delta in an SSE chunk.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Zero-based tool-call index, matching OpenAI's streaming shape.
    pub index: u32,
    /// Call id, present on the first delta of a tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Discriminator, present on the first delta of a tool call.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Function fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Function fragment inside a [`ToolCallDelta`].
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ToolCallFunctionDelta {
    /// Function name, present on the first delta of a tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Fragment of the JSON-encoded arguments string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
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
    /// Optional tool definitions; requests without tools keep the legacy response shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// How the service may use the declared tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// One choice in a non-streaming chat response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ChatChoice {
    /// Zero-based choice index.
    pub index: u32,
    /// Generated message; the role is always `assistant`.
    pub message: ChatMessage,
    /// `stop`, `length` or `tool_calls`, matching OpenAI's finish-reason vocabulary.
    pub finish_reason: Option<String>,
    /// Tool calls emitted with this choice, absent for ordinary text responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
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
    /// Incremental tool-call fragments; absent on ordinary text streams.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

impl ChatDelta {
    /// Creates a content-only delta.
    #[must_use]
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            role: None,
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    /// Creates a tool-call delta.
    #[must_use]
    pub fn tool_calls(calls: Vec<ToolCallDelta>) -> Self {
        Self {
            role: None,
            content: None,
            tool_calls: Some(calls),
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
    fn tool_fields_deserialise_and_legacy_requests_omit_them() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{
                    "role":"assistant",
                    "content":null,
                    "tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{}"}}],
                    "tool_call_id":"c0",
                    "unknown_message_field":true
                }],
                "tools":[{"type":"function","function":{"name":"calc","description":null,"parameters":{"type":"object","properties":{}}},"unknown_tool_field":1}],
                "tool_choice":{"type":"function","function":{"name":"calc"},"unknown_choice_field":2},
                "parallel_tool_calls":false
            }"#,
        )
        .unwrap();
        assert_eq!(request.messages[0].content, "");
        assert_eq!(request.messages[0].tool_call_id.as_deref(), Some("c0"));
        let call = &request.messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.name, "calc");
        assert_eq!(call.function.arguments, "{}");
        assert_eq!(request.tools.as_ref().unwrap()[0].function.name, "calc");
        assert!(matches!(request.tool_choice, Some(ToolChoice::Function(_))));

        assert_eq!(
            serde_json::from_str::<ToolChoice>(r#""required""#).unwrap(),
            ToolChoice::Mode(ToolChoiceMode::Required)
        );
        assert_eq!(
            serde_json::from_str::<ToolChoice>(r#""sometimes""#).unwrap(),
            ToolChoice::Unsupported("sometimes".to_owned())
        );

        let minimal: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        let json = serde_json::to_value(minimal).unwrap();
        assert!(!json.as_object().unwrap().contains_key("tools"));
        assert!(!json.as_object().unwrap().contains_key("tool_choice"));
        assert!(
            !json["messages"][0]
                .as_object()
                .unwrap()
                .contains_key("tool_calls")
        );
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
                tool_calls: None,
            }],
            Usage::new(3, 1),
        );
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["usage"]["total_tokens"], 4);
        assert!(
            !json["choices"][0]
                .as_object()
                .unwrap()
                .contains_key("tool_calls")
        );
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
