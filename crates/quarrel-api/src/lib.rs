//! OpenAI-compatible HTTP contract and deterministic reference inference for `quarrel-rs`.
//!
//! This crate owns the wire DTOs (also used to generate the OpenAPI document), a tiny reference
//! tokenizer, an inference service backed by `quarrel-engine`'s `ReferenceBackend`, a bounded
//! per-key rate limiter, the SSE event plumbing, the Ollama compatibility DTOs and the embedded
//! WebUI asset. The transport-specific axum handlers live in `quarrel-server`.
//!
//! `ReferenceBackend` is a deterministic fixture, **not** a Go-parity model. All generated text and
//! embeddings are stable and finite but carry no semantic meaning.

#![deny(missing_docs)]

pub mod dto;
pub mod error;
pub mod ratelimit;
pub mod service;
pub mod tokenizer;
pub mod webui;

pub use dto::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ChatMessage, ChatStreamChoice, CompletionChoice, CompletionRequest, CompletionResponse,
    EmbeddingList, EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingUsage,
    EncodingFormat, ErrorDetail, ErrorEnvelope, FunctionDefinition, FunctionReference,
    HealthResponse, Model, ModelList, NamedToolChoice, OllamaChatMessage, OllamaChatRequest,
    OllamaChatResponse, OllamaGenerateRequest, OllamaGenerateResponse, OllamaOptions,
    ReadyResponse, StringOrList, Tool, ToolCall, ToolCallDelta, ToolCallFunction,
    ToolCallFunctionDelta, ToolChoice, ToolChoiceMode, Usage,
};
pub use error::ApiError;
pub use ratelimit::RateLimiter;
pub use service::{ChatEventSse, ChatEventStream, InferenceService, OllamaEventStream};
pub use webui::WEBUI_HTML;
