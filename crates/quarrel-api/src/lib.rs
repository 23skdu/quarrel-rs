//! OpenAI-compatible HTTP contract and deterministic reference inference for `quarrel-rs`.
//!
//! This crate owns the wire DTOs (also used to generate the OpenAPI document), a tiny reference
//! tokenizer, an inference service backed by `quarrel-engine`'s `ReferenceBackend`, a bounded
//! per-key rate limiter and the SSE event plumbing. The transport-specific axum handlers live in
//! `quarrel-server`.
//!
//! `ReferenceBackend` is a deterministic fixture, **not** a Go-parity model. All generated text and
//! embeddings are stable and finite but carry no semantic meaning.

#![deny(missing_docs)]

pub mod dto;
pub mod error;
pub mod ratelimit;
pub mod service;
pub mod tokenizer;

pub use dto::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ChatMessage, ChatStreamChoice, CompletionChoice, CompletionRequest, CompletionResponse,
    EmbeddingList, EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingUsage,
    EncodingFormat, ErrorDetail, ErrorEnvelope, HealthResponse, Model, ModelList, ReadyResponse,
    StringOrList, Usage,
};
pub use error::ApiError;
pub use ratelimit::RateLimiter;
pub use service::{ChatEventSse, ChatEventStream, InferenceService};
