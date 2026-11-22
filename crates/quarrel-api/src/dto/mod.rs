//! OpenAI-compatible wire types shared by the HTTP handlers and their clients.
//!
//! Every type derives `Serialize`, `Deserialize` and `utoipa::ToSchema`; types without floating
//! point fields also derive `PartialEq` and `Eq` so fixtures can be compared exactly. Unknown JSON
//! fields are ignored on deserialization, as required by the OpenAI compatibility contract.

pub mod chat;
pub mod common;
pub mod completion;
pub mod embedding;
pub mod error;
pub mod model;
pub mod system;

pub use chat::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ChatMessage, ChatStreamChoice,
};
pub use common::StringOrList;
pub use completion::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};
pub use embedding::{
    EmbeddingList, EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingUsage,
    EncodingFormat,
};
pub use error::{ErrorDetail, ErrorEnvelope};
pub use model::{Model, ModelList};
pub use system::{HealthResponse, ReadyResponse};
