//! Native, mmap-backed GGUF inference for real Gemma 3 and SmolLM2 models.
//!
//! This crate does not use a surrogate backend or an FFI runtime. It reads GGUF
//! tensors through [`quarrel_gguf`], executes transformer blocks with the
//! canonical quantized kernels, and builds tokenizers from the GGUF metadata
//! arrays embedded in the model file.

#![deny(missing_docs)]

pub mod error;
pub mod forward;
pub mod generate;
pub mod model;
pub mod tokenizer;

pub use error::RuntimeError;
pub use forward::{InferenceSession, KvCache};
pub use generate::{GenerationOptions, generate};
pub use model::{Architecture, Model, ModelConfig, RopeType};
pub use tokenizer::{ChatMessage, Tokenizer, TokenizerKind};
