//! Reference inference service backing the OpenAI-compatible handlers.
//!
//! Generation goes through [`ReferenceBackend`], a deterministic fixture from `quarrel-engine`.
//! It is intentionally **not** a Go-parity model: responses are stable and finite but carry no
//! semantic meaning. Model ids, however, are real: they come from `quarrel-core`'s GGUF resolver.

use std::{
    collections::VecDeque,
    convert::Infallible,
    path::{Path, PathBuf},
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Bytes,
    response::sse::{Event, Sse},
};
use futures::Stream;
use quarrel_core::{
    CoreError, GenerateRequest, ModelId, RequestId, SamplingParams, Token, chat_template,
    resolver::ModelResolver,
};
use quarrel_engine::{
    EngineError,
    backend::{Backend, ReferenceBackend, generate},
};
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};

use crate::{
    dto::{
        ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
        ChatMessage, ChatStreamChoice, CompletionChoice, CompletionRequest, CompletionResponse,
        EmbeddingList, EmbeddingObject, EmbeddingPayload, EmbeddingRequest, EmbeddingUsage,
        EncodingFormat, Model, ModelList, OllamaChatMessage, OllamaChatRequest, OllamaChatResponse,
        OllamaGenerateRequest, OllamaGenerateResponse, OllamaOptions, StringOrList, Usage,
        ollama::format_timestamp,
    },
    error::ApiError,
    tokenizer,
};

/// Default delay between two SSE chunks.
pub const DEFAULT_CHUNK_DELAY: Duration = Duration::from_millis(20);
/// Default interval between SSE heartbeat comments.
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(15);
/// SSE channel capacity; the bounded channel applies backpressure to the producer.
const EVENT_CHANNEL_CAPACITY: usize = 32;
/// Prefix of chat request identifiers.
const CHAT_ID_PREFIX: &str = "chatcmpl";
/// Prefix of legacy completion request identifiers.
const COMPLETION_ID_PREFIX: &str = "cmpl";

/// Boxed SSE event stream used by both chat and completion streaming handlers.
pub type ChatEventStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;
/// SSE response carrying `chat.completion.chunk` events.
pub type ChatEventSse = Sse<ChatEventStream>;
/// Newline-delimited JSON byte stream used by the Ollama compatibility shims.
pub type OllamaEventStream = Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>>;

/// Stateless service that resolves models and runs deterministic reference inference.
#[derive(Debug, Clone)]
pub struct InferenceService {
    models_dir: PathBuf,
    backend: ReferenceBackend,
    created_at: u64,
    heartbeat: Duration,
    chunk_delay: Duration,
}

impl InferenceService {
    /// Creates a service over a GGUF models directory using wall-clock timestamps.
    #[must_use]
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        Self {
            models_dir: models_dir.into(),
            backend: ReferenceBackend::new(),
            created_at,
            heartbeat: DEFAULT_HEARTBEAT,
            chunk_delay: DEFAULT_CHUNK_DELAY,
        }
    }

    /// Overrides the timestamp written to responses; useful for reproducible contract fixtures.
    #[must_use]
    pub const fn with_created_at(mut self, created_at: u64) -> Self {
        self.created_at = created_at;
        self
    }

    /// Overrides the SSE pacing. Short intervals make streaming behaviour testable.
    #[must_use]
    pub const fn with_timing(mut self, heartbeat: Duration, chunk_delay: Duration) -> Self {
        self.heartbeat = heartbeat;
        self.chunk_delay = chunk_delay;
        self
    }

    /// Directory searched for GGUF models.
    #[must_use]
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Discovers models with `quarrel_core::resolver` and returns the OpenAI model list.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::Unavailable`] when an existing models directory cannot be read.
    pub fn models(&self) -> Result<ModelList, ApiError> {
        let data = self
            .discover()?
            .into_iter()
            .map(|id| Model::new(id.as_str()))
            .collect();
        Ok(ModelList::new(data))
    }

    /// Validates a model id against the models directory.
    ///
    /// Unknown names produce [`ApiError::InvalidModel`] (HTTP 400) rather than a panic. Substring
    /// matching is deliberately *not* used here: OpenAI clients expect exact model ids.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is empty, cannot be discovered, or the directory is unreadable.
    pub fn validate_model(&self, model: &str) -> Result<ModelId, ApiError> {
        let id = ModelId::new(model).map_err(|_| ApiError::InvalidModel(model.to_owned()))?;
        if self
            .discover()?
            .iter()
            .any(|candidate| candidate.as_str() == id.as_str())
        {
            Ok(id)
        } else {
            Err(ApiError::InvalidModel(model.to_owned()))
        }
    }

    /// Generates one non-streaming chat completion.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an empty conversation, an unknown model, invalid sampling fields
    /// or an engine failure.
    pub fn chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        if request.messages.is_empty() {
            return Err(ApiError::BadRequest(
                "messages must not be empty".to_owned(),
            ));
        }
        let model = self.validate_model(&request.model)?;
        let messages: Vec<chat_template::ChatMessage> = request
            .messages
            .iter()
            .map(|message| chat_template::ChatMessage {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect();
        let prompt = chat_template::fallback(&messages);
        let tokens = self.generate(&model, &prompt, request)?;

        let text = apply_stops(
            &tokenizer::decode(&token_ids_of(&tokens)),
            request.stop.as_ref(),
        );
        let finish_reason = finish_reason(tokens.len(), request.max_tokens, text.stopped);
        let usage = Usage::new(encode_prompt(&prompt)?.len() as u32, tokens.len() as u32);
        let id = self.request_id(
            CHAT_ID_PREFIX,
            model.as_str(),
            &prompt,
            &request.temperature,
            &request.seed,
        );

        Ok(ChatCompletionResponse::new(
            id,
            self.created_at,
            model.as_str().to_owned(),
            vec![ChatChoice {
                index: 0,
                message: ChatMessage::new("assistant", text.value),
                finish_reason: Some(finish_reason.to_owned()),
            }],
            usage,
        ))
    }

    /// Generates a legacy text completion, one choice per prompt string.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an empty prompt list, an unknown model, invalid sampling fields or
    /// an engine failure.
    pub fn completion(&self, request: &CompletionRequest) -> Result<CompletionResponse, ApiError> {
        let prompts = request.prompt.as_slice();
        if prompts.is_empty() {
            return Err(ApiError::BadRequest("prompt must not be empty".to_owned()));
        }
        let model = self.validate_model(&request.model)?;
        let params = sampling_params(
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
            request.max_tokens,
            None,
        )?;
        let seed = request.seed.unwrap_or(0);
        let mut choices = Vec::with_capacity(prompts.len());
        let mut prompt_tokens = 0_u32;
        let mut completion_tokens = 0_u32;

        for (index, prompt) in prompts.iter().enumerate() {
            let ids = encode_prompt(prompt)?;
            prompt_tokens += ids.len() as u32;
            let tokens = self.generate_tokens(&model, &ids, &params, seed + index as u64, false)?;
            completion_tokens += tokens.len() as u32;
            choices.push(CompletionChoice {
                index: index as u32,
                text: tokenizer::decode(&token_ids_of(&tokens)),
                finish_reason: Some(
                    finish_reason(tokens.len(), request.max_tokens, false).to_owned(),
                ),
            });
        }
        let id = self.request_id(
            COMPLETION_ID_PREFIX,
            model.as_str(),
            &prompts.join("\n"),
            &request.temperature,
            &request.seed,
        );

        Ok(CompletionResponse::new(
            id,
            self.created_at,
            model.as_str().to_owned(),
            choices,
            Usage::new(prompt_tokens, completion_tokens),
        ))
    }

    /// Computes deterministic embeddings from `ReferenceBackend` logits.
    ///
    /// Each input is tokenized, passed through the backend once, and L2-normalised. The resulting
    /// vector has 32 components (the reference vocabulary size) and is returned either as floats or
    /// as base64-encoded little-endian `f32` bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an empty input list, an unknown model or an engine failure.
    pub fn embedding(&self, request: &EmbeddingRequest) -> Result<EmbeddingList, ApiError> {
        let inputs = request.input.as_slice();
        if inputs.is_empty() {
            return Err(ApiError::BadRequest("input must not be empty".to_owned()));
        }
        let model = self.validate_model(&request.model)?;
        let mut data = Vec::with_capacity(inputs.len());
        let mut prompt_tokens = 0_u32;

        for (index, input) in inputs.iter().enumerate() {
            let ids = encode_prompt(input)?;
            prompt_tokens += ids.len() as u32;
            let logits = self
                .backend
                .forward(&ids)
                .map_err(|error| map_engine_error(&error))?;
            let values = normalise(logits);
            let payload = match request.encoding_format.unwrap_or(EncodingFormat::Float) {
                EncodingFormat::Float => EmbeddingPayload::Float(values),
                EncodingFormat::Base64 => EmbeddingPayload::Base64(encode_f32_base64(&values)),
            };
            data.push(EmbeddingObject::new(index as u32, payload));
        }

        Ok(EmbeddingList::new(
            data,
            model.as_str().to_owned(),
            EmbeddingUsage::new(prompt_tokens),
        ))
    }

    /// Prepares a streamed chat response.
    ///
    /// Tokens are generated before the response starts so validation errors are still regular JSON
    /// 400s; the SSE stream then paces the precomputed chunks, emits heartbeat comments and stops
    /// as soon as the client drops the connection.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::chat`].
    pub fn chat_stream(&self, request: &ChatCompletionRequest) -> Result<ChatEventSse, ApiError> {
        let events = self.chat_events(request)?;
        let (receiver, _producer) = spawn_events(events, self.chunk_delay, self.heartbeat);
        Ok(sse_from_receiver(receiver))
    }

    /// Returns the payloads of the SSE stream as strings, with `[DONE]` as the final entry.
    ///
    /// This is the contract-level view of [`Self::chat_stream`] and is used by replay tests to
    /// verify chunk sequence without consuming a live HTTP body.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::chat_stream`].
    pub fn chat_stream_events(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Vec<String>, ApiError> {
        Ok(self
            .chat_events(request)?
            .into_iter()
            .map(|event| match event {
                StreamEvent::Data(data) => data,
                StreamEvent::Done => "[DONE]".to_owned(),
                StreamEvent::Heartbeat => ": heartbeat".to_owned(),
            })
            .collect())
    }

    /// Generates a non-streaming Ollama-shaped `/api/generate` response.
    ///
    /// The prompt (and optional system instruction) are rendered through the same chat template as
    /// OpenAI requests, then generated by the same `ReferenceBackend`.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as [`Self::chat`], including `InvalidModel` for unknown ids.
    pub fn ollama_generate(
        &self,
        request: &OllamaGenerateRequest,
    ) -> Result<OllamaGenerateResponse, ApiError> {
        let response = self.chat(&generate_chat_request(request))?;
        ollama_generate_response(&response)
    }

    /// Generates a non-streaming Ollama-shaped `/api/chat` response.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as [`Self::chat`], including `BadRequest` for an empty
    /// conversation and `InvalidModel` for unknown ids.
    pub fn ollama_chat(&self, request: &OllamaChatRequest) -> Result<OllamaChatResponse, ApiError> {
        let response = self.chat(&chat_request_from_ollama(request))?;
        ollama_chat_response(&response)
    }

    /// Prepares an Ollama `/api/generate` NDJSON stream.
    ///
    /// Chunk generation and pacing reuse the SSE event producer, but the wire framing is Ollama's
    /// newline-delimited JSON: each line is one response object and the last carries `done: true`.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as [`Self::chat_stream`]; validation happens before the stream
    /// is returned so unknown models remain ordinary JSON 400s.
    pub fn ollama_generate_stream(
        &self,
        request: &OllamaGenerateRequest,
    ) -> Result<OllamaEventStream, ApiError> {
        let events = self.chat_events(&generate_chat_request(request))?;
        Ok(ndjson_stream(
            ollama_lines(events, OllamaLineKind::Generate)?,
            self.chunk_delay,
            self.heartbeat,
        ))
    }

    /// Prepares an Ollama `/api/chat` NDJSON stream.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as [`Self::chat_stream`].
    pub fn ollama_chat_stream(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<OllamaEventStream, ApiError> {
        let events = self.chat_events(&chat_request_from_ollama(request))?;
        Ok(ndjson_stream(
            ollama_lines(events, OllamaLineKind::Chat)?,
            self.chunk_delay,
            self.heartbeat,
        ))
    }

    fn chat_events(&self, request: &ChatCompletionRequest) -> Result<Vec<StreamEvent>, ApiError> {
        if request.messages.is_empty() {
            return Err(ApiError::BadRequest(
                "messages must not be empty".to_owned(),
            ));
        }
        let model = self.validate_model(&request.model)?;
        let messages: Vec<chat_template::ChatMessage> = request
            .messages
            .iter()
            .map(|message| chat_template::ChatMessage {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect();
        let prompt = chat_template::fallback(&messages);
        let tokens = self.generate(&model, &prompt, request)?;
        let text = apply_stops(
            &tokenizer::decode(&token_ids_of(&tokens)),
            request.stop.as_ref(),
        );
        let finish_reason = finish_reason(tokens.len(), request.max_tokens, text.stopped);
        let id = self.request_id(
            CHAT_ID_PREFIX,
            model.as_str(),
            &prompt,
            &request.temperature,
            &request.seed,
        );

        let mut events = Vec::with_capacity(tokens.len() + 3);
        events.push(chunk_event(&ChatCompletionChunk::new(
            id.clone(),
            self.created_at,
            model.as_str().to_owned(),
            vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_owned()),
                    content: None,
                },
                finish_reason: None,
            }],
        )));
        for character in text.value.chars() {
            events.push(chunk_event(&ChatCompletionChunk::new(
                id.clone(),
                self.created_at,
                model.as_str().to_owned(),
                vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatDelta::content(character.to_string()),
                    finish_reason: None,
                }],
            )));
        }
        events.push(chunk_event(&ChatCompletionChunk::new(
            id,
            self.created_at,
            model.as_str().to_owned(),
            vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta::default(),
                finish_reason: Some(finish_reason.to_owned()),
            }],
        )));
        events.push(StreamEvent::Done);
        Ok(events)
    }

    fn generate(
        &self,
        model: &ModelId,
        prompt: &str,
        request: &ChatCompletionRequest,
    ) -> Result<Vec<Token>, ApiError> {
        let ids = encode_prompt(prompt)?;
        let params = sampling_params(
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
            request.max_tokens,
            request.stop.as_ref(),
        )?;
        self.generate_tokens(model, &ids, &params, request.seed.unwrap_or(0), true)
    }

    fn generate_tokens(
        &self,
        model: &ModelId,
        ids: &[i32],
        params: &SamplingParams,
        seed: u64,
        stream: bool,
    ) -> Result<Vec<Token>, ApiError> {
        let request_id = RequestId::new("http-reference")
            .map_err(|error| ApiError::BadRequest(error.to_string()))?;
        let engine_request = GenerateRequest::new(
            request_id,
            model.clone(),
            ids.to_vec(),
            params.clone(),
            stream,
        )
        .map_err(|error| match error {
            CoreError::InvalidParams(_) | CoreError::EmptyPrompt => {
                ApiError::BadRequest(error.to_string())
            }
            other => ApiError::Unavailable(other.to_string()),
        })?;
        let limit = engine_request.params.max_tokens as usize;
        generate(
            &self.backend,
            &engine_request.token_ids,
            &engine_request.params,
            seed,
            limit,
        )
        .map_err(|error| map_engine_error(&error))
    }

    fn request_id(
        &self,
        prefix: &str,
        model: &str,
        prompt: &str,
        temperature: &Option<f32>,
        seed: &Option<u64>,
    ) -> String {
        stable_id(
            prefix,
            &[
                model,
                prompt,
                &format!("{temperature:?}"),
                &format!("{seed:?}"),
            ],
        )
    }

    fn discover(&self) -> Result<Vec<ModelId>, ApiError> {
        ModelResolver::new(vec![self.models_dir.clone()])
            .discover()
            .map_err(|error| match error {
                CoreError::Resolver(message) => ApiError::Unavailable(message),
                other => ApiError::BadRequest(other.to_string()),
            })
    }
}

fn token_ids_of(tokens: &[Token]) -> Vec<i32> {
    tokens.iter().map(|token| token.id).collect()
}

fn encode_prompt(prompt: &str) -> Result<Vec<i32>, ApiError> {
    let ids = tokenizer::encode(prompt);
    if ids.is_empty() {
        return Err(ApiError::BadRequest(
            "prompt must tokenize to at least one token".to_owned(),
        ));
    }
    Ok(ids)
}

fn sampling_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    repetition_penalty: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<&StringOrList>,
) -> Result<SamplingParams, ApiError> {
    let mut params = SamplingParams::new();
    if let Some(temperature) = temperature {
        params.temperature = temperature;
    }
    if let Some(top_p) = top_p {
        params.top_p = top_p;
    }
    if let Some(top_k) = top_k {
        params.top_k = top_k;
    }
    if let Some(repetition_penalty) = repetition_penalty {
        params.repetition_penalty = repetition_penalty;
    }
    if let Some(max_tokens) = max_tokens {
        params.max_tokens = max_tokens;
    }
    if let Some(stop) = stop {
        params.stop_token_ids = stop
            .as_slice()
            .iter()
            .filter_map(|text| match tokenizer::encode(text).as_slice() {
                [id] => Some(*id),
                _ => None,
            })
            .collect();
    }
    params
        .validate()
        .map_err(|error| ApiError::BadRequest(error.to_string()))?;
    Ok(params)
}

#[derive(Debug, PartialEq, Eq)]
struct TextWithStop {
    value: String,
    stopped: bool,
}

fn apply_stops(text: &str, stop: Option<&StringOrList>) -> TextWithStop {
    let Some(stop) = stop else {
        return TextWithStop {
            value: text.to_owned(),
            stopped: false,
        };
    };
    let mut end = text.len();
    for needle in stop.as_slice() {
        if !needle.is_empty()
            && let Some(index) = text.find(needle)
        {
            end = end.min(index);
        }
    }
    TextWithStop {
        value: text[..end].to_owned(),
        stopped: end < text.len(),
    }
}

fn finish_reason(generated: usize, limit: Option<u32>, stopped: bool) -> &'static str {
    if stopped || limit.is_some_and(|limit| generated < limit as usize) {
        "stop"
    } else {
        "length"
    }
}

fn normalise(values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        values.into_iter().map(|value| value / norm).collect()
    } else {
        values
    }
}

/// Encodes `f32` values as base64 of their little-endian IEEE-754 representation (no new crate).
fn encode_f32_base64(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    encode_base64(&bytes)
}

/// Standard padded base64 encoding for arbitrary bytes.
fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = *chunk.get(1).unwrap_or(&0);
        let third = *chunk.get(2).unwrap_or(&0);
        output.push(char::from(ALPHABET[usize::from(first >> 2)]));
        output.push(char::from(
            ALPHABET[usize::from((first & 0x03) << 4 | second >> 4)],
        ));
        output.push(if chunk.len() > 1 {
            char::from(ALPHABET[usize::from((second & 0x0f) << 2 | third >> 6)])
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            char::from(ALPHABET[usize::from(third & 0x3f)])
        } else {
            '='
        });
    }
    output
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{prefix}-{hash:016x}")
}

/// A protocol-level SSE event. Keeping this as data (rather than axum's opaque `Event`) makes the
/// emitted sequence straightforward to assert in unit tests.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamEvent {
    /// A JSON-encoded `chat.completion.chunk`.
    Data(String),
    /// The terminal `[DONE]` marker.
    Done,
    /// An idle-stream heartbeat comment.
    Heartbeat,
}

fn chunk_event(chunk: &ChatCompletionChunk) -> StreamEvent {
    StreamEvent::Data(
        serde_json::to_string(chunk)
            .unwrap_or_else(|_| r#"{"object":"chat.completion.chunk"}"#.to_owned()),
    )
}

fn generate_chat_request(request: &OllamaGenerateRequest) -> ChatCompletionRequest {
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = &request.system {
        messages.push(ChatMessage::new("system", system.clone()));
    }
    messages.push(ChatMessage::new("user", request.prompt.clone()));
    let OllamaOptions {
        temperature,
        top_p,
        top_k,
        num_predict,
        seed,
        stop,
        repeat_penalty,
    } = request.options.clone().unwrap_or_default();
    ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        stream: None,
        temperature,
        top_p,
        top_k,
        repetition_penalty: repeat_penalty,
        max_tokens: num_predict,
        seed,
        stop,
    }
}

fn chat_request_from_ollama(request: &OllamaChatRequest) -> ChatCompletionRequest {
    let messages = request
        .messages
        .iter()
        .map(|message| ChatMessage::new(message.role.clone(), message.content.clone()))
        .collect();
    let OllamaOptions {
        temperature,
        top_p,
        top_k,
        num_predict,
        seed,
        stop,
        repeat_penalty,
    } = request.options.clone().unwrap_or_default();
    ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        stream: None,
        temperature,
        top_p,
        top_k,
        repetition_penalty: repeat_penalty,
        max_tokens: num_predict,
        seed,
        stop,
    }
}

fn response_choice(response: &ChatCompletionResponse) -> Result<&ChatChoice, ApiError> {
    response
        .choices
        .first()
        .ok_or_else(|| ApiError::Engine("chat response has no choices".to_owned()))
}

fn chunk_choice(chunk: &ChatCompletionChunk) -> Result<&ChatStreamChoice, ApiError> {
    chunk
        .choices
        .first()
        .ok_or_else(|| ApiError::Engine("stream chunk has no choices".to_owned()))
}

fn ollama_generate_response(
    response: &ChatCompletionResponse,
) -> Result<OllamaGenerateResponse, ApiError> {
    let choice = response_choice(response)?;
    Ok(OllamaGenerateResponse {
        model: response.model.clone(),
        created_at: format_timestamp(response.created),
        response: choice.message.content.clone(),
        done: true,
        done_reason: choice.finish_reason.clone(),
        prompt_eval_count: Some(response.usage.prompt_tokens),
        eval_count: Some(response.usage.completion_tokens),
    })
}

fn ollama_chat_response(response: &ChatCompletionResponse) -> Result<OllamaChatResponse, ApiError> {
    let choice = response_choice(response)?;
    Ok(OllamaChatResponse {
        model: response.model.clone(),
        created_at: format_timestamp(response.created),
        message: OllamaChatMessage::new("assistant", choice.message.content.clone()),
        done: true,
        done_reason: choice.finish_reason.clone(),
        prompt_eval_count: Some(response.usage.prompt_tokens),
        eval_count: Some(response.usage.completion_tokens),
    })
}

/// Converts the precomputed OpenAI SSE chunks into Ollama JSON lines.
///
/// Heartbeat comments are dropped because Ollama clients expect only JSON lines. The `[DONE]`
/// sentinel is retained internally to terminate the producer, then filtered out by
/// [`ndjson_stream`].
fn ollama_lines(
    events: Vec<StreamEvent>,
    kind: OllamaLineKind,
) -> Result<Vec<StreamEvent>, ApiError> {
    let mut lines = Vec::with_capacity(events.len());
    for event in events {
        match event {
            StreamEvent::Data(data) => {
                let chunk: ChatCompletionChunk = serde_json::from_str(&data)
                    .map_err(|error| ApiError::Engine(error.to_string()))?;
                let line = match kind {
                    OllamaLineKind::Generate => {
                        serde_json::to_string(&ollama_generate_chunk(&chunk)?)
                    }
                    OllamaLineKind::Chat => serde_json::to_string(&ollama_chat_chunk(&chunk)?),
                }
                .map_err(|error| ApiError::Engine(error.to_string()))?;
                lines.push(StreamEvent::Data(line));
            }
            StreamEvent::Done => lines.push(StreamEvent::Done),
            StreamEvent::Heartbeat => {}
        }
    }
    Ok(lines)
}

fn ollama_generate_chunk(chunk: &ChatCompletionChunk) -> Result<OllamaGenerateResponse, ApiError> {
    let choice = chunk_choice(chunk)?;
    Ok(OllamaGenerateResponse {
        model: chunk.model.clone(),
        created_at: format_timestamp(chunk.created),
        response: choice.delta.content.clone().unwrap_or_default(),
        done: choice.finish_reason.is_some(),
        done_reason: choice.finish_reason.clone(),
        prompt_eval_count: None,
        eval_count: None,
    })
}

fn ollama_chat_chunk(chunk: &ChatCompletionChunk) -> Result<OllamaChatResponse, ApiError> {
    let choice = chunk_choice(chunk)?;
    Ok(OllamaChatResponse {
        model: chunk.model.clone(),
        created_at: format_timestamp(chunk.created),
        message: OllamaChatMessage::new(
            choice
                .delta
                .role
                .clone()
                .unwrap_or_else(|| "assistant".to_owned()),
            choice.delta.content.clone().unwrap_or_default(),
        ),
        done: choice.finish_reason.is_some(),
        done_reason: choice.finish_reason.clone(),
        prompt_eval_count: None,
        eval_count: None,
    })
}

/// Reuses the bounded SSE producer to pace Ollama lines, dropping internal markers and heartbeats.
fn ndjson_stream(
    lines: Vec<StreamEvent>,
    chunk_delay: Duration,
    heartbeat: Duration,
) -> OllamaEventStream {
    let (receiver, _producer) = spawn_events(lines, chunk_delay, heartbeat);
    let stream = futures::stream::unfold(receiver, |mut receiver| async move {
        loop {
            let event = receiver.recv().await?;
            if let StreamEvent::Data(line) = event {
                let bytes = Bytes::from(format!("{line}\n"));
                return Some((Ok(bytes), receiver));
            }
        }
    });
    Box::pin(stream)
}

#[derive(Debug, Clone, Copy)]
enum OllamaLineKind {
    Generate,
    Chat,
}

/// Starts the event producer and returns its receiver plus the producer task handle.
///
/// The producer races client disconnects against heartbeat and chunk timers with `tokio::select!`,
/// so no work continues once the receiver (owned by the SSE body) is dropped.
fn spawn_events(
    events: Vec<StreamEvent>,
    chunk_delay: Duration,
    heartbeat: Duration,
) -> (mpsc::Receiver<StreamEvent>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let mut pending: VecDeque<(StreamEvent, bool)> =
        events.into_iter().map(|event| (event, false)).collect();
    if let Some(last) = pending.back_mut() {
        last.1 = true;
    }

    let task = tokio::spawn(async move {
        let mut timer = tokio::time::interval(heartbeat);
        timer.tick().await;
        let mut next_chunk = Instant::now();
        loop {
            tokio::select! {
                _ = tx.closed() => break,
                _ = timer.tick() => {
                    if tx.send(StreamEvent::Heartbeat).await.is_err() {
                        break;
                    }
                }
                _ = tokio::time::sleep_until(next_chunk), if !pending.is_empty() => {
                    let (event, last) = pending.pop_front().expect("pending is not empty");
                    if tx.send(event).await.is_err() {
                        break;
                    }
                    if last {
                        break;
                    }
                    next_chunk = Instant::now() + chunk_delay;
                }
            }
        }
    });
    (rx, task)
}

fn sse_from_receiver(rx: mpsc::Receiver<StreamEvent>) -> ChatEventSse {
    let stream = Box::pin(futures::stream::unfold(rx, |mut rx| async move {
        let event = rx.recv().await?;
        let event = match event {
            StreamEvent::Data(data) => Event::default().data(data),
            StreamEvent::Done => Event::default().data("[DONE]"),
            StreamEvent::Heartbeat => Event::DEFAULT_KEEP_ALIVE,
        };
        Some((Ok::<Event, Infallible>(event), rx))
    }));
    Sse::new(stream)
}

fn map_engine_error(error: &EngineError) -> ApiError {
    ApiError::Engine(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn fixture_models_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models")
    }

    fn service() -> InferenceService {
        InferenceService::new(fixture_models_dir()).with_created_at(0)
    }

    fn chat_request(model: &str, prompt: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_owned(),
            messages: vec![ChatMessage::new("user", prompt)],
            stream: None,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(8),
            repetition_penalty: Some(1.0),
            max_tokens: Some(5),
            seed: Some(42),
            stop: None,
        }
    }

    #[test]
    fn discover_returns_deterministic_openai_models() {
        let models = service().models().unwrap();
        assert_eq!(models.object, "list");
        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].id, "reference-tiny");
        assert_eq!(models.data[0].object, "model");
        assert_eq!(models.data[0].owned_by, "quarrel-rs");
    }

    #[test]
    fn empty_and_missing_directories_yield_empty_lists() {
        for dir in [
            "tests/fixtures/empty_models",
            "tests/fixtures/does-not-exist",
        ] {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dir);
            assert!(
                InferenceService::new(path)
                    .models()
                    .unwrap()
                    .data
                    .is_empty()
            );
        }
    }

    #[test]
    fn unknown_model_is_a_typed_400() {
        let error = service().validate_model("not-a-model").unwrap_err();
        assert_eq!(error, ApiError::InvalidModel("not-a-model".to_owned()));
        assert_eq!(error.error_type(), "invalid_request_error");
        assert!(service().validate_model("").is_err());
    }

    #[test]
    fn chat_and_completion_are_deterministic_and_well_shaped() {
        let service = service();
        let request = chat_request("reference-tiny", "hello");
        let first = service.chat(&request).unwrap();
        assert_eq!(first, service.chat(&request).unwrap());
        assert_eq!(first.object, "chat.completion");
        assert_eq!(first.model, "reference-tiny");
        assert_eq!(first.created, 0);
        assert_eq!(first.choices.len(), 1);
        assert_eq!(first.choices[0].message.role, "assistant");
        assert_eq!(first.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(first.usage.prompt_tokens, 11);
        assert_eq!(first.usage.completion_tokens, 5);
        assert_eq!(first.usage.total_tokens, 16);

        let completion = service
            .completion(&CompletionRequest {
                model: "reference-tiny".to_owned(),
                prompt: StringOrList::List(vec!["hello".to_owned(), "world".to_owned()]),
                stream: None,
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(8),
                repetition_penalty: Some(1.0),
                max_tokens: Some(4),
                seed: Some(7),
            })
            .unwrap();
        assert_eq!(completion.object, "text_completion");
        assert_eq!(completion.choices.len(), 2);
        assert_eq!(completion.choices[0].index, 0);
        assert_eq!(completion.choices[1].index, 1);
        assert_eq!(completion.usage.prompt_tokens, 10);
    }

    #[test]
    fn invalid_inputs_and_parameters_are_rejected() {
        let service = service();
        let mut request = chat_request("reference-tiny", "hello");
        request.messages.clear();
        assert!(matches!(
            service.chat(&request),
            Err(ApiError::BadRequest(_))
        ));

        let mut request = chat_request("reference-tiny", "hello");
        request.temperature = Some(5.0);
        assert!(matches!(
            service.chat(&request),
            Err(ApiError::BadRequest(_))
        ));

        let mut request = chat_request("reference-tiny", "hello");
        request.messages[0].content.clear();
        assert!(service.chat(&request).is_ok());

        let completion = CompletionRequest {
            model: "reference-tiny".to_owned(),
            prompt: StringOrList::Single(String::new()),
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            max_tokens: None,
            seed: None,
        };
        assert!(matches!(
            service.completion(&completion),
            Err(ApiError::BadRequest(_))
        ));

        let embedding = EmbeddingRequest::single("missing", "hello");
        assert_eq!(
            service.embedding(&embedding).unwrap_err(),
            ApiError::InvalidModel("missing".to_owned())
        );
    }

    #[test]
    fn stop_strings_truncate_and_mark_finish_reason() {
        let service = service();
        let mut request = chat_request("reference-tiny", "hello");
        request.max_tokens = Some(8);
        let baseline = service.chat(&request).unwrap();
        let text = baseline.choices[0].message.content.clone();
        assert!(!text.is_empty());
        assert_eq!(
            apply_stops(&text, None),
            TextWithStop {
                value: text.clone(),
                stopped: false,
            }
        );

        // A multi-token stop string does not become a stop token id, so generation is unchanged
        // and the whole prefix is trimmed after decoding.
        request.stop = Some(StringOrList::Single(text.clone()));
        let response = service.chat(&request).unwrap();
        assert_eq!(response.choices[0].message.content, "");
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));

        assert_eq!(
            apply_stops("abcdef", Some(&StringOrList::Single("cd".to_owned()))),
            TextWithStop {
                value: "ab".to_owned(),
                stopped: true,
            }
        );
    }

    #[test]
    fn embeddings_are_normalised_and_support_both_encodings() {
        let service = service();
        let request = EmbeddingRequest::single("reference-tiny", "hello");
        let list = service.embedding(&request).unwrap();
        assert_eq!(list.object, "list");
        assert_eq!(list.model, "reference-tiny");
        assert_eq!(list.usage.prompt_tokens, 5);
        assert_eq!(list.data.len(), 1);
        let EmbeddingPayload::Float(values) = &list.data[0].embedding else {
            panic!("default encoding must be float");
        };
        assert_eq!(values.len(), 32);
        assert!(values.iter().all(|value| value.is_finite()));
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");

        let base64_request = EmbeddingRequest {
            model: "reference-tiny".to_owned(),
            input: StringOrList::List(vec!["hello".to_owned(), "world".to_owned()]),
            encoding_format: Some(EncodingFormat::Base64),
        };
        let base64 = service.embedding(&base64_request).unwrap();
        assert_eq!(base64.data.len(), 2);
        for object in &base64.data {
            let EmbeddingPayload::Base64(encoded) = &object.embedding else {
                panic!("base64 encoding requested");
            };
            assert_eq!(encoded.len(), (32_usize * 4).div_ceil(3) * 4);
            assert!(encoded.bytes().all(|byte| byte.is_ascii_alphanumeric()
                || byte == b'+'
                || byte == b'/'
                || byte == b'='));
        }
    }

    #[test]
    fn base64_encoder_matches_reference_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode_f32_base64(&[1.0_f32]), "AACAPw==");
    }

    #[test]
    fn stable_ids_are_reproducible_and_distinct() {
        assert_eq!(
            stable_id("cmpl", &["a", "b"]),
            stable_id("cmpl", &["a", "b"])
        );
        assert_ne!(
            stable_id("cmpl", &["a", "b"]),
            stable_id("cmpl", &["b", "a"])
        );
        assert_ne!(stable_id("cmpl", &["a"]), stable_id("chatcmpl", &["a"]));
    }

    #[tokio::test]
    async fn stream_pacing_heartbeat_done_and_cancellation() {
        let events = vec![
            StreamEvent::Data("one".to_owned()),
            StreamEvent::Data("two".to_owned()),
            StreamEvent::Done,
        ];
        let (mut rx, task) =
            spawn_events(events, Duration::from_millis(5), Duration::from_millis(2));
        let mut received = Vec::new();
        for _ in 0..64 {
            let event = tokio::time::timeout(Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap();
            received.push(event);
            if received.last() == Some(&StreamEvent::Done) {
                break;
            }
        }
        assert_eq!(received.last(), Some(&StreamEvent::Done));
        assert!(received.contains(&StreamEvent::Data("one".to_owned())));
        assert!(received.contains(&StreamEvent::Heartbeat));
        drop(rx);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
    }

    #[test]
    fn chat_events_have_role_content_finish_and_done_shapes() {
        let mut request = chat_request("reference-tiny", "hello");
        request.max_tokens = Some(2);
        let events = service().chat_events(&request).unwrap();
        assert!(events.len() >= 3);
        assert_eq!(events.last(), Some(&StreamEvent::Done));

        let first = match &events[0] {
            StreamEvent::Data(data) => serde_json::from_str::<ChatCompletionChunk>(data).unwrap(),
            other => panic!("unexpected first event: {other:?}"),
        };
        assert_eq!(first.object, "chat.completion.chunk");
        assert_eq!(first.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(first.choices[0].delta.content.is_none());

        let last = match &events[events.len() - 2] {
            StreamEvent::Data(data) => serde_json::from_str::<ChatCompletionChunk>(data).unwrap(),
            other => panic!("unexpected finish event: {other:?}"),
        };
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("length"));
        assert!(last.choices[0].delta.content.is_none());

        for event in &events[1..events.len() - 2] {
            let StreamEvent::Data(data) = event else {
                panic!("unexpected content event: {event:?}");
            };
            let chunk = serde_json::from_str::<ChatCompletionChunk>(data).unwrap();
            assert!(chunk.choices[0].delta.content.is_some());
        }
    }

    fn ollama_generate_request() -> OllamaGenerateRequest {
        serde_json::from_str(
            r#"{"model":"reference-tiny","prompt":"hello","system":"be brief","stream":false,"options":{"temperature":0.7,"top_p":0.9,"top_k":8,"seed":42,"num_predict":5,"repeat_penalty":1.0}}"#,
        )
        .unwrap()
    }

    fn ollama_chat_request() -> OllamaChatRequest {
        serde_json::from_str(
            r#"{"model":"reference-tiny","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"}],"stream":false,"options":{"temperature":0.7,"top_p":0.9,"top_k":8,"seed":42,"num_predict":5,"repeat_penalty":1.0}}"#,
        )
        .unwrap()
    }

    #[test]
    fn ollama_generate_and_chat_use_the_same_reference_backend() {
        let service = service();
        let generate = service.ollama_generate(&ollama_generate_request()).unwrap();
        assert_eq!(generate.model, "reference-tiny");
        assert_eq!(generate.created_at, "1970-01-01T00:00:00Z");
        assert!(generate.done);
        assert_eq!(generate.done_reason.as_deref(), Some("length"));
        assert!(generate.prompt_eval_count.unwrap() > 0);
        assert_eq!(generate.eval_count, Some(5));
        assert_eq!(
            generate,
            service.ollama_generate(&ollama_generate_request()).unwrap()
        );

        let chat = service.ollama_chat(&ollama_chat_request()).unwrap();
        assert_eq!(chat.message.role, "assistant");
        assert_eq!(chat.message.content, generate.response);
        assert!(chat.done);
        assert_eq!(chat.prompt_eval_count, generate.prompt_eval_count);
        assert_eq!(chat.eval_count, generate.eval_count);
    }

    #[test]
    fn ollama_unknown_models_and_empty_messages_are_typed_errors() {
        let mut request = ollama_generate_request();
        request.model = "missing".to_owned();
        assert_eq!(
            service().ollama_generate(&request).unwrap_err(),
            ApiError::InvalidModel("missing".to_owned())
        );
        assert_eq!(
            service().ollama_chat(&ollama_chat_request()).unwrap().model,
            "reference-tiny"
        );
        let mut request = ollama_chat_request();
        request.messages.clear();
        assert!(matches!(
            service().ollama_chat(&request),
            Err(ApiError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn ollama_streams_are_newline_delimited_json_with_final_done() {
        use futures::StreamExt;

        let service = service().with_timing(Duration::from_millis(1), Duration::from_millis(0));
        let mut request = ollama_generate_request();
        request.stream = Some(true);
        let mut stream = service.ollama_generate_stream(&request).unwrap();
        let mut output = String::new();
        while let Some(bytes) = stream.next().await {
            output.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
        }
        let chunks: Vec<OllamaGenerateResponse> = output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(chunks.len() > 2);
        assert!(
            chunks[..chunks.len() - 1]
                .iter()
                .all(|chunk| !chunk.done && chunk.response.len() <= 1)
        );
        assert!(chunks.last().unwrap().done);
        assert_eq!(
            chunks.last().unwrap().done_reason.as_deref(),
            Some("length")
        );
        assert_eq!(chunks.last().unwrap().response, "");

        let mut request = ollama_chat_request();
        request.stream = Some(true);
        let mut stream = service.ollama_chat_stream(&request).unwrap();
        let mut output = String::new();
        while let Some(bytes) = stream.next().await {
            output.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
        }
        let chunks: Vec<OllamaChatResponse> = output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(chunks.len() > 2);
        assert_eq!(chunks[0].message.role, "assistant");
        assert!(chunks.last().unwrap().done);
        assert_eq!(chunks.last().unwrap().message.content, "");
        let streamed: String = chunks[1..chunks.len() - 1]
            .iter()
            .map(|chunk| chunk.message.content.as_str())
            .collect();
        assert!(!streamed.is_empty());
    }
}
