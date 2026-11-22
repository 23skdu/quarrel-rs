//! Arrow Flight Agent plane for `quarrel-rs`.
//!
//! This crate implements the schemas, zero-copy-friendly adapters and gRPC
//! service described by `api/flight-schema.md`. All payloads stay as Arrow
//! `RecordBatch`es until they reach an adapter.

mod adapters;
mod error;
mod ipc;
mod schema;
mod service;

pub use adapters::{
    DEFAULT_MODEL_ID, FINISH_REASON_LENGTH, decode_embedding, decode_finished,
    decode_generate_request, decode_generate_response, decode_kv_block, decode_top_logprobs,
    embedding_response_schema_for, encode_embedding, encode_generate_request,
    encode_generate_response, encode_generate_response_with_logprobs, encode_kv_block,
    generate_response_token_schema, kv_blocks_schema_for,
};
pub use error::FlightError;
pub use ipc::{batch_to_ipc, ipc_to_batch};
pub use schema::{
    DEFAULT_EMBEDDING_DIM, DEFAULT_KV_BLOCK_BYTES, canonical_fields_json, embed_request_schema,
    embed_response_schema, generate_request_schema, generate_response_schema, kv_blocks_schema,
    schema_hash,
};
pub use service::{
    DEFAULT_QUEUE_DEPTH, GENERATE_PATH, HEALTH_ACTION, QuarrelFlightService, flight_server,
};

/// Arrow `RecordBatch`, re-exported for convenience of Flight clients.
pub use arrow::record_batch::RecordBatch;
