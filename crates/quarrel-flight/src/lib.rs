//! Arrow Flight Agent plane for `quarrel-rs`.
//!
//! This crate implements the schemas, zero-copy-friendly adapters and gRPC
//! service described by `api/flight-schema.md`. All payloads stay as Arrow
//! `RecordBatch`es until they reach an adapter.

#![deny(missing_docs)]

mod adapters;
mod error;
mod ipc;
mod pool;
mod schema;
mod service;
mod stream;

pub use adapters::{
    DEFAULT_MODEL_ID, FINISH_REASON_LENGTH, GenerateRequestView, GenerateResponseBuilder,
    assert_kv_dtype_endianness, decode_embed_request, decode_embed_request_rows, decode_embedding,
    decode_finished, decode_generate_request, decode_generate_request_view,
    decode_generate_response, decode_kv_block, decode_top_logprobs, embedding_matrix_values,
    embedding_response_schema_for, embedding_values, encode_embed_request, encode_embedding,
    encode_generate_request, encode_generate_response, encode_generate_response_from_arrays,
    encode_generate_response_with_logprobs, encode_kv_block, generate_response_token_schema,
    kv_block_payload, kv_blocks_schema_for, parse_kv_dtype, request_stop_token_ids,
    request_token_ids, response_logprobs, response_token_ids,
};
pub use error::FlightError;
pub use ipc::{batch_to_ipc, ipc_to_batch};
pub use pool::{PooledResponseBuilder, ResponseBatchPool};
pub use schema::{
    DEFAULT_EMBEDDING_DIM, DEFAULT_KV_BLOCK_BYTES, canonical_fields_json, embed_request_schema,
    embed_response_schema, generate_request_schema, generate_response_schema, kv_blocks_schema,
    schema_hash,
};
pub use service::{
    DEFAULT_QUEUE_DEPTH, EMBED_PATH, GENERATE_PATH, HEALTH_ACTION, QuarrelFlightService,
    flight_server,
};
pub use stream::{RecordBatchStream, record_batch_stream, record_batch_stream_from_iter};

/// Arrow `RecordBatch`, re-exported for convenience of Flight clients.
pub use arrow::record_batch::RecordBatch;
