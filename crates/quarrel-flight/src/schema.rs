//! Arrow schemas and the stable schema hash used by the Flight plane.
//!
//! The field names and types here are the normative contract in
//! `api/flight-schema.md`.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit, UnionMode};

/// Default width of the `embedding` fixed-size list in [`embed_response_schema`].
///
/// [`encode_embedding`](crate::encode_embedding) uses the actual length of the
/// supplied embedding as `D`, so embeddings of any dimension round-trip. This
/// constant only fixes the canonical schema returned by the helper (and hence
/// the value reported by [`schema_hash`](crate::schema_hash)).
pub const DEFAULT_EMBEDDING_DIM: i32 = 4;

/// Default width of the raw `blocks` fixed-size list in [`kv_blocks_schema`].
pub const DEFAULT_KV_BLOCK_BYTES: i32 = 16;

/// `/quarrel/v1/generate` request batch schema.
pub fn generate_request_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new(
            "token_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ),
        Field::new("max_tokens", DataType::UInt32, false),
        Field::new("temperature", DataType::Float32, false),
        Field::new("top_p", DataType::Float32, false),
        Field::new("top_k", DataType::UInt32, false),
        Field::new(
            "stop_token_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Field::new("stream", DataType::Boolean, false),
    ]))
}

/// `/quarrel/v1/generate` response batch schema.
pub fn generate_response_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("token_id", DataType::Int32, false),
        Field::new("logprob", DataType::Float32, false),
        Field::new(
            "top_logprobs",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![
                    Field::new("token_id", DataType::Int32, false),
                    Field::new("logprob", DataType::Float32, false),
                ])),
                true,
            ))),
            true,
        ),
        Field::new_dictionary("finish_reason", DataType::UInt16, DataType::Utf8, true),
        Field::new("finished", DataType::Boolean, false),
    ]))
}

/// `/quarrel/v1/embed` request batch schema.
pub fn embed_request_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("token_id", DataType::Int32, false),
    ]))
}

/// `/quarrel/v1/embed` response batch schema.
pub fn embed_response_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("token_id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DEFAULT_EMBEDDING_DIM,
            ),
            false,
        ),
    ]))
}

/// `/quarrel/v1/kv/blocks` batch schema.
pub fn kv_blocks_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("block_id", DataType::UInt32, false),
        Field::new("layer_id", DataType::UInt16, false),
        Field::new_dictionary("dtype", DataType::UInt8, DataType::Utf8, false),
        Field::new(
            "blocks",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                DEFAULT_KV_BLOCK_BYTES,
            ),
            false,
        ),
    ]))
}

/// Return a deterministic, human-readable canonical representation of a schema.
///
/// Dictionary ids are intentionally omitted: they are IPC bookkeeping, not part
/// of the wire contract. Field metadata, when present, is sorted by key.
fn canonical_schema_json(schema: &Schema) -> String {
    let mut out = String::from("{\"fields\":[");
    for (i, field) in schema.fields().iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        push_field(&mut out, field);
    }
    out.push_str("]}");
    out
}

fn push_field(out: &mut String, field: &Field) {
    out.push_str("{\"name\":\"");
    push_escaped(out, field.name());
    out.push_str("\",\"nullable\":");
    out.push_str(if field.is_nullable() { "true" } else { "false" });
    out.push_str(",\"type\":");
    push_type(out, field.data_type());
    if !field.metadata().is_empty() {
        let mut metadata: Vec<_> = field.metadata().iter().collect();
        metadata.sort_by(|a, b| a.0.cmp(b.0));
        out.push_str(",\"metadata\":[");
        for (i, (key, value)) in metadata.iter().enumerate() {
            if i != 0 {
                out.push(',');
            }
            out.push_str("[\"");
            push_escaped(out, key);
            out.push_str("\",\"");
            push_escaped(out, value);
            out.push_str("\"]");
        }
        out.push(']');
    }
    out.push('}');
}

fn push_type(out: &mut String, data_type: &DataType) {
    match data_type {
        DataType::Null => out.push_str("{\"name\":\"null\"}"),
        DataType::Boolean => out.push_str("{\"name\":\"bool\"}"),
        DataType::Int8 => out.push_str("{\"name\":\"int8\"}"),
        DataType::Int16 => out.push_str("{\"name\":\"int16\"}"),
        DataType::Int32 => out.push_str("{\"name\":\"int32\"}"),
        DataType::Int64 => out.push_str("{\"name\":\"int64\"}"),
        DataType::UInt8 => out.push_str("{\"name\":\"uint8\"}"),
        DataType::UInt16 => out.push_str("{\"name\":\"uint16\"}"),
        DataType::UInt32 => out.push_str("{\"name\":\"uint32\"}"),
        DataType::UInt64 => out.push_str("{\"name\":\"uint64\"}"),
        DataType::Float16 => out.push_str("{\"name\":\"float16\"}"),
        DataType::Float32 => out.push_str("{\"name\":\"float32\"}"),
        DataType::Float64 => out.push_str("{\"name\":\"float64\"}"),
        DataType::Utf8 => out.push_str("{\"name\":\"utf8\"}"),
        DataType::LargeUtf8 => out.push_str("{\"name\":\"large_utf8\"}"),
        DataType::Binary => out.push_str("{\"name\":\"binary\"}"),
        DataType::LargeBinary => out.push_str("{\"name\":\"large_binary\"}"),
        DataType::FixedSizeBinary(size) => {
            out.push_str("{\"name\":\"fixed_size_binary\",\"size\":");
            out.push_str(&size.to_string());
            out.push('}');
        }
        DataType::List(field) => {
            out.push_str("{\"name\":\"list\",\"item\":");
            push_field(out, field);
            out.push('}');
        }
        DataType::LargeList(field) => {
            out.push_str("{\"name\":\"large_list\",\"item\":");
            push_field(out, field);
            out.push('}');
        }
        DataType::FixedSizeList(field, size) => {
            out.push_str("{\"name\":\"fixed_size_list\",\"item\":");
            push_field(out, field);
            out.push_str(",\"size\":");
            out.push_str(&size.to_string());
            out.push('}');
        }
        DataType::Struct(fields) => {
            out.push_str("{\"name\":\"struct\",\"fields\":[");
            for (i, field) in fields.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                push_field(out, field);
            }
            out.push_str("]}");
        }
        DataType::Dictionary(key, value) => {
            out.push_str("{\"name\":\"dictionary\",\"key\":");
            push_type(out, key);
            out.push_str(",\"value\":");
            push_type(out, value);
            out.push('}');
        }
        DataType::Timestamp(unit, tz) => {
            out.push_str("{\"name\":\"timestamp\",\"unit\":\"");
            out.push_str(match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            });
            out.push_str("\",\"timezone\":");
            match tz {
                Some(tz) => {
                    out.push('"');
                    push_escaped(out, tz);
                    out.push('"');
                }
                None => out.push_str("null"),
            }
            out.push('}');
        }
        DataType::Union(fields, mode) => {
            out.push_str("{\"name\":\"union\",\"mode\":\"");
            out.push_str(match mode {
                UnionMode::Sparse => "sparse",
                UnionMode::Dense => "dense",
            });
            out.push_str("\",\"fields\":[");
            for (i, (_, field)) in fields.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                push_field(out, field);
            }
            out.push_str("]}");
        }
        other => {
            out.push_str("{\"name\":\"");
            push_escaped(out, &format!("{other:?}"));
            out.push_str("\"}");
        }
    }
}

fn push_escaped(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// Stable hex digest of a schema's canonical field JSON.
///
/// Uses FNV-1a (64-bit), so the result is deterministic across processes and
/// Arrow versions as long as the fields themselves do not change.
pub fn schema_hash(schema: &Schema) -> String {
    let json = canonical_schema_json(schema);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in json.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Canonical field JSON for a field collection (used by tests and diagnostics).
#[doc(hidden)]
pub fn canonical_fields_json(fields: &Fields) -> String {
    let mut out = String::from("{\"fields\":[");
    for (i, field) in fields.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        push_field(&mut out, field);
    }
    out.push_str("]}");
    out
}
