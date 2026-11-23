//! Test-only GGUF writer used to build deterministic v1/v2/v3 fixtures.

use crate::dtype::GgufVersion;
use crate::header::MetadataValue;

fn u32_le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn u64_le(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn count(out: &mut Vec<u8>, version: GgufVersion, value: u64) {
    if version == GgufVersion::V1 {
        u32_le(
            out,
            u32::try_from(value).expect("fixture count fits in u32"),
        );
    } else {
        u64_le(out, value);
    }
}

fn string(out: &mut Vec<u8>, version: GgufVersion, value: &str) {
    count(out, version, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// Encode a metadata value, including its type tag.
fn value(out: &mut Vec<u8>, version: GgufVersion, metadata: &MetadataValue) {
    u32_le(out, element_type(Some(metadata)));
    payload(out, version, metadata, false);
}

/// Encode the value bytes only. Array elements have no per-element type tag.
fn payload(out: &mut Vec<u8>, version: GgufVersion, metadata: &MetadataValue, in_array: bool) {
    match metadata {
        MetadataValue::Uint8(v) => out.push(*v),
        MetadataValue::Int8(v) => out.push(*v as u8),
        MetadataValue::Uint16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::Int16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::Uint32(v) => u32_le(out, *v),
        MetadataValue::Int32(v) => u32_le(out, *v as u32),
        MetadataValue::Float32(v) => u32_le(out, v.to_bits()),
        MetadataValue::Bool(v) => out.push(u8::from(*v)),
        MetadataValue::String(v) => {
            if in_array || version != GgufVersion::V1 {
                u64_le(out, v.len() as u64);
            } else {
                u32_le(out, v.len() as u32);
            }
            out.extend_from_slice(v.as_bytes());
        }
        MetadataValue::Array(values) => {
            u32_le(out, element_type(values.first()));
            u64_le(out, values.len() as u64);
            for item in values {
                payload(out, version, item, true);
            }
        }
        MetadataValue::Uint64(v) => u64_le(out, *v),
        MetadataValue::Int64(v) => u64_le(out, *v as u64),
        MetadataValue::Float64(v) => u64_le(out, v.to_bits()),
    }
}

fn element_type(first: Option<&MetadataValue>) -> u32 {
    match first {
        Some(MetadataValue::Uint8(_)) => 0,
        Some(MetadataValue::Int8(_)) => 1,
        Some(MetadataValue::Uint16(_)) => 2,
        Some(MetadataValue::Int16(_)) => 3,
        Some(MetadataValue::Uint32(_)) => 4,
        Some(MetadataValue::Int32(_)) => 5,
        Some(MetadataValue::Float32(_)) => 6,
        Some(MetadataValue::Bool(_)) => 7,
        Some(MetadataValue::String(_)) => 8,
        Some(MetadataValue::Array(_)) => 9,
        Some(MetadataValue::Uint64(_)) => 10,
        Some(MetadataValue::Int64(_)) => 11,
        Some(MetadataValue::Float64(_)) => 12,
        None => 0,
    }
}

/// Build a deterministic GGUF fixture with metadata and two tensors.
pub(super) fn fixture_bytes(version: GgufVersion) -> Vec<u8> {
    let metadata = vec![
        (
            "general.architecture".to_owned(),
            MetadataValue::String("llama".to_owned()),
        ),
        (
            "general.name".to_owned(),
            MetadataValue::String("quarrel-test".to_owned()),
        ),
        ("general.file_type".to_owned(), MetadataValue::Uint32(8)),
        (
            "llama.context_length".to_owned(),
            MetadataValue::Uint64(4096),
        ),
        (
            "llama.embedding_length".to_owned(),
            MetadataValue::Uint32(32),
        ),
        ("test.negative".to_owned(), MetadataValue::Int32(-17)),
        ("test.float".to_owned(), MetadataValue::Float32(0.5)),
        ("test.bool".to_owned(), MetadataValue::Bool(true)),
        (
            "test.numbers".to_owned(),
            MetadataValue::Array(vec![
                MetadataValue::Uint8(1),
                MetadataValue::Uint8(2),
                MetadataValue::Uint8(3),
            ]),
        ),
        (
            "test.strings".to_owned(),
            MetadataValue::Array(vec![
                MetadataValue::String("alpha".to_owned()),
                MetadataValue::String("beta".to_owned()),
            ]),
        ),
    ];

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    u32_le(&mut out, version.number());
    count(&mut out, version, 2); // tensor count
    count(&mut out, version, metadata.len() as u64); // kv count

    for (key, metadata_value) in &metadata {
        string(&mut out, version, key);
        value(&mut out, version, metadata_value);
    }

    // "token_embeddings.weight": 4 x 8 = 32 elements, one Q8_0 block.
    string(&mut out, version, "token_embeddings.weight");
    u32_le(&mut out, 2);
    u64_le(&mut out, 4);
    u64_le(&mut out, 8);
    u32_le(&mut out, dtype_code::Q8_0);
    u64_le(&mut out, 0);

    // "output.weight": 2 x 8 = 16 elements, one padded Q4_0 block.
    string(&mut out, version, "output.weight");
    u32_le(&mut out, 2);
    u64_le(&mut out, 2);
    u64_le(&mut out, 8);
    u32_le(&mut out, dtype_code::Q4_0);
    u64_le(&mut out, 34);

    let data_offset = (out.len() + 31) & !31;
    out.resize(data_offset, 0);

    // Q8_0: scale 0.5 (f16 0x3800), values 2, -4, 0, 1, then zeros.
    out.extend_from_slice(&[0x00, 0x38]);
    out.extend_from_slice(&[2u8, (-4i8) as u8, 0, 1]);
    out.resize(data_offset + 34, 0);

    // Q4_0: scale 2.0 (f16 0x4000), nibble bytes 0x9a (values 2, 4),
    // 0x07 (values -2, -16), then zeros.
    out.extend_from_slice(&[0x00, 0x40, 0x9a, 0x07]);
    out.resize(data_offset + 34 + 18, 0);

    out
}

/// Raw GGUF dtype ids, kept local so the fixture does not depend on the
/// public `DType` mapping it is meant to test.
mod dtype_code {
    pub(super) const Q4_0: u32 = 2;
    pub(super) const Q8_0: u32 = 8;
}
