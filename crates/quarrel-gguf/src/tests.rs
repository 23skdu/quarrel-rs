//! Unit, property and snapshot tests for the GGUF reader.

use crate::dtype::GgufVersion;
use crate::fixture::fixture_bytes;
use crate::quant::{
    dequantize_f16, dequantize_q4_0, dequantize_q8_0, dot_q4_0, dot_q8_0, embedding_lookup_q8_0,
};
use crate::{DType, GgufFile, GgufMmap, MetadataValue};
use half::f16;
use proptest::prelude::*;

fn fixture(version: GgufVersion) -> Vec<u8> {
    fixture_bytes(version)
}

fn check_fixture(version: GgufVersion) {
    let bytes = fixture(version);
    let file = GgufFile::from_bytes(&bytes).expect("fixture must parse");
    let header = file.header();

    assert_eq!(header.version, version);
    assert_eq!(header.tensor_count, 2);
    assert_eq!(header.data_offset % 32, 0);
    assert_eq!(header.metadata.len(), 10);
    assert_eq!(
        header.metadata.get("general.architecture"),
        Some(&MetadataValue::String("llama".to_owned()))
    );
    assert_eq!(
        header.metadata.get("general.name"),
        Some(&MetadataValue::String("quarrel-test".to_owned()))
    );
    assert_eq!(
        header.metadata.get("general.file_type"),
        Some(&MetadataValue::Uint32(8))
    );
    assert_eq!(
        header.metadata.get("llama.context_length"),
        Some(&MetadataValue::Uint64(4096))
    );
    assert_eq!(
        header.metadata.get("test.negative"),
        Some(&MetadataValue::Int32(-17))
    );
    assert_eq!(
        header.metadata.get("test.float"),
        Some(&MetadataValue::Float32(0.5))
    );
    assert_eq!(
        header.metadata.get("test.bool"),
        Some(&MetadataValue::Bool(true))
    );
    assert_eq!(
        header.metadata.get("test.numbers"),
        Some(&MetadataValue::Array(vec![
            MetadataValue::Uint8(1),
            MetadataValue::Uint8(2),
            MetadataValue::Uint8(3),
        ]))
    );
    assert_eq!(
        header.metadata.get("test.strings"),
        Some(&MetadataValue::Array(vec![
            MetadataValue::String("alpha".to_owned()),
            MetadataValue::String("beta".to_owned()),
        ]))
    );

    assert_eq!(header.tensors.len(), 2);
    let embeddings = &header.tensors[0];
    assert_eq!(embeddings.name, "token_embeddings.weight");
    assert_eq!(embeddings.shape, vec![4, 8]);
    assert_eq!(embeddings.dtype, DType::Q8_0);
    assert_eq!(embeddings.offset, 0);
    assert_eq!(embeddings.byte_len, 34);

    let output = &header.tensors[1];
    assert_eq!(output.name, "output.weight");
    assert_eq!(output.shape, vec![2, 8]);
    assert_eq!(output.dtype, DType::Q4_0);
    assert_eq!(output.offset, 34);
    assert_eq!(output.byte_len, 18);

    let data_offset = header.data_offset as usize;
    assert!(
        bytes[data_offset - 32..data_offset]
            .iter()
            .all(|&byte| byte == 0)
    );

    let view = file
        .tensor("token_embeddings.weight")
        .expect("tensor exists");
    assert_eq!(view.name(), "token_embeddings.weight");
    assert_eq!(view.shape(), &[4, 8]);
    assert_eq!(view.dtype(), DType::Q8_0);
    assert_eq!(view.element_count(), 32);
    assert_eq!(view.as_bytes(), &bytes[data_offset..data_offset + 34]);
    assert_eq!(
        file.tensor_bytes("output.weight").expect("tensor exists"),
        &bytes[data_offset + 34..data_offset + 52]
    );

    assert!(matches!(
        file.tensor("missing"),
        Err(crate::GgufError::TensorNotFound(_))
    ));
    assert!(matches!(
        file.tensor_bytes("missing"),
        Err(crate::GgufError::TensorNotFound(_))
    ));
}

#[test]
fn round_trip_v1() {
    check_fixture(GgufVersion::V1);
}

#[test]
fn round_trip_v2() {
    check_fixture(GgufVersion::V2);
}

#[test]
fn round_trip_v3() {
    check_fixture(GgufVersion::V3);
}

#[test]
fn version_numbers() {
    assert_eq!(GgufVersion::V1.number(), 1);
    assert_eq!(GgufVersion::V2.number(), 2);
    assert_eq!(GgufVersion::V3.number(), 3);
}

#[test]
fn dtype_mapping_and_block_layout() {
    for (code, dtype) in [
        (0, DType::F32),
        (1, DType::F16),
        (2, DType::Q4_0),
        (6, DType::Q5_0),
        (8, DType::Q8_0),
        (10, DType::Q2_K),
        (11, DType::Q3_K),
        (12, DType::Q4_K),
        (13, DType::Q5_K),
        (14, DType::Q6_K),
        (30, DType::BF16),
    ] {
        assert_eq!(DType::from_u32(code), dtype);
        assert_eq!(dtype.as_u32(), code);
    }
    assert_eq!(DType::from_u32(3), DType::Unknown(3));
    assert_eq!(DType::from_u32(28), DType::Unknown(28));
    assert_eq!(DType::Unknown(999).as_u32(), 999);

    assert_eq!(
        (DType::F32.block_bytes(), DType::F32.element_count()),
        (4, 1)
    );
    assert_eq!(
        (DType::F16.block_bytes(), DType::F16.element_count()),
        (2, 1)
    );
    assert_eq!(
        (DType::BF16.block_bytes(), DType::BF16.element_count()),
        (2, 1)
    );
    assert_eq!(
        (DType::Q8_0.block_bytes(), DType::Q8_0.element_count()),
        (34, 32)
    );
    assert_eq!(
        (DType::Q4_0.block_bytes(), DType::Q4_0.element_count()),
        (18, 32)
    );
    assert_eq!(
        (DType::Q5_0.block_bytes(), DType::Q5_0.element_count()),
        (22, 32)
    );
    assert_eq!(
        (DType::Q2_K.block_bytes(), DType::Q2_K.element_count()),
        (84, 256)
    );
    assert_eq!(
        (DType::Q3_K.block_bytes(), DType::Q3_K.element_count()),
        (110, 256)
    );
    assert_eq!(
        (DType::Q4_K.block_bytes(), DType::Q4_K.element_count()),
        (144, 256)
    );
    assert_eq!(
        (DType::Q5_K.block_bytes(), DType::Q5_K.element_count()),
        (176, 256)
    );
    assert_eq!(
        (DType::Q6_K.block_bytes(), DType::Q6_K.element_count()),
        (210, 256)
    );
    // `general.file_type` uses the ggml ftype enum, not tensor dtype ids.
    assert_eq!(DType::Q8_0.file_type(), 7);
    assert_eq!(DType::Q4_K.file_type(), 12);
    assert_eq!(DType::Q6_K.file_type(), 14);

    assert_eq!(DType::Unknown(3).block_bytes(), 0);
    assert_eq!(DType::Unknown(3).element_count(), 0);
}

#[test]
fn truncated_fixtures_fail_at_every_prefix() {
    for version in [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3] {
        let bytes = fixture(version);
        for len in 0..bytes.len() {
            let result = GgufFile::from_bytes(&bytes[..len]);
            assert!(
                result.is_err(),
                "prefix of length {len} unexpectedly parsed"
            );
        }
        assert!(GgufFile::from_bytes(&bytes).is_ok());
    }
}

#[test]
fn malformed_headers_are_rejected() {
    assert!(matches!(
        GgufFile::from_bytes(&[]),
        Err(crate::GgufError::Truncated { .. })
    ));
    let mut bad_magic = fixture(GgufVersion::V1);
    bad_magic[0] = b'X';
    assert!(matches!(
        GgufFile::from_bytes(&bad_magic),
        Err(crate::GgufError::InvalidMagic)
    ));

    let mut bad_version = fixture(GgufVersion::V1);
    bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        GgufFile::from_bytes(&bad_version),
        Err(crate::GgufError::UnsupportedVersion(99))
    ));

    let mut huge_v1 = b"GGUF".to_vec();
    huge_v1.extend_from_slice(&1u32.to_le_bytes());
    huge_v1.extend_from_slice(&u32::MAX.to_le_bytes());
    huge_v1.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(GgufFile::from_bytes(&huge_v1).is_err());

    let mut huge_v2 = b"GGUF".to_vec();
    huge_v2.extend_from_slice(&3u32.to_le_bytes());
    huge_v2.extend_from_slice(&u64::MAX.to_le_bytes());
    huge_v2.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(GgufFile::from_bytes(&huge_v2).is_err());

    let mut bad_offset = fixture(GgufVersion::V1);
    let data_offset = GgufFile::from_bytes(&bad_offset)
        .expect("fixture is valid")
        .header()
        .data_offset as usize;
    // Find the second tensor's relative offset (34) in the descriptor area.
    let offset_pos = bad_offset[..data_offset]
        .windows(8)
        .enumerate()
        .filter(|(_, window)| *window == 34u64.to_le_bytes())
        .map(|(index, _)| index)
        .next_back()
        .expect("relative tensor offset in fixture");
    let file_len = bad_offset.len() as u64;
    bad_offset[offset_pos..offset_pos + 8].copy_from_slice(&file_len.to_le_bytes());
    assert!(matches!(
        GgufFile::from_bytes(&bad_offset),
        Err(crate::GgufError::BadTensorOffset { .. })
    ));
}

#[test]
fn unknown_dtype_descriptor_is_preserved() {
    let mut bytes = fixture(GgufVersion::V2);
    let file = GgufFile::from_bytes(&bytes).expect("fixture is valid");
    let data_offset = file.header().data_offset as usize;
    // Second tensor descriptor: name, dims, dtype, offset. Locate dtype id by
    // scanning for the raw q4_0 id in the descriptor area before data_offset.
    let q4_code = DType::Q4_0.as_u32().to_le_bytes();
    let positions = bytes[..data_offset]
        .windows(4)
        .enumerate()
        .filter(|(_, window)| *window == q4_code)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let dtype_pos = *positions.last().expect("q4_0 dtype id in descriptor");
    bytes[dtype_pos..dtype_pos + 4].copy_from_slice(&777u32.to_le_bytes());

    let file = GgufFile::from_bytes(&bytes).expect("unknown dtype still parses");
    assert_eq!(file.header().tensors[1].dtype, DType::Unknown(777));
    assert_eq!(file.header().tensors[1].byte_len, 0);
}

fn q8_bytes(scale: f16, values: &[i8]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks(32) {
        out.extend_from_slice(&scale.to_le_bytes());
        for &value in block {
            out.push(value as u8);
        }
    }
    out
}

fn q4_bytes(scale: f16, values: &[i8]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut out = Vec::with_capacity(values.len() / 32 * 18);
    for block in values.chunks(32) {
        out.extend_from_slice(&scale.to_le_bytes());
        for pair in block.as_chunks::<2>().0 {
            let low = (pair[0] + 8) as u8 & 0x0f;
            let high = (pair[1] + 8) as u8 & 0x0f;
            out.push(low | (high << 4));
        }
    }
    out
}

#[test]
fn dequantize_q8_0_hand_computed() {
    let mut values = [0i8; 32];
    values[..4].copy_from_slice(&[2, -4, 0, 1]);
    let bytes = q8_bytes(f16::from_f32(0.5), &values);
    let mut out = [f32::NAN; 32];
    dequantize_q8_0(&bytes, &mut out).expect("valid q8_0 block");
    assert_eq!(&out[..4], &[1.0, -2.0, 0.0, 0.5]);
    assert!(out[4..].iter().all(|&value| value == 0.0));
}

#[test]
fn dot_q8_0_hand_computed() {
    let mut values = [0i8; 32];
    values[..4].copy_from_slice(&[2, -4, 0, 1]);
    let bytes = q8_bytes(f16::from_f32(0.5), &values);
    let mut lhs = [0.0f32; 32];
    lhs[..4].copy_from_slice(&[1.0, 0.5, 3.0, -2.0]);
    assert_eq!(dot_q8_0(&bytes, &lhs).expect("valid dot"), -1.0);
}

#[test]
fn dequantize_q4_0_hand_computed() {
    let mut values = [-8i8; 32];
    values[..4].copy_from_slice(&[2, 4, -2, -8]);
    let bytes = q4_bytes(f16::from_f32(2.0), &values);
    let mut out = [f32::NAN; 32];
    dequantize_q4_0(&bytes, &mut out).expect("valid q4_0 block");
    assert_eq!(&out[..4], &[4.0, 8.0, -4.0, -16.0]);
    assert!(out[4..].iter().all(|&value| value == -16.0));
}

#[test]
fn dot_q4_0_hand_computed() {
    let mut values = [-8i8; 32];
    values[..4].copy_from_slice(&[2, 4, -2, -8]);
    let bytes = q4_bytes(f16::from_f32(2.0), &values);
    let mut lhs = [0.0f32; 32];
    lhs[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dot_q4_0(&bytes, &lhs).expect("valid dot"), -56.0);
}

#[test]
fn dequantize_f16_hand_computed() {
    let bytes = [
        0x00, 0x3c, // 1.0
        0x00, 0xc0, // -2.0
        0x00, 0x38, // 0.5
    ];
    let mut out = [f32::NAN; 4];
    dequantize_f16(&bytes, &mut out).expect("valid f16 data");
    assert_eq!(&out[..3], &[1.0, -2.0, 0.5]);
    assert!(out[3].is_nan());
}

#[test]
fn embedding_lookup_q8_0_reads_only_one_row() {
    let row0 = [4i8; 32];
    let mut row1 = [0i8; 32];
    row1[..3].copy_from_slice(&[1, -1, 2]);
    let mut bytes = q8_bytes(f16::from_f32(0.5), &row0);
    bytes.extend_from_slice(&q8_bytes(f16::from_f32(1.0), &row1));

    let mut out = [f32::NAN; 32];
    embedding_lookup_q8_0(&bytes, 1, 32, &mut out).expect("second row exists");
    assert_eq!(&out[..3], &[1.0, -1.0, 2.0]);
    assert!(out[3..].iter().all(|&value| value == 0.0));

    embedding_lookup_q8_0(&bytes, 0, 32, &mut out).expect("first row exists");
    assert!(out.iter().all(|&value| value == 2.0));
}

#[test]
fn quant_error_cases() {
    let bytes = q8_bytes(f16::from_f32(1.0), &[1i8; 32]);
    let mut short = [0.0f32; 31];
    assert!(matches!(
        dequantize_q8_0(&bytes, &mut short),
        Err(crate::GgufError::OutputTooSmall {
            needed: 32,
            actual: 31
        })
    ));
    assert!(matches!(
        dequantize_q8_0(&bytes[..33], &mut [0.0f32; 32]),
        Err(crate::GgufError::InvalidBlockLength { .. })
    ));
    assert!(matches!(
        dot_q8_0(&bytes, &[1.0f32; 31]),
        Err(crate::GgufError::LengthMismatch { quant: 32, lhs: 31 })
    ));
    assert!(matches!(
        embedding_lookup_q8_0(&bytes, 1, 32, &mut [0.0f32; 32]),
        Err(crate::GgufError::RowOutOfRange { index: 1, rows: 1 })
    ));
    assert!(matches!(
        embedding_lookup_q8_0(&bytes, 0, 30, &mut [0.0f32; 30]),
        Err(crate::GgufError::InvalidQuantArgument(_))
    ));
    assert!(matches!(
        dequantize_q4_0(&[0u8; 17], &mut [0.0f32; 32]),
        Err(crate::GgufError::InvalidBlockLength { .. })
    ));
    assert!(matches!(
        dequantize_f16(&[0u8; 3], &mut [0.0f32; 2]),
        Err(crate::GgufError::InvalidBlockLength { .. })
    ));
}

#[test]
fn mmap_open_roundtrip() {
    for version in [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3] {
        let path = std::env::temp_dir().join(format!(
            "quarrel-gguf-{}-{}.gguf",
            std::process::id(),
            version.number()
        ));
        std::fs::write(&path, fixture(version)).expect("write fixture");
        let mmap = GgufMmap::open(&path).expect("open fixture");
        assert_eq!(mmap.header().version, version);
        assert_eq!(mmap.metadata_tensor_count(), 2);
        let file = mmap.file();
        assert_eq!(file.header().tensor_count, 2);
        assert_eq!(file.tensor_bytes("output.weight").unwrap().len(), 18);
        drop(file);
        drop(mmap);
        std::fs::remove_file(&path).expect("remove fixture");
    }
}

#[test]
fn snapshot_of_parsed_header() {
    let bytes = fixture(GgufVersion::V3);
    let file = GgufFile::from_bytes(&bytes).expect("fixture must parse");
    insta::assert_debug_snapshot!(file.header());
}

proptest! {
    #[test]
    fn arbitrary_garbage_is_rejected(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
        let mut bytes = bytes;
        if bytes.len() >= 4 {
            // Ensure this is never mistaken for a GGUF file.
            bytes[..4].copy_from_slice(b"NOPE");
        }
        prop_assert!(GgufFile::from_bytes(&bytes).is_err());
    }

    #[test]
    fn huge_length_prefixes_do_not_panic(
        len in any::<u64>(),
        count in any::<u64>(),
        tail in prop::collection::vec(any::<u8>(), 0..32),
    ) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(b"key");
        bytes.extend_from_slice(&tail);
        let _ = GgufFile::from_bytes(&bytes);
    }

    #[test]
    fn dequantizers_respect_output_length(extra in 0usize..8, short_by in 1usize..32) {
        let mut values = [0i8; 32];
        values[..4].copy_from_slice(&[2, -4, 0, 1]);
        let q8 = q8_bytes(f16::from_f32(0.5), &values);

        let mut out = vec![123.0f32; 32 + extra];
        dequantize_q8_0(&q8, &mut out).expect("valid q8 data");
        prop_assert_eq!(&out[..4], &[1.0, -2.0, 0.0, 0.5]);
        prop_assert!(out[4..32].iter().all(|&v| v == 0.0));
        prop_assert!(out[32..].iter().all(|&v| v == 123.0));

        let mut short = vec![123.0f32; 32 - short_by];
        prop_assert!(dequantize_q8_0(&q8, &mut short).is_err());
        prop_assert!(short.iter().all(|&v| v == 123.0));

        let mut q4_values = [-8i8; 32];
        q4_values[..2].copy_from_slice(&[2, 4]);
        let q4 = q4_bytes(f16::from_f32(2.0), &q4_values);
        let mut out = vec![99.0f32; 32 + extra];
        dequantize_q4_0(&q4, &mut out).expect("valid q4 data");
        prop_assert_eq!(&out[..2], &[4.0, 8.0]);
        prop_assert!(out[2..32].iter().all(|&v| v == -16.0));
        prop_assert!(out[32..].iter().all(|&v| v == 99.0));

        let f16_bytes = [0x00, 0x3c, 0x00, 0xc0];
        let mut out = vec![7.0f32; 2 + extra];
        dequantize_f16(&f16_bytes, &mut out).expect("valid f16 data");
        prop_assert_eq!(&out[..2], &[1.0, -2.0]);
        prop_assert!(out[2..].iter().all(|&v| v == 7.0));
    }
}

#[test]
fn public_types_implement_serde_traits() {
    fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
    assert_serde::<MetadataValue>();
    assert_serde::<DType>();
    assert_serde::<crate::GgufHeader>();
    assert_serde::<crate::TensorDescriptor>();
}
