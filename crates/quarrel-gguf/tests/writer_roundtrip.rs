//! Round-trip and error tests for the deterministic GGUF writer.

use half::f16;
use quarrel_gguf::quant::{dequantize_f16, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0};
use quarrel_gguf::writer::{GgufWriter, requantize_gguf};
use quarrel_gguf::{DType, GgufError, GgufFile, GgufVersion, MetadataValue};

const ELEMENTS: usize = 256;

fn values() -> Vec<f32> {
    (0..ELEMENTS)
        .map(|index| (index as f32 - 127.5) / 127.5)
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn source_fixture(version: GgufVersion) -> Vec<u8> {
    let values = values();
    let mut writer = GgufWriter::new(version);
    writer
        .set_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        )
        .unwrap();
    writer
        .set_metadata("general.file_type", MetadataValue::Uint32(0))
        .unwrap();
    writer
        .set_metadata("test.bool", MetadataValue::Bool(true))
        .unwrap();
    writer
        .set_metadata("test.int8", MetadataValue::Int8(-3))
        .unwrap();
    writer
        .set_metadata("test.uint16", MetadataValue::Uint16(60_000))
        .unwrap();
    writer
        .set_metadata("test.int16", MetadataValue::Int16(-1234))
        .unwrap();
    writer
        .set_metadata("test.int64", MetadataValue::Int64(-9_000_000_000))
        .unwrap();
    writer
        .set_metadata("test.float64", MetadataValue::Float64(0.25))
        .unwrap();
    writer
        .set_metadata(
            "test.array",
            MetadataValue::Array(vec![MetadataValue::Int16(-1), MetadataValue::Int16(2)]),
        )
        .unwrap();
    writer
        .set_metadata(
            "test.strings",
            MetadataValue::Array(vec![
                MetadataValue::String("alpha".into()),
                MetadataValue::String("beta".into()),
            ]),
        )
        .unwrap();
    writer
        .add_tensor("w0", vec![16, 16], DType::F32, f32_bytes(&values))
        .unwrap();
    writer
        .add_tensor("w1", vec![ELEMENTS as u64], DType::F32, f32_bytes(&values))
        .unwrap();
    writer.to_bytes().unwrap()
}

fn align_up(value: u64) -> u64 {
    (value + 31) & !31
}

fn expected_byte_len(target: DType) -> u64 {
    match target {
        DType::F32 => 1024,
        DType::F16 => 512,
        DType::Q8_0 => 272,
        DType::Q4_K => 144,
        DType::Q6_K => 210,
        other => panic!("unexpected target dtype {other:?}"),
    }
}

fn decode_tensor(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect(),
        DType::F16 => {
            let mut out = vec![0.0; bytes.len() / 2];
            dequantize_f16(bytes, &mut out).unwrap();
            out
        }
        DType::Q8_0 => {
            let mut out = vec![0.0; bytes.len() / 34 * 32];
            dequantize_q8_0(bytes, &mut out).unwrap();
            out
        }
        DType::Q4_K => {
            let mut out = vec![0.0; bytes.len() / 144 * 256];
            dequantize_q4_k(bytes, &mut out).unwrap();
            out
        }
        DType::Q6_K => {
            let mut out = vec![0.0; bytes.len() / 210 * 256];
            dequantize_q6_k(bytes, &mut out).unwrap();
            out
        }
        other => panic!("unexpected dtype {other:?}"),
    }
}

fn tolerance(target: DType) -> f32 {
    match target {
        DType::F32 => 0.0,
        DType::F16 => 0.001,
        DType::Q8_0 => 0.02,
        DType::Q4_K => 0.15,
        DType::Q6_K => 0.03,
        other => panic!("unexpected target dtype {other:?}"),
    }
}

#[test]
fn round_trip_all_targets_and_versions() {
    for version in [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3] {
        let source = source_fixture(version);
        for target in [
            DType::F32,
            DType::F16,
            DType::Q8_0,
            DType::Q4_K,
            DType::Q6_K,
        ] {
            let output = requantize_gguf(&source, target).expect("valid requantization");
            let file = GgufFile::from_bytes(&output).expect("writer produced a valid GGUF file");
            let header = file.header();

            assert_eq!(header.version, version, "version preserved for {target:?}");
            assert_eq!(header.tensor_count, 2);
            assert_eq!(header.data_offset % 32, 0);
            assert_eq!(
                header.metadata.get("general.architecture"),
                Some(&MetadataValue::String("llama".into()))
            );
            assert_eq!(
                header.metadata.get("general.file_type"),
                Some(&MetadataValue::Uint32(target.file_type()))
            );
            assert_eq!(
                header.metadata.get("test.bool"),
                Some(&MetadataValue::Bool(true))
            );
            assert_eq!(
                header.metadata.get("test.int8"),
                Some(&MetadataValue::Int8(-3))
            );
            assert_eq!(
                header.metadata.get("test.uint16"),
                Some(&MetadataValue::Uint16(60_000))
            );
            assert_eq!(
                header.metadata.get("test.int16"),
                Some(&MetadataValue::Int16(-1234))
            );
            assert_eq!(
                header.metadata.get("test.int64"),
                Some(&MetadataValue::Int64(-9_000_000_000))
            );
            assert_eq!(
                header.metadata.get("test.float64"),
                Some(&MetadataValue::Float64(0.25))
            );
            assert_eq!(
                header.metadata.get("test.array"),
                Some(&MetadataValue::Array(vec![
                    MetadataValue::Int16(-1),
                    MetadataValue::Int16(2)
                ]))
            );
            assert_eq!(
                header.metadata.get("test.strings"),
                Some(&MetadataValue::Array(vec![
                    MetadataValue::String("alpha".into()),
                    MetadataValue::String("beta".into())
                ]))
            );

            let byte_len = expected_byte_len(target);
            let offsets = [0, align_up(byte_len)];
            for (index, descriptor) in header.tensors.iter().enumerate() {
                assert_eq!(descriptor.name, if index == 0 { "w0" } else { "w1" });
                assert_eq!(descriptor.dtype, target);
                assert_eq!(
                    descriptor.shape,
                    if index == 0 {
                        vec![16, 16]
                    } else {
                        vec![ELEMENTS as u64]
                    }
                );
                assert_eq!(descriptor.offset, offsets[index], "offset for {target:?}");
                assert_eq!(descriptor.offset % 32, 0);
                assert_eq!(descriptor.byte_len, byte_len);
                let raw = file.tensor_bytes(&descriptor.name).unwrap();
                assert_eq!(raw.len() as u64, descriptor.byte_len);
            }

            let decoded = decode_tensor(file.tensor_bytes("w0").unwrap(), target);
            assert_eq!(decoded.len(), ELEMENTS);
            assert!(
                decoded.iter().all(|value| value.is_finite()),
                "non-finite dequantized value for {target:?}"
            );
            let max_error = values()
                .iter()
                .zip(decoded.iter())
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= tolerance(target),
                "max error {max_error} exceeds {} for {target:?}",
                tolerance(target)
            );
        }
    }
}

#[test]
fn requantizes_f16_source_tensors() {
    let expected_values = values();
    let data: Vec<u8> = expected_values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect();
    let mut writer = GgufWriter::new(GgufVersion::V2);
    writer
        .add_tensor("w0", vec![ELEMENTS as u64], DType::F16, data)
        .unwrap();
    let source = writer.to_bytes().unwrap();

    let output = requantize_gguf(&source, DType::Q8_0).unwrap();
    let file = GgufFile::from_bytes(&output).unwrap();
    assert_eq!(file.header().tensors[0].dtype, DType::Q8_0);
    let decoded = decode_tensor(file.tensor_bytes("w0").unwrap(), DType::Q8_0);
    assert!(decoded.iter().all(|value| value.is_finite()));
    let max_error = expected_values
        .iter()
        .zip(decoded.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 0.03, "max error {max_error}");
}

#[test]
fn requantization_is_byte_identical() {
    for version in [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3] {
        let source = source_fixture(version);
        for target in [
            DType::F32,
            DType::F16,
            DType::Q8_0,
            DType::Q4_K,
            DType::Q6_K,
        ] {
            let first = requantize_gguf(&source, target).unwrap();
            let second = requantize_gguf(&source, target).unwrap();
            assert_eq!(first, second, "non-deterministic output for {target:?}");
        }
    }
}

#[test]
fn writer_rejects_malformed_inputs() {
    let mut writer = GgufWriter::new(GgufVersion::V1);
    assert!(matches!(
        writer.add_tensor("bad", vec![0], DType::F32, vec![]),
        Err(GgufError::InvalidShape(_))
    ));
    assert!(
        writer
            .add_tensor("bad", vec![4], DType::F32, vec![0; 16])
            .is_ok()
    );
    assert!(matches!(
        writer.add_tensor("bad", vec![4], DType::F32, vec![0; 16]),
        Err(GgufError::DuplicateTensorName(_))
    ));
    assert!(matches!(
        writer.add_tensor("size", vec![4], DType::F32, vec![0; 15]),
        Err(GgufError::InvalidTensorLength { .. })
    ));
    assert!(matches!(
        writer.add_tensor("unknown", vec![4], DType::Unknown(99), vec![0; 16]),
        Err(GgufError::UnsupportedDType(_))
    ));
    assert!(matches!(
        writer.set_metadata(
            "mixed",
            MetadataValue::Array(vec![MetadataValue::Uint8(1), MetadataValue::Int8(2)])
        ),
        Err(GgufError::EncodeError { .. })
    ));
    assert!(matches!(
        writer.set_metadata(
            "nested",
            MetadataValue::Array(vec![MetadataValue::Array(vec![])])
        ),
        Err(GgufError::NestedArray)
    ));
    assert!(matches!(
        writer.set_metadata("nan", MetadataValue::Float32(f32::NAN)),
        Err(GgufError::EncodeError { .. })
    ));
    assert!(matches!(
        writer.set_metadata("", MetadataValue::Bool(true)),
        Err(GgufError::EncodeError { .. })
    ));
}

#[test]
fn requantizer_rejects_unsupported_inputs() {
    let source = source_fixture(GgufVersion::V3);
    assert!(matches!(
        requantize_gguf(&source, DType::Q2_K),
        Err(GgufError::UnsupportedDType(_))
    ));
    assert!(matches!(
        requantize_gguf(&[], DType::F16),
        Err(GgufError::Truncated { .. })
    ));

    let mut writer = GgufWriter::new(GgufVersion::V3);
    writer
        .add_tensor("q", vec![32], DType::Q8_0, vec![0; 34])
        .unwrap();
    let quantized = writer.to_bytes().unwrap();
    assert!(matches!(
        requantize_gguf(&quantized, DType::F32),
        Err(GgufError::UnsupportedRequantization { .. })
    ));
}

#[test]
fn k_quant_dequantizers_validate_block_lengths() {
    let mut out = [0.0f32; 256];
    assert!(matches!(
        dequantize_q4_k(&[0u8; 143], &mut out),
        Err(GgufError::InvalidBlockLength { .. })
    ));
    assert!(matches!(
        dequantize_q6_k(&[0u8; 209], &mut out),
        Err(GgufError::InvalidBlockLength { .. })
    ));
    assert!(matches!(
        dequantize_q4_k(&[0u8; 144], &mut [0.0; 255]),
        Err(GgufError::OutputTooSmall { .. })
    ));
    assert!(matches!(
        dequantize_q6_k(&[0u8; 210], &mut [0.0; 255]),
        Err(GgufError::OutputTooSmall { .. })
    ));
}
