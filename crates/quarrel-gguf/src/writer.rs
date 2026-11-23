//! Deterministic GGUF v1-v3 writer and F32/F16 requantization helpers.
//!
//! [`GgufWriter`] serializes typed metadata and already-encoded tensor payloads
//! into a canonical little-endian GGUF byte stream. Tensor data is padded to
//! 32-byte boundaries, both at the start of the data section and before every
//! tensor, so the same inputs always produce byte-identical output.
//!
//! [`requantize_gguf`] reads a GGUF file whose tensors are F32 or F16 and
//! writes a new file with the same version, metadata, names and shapes using
//! one of `F32`, `F16`, `Q8_0`, `Q4_K` or `Q6_K` tensors.

use crate::DType;
use crate::GgufError;
use crate::GgufFile;
use crate::GgufVersion;
use crate::MetadataValue;
use crate::header::tensor_byte_len;
use crate::quant;
use half::f16;
use std::collections::BTreeMap;
use std::io::Write;

const MAGIC: &[u8; 4] = b"GGUF";
const ALIGNMENT: u64 = 32;
const ZEROS: [u8; ALIGNMENT as usize] = [0; ALIGNMENT as usize];

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    shape: Vec<u64>,
    dtype: DType,
    byte_len: u64,
    data: Vec<u8>,
}

#[derive(Debug)]
struct Layout {
    header: Vec<u8>,
    header_len: u64,
    data_offset: u64,
    offsets: Vec<u64>,
    final_len: u64,
}

/// Builder for deterministic GGUF v1-v3 files.
///
/// Metadata is emitted in key order and tensor payloads are stored in insertion
/// order, each starting at a 32-byte-aligned offset relative to the data
/// section. No operation in this type panics on invalid caller input; errors are
/// returned as [`GgufError`].
#[derive(Debug, Clone)]
pub struct GgufWriter {
    version: GgufVersion,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorSpec>,
}

impl GgufWriter {
    /// Create an empty writer for `version`.
    pub fn new(version: GgufVersion) -> Self {
        Self {
            version,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }

    /// File format version this writer will emit.
    pub fn version(&self) -> GgufVersion {
        self.version
    }

    /// Number of tensor descriptors added so far.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Insert or replace a metadata entry.
    pub fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: MetadataValue,
    ) -> Result<&mut Self, GgufError> {
        let key = key.into();
        validate_metadata(self.version, &key, &value)?;
        self.metadata.insert(key, value);
        Ok(self)
    }

    /// Add a tensor with an already-encoded little-endian payload.
    ///
    /// `data.len()` must equal the byte length implied by `shape` and `dtype`,
    /// including quant-block padding.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: DType,
        data: Vec<u8>,
    ) -> Result<&mut Self, GgufError> {
        let name = name.into();
        if name.is_empty() {
            return Err(GgufError::EncodeError {
                what: "tensor name",
                reason: "names must not be empty".to_owned(),
            });
        }
        if self.tensors.iter().any(|tensor| tensor.name == name) {
            return Err(GgufError::DuplicateTensorName(name));
        }
        if shape.is_empty() || shape.len() > 4 || shape.contains(&0) {
            return Err(GgufError::InvalidShape(format!(
                "tensor {name:?} must have 1..=4 non-zero dimensions"
            )));
        }
        if dtype.block_bytes() == 0 || dtype.element_count() == 0 {
            return Err(GgufError::UnsupportedDType(dtype));
        }
        let byte_len = tensor_byte_len(dtype, &shape, &name)?;
        let actual = u64::try_from(data.len()).map_err(|_| GgufError::Overflow)?;
        if actual != byte_len {
            return Err(GgufError::InvalidTensorLength {
                name,
                expected: byte_len,
                actual,
            });
        }
        self.tensors.push(TensorSpec {
            name,
            shape,
            dtype,
            byte_len,
            data,
        });
        Ok(self)
    }

    /// Serialize the file into a freshly allocated byte vector.
    pub fn to_bytes(&self) -> Result<Vec<u8>, GgufError> {
        let layout = self.layout()?;
        let mut out = Vec::new();
        out.try_reserve(usize::try_from(layout.final_len).map_err(|_| GgufError::Overflow)?)
            .map_err(|_| GgufError::Overflow)?;
        self.write_layout(&layout, &mut out)?;
        Ok(out)
    }

    /// Write the serialized file into `writer`.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), GgufError> {
        let layout = self.layout()?;
        self.write_layout(&layout, writer)
    }

    fn write_layout<W: Write>(&self, layout: &Layout, writer: &mut W) -> Result<(), GgufError> {
        writer.write_all(&layout.header)?;
        let mut position = layout.header_len;
        let padding = layout
            .data_offset
            .checked_sub(position)
            .ok_or(GgufError::Overflow)?;
        write_zeros(writer, padding)?;
        position = layout.data_offset;

        for (spec, relative_offset) in self.tensors.iter().zip(layout.offsets.iter()) {
            let start = layout
                .data_offset
                .checked_add(*relative_offset)
                .ok_or(GgufError::Overflow)?;
            let padding = start.checked_sub(position).ok_or(GgufError::Overflow)?;
            write_zeros(writer, padding)?;
            writer.write_all(&spec.data)?;
            position = start
                .checked_add(spec.byte_len)
                .ok_or(GgufError::Overflow)?;
            let aligned = align_up(position)?;
            let padding = aligned.checked_sub(position).ok_or(GgufError::Overflow)?;
            write_zeros(writer, padding)?;
            position = aligned;
        }
        if position != layout.final_len {
            return Err(GgufError::Overflow);
        }
        Ok(())
    }

    fn layout(&self) -> Result<Layout, GgufError> {
        let tensor_count = u64::try_from(self.tensors.len()).map_err(|_| GgufError::Overflow)?;
        let kv_count = u64::try_from(self.metadata.len()).map_err(|_| GgufError::Overflow)?;

        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut relative = 0u64;
        for spec in &self.tensors {
            relative = align_up(relative)?;
            offsets.push(relative);
            relative = relative
                .checked_add(spec.byte_len)
                .ok_or(GgufError::Overflow)?;
        }
        let data_len = align_up(relative)?;

        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&self.version.number().to_le_bytes());
        write_count(&mut header, self.version, tensor_count, "tensor count")?;
        write_count(&mut header, self.version, kv_count, "metadata count")?;
        for (key, value) in &self.metadata {
            write_string(&mut header, self.version, key, "metadata key")?;
            write_value(&mut header, self.version, value)?;
        }
        for (spec, offset) in self.tensors.iter().zip(offsets.iter()) {
            write_string(&mut header, self.version, &spec.name, "tensor name")?;
            let dims = u32::try_from(spec.shape.len()).map_err(|_| GgufError::Overflow)?;
            header.extend_from_slice(&dims.to_le_bytes());
            for &dim in &spec.shape {
                header.extend_from_slice(&dim.to_le_bytes());
            }
            header.extend_from_slice(&spec.dtype.as_u32().to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
        }

        let header_len = u64::try_from(header.len()).map_err(|_| GgufError::Overflow)?;
        let data_offset = align_up(header_len)?;
        let final_len = data_offset
            .checked_add(data_len)
            .ok_or(GgufError::Overflow)?;
        Ok(Layout {
            header,
            header_len,
            data_offset,
            offsets,
            final_len,
        })
    }
}

fn align_up(value: u64) -> Result<u64, GgufError> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|sum| sum & !(ALIGNMENT - 1))
        .ok_or(GgufError::Overflow)
}

fn write_zeros<W: Write>(writer: &mut W, mut len: u64) -> Result<(), GgufError> {
    while len >= ALIGNMENT {
        writer.write_all(&ZEROS)?;
        len -= ALIGNMENT;
    }
    if len > 0 {
        // The remaining length is guaranteed to be smaller than ALIGNMENT.
        writer.write_all(&ZEROS[..len as usize])?;
    }
    Ok(())
}

fn write_count(
    out: &mut Vec<u8>,
    version: GgufVersion,
    value: u64,
    what: &'static str,
) -> Result<(), GgufError> {
    if version == GgufVersion::V1 {
        let narrowed = u32::try_from(value).map_err(|_| GgufError::EncodeError {
            what,
            reason: format!("{value} does not fit in a 32-bit v1 count"),
        })?;
        out.extend_from_slice(&narrowed.to_le_bytes());
    } else {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn write_string(
    out: &mut Vec<u8>,
    version: GgufVersion,
    value: &str,
    what: &'static str,
) -> Result<(), GgufError> {
    let len = u64::try_from(value.len()).map_err(|_| GgufError::Overflow)?;
    write_count(out, version, len, what)?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn value_type(value: &MetadataValue) -> u32 {
    match value {
        MetadataValue::Uint8(_) => 0,
        MetadataValue::Int8(_) => 1,
        MetadataValue::Uint16(_) => 2,
        MetadataValue::Int16(_) => 3,
        MetadataValue::Uint32(_) => 4,
        MetadataValue::Int32(_) => 5,
        MetadataValue::Float32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::Array(_) => 9,
        MetadataValue::Uint64(_) => 10,
        MetadataValue::Int64(_) => 11,
        MetadataValue::Float64(_) => 12,
    }
}

fn validate_metadata(
    version: GgufVersion,
    key: &str,
    value: &MetadataValue,
) -> Result<(), GgufError> {
    if key.is_empty() {
        return Err(GgufError::EncodeError {
            what: "metadata key",
            reason: "keys must not be empty".to_owned(),
        });
    }
    match value {
        MetadataValue::Array(items) => {
            if items
                .iter()
                .any(|item| matches!(item, MetadataValue::Array(_)))
            {
                return Err(GgufError::NestedArray);
            }
            if let Some(first) = items.first() {
                let first_type = value_type(first);
                if items.iter().any(|item| value_type(item) != first_type) {
                    return Err(GgufError::EncodeError {
                        what: "metadata array",
                        reason: "all elements must have the same value type".to_owned(),
                    });
                }
            }
            for item in items {
                validate_scalar(version, item)?;
            }
        }
        scalar => validate_scalar(version, scalar)?,
    }
    Ok(())
}

fn validate_scalar(version: GgufVersion, value: &MetadataValue) -> Result<(), GgufError> {
    match value {
        MetadataValue::Float32(number) if !number.is_finite() => Err(GgufError::EncodeError {
            what: "metadata float32",
            reason: "NaN and infinity are not deterministic metadata values".to_owned(),
        }),
        MetadataValue::Float64(number) if !number.is_finite() => Err(GgufError::EncodeError {
            what: "metadata float64",
            reason: "NaN and infinity are not deterministic metadata values".to_owned(),
        }),
        MetadataValue::String(text) => {
            let len = u64::try_from(text.len()).map_err(|_| GgufError::Overflow)?;
            if version == GgufVersion::V1 && len > u64::from(u32::MAX) {
                return Err(GgufError::EncodeError {
                    what: "metadata string",
                    reason: "v1 string length exceeds the 32-bit length prefix".to_owned(),
                });
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn write_value(
    out: &mut Vec<u8>,
    version: GgufVersion,
    value: &MetadataValue,
) -> Result<(), GgufError> {
    out.extend_from_slice(&value_type(value).to_le_bytes());
    match value {
        MetadataValue::Array(items) => {
            let element_type = items.first().map(value_type).unwrap_or(0);
            out.extend_from_slice(&element_type.to_le_bytes());
            let len = u64::try_from(items.len()).map_err(|_| GgufError::Overflow)?;
            out.extend_from_slice(&len.to_le_bytes());
            for item in items {
                write_payload(out, version, item, true)?;
            }
        }
        scalar => write_payload(out, version, scalar, false)?,
    }
    Ok(())
}

fn write_payload(
    out: &mut Vec<u8>,
    version: GgufVersion,
    value: &MetadataValue,
    in_array: bool,
) -> Result<(), GgufError> {
    match value {
        MetadataValue::Uint8(number) => out.push(*number),
        MetadataValue::Int8(number) => out.push(*number as u8),
        MetadataValue::Uint16(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Int16(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Uint32(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Int32(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Float32(number) => out.extend_from_slice(&number.to_bits().to_le_bytes()),
        MetadataValue::Bool(flag) => out.push(u8::from(*flag)),
        MetadataValue::String(text) => {
            if in_array || version != GgufVersion::V1 {
                let len = u64::try_from(text.len()).map_err(|_| GgufError::Overflow)?;
                out.extend_from_slice(&len.to_le_bytes());
            } else {
                write_string(out, version, text, "metadata string")?;
                return Ok(());
            }
            out.extend_from_slice(text.as_bytes());
        }
        MetadataValue::Array(_) => return Err(GgufError::NestedArray),
        MetadataValue::Uint64(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Int64(number) => out.extend_from_slice(&number.to_le_bytes()),
        MetadataValue::Float64(number) => out.extend_from_slice(&number.to_bits().to_le_bytes()),
    }
    Ok(())
}

fn check_finite_values(values: &[f32]) -> Result<(), GgufError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GgufError::NonFiniteValue {
                role: "input",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn decode_source_tensor(bytes: &[u8], dtype: DType, target: DType) -> Result<Vec<f32>, GgufError> {
    match dtype {
        DType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(GgufError::InvalidBlockLength {
                    block_bytes: 4,
                    actual: bytes.len(),
                });
            }
            let mut values = Vec::new();
            values
                .try_reserve(bytes.len() / 4)
                .map_err(|_| GgufError::Overflow)?;
            for quad in bytes.as_chunks::<4>().0 {
                values.push(f32::from_le_bytes(*quad));
            }
            check_finite_values(&values)?;
            Ok(values)
        }
        DType::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(GgufError::InvalidBlockLength {
                    block_bytes: 2,
                    actual: bytes.len(),
                });
            }
            let mut values = Vec::new();
            values
                .try_reserve(bytes.len() / 2)
                .map_err(|_| GgufError::Overflow)?;
            values.resize(bytes.len() / 2, 0.0);
            quant::dequantize_f16(bytes, &mut values)?;
            check_finite_values(&values)?;
            Ok(values)
        }
        other => Err(GgufError::UnsupportedRequantization {
            from: other,
            to: target,
        }),
    }
}

fn encode_target_tensor(values: &[f32], target: DType) -> Result<Vec<u8>, GgufError> {
    match target {
        DType::F32 => {
            let mut out = Vec::new();
            out.try_reserve(values.len().checked_mul(4).ok_or(GgufError::Overflow)?)
                .map_err(|_| GgufError::Overflow)?;
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(out)
        }
        DType::F16 => {
            let mut out = Vec::new();
            out.try_reserve(values.len().checked_mul(2).ok_or(GgufError::Overflow)?)
                .map_err(|_| GgufError::Overflow)?;
            for (index, &value) in values.iter().enumerate() {
                let encoded = f16::from_f32(value);
                if !encoded.is_finite() {
                    return Err(GgufError::QuantizationRange {
                        dtype: DType::F16,
                        index,
                        value,
                    });
                }
                out.extend_from_slice(&encoded.to_le_bytes());
            }
            Ok(out)
        }
        DType::Q8_0 => quant::quantize_f32_to_q8_0(values),
        DType::Q4_K => quant::quantize_f32_to_q4_k(values),
        DType::Q6_K => quant::quantize_f32_to_q6_k(values),
        other => Err(GgufError::UnsupportedDType(other)),
    }
}

/// Requantize every F32/F16 tensor in `bytes` to `target` and return a new
/// deterministic GGUF byte stream.
///
/// The input version, metadata, tensor names, shapes and order are preserved.
/// A numeric `general.file_type` entry is updated to match the target dtype.
pub fn requantize_gguf(bytes: &[u8], target: DType) -> Result<Vec<u8>, GgufError> {
    if !matches!(
        target,
        DType::F32 | DType::F16 | DType::Q8_0 | DType::Q4_K | DType::Q6_K
    ) {
        return Err(GgufError::UnsupportedDType(target));
    }
    let file = GgufFile::from_bytes(bytes)?;
    let mut writer = GgufWriter::new(file.header().version);

    for (key, value) in &file.header().metadata {
        let updated = if key == "general.file_type" && matches!(value, MetadataValue::Uint32(_)) {
            MetadataValue::Uint32(target.file_type())
        } else {
            value.clone()
        };
        writer.set_metadata(key.clone(), updated)?;
    }

    for descriptor in &file.header().tensors {
        let raw = file.tensor_bytes(&descriptor.name)?;
        let values = decode_source_tensor(raw, descriptor.dtype, target)?;
        let encoded = encode_target_tensor(&values, target)?;
        writer.add_tensor(
            descriptor.name.clone(),
            descriptor.shape.clone(),
            target,
            encoded,
        )?;
    }
    writer.to_bytes()
}
