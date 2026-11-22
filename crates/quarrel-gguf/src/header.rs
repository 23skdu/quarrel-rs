//! Safe little-endian parser for GGUF v1-v3 headers.

use crate::dtype::{DType, GgufVersion};
use crate::error::GgufError;
use bytes::Buf;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"GGUF";
const ALIGNMENT: u64 = 32;

/// A scalar or array value stored in a GGUF metadata entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// GGUF value type 0.
    Uint8(u8),
    /// GGUF value type 1.
    Int8(i8),
    /// GGUF value type 2.
    Uint16(u16),
    /// GGUF value type 3.
    Int16(i16),
    /// GGUF value type 4.
    Uint32(u32),
    /// GGUF value type 5.
    Int32(i32),
    /// GGUF value type 6.
    Float32(f32),
    /// GGUF value type 7 (a single byte, nonzero means true).
    Bool(bool),
    /// GGUF value type 8.
    String(String),
    /// GGUF value type 9, including its element type tag.
    Array(Vec<MetadataValue>),
    /// GGUF value type 10.
    Uint64(u64),
    /// GGUF value type 11.
    Int64(i64),
    /// GGUF value type 12.
    Float64(f64),
}

/// One entry from the tensor descriptor table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Tensor name, as stored in the file.
    pub name: String,
    /// Tensor dimensions, in GGUF order (innermost dimension first).
    pub shape: Vec<u64>,
    /// Element type.
    pub dtype: DType,
    /// Byte offset relative to [`GgufHeader::data_offset`].
    pub offset: u64,
    /// Byte length computed from `shape` and `dtype`.
    pub byte_len: u64,
}

/// Fully parsed GGUF header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufHeader {
    /// File format version.
    pub version: GgufVersion,
    /// Number of tensors described by the header.
    pub tensor_count: u64,
    /// Metadata key/value pairs, sorted by key.
    pub metadata: BTreeMap<String, MetadataValue>,
    /// Tensor descriptors, in file order.
    pub tensors: Vec<TensorDescriptor>,
    /// 32-byte-aligned offset at which tensor data starts.
    pub data_offset: u64,
}

/// Parse a GGUF header from the beginning of `bytes` and validate the tensor
/// data ranges against the available input.
pub(crate) fn parse_header(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
    let mut reader = Reader::new(bytes);

    if reader.take(MAGIC.len())? != MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version_raw = reader.u32()?;
    let version =
        GgufVersion::from_u32(version_raw).ok_or(GgufError::UnsupportedVersion(version_raw))?;

    let (tensor_count, kv_count) = if version == GgufVersion::V1 {
        (u64::from(reader.u32()?), u64::from(reader.u32()?))
    } else {
        (reader.u64()?, reader.u64()?)
    };

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = reader.string(version.string_len_bytes())?;
        let value = reader.value(version, false)?;
        // Later entries with the same key overwrite earlier ones, mirroring
        // the behaviour of most GGUF readers.
        metadata.insert(key, value);
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader.string(version.string_len_bytes())?;
        let n_dims = reader.u32()?;
        if !(1..=4).contains(&n_dims) {
            return Err(GgufError::InvalidShape(format!(
                "tensor {name:?} has {n_dims} dimensions (expected 1..=4)"
            )));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = reader.u64()?;
            if dim == 0 {
                return Err(GgufError::InvalidShape(format!(
                    "tensor {name:?} has a zero-sized dimension"
                )));
            }
            shape.push(dim);
        }
        let dtype = DType::from_u32(reader.u32()?);
        let offset = reader.u64()?;
        let byte_len = tensor_byte_len(dtype, &shape, &name)?;
        tensors.push(TensorDescriptor {
            name,
            shape,
            dtype,
            offset,
            byte_len,
        });
    }

    let header_end = reader.position();
    let data_offset = align_up(header_end, ALIGNMENT)?;
    let file_len = bytes.len() as u64;
    if data_offset > file_len {
        return Err(GgufError::Truncated {
            offset: header_end,
            needed: data_offset - file_len,
        });
    }

    for tensor in &tensors {
        let start = data_offset
            .checked_add(tensor.offset)
            .ok_or(GgufError::Overflow)?;
        let end = start
            .checked_add(tensor.byte_len)
            .ok_or(GgufError::Overflow)?;
        if end > file_len {
            return Err(GgufError::BadTensorOffset {
                name: tensor.name.clone(),
                data_offset,
                offset: tensor.offset,
                byte_len: tensor.byte_len,
                file_len,
            });
        }
    }

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata,
        tensors,
        data_offset,
    })
}

/// Byte length of a tensor, using quant block sizes where applicable.
pub(crate) fn tensor_byte_len(dtype: DType, shape: &[u64], name: &str) -> Result<u64, GgufError> {
    let elements = shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            GgufError::InvalidShape(format!(
                "tensor {name:?} has dimensions whose product overflows u64"
            ))
        })?;

    let block_bytes = dtype.block_bytes() as u64;
    let block_elements = dtype.element_count() as u64;
    if block_bytes == 0 || block_elements == 0 {
        // Unknown dtype: keep the descriptor but do not pretend to know its size.
        return Ok(0);
    }

    let blocks = elements.div_ceil(block_elements);
    blocks.checked_mul(block_bytes).ok_or(GgufError::Overflow)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(GgufError::Overflow)
}

/// Bounds-checked cursor over a byte slice. All multi-byte reads are little-endian.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn position(&self) -> u64 {
        self.pos as u64
    }

    fn ensure(&self, len: usize) -> Result<(), GgufError> {
        if len <= self.buf.len() {
            Ok(())
        } else {
            Err(GgufError::Truncated {
                offset: self.position(),
                needed: (len - self.buf.len()) as u64,
            })
        }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        self.ensure(len)?;
        let (head, tail) = self.buf.split_at(len);
        self.buf = tail;
        self.pos += len;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        self.ensure(1)?;
        let value = self.buf.get_u8();
        self.pos += 1;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        self.ensure(2)?;
        let value = self.buf.get_u16_le();
        self.pos += 2;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        self.ensure(4)?;
        let value = self.buf.get_u32_le();
        self.pos += 4;
        Ok(value)
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        self.ensure(8)?;
        let value = self.buf.get_u64_le();
        self.pos += 8;
        Ok(value)
    }

    fn string(&mut self, len_bytes: usize) -> Result<String, GgufError> {
        let len = if len_bytes == 4 {
            u64::from(self.u32()?)
        } else {
            self.u64()?
        };
        let len = usize::try_from(len).map_err(|_| GgufError::Truncated {
            offset: self.position(),
            needed: u64::MAX,
        })?;
        let bytes = self.take(len)?;
        let text = std::str::from_utf8(bytes)?;
        Ok(text.to_owned())
    }

    fn value(&mut self, version: GgufVersion, in_array: bool) -> Result<MetadataValue, GgufError> {
        let value_type = self.u32()?;
        self.typed_value(value_type, version, in_array)
    }

    fn typed_value(
        &mut self,
        value_type: u32,
        version: GgufVersion,
        in_array: bool,
    ) -> Result<MetadataValue, GgufError> {
        let value = match value_type {
            0 => MetadataValue::Uint8(self.u8()?),
            1 => MetadataValue::Int8(self.u8()? as i8),
            2 => MetadataValue::Uint16(self.u16()?),
            3 => MetadataValue::Int16(self.u16()? as i16),
            4 => MetadataValue::Uint32(self.u32()?),
            5 => MetadataValue::Int32(self.u32()? as i32),
            6 => MetadataValue::Float32(f32::from_bits(self.u32()?)),
            7 => MetadataValue::Bool(self.u8()? != 0),
            8 => {
                // Array strings use 64-bit lengths even in v1 files.
                let width = if in_array {
                    8
                } else {
                    version.string_len_bytes()
                };
                MetadataValue::String(self.string(width)?)
            }
            9 => {
                if in_array {
                    return Err(GgufError::NestedArray);
                }
                let element_type = self.u32()?;
                let len = self.u64()?;
                let min_bytes = min_value_bytes(element_type)
                    .ok_or(GgufError::UnsupportedValueType(element_type))?;
                let remaining = self.buf.len() as u64;
                if len > remaining / min_bytes {
                    return Err(GgufError::ArrayTooLong { len, remaining });
                }
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(self.typed_value(element_type, version, true)?);
                }
                MetadataValue::Array(values)
            }
            10 => MetadataValue::Uint64(self.u64()?),
            11 => MetadataValue::Int64(self.u64()? as i64),
            12 => MetadataValue::Float64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::UnsupportedValueType(other)),
        };
        Ok(value)
    }
}

/// Minimum encoded size of one value of the given type, used to bound array
/// allocations before reading untrusted lengths.
fn min_value_bytes(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        // String length prefix, even in v1 arrays.
        8 => Some(8),
        10..=12 => Some(8),
        _ => None,
    }
}
