//! GGUF format versions and tensor data types.

use serde::{Deserialize, Serialize};

/// GGUF file format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GgufVersion {
    /// Version 1: 32-bit counts and string lengths.
    V1,
    /// Version 2: 64-bit counts and string lengths.
    V2,
    /// Version 3: same layout as version 2.
    V3,
}

impl GgufVersion {
    /// Numeric version as stored in the file header.
    pub const fn number(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    pub(crate) const fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    /// Width of ordinary string length prefixes in this version, in bytes.
    pub(crate) const fn string_len_bytes(self) -> usize {
        match self {
            Self::V1 => 4,
            Self::V2 | Self::V3 => 8,
        }
    }
}

/// Tensor element type as encoded in a GGUF tensor descriptor.
///
/// Types this reader does not have a dedicated variant for are preserved as
/// [`DType::Unknown`] so headers can still be parsed round-trip.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit IEEE float (GGUF type 0).
    F32,
    /// 16-bit IEEE float (GGUF type 1).
    F16,
    /// 16-bit "brain" float (GGUF type 28).
    BF16,
    /// 8-bit quantization, 32 values per 34-byte block (GGUF type 6).
    Q8_0,
    /// 4-bit quantization, 32 values per 18-byte block (GGUF type 2).
    Q4_0,
    /// K-quantization, 256 values per 144-byte block (GGUF type 10).
    Q4_K,
    /// K-quantization, 256 values per 210-byte block (GGUF type 12).
    Q6_K,
    /// K-quantization, 256 values per 176-byte block (GGUF type 11).
    Q5_K,
    /// K-quantization, 256 values per 110-byte block (GGUF type 9).
    Q3_K,
    /// K-quantization, 256 values per 84-byte block (GGUF type 8).
    Q2_K,
    /// Any other GGUF type id.
    Unknown(u32),
}

impl DType {
    /// Map a raw GGUF type id to a dtype, preserving unknown ids.
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            6 => Self::Q8_0,
            8 => Self::Q2_K,
            9 => Self::Q3_K,
            10 => Self::Q4_K,
            11 => Self::Q5_K,
            12 => Self::Q6_K,
            28 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Raw GGUF type id.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 6,
            Self::Q2_K => 8,
            Self::Q3_K => 9,
            Self::Q4_K => 10,
            Self::Q5_K => 11,
            Self::Q6_K => 12,
            Self::BF16 => 28,
            Self::Unknown(other) => other,
        }
    }

    /// Size in bytes of one storage block (one element for unquantized types).
    ///
    /// Returns `0` for unknown types, whose block layout cannot be inferred.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Unknown(_) => 0,
        }
    }

    /// Number of logical elements represented by one storage block.
    ///
    /// Returns `0` for unknown types. Together with [`DType::block_bytes`] this
    /// allows computing tensor byte lengths as
    /// `ceil(elements / element_count) * block_bytes`.
    pub const fn element_count(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
            Self::Unknown(_) => 0,
        }
    }

    /// Whether this dtype uses fixed-size quant blocks.
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q8_0
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
        )
    }
}
