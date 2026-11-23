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
    /// 16-bit "brain" float (GGUF type 30).
    BF16,
    /// 8-bit quantization, 32 values per 34-byte block (GGUF type 8).
    Q8_0,
    /// 4-bit quantization, 32 values per 18-byte block (GGUF type 2).
    Q4_0,
    /// 5-bit quantization, 32 values per 22-byte block (GGUF type 6).
    Q5_0,
    /// K-quantization, 256 values per 144-byte block (GGUF type 12).
    Q4_K,
    /// K-quantization, 256 values per 210-byte block (GGUF type 14).
    Q6_K,
    /// K-quantization, 256 values per 176-byte block (GGUF type 13).
    Q5_K,
    /// K-quantization, 256 values per 110-byte block (GGUF type 11).
    Q3_K,
    /// K-quantization, 256 values per 84-byte block (GGUF type 10).
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
            6 => Self::Q5_0,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Raw GGUF type id.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q5_0 => 6,
            Self::Q8_0 => 8,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 30,
            Self::Unknown(other) => other,
        }
    }

    /// `general.file_type` value used by ggml for this quantization.
    ///
    /// This is a different enum from the tensor dtype ids above; for example
    /// `Q8_0` tensors use dtype 8 but file type 7. Unknown dtypes fall back to
    /// their raw id so requantization never invents a value.
    pub const fn file_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q5_0 => 8,
            Self::Q8_0 => 7,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 24,
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
            Self::Q5_0 => 22,
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
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
            Self::Unknown(_) => 0,
        }
    }

    /// Whether this dtype uses fixed-size quant blocks.
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q5_0
                | Self::Q8_0
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_numbers_and_string_widths_round_trip() {
        for (version, number, string_bytes) in [
            (GgufVersion::V1, 1, 4),
            (GgufVersion::V2, 2, 8),
            (GgufVersion::V3, 3, 8),
        ] {
            assert_eq!(version.number(), number);
            assert_eq!(GgufVersion::from_u32(number), Some(version));
            assert_eq!(version.string_len_bytes(), string_bytes);
        }
        assert_eq!(GgufVersion::from_u32(0), None);
        assert_eq!(GgufVersion::from_u32(4), None);
    }

    #[test]
    fn every_dtype_maps_to_its_canonical_ids_and_storage_layout() {
        // (dtype, tensor id, ggml file type, block bytes, elements/block)
        let cases = [
            (DType::F32, 0, 0, 4, 1),
            (DType::F16, 1, 1, 2, 1),
            (DType::Q4_0, 2, 2, 18, 32),
            (DType::Q5_0, 6, 8, 22, 32),
            (DType::Q8_0, 8, 7, 34, 32),
            (DType::Q2_K, 10, 10, 84, 256),
            (DType::Q3_K, 11, 11, 110, 256),
            (DType::Q4_K, 12, 12, 144, 256),
            (DType::Q5_K, 13, 13, 176, 256),
            (DType::Q6_K, 14, 14, 210, 256),
            (DType::BF16, 30, 24, 2, 1),
            (DType::Unknown(99), 99, 99, 0, 0),
        ];
        for &(dtype, id, file_type, block_bytes, elements) in &cases {
            assert_eq!(DType::from_u32(id), dtype);
            assert_eq!(dtype.as_u32(), id);
            assert_eq!(dtype.file_type(), file_type);
            assert_eq!(dtype.block_bytes(), block_bytes);
            assert_eq!(dtype.element_count(), elements);
        }
        assert!(!DType::F32.is_quantized());
        assert!(!DType::BF16.is_quantized());
        assert!(!DType::Unknown(1).is_quantized());
        for dtype in [
            DType::Q4_0,
            DType::Q5_0,
            DType::Q8_0,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
        ] {
            assert!(dtype.is_quantized(), "{dtype:?} must be quantized");
        }
    }
}
