//! Allocation-free scalar kernels for the quant formats used by this reader.
//!
//! All kernels write into caller-provided buffers and never allocate. Input
//! buffers are expected to hold whole blocks; a partial trailing block is an
//! error rather than being silently ignored.

use crate::error::GgufError;
use half::f16;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_0_BLOCK_BYTES: usize = 18;
const BLOCK_ELEMENTS: usize = 32;

fn check_blocks(bytes: &[u8], block_bytes: usize, out_len: usize) -> Result<usize, GgufError> {
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes,
            actual: bytes.len(),
        });
    }
    let blocks = bytes.len() / block_bytes;
    let needed = blocks
        .checked_mul(BLOCK_ELEMENTS)
        .ok_or(GgufError::Overflow)?;
    if out_len < needed {
        return Err(GgufError::OutputTooSmall {
            needed,
            actual: out_len,
        });
    }
    Ok(blocks)
}

fn check_lhs(blocks: usize, lhs_len: usize) -> Result<(), GgufError> {
    let needed = blocks
        .checked_mul(BLOCK_ELEMENTS)
        .ok_or(GgufError::Overflow)?;
    if lhs_len != needed {
        return Err(GgufError::LengthMismatch {
            quant: needed,
            lhs: lhs_len,
        });
    }
    Ok(())
}

fn scale_of(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[0], block[1]]).to_f32()
}

/// Dequantize Q8_0 blocks (f16 scale followed by 32 signed bytes) into `out`.
///
/// Extra capacity in `out` is left untouched. The number of written values is
/// `bytes.len() / 34 * 32`.
pub fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = check_blocks(bytes, Q8_0_BLOCK_BYTES, out.len())?;
    let byte_blocks = bytes.as_chunks::<Q8_0_BLOCK_BYTES>().0;
    let out_blocks = out.as_chunks_mut::<BLOCK_ELEMENTS>().0;
    for (block, out_block) in byte_blocks.iter().zip(out_blocks.iter_mut()) {
        let scale = scale_of(block);
        for (&value, dst) in block[2..].iter().zip(out_block.iter_mut()) {
            *dst = f32::from(value as i8) * scale;
        }
    }
    debug_assert!(blocks == byte_blocks.len() && out_blocks.len() >= blocks);
    Ok(())
}

/// Dequantize Q4_0 blocks (f16 scale plus 16 packed nibbles) into `out`.
///
/// Each nibble is interpreted as a signed 4-bit value in `[-8, 7]`; the low
/// nibble of a byte holds the earlier element. Extra capacity in `out` is left
/// untouched.
pub fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = check_blocks(bytes, Q4_0_BLOCK_BYTES, out.len())?;
    let byte_blocks = bytes.as_chunks::<Q4_0_BLOCK_BYTES>().0;
    let out_blocks = out.as_chunks_mut::<BLOCK_ELEMENTS>().0;
    for (block, out_block) in byte_blocks.iter().zip(out_blocks.iter_mut()) {
        let scale = scale_of(block);
        for (&packed, pair) in block[2..].iter().zip(out_block.as_chunks_mut::<2>().0) {
            let low = (packed & 0x0f) as i8 - 8;
            let high = (packed >> 4) as i8 - 8;
            pair[0] = f32::from(low) * scale;
            pair[1] = f32::from(high) * scale;
        }
    }
    debug_assert!(blocks == byte_blocks.len() && out_blocks.len() >= blocks);
    Ok(())
}

/// Compute the dot product of Q8_0 data with `lhs`, dequantizing on the fly.
pub fn dot_q8_0(bytes: &[u8], lhs: &[f32]) -> Result<f32, GgufError> {
    if !bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: Q8_0_BLOCK_BYTES,
            actual: bytes.len(),
        });
    }
    let blocks = bytes.len() / Q8_0_BLOCK_BYTES;
    check_lhs(blocks, lhs.len())?;

    let mut acc = 0.0f32;
    for (block, lhs_block) in bytes
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(lhs.as_chunks::<BLOCK_ELEMENTS>().0)
    {
        let scale = scale_of(block);
        for (&value, &l) in block[2..].iter().zip(lhs_block) {
            acc = (f32::from(value as i8) * scale).mul_add(l, acc);
        }
    }
    Ok(acc)
}

/// Compute the dot product of Q4_0 data with `lhs`, dequantizing on the fly.
pub fn dot_q4_0(bytes: &[u8], lhs: &[f32]) -> Result<f32, GgufError> {
    if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: Q4_0_BLOCK_BYTES,
            actual: bytes.len(),
        });
    }
    let blocks = bytes.len() / Q4_0_BLOCK_BYTES;
    check_lhs(blocks, lhs.len())?;

    let mut acc = 0.0f32;
    for (block, lhs_block) in bytes
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(lhs.as_chunks::<BLOCK_ELEMENTS>().0)
    {
        let scale = scale_of(block);
        for (index, &packed) in block[2..].iter().enumerate() {
            let low = f32::from((packed & 0x0f) as i8 - 8) * scale;
            let high = f32::from((packed >> 4) as i8 - 8) * scale;
            acc = low.mul_add(lhs_block[2 * index], acc);
            acc = high.mul_add(lhs_block[2 * index + 1], acc);
        }
    }
    Ok(acc)
}

/// Decode little-endian IEEE half-precision values into `out`.
pub fn dequantize_f16(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: 2,
            actual: bytes.len(),
        });
    }
    let needed = bytes.len() / 2;
    if out.len() < needed {
        return Err(GgufError::OutputTooSmall {
            needed,
            actual: out.len(),
        });
    }
    for (pair, dst) in bytes.as_chunks::<2>().0.iter().zip(out) {
        *dst = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
    }
    Ok(())
}

/// Dequantize a single Q8_0 embedding row without touching the rest of a
/// quantized weight matrix.
///
/// `dim` must be a nonzero multiple of 32 and the tensor byte length must be
/// an exact number of rows. Only the `index`-th row is read and written.
pub fn embedding_lookup_q8_0(
    bytes: &[u8],
    index: usize,
    dim: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if dim == 0 || !dim.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(GgufError::InvalidQuantArgument(format!(
            "embedding dim {dim} must be a nonzero multiple of {BLOCK_ELEMENTS}"
        )));
    }
    let row_bytes = (dim / BLOCK_ELEMENTS)
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(GgufError::Overflow)?;
    if !bytes.len().is_multiple_of(row_bytes) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: row_bytes,
            actual: bytes.len(),
        });
    }
    let rows = bytes.len() / row_bytes;
    if index >= rows {
        return Err(GgufError::RowOutOfRange { index, rows });
    }
    if out.len() < dim {
        return Err(GgufError::OutputTooSmall {
            needed: dim,
            actual: out.len(),
        });
    }
    let start = index * row_bytes;
    dequantize_q8_0(&bytes[start..start + row_bytes], &mut out[..dim])
}
