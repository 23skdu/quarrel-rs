//! Allocation-free dequantization kernels and deterministic encoders for the
//! quant formats used by this crate.
//!
//! Decoder kernels write into caller-provided buffers and never allocate. The
//! encoder helpers allocate only their returned byte buffer and use the same
//! canonical ggml block layouts as the decoders. Input buffers are expected to
//! hold whole blocks; a partial trailing block is an error rather than being
//! silently ignored.

use crate::DType;
use crate::error::GgufError;
use half::f16;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const BLOCK_ELEMENTS: usize = 32;
const K_BLOCK_ELEMENTS: usize = 256;

fn check_blocks_with(
    bytes: &[u8],
    block_bytes: usize,
    block_elements: usize,
    out_len: usize,
) -> Result<usize, GgufError> {
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes,
            actual: bytes.len(),
        });
    }
    let blocks = bytes.len() / block_bytes;
    let needed = blocks
        .checked_mul(block_elements)
        .ok_or(GgufError::Overflow)?;
    if out_len < needed {
        return Err(GgufError::OutputTooSmall {
            needed,
            actual: out_len,
        });
    }
    Ok(blocks)
}

fn check_blocks(bytes: &[u8], block_bytes: usize, out_len: usize) -> Result<usize, GgufError> {
    check_blocks_with(bytes, block_bytes, BLOCK_ELEMENTS, out_len)
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

/// Dequantize Q4_K blocks (144 bytes each, 256 values per block) into `out`.
///
/// This uses the same canonical layout as [`crate::matvec::matvec_q4_k`].
pub fn dequantize_q4_k(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = check_blocks_with(bytes, Q4_K_BLOCK_BYTES, K_BLOCK_ELEMENTS, out.len())?;
    for (block, chunk) in bytes
        .as_chunks::<Q4_K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<K_BLOCK_ELEMENTS>().0)
    {
        crate::matvec::decode_q4_k(block, 0, chunk)?;
    }
    debug_assert_eq!(blocks, bytes.len() / Q4_K_BLOCK_BYTES);
    Ok(())
}

/// Dequantize Q6_K blocks (210 bytes each, 256 values per block) into `out`.
///
/// This uses the same canonical layout as [`crate::matvec::matvec_q6_k`].
pub fn dequantize_q6_k(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = check_blocks_with(bytes, Q6_K_BLOCK_BYTES, K_BLOCK_ELEMENTS, out.len())?;
    for (block, chunk) in bytes
        .as_chunks::<Q6_K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<K_BLOCK_ELEMENTS>().0)
    {
        crate::matvec::decode_q6_k(block, 0, chunk)?;
    }
    debug_assert_eq!(blocks, bytes.len() / Q6_K_BLOCK_BYTES);
    Ok(())
}

fn check_quant_input(values: &[f32], block_elements: usize) -> Result<(), GgufError> {
    if values.is_empty() {
        return Err(GgufError::InvalidQuantArgument(
            "cannot quantize an empty value slice".to_owned(),
        ));
    }
    if !values.len().is_multiple_of(block_elements) {
        return Err(GgufError::InvalidQuantArgument(format!(
            "input length {} must be a multiple of {block_elements}",
            values.len()
        )));
    }
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

fn reserve_bytes(len: usize) -> Result<Vec<u8>, GgufError> {
    let mut out = Vec::new();
    out.try_reserve(len).map_err(|_| GgufError::Overflow)?;
    Ok(out)
}

fn finite_f16(value: f32, dtype: DType, index: usize, offending: f32) -> Result<f16, GgufError> {
    let encoded = f16::from_f32(value);
    if !encoded.is_finite() || encoded.to_f32() <= 0.0 {
        return Err(GgufError::QuantizationRange {
            dtype,
            index,
            value: offending,
        });
    }
    Ok(encoded)
}

/// Encode `f32` values as canonical Q8_0 blocks (34 bytes per 32 values).
pub fn quantize_f32_to_q8_0(values: &[f32]) -> Result<Vec<u8>, GgufError> {
    check_quant_input(values, BLOCK_ELEMENTS)?;
    let mut out = reserve_bytes(
        values
            .len()
            .checked_div(BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES))
            .ok_or(GgufError::Overflow)?,
    )?;

    for (block_index, block) in values.as_chunks::<BLOCK_ELEMENTS>().0.iter().enumerate() {
        let amax = block
            .iter()
            .fold(0.0f32, |acc, &value| acc.max(value.abs()));
        let scale = finite_f16(
            if amax == 0.0 { 1.0 } else { amax / 127.0 },
            DType::Q8_0,
            block_index * BLOCK_ELEMENTS,
            amax,
        )?;
        let scale = scale.to_f32();
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for &value in block {
            let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
            out.push(quantized as u8);
        }
    }
    Ok(out)
}

/// Encode `f32` values as canonical Q4_K super-blocks (144 bytes per 256 values).
///
/// Each 32-value subgroup gets its own six-bit scale and minimum, with the
/// shared `d`/`dmin` f16 scales stored in each block header.
pub fn quantize_f32_to_q4_k(values: &[f32]) -> Result<Vec<u8>, GgufError> {
    check_quant_input(values, K_BLOCK_ELEMENTS)?;
    let mut out = reserve_bytes(
        values
            .len()
            .checked_div(K_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(Q4_K_BLOCK_BYTES))
            .ok_or(GgufError::Overflow)?,
    )?;

    for (block_index, block) in values.as_chunks::<K_BLOCK_ELEMENTS>().0.iter().enumerate() {
        let base = block_index * K_BLOCK_ELEMENTS;
        let mut group_min = [0.0f32; 8];
        let mut group_max = [0.0f32; 8];
        for (group, (min, max)) in group_min.iter_mut().zip(group_max.iter_mut()).enumerate() {
            let start = group * BLOCK_ELEMENTS;
            let slice = &block[start..start + BLOCK_ELEMENTS];
            let lo = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            *min = lo;
            *max = hi;
        }
        let global_min = group_min.iter().copied().fold(f32::INFINITY, f32::min);
        let dmin_f16 = if global_min < 0.0 {
            finite_f16(-global_min / 63.0, DType::Q4_K, base, global_min)?
        } else {
            f16::from_f32(0.0)
        };
        let dmin = dmin_f16.to_f32();

        let mut min_params = [0u8; 8];
        let mut d_candidate = 0.0f32;
        for (group, (&lo, &hi)) in group_min.iter().zip(group_max.iter()).enumerate() {
            min_params[group] = if dmin > 0.0 {
                (-lo / dmin).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            let upper = hi + dmin * f32::from(min_params[group]);
            d_candidate = d_candidate.max(upper / (63.0 * 15.0));
        }
        let d_f16 = finite_f16(
            if d_candidate == 0.0 { 1.0 } else { d_candidate },
            DType::Q4_K,
            base,
            d_candidate,
        )?;
        let d = d_f16.to_f32();

        let mut scale_params = [0u8; 8];
        let mut qs = [0u8; 128];
        for (group, (&_lo, &hi)) in group_min.iter().zip(group_max.iter()).enumerate() {
            let offset = dmin * f32::from(min_params[group]);
            let wanted = (hi + offset) / (15.0 * d);
            scale_params[group] = if wanted <= 0.0 {
                0
            } else {
                wanted.ceil().clamp(1.0, 63.0) as u8
            };
            let step = d * f32::from(scale_params[group]);
            for lane in 0..BLOCK_ELEMENTS {
                let n = group * BLOCK_ELEMENTS + lane;
                let quantized = if step == 0.0 {
                    0u8
                } else {
                    ((block[n] + offset) / step).round().clamp(0.0, 15.0) as u8
                };
                let byte = qs
                    .get_mut(32 * (n / 64) + n % 32)
                    .ok_or(GgufError::Overflow)?;
                if n % 64 < 32 {
                    *byte |= quantized;
                } else {
                    *byte |= quantized << 4;
                }
            }
        }

        let mut packed = [0u8; 12];
        for group in 0..4 {
            packed[group] = scale_params[group] & 0x3f;
            packed[group + 4] = min_params[group] & 0x3f;
        }
        for group in 4..8 {
            packed[group + 4] = (scale_params[group] & 0x0f) | ((min_params[group] & 0x0f) << 4);
            packed[group - 4] |= ((scale_params[group] >> 4) & 0x03) << 6;
            packed[group] |= ((min_params[group] >> 4) & 0x03) << 6;
        }

        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());
        out.extend_from_slice(&packed);
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Encode `f32` values as canonical Q6_K super-blocks (210 bytes per 256 values).
///
/// Each 16-value subgroup gets a signed scale; the shared `d` f16 scale is
/// stored in each block header.
pub fn quantize_f32_to_q6_k(values: &[f32]) -> Result<Vec<u8>, GgufError> {
    check_quant_input(values, K_BLOCK_ELEMENTS)?;
    let mut out = reserve_bytes(
        values
            .len()
            .checked_div(K_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(Q6_K_BLOCK_BYTES))
            .ok_or(GgufError::Overflow)?,
    )?;

    for (block_index, block) in values.as_chunks::<K_BLOCK_ELEMENTS>().0.iter().enumerate() {
        let base = block_index * K_BLOCK_ELEMENTS;
        let amax = block
            .iter()
            .fold(0.0f32, |acc, &value| acc.max(value.abs()));
        let d_f16 = finite_f16(
            if amax == 0.0 {
                1.0
            } else {
                amax / (30.0 * 127.0)
            },
            DType::Q6_K,
            base,
            amax,
        )?;
        let d = d_f16.to_f32();

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for (group, scales_entry) in scales.iter_mut().enumerate() {
            let start = group * 16;
            let slice = &block[start..start + 16];
            let group_amax = slice
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs()));
            let wanted = group_amax / (31.0 * d);
            if wanted > 127.0 {
                return Err(GgufError::QuantizationRange {
                    dtype: DType::Q6_K,
                    index: base + start,
                    value: group_amax,
                });
            }
            let scale = wanted.ceil().max(1.0) as i8;
            *scales_entry = scale;
            let step = d * f32::from(scale);

            for (lane, &value) in slice.iter().enumerate() {
                let n = start + lane;
                let biased = (value / step).round().clamp(-31.0, 31.0) as i16 + 32;
                let quantized = biased as u8;
                let low = quantized & 0x0f;
                let high = (quantized >> 4) & 0x03;
                let half = n / 128;
                let plane = (n % 128) / 32;
                let l = n % 32;
                match plane {
                    0 => {
                        ql[64 * half + l] |= low;
                        qh[32 * half + l] |= high;
                    }
                    1 => {
                        ql[64 * half + 32 + l] |= low;
                        qh[32 * half + l] |= high << 2;
                    }
                    2 => {
                        ql[64 * half + l] |= low << 4;
                        qh[32 * half + l] |= high << 4;
                    }
                    _ => {
                        ql[64 * half + 32 + l] |= low << 4;
                        qh[32 * half + l] |= high << 6;
                    }
                }
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(scales.iter().flat_map(|scale| scale.to_le_bytes()));
        out.extend_from_slice(&d_f16.to_le_bytes());
    }
    Ok(out)
}
