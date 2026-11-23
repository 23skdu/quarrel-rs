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
const Q5_0_BLOCK_BYTES: usize = 22;
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

/// Dequantize Q5_0 blocks (f16 scale, 32 high bits and 16 packed nibbles).
///
/// Each block holds 32 values. Extra capacity in `out` is left untouched.
pub fn dequantize_q5_0(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = check_blocks_with(bytes, Q5_0_BLOCK_BYTES, BLOCK_ELEMENTS, out.len())?;
    for (block, chunk) in bytes
        .as_chunks::<Q5_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_ELEMENTS>().0)
    {
        crate::matvec::decode_q5_0(block, 0, chunk)?;
    }
    debug_assert_eq!(blocks, bytes.len() / Q5_0_BLOCK_BYTES);
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

/// Decode little-endian IEEE binary32 values into `out`.
pub fn dequantize_f32(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: 4,
            actual: bytes.len(),
        });
    }
    if out.len() < bytes.len() / 4 {
        return Err(GgufError::OutputTooSmall {
            needed: bytes.len() / 4,
            actual: out.len(),
        });
    }
    for (quad, dst) in bytes.as_chunks::<4>().0.iter().zip(out.iter_mut()) {
        *dst = f32::from_le_bytes(*quad);
    }
    Ok(())
}

/// Decode little-endian bfloat16 values into `out`.
pub fn dequantize_bf16(bytes: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes: 2,
            actual: bytes.len(),
        });
    }
    if out.len() < bytes.len() / 2 {
        return Err(GgufError::OutputTooSmall {
            needed: bytes.len() / 2,
            actual: out.len(),
        });
    }
    for (pair, dst) in bytes.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
        *dst = half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32();
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

/// Dequantize one logical row of any tensor supported by the runtime kernels.
///
/// GGUF quant blocks run over the flattened tensor rather than stopping at row
/// boundaries, so a 640-wide Gemma 3 embedding row can begin and end inside
/// different 256-element Q4_K blocks. This helper decodes only the blocks that
/// overlap the requested row and copies their values into `out`; the rest of
/// the tensor is never materialized.
pub fn embedding_lookup(
    bytes: &[u8],
    dtype: DType,
    index: usize,
    dim: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if dim == 0 {
        return Err(GgufError::InvalidQuantArgument(
            "embedding row width must be non-zero".to_owned(),
        ));
    }
    let block_bytes = dtype.block_bytes();
    let block_elements = dtype.element_count();
    if block_bytes == 0 || block_elements == 0 {
        return Err(GgufError::UnsupportedDType(dtype));
    }
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(GgufError::InvalidBlockLength {
            block_bytes,
            actual: bytes.len(),
        });
    }
    let blocks = bytes.len() / block_bytes;
    let total_elements = blocks
        .checked_mul(block_elements)
        .ok_or(GgufError::Overflow)?;
    if !total_elements.is_multiple_of(dim) {
        return Err(GgufError::InvalidQuantArgument(format!(
            "tensor has {total_elements} elements, which is not a multiple of row width {dim}"
        )));
    }
    let rows = total_elements / dim;
    if index >= rows {
        return Err(GgufError::RowOutOfRange { index, rows });
    }
    if out.len() < dim {
        return Err(GgufError::OutputTooSmall {
            needed: dim,
            actual: out.len(),
        });
    }

    let first_element = index.checked_mul(dim).ok_or(GgufError::Overflow)?;
    let end_element = first_element.checked_add(dim).ok_or(GgufError::Overflow)?;
    let first_block = first_element / block_elements;
    let last_block = (end_element - 1) / block_elements;
    let mut scratch = [0.0f32; 256];
    for block_index in first_block..=last_block {
        let byte_start = block_index
            .checked_mul(block_bytes)
            .ok_or(GgufError::Overflow)?;
        let block = &bytes[byte_start..byte_start + block_bytes];
        let block_base = block_index
            .checked_mul(block_elements)
            .ok_or(GgufError::Overflow)?;
        crate::matvec::decode_block(dtype, block, block_base, &mut scratch)?;
        let lo = first_element.max(block_base);
        let hi = end_element.min(block_base + block_elements);
        out[lo - first_element..hi - first_element]
            .copy_from_slice(&scratch[lo - block_base..hi - block_base]);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn q8_block(scale: f16, values: &[i8; BLOCK_ELEMENTS]) -> Vec<u8> {
        let mut out = scale.to_le_bytes().to_vec();
        out.extend(values.iter().map(|value| *value as u8));
        out
    }

    fn q4_block(scale: f16, values: &[i8; BLOCK_ELEMENTS]) -> Vec<u8> {
        let mut out = scale.to_le_bytes().to_vec();
        for pair in values.as_chunks::<2>().0 {
            let low = (pair[0] + 8) as u8 & 0x0f;
            let high = (pair[1] + 8) as u8 & 0x0f;
            out.push(low | (high << 4));
        }
        out
    }

    fn q5_block(scale: f16, values: &[i16; BLOCK_ELEMENTS]) -> Vec<u8> {
        let mut qh = 0u32;
        let mut qs = [0u8; BLOCK_ELEMENTS / 2];
        for (lane, &value) in values.iter().enumerate() {
            let biased = (value + 16) as u8;
            assert!(biased < 32);
            if lane < BLOCK_ELEMENTS / 2 {
                qs[lane] |= biased & 0x0f;
                qh |= u32::from((biased >> 4) & 1) << lane;
            } else {
                let index = lane - BLOCK_ELEMENTS / 2;
                qs[index] |= (biased & 0x0f) << 4;
                qh |= u32::from((biased >> 4) & 1) << (index + BLOCK_ELEMENTS / 2);
            }
        }
        let mut out = scale.to_le_bytes().to_vec();
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
        out
    }

    fn q4_k_block(d: f16, dmin: f16, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut out = d.to_le_bytes().to_vec();
        out.extend_from_slice(&dmin.to_le_bytes());
        out.extend_from_slice(scales);
        out.extend_from_slice(qs);
        out
    }

    fn q6_k_block(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], d: f16) -> Vec<u8> {
        let mut out = ql.to_vec();
        out.extend_from_slice(qh);
        out.extend(scales.iter().flat_map(|value| value.to_le_bytes()));
        out.extend_from_slice(&d.to_le_bytes());
        out
    }

    #[test]
    fn unquantized_dequantizers_decode_and_reject_bad_inputs() {
        let f32_values = [1.5f32, -2.25, 0.0];
        let f32_bytes: Vec<u8> = f32_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = [0.0f32; 4];
        dequantize_f32(&f32_bytes, &mut out).unwrap();
        assert_eq!(&out[..3], &f32_values);
        assert!(dequantize_f32(&f32_bytes[..3], &mut out).is_err());
        assert!(dequantize_f32(&f32_bytes, &mut out[..2]).is_err());

        let f16_bytes: Vec<u8> = f32_values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        dequantize_f16(&f16_bytes, &mut out).unwrap();
        assert_eq!(&out[..3], &f32_values);
        assert!(dequantize_f16(&f16_bytes[..1], &mut out).is_err());
        assert!(dequantize_f16(&f16_bytes, &mut out[..2]).is_err());

        let bf16_bytes: Vec<u8> = f32_values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        dequantize_bf16(&bf16_bytes, &mut out).unwrap();
        assert_eq!(&out[..3], &f32_values);
        assert!(dequantize_bf16(&bf16_bytes[..1], &mut out).is_err());
        assert!(dequantize_bf16(&bf16_bytes, &mut out[..2]).is_err());
    }

    #[test]
    fn simple_quantizers_decode_all_block_planes() {
        let mut q8_values = [-128i8; BLOCK_ELEMENTS];
        q8_values[0] = -1;
        q8_values[1] = 0;
        q8_values[2] = 1;
        q8_values[31] = 127;
        let q8 = q8_block(f16::from_f32(0.5), &q8_values);
        let mut decoded = [0.0f32; BLOCK_ELEMENTS];
        dequantize_q8_0(&q8, &mut decoded).unwrap();
        assert_eq!(decoded[0], -0.5);
        assert_eq!(decoded[31], 63.5);
        let lhs = [1.0f32; BLOCK_ELEMENTS];
        let expected: f32 = decoded.iter().sum();
        assert_eq!(dot_q8_0(&q8, &lhs).unwrap(), expected);
        assert!(dequantize_q8_0(&q8[..33], &mut decoded).is_err());
        assert!(dequantize_q8_0(&q8, &mut decoded[..8]).is_err());
        assert!(dot_q8_0(&q8[..33], &lhs).is_err());
        assert!(dot_q8_0(&q8, &lhs[..8]).is_err());

        let mut q4_values = [-8i8; BLOCK_ELEMENTS];
        for (index, value) in q4_values.iter_mut().enumerate() {
            *value = (index as i8 % 16) - 8;
        }
        let q4 = q4_block(f16::from_f32(2.0), &q4_values);
        dequantize_q4_0(&q4, &mut decoded).unwrap();
        for (index, &value) in q4_values.iter().enumerate() {
            assert_eq!(decoded[index], f32::from(value) * 2.0);
        }
        let expected: f32 = decoded.iter().sum();
        assert_eq!(dot_q4_0(&q4, &lhs).unwrap(), expected);
        assert!(dequantize_q4_0(&q4[..17], &mut decoded).is_err());
        assert!(dequantize_q4_0(&q4, &mut decoded[..8]).is_err());
        assert!(dot_q4_0(&q4[..17], &lhs).is_err());
        assert!(dot_q4_0(&q4, &lhs[..8]).is_err());

        let mut q5_values = [-16i16; BLOCK_ELEMENTS];
        for (index, value) in q5_values.iter_mut().enumerate() {
            *value = (index as i16 % 32) - 16;
        }
        let q5 = q5_block(f16::from_f32(0.25), &q5_values);
        dequantize_q5_0(&q5, &mut decoded).unwrap();
        for (index, &value) in q5_values.iter().enumerate() {
            assert_eq!(decoded[index], f32::from(value) * 0.25);
        }
        assert!(dequantize_q5_0(&q5[..21], &mut decoded).is_err());
        assert!(dequantize_q5_0(&q5, &mut decoded[..8]).is_err());
    }

    #[test]
    fn k_quant_dequantizers_decode_all_groups() {
        let mut scales = [0x23u8; 16];
        scales[1] = 0x41;
        scales[8] = 0x9c;
        let qs = [0xe4u8; 64];
        let q2 = {
            let mut out = scales.to_vec();
            out.extend_from_slice(&qs);
            out.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
            out.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
            out
        };
        let mut decoded = [0.0f32; K_BLOCK_ELEMENTS];
        let mut out = [0.0f32; 1];
        let input = [1.0f32; K_BLOCK_ELEMENTS];
        crate::matvec::matvec_q2_k(&q2, &input, &mut out).unwrap();
        assert!(out[0].is_finite());

        let hmask = [0xa5u8; 32];
        let q3_scales = [0x81u8; 12];
        let q3 = {
            let mut out = hmask.to_vec();
            out.extend_from_slice(&qs);
            out.extend_from_slice(&q3_scales);
            out.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
            out
        };
        // Q3_K has no public dequantizer; the matvec kernel shares its decoder.
        crate::matvec::matvec_q3_k(&q3, &input, &mut out).unwrap();
        assert!(out[0].is_finite());

        let mut k_scales = [0x26u8; 12];
        k_scales[0] = 2;
        k_scales[4] = 1;
        k_scales[8] = 0x15;
        let k_qs = [0x7fu8; 128];
        let q4k = q4_k_block(f16::from_f32(0.25), f16::from_f32(0.5), &k_scales, &k_qs);
        dequantize_q4_k(&q4k, &mut decoded).unwrap();
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(decoded.iter().any(|&value| value != 0.0));
        let bad_q4k = q4_k_block(f16::INFINITY, f16::from_f32(0.5), &k_scales, &k_qs);
        assert!(dequantize_q4_k(&bad_q4k, &mut decoded).is_err());
        assert!(dequantize_q4_k(&bad_q4k[..143], &mut decoded).is_err());
        assert!(dequantize_q4_k(&q4k, &mut decoded[..8]).is_err());

        let qh = [0x5au8; 32];
        let q5k = {
            let mut out = q4_k_block(f16::from_f32(0.25), f16::from_f32(0.5), &k_scales, &k_qs);
            // Rewrite the tail as Q5_K's 32 high-bit bytes plus the low nibbles.
            out.truncate(16);
            out.extend_from_slice(&qh);
            out.extend_from_slice(&k_qs);
            out
        };
        crate::matvec::matvec_q5_k(&q5k, &input, &mut out).unwrap();
        assert!(out[0].is_finite());

        let mut ql = [0x21u8; 128];
        ql[64] = 0x43;
        let mut qh6 = [0xedu8; 64];
        qh6[32] = 0x08;
        let mut k6_scales = [2i8; 16];
        k6_scales[3] = -3;
        k6_scales[11] = 4;
        let q6k = q6_k_block(&ql, &qh6, &k6_scales, f16::from_f32(0.25));
        dequantize_q6_k(&q6k, &mut decoded).unwrap();
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(decoded.iter().any(|&value| value != 0.0));
        let bad_q6k = q6_k_block(&ql, &qh6, &k6_scales, f16::NEG_INFINITY);
        assert!(dequantize_q6_k(&bad_q6k, &mut decoded).is_err());
        assert!(dequantize_q6_k(&q6k[..209], &mut decoded).is_err());
        assert!(dequantize_q6_k(&q6k, &mut decoded[..8]).is_err());
    }

    #[test]
    fn embedding_lookup_handles_every_dtype_and_straddling_blocks() {
        let values: Vec<f32> = (0..(2 * K_BLOCK_ELEMENTS))
            .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
            .collect();
        let input = [1.0f32; K_BLOCK_ELEMENTS];
        let mut out = [0.0f32; K_BLOCK_ELEMENTS];

        let q4k = quantize_f32_to_q4_k(&values).unwrap();
        embedding_lookup(&q4k, DType::Q4_K, 1, K_BLOCK_ELEMENTS, &mut out).unwrap();
        let mut all = [0.0f32; 2 * K_BLOCK_ELEMENTS];
        dequantize_q4_k(&q4k, &mut all).unwrap();
        assert_eq!(out, all[K_BLOCK_ELEMENTS..]);

        let q6k = quantize_f32_to_q6_k(&values).unwrap();
        embedding_lookup(&q6k, DType::Q6_K, 0, K_BLOCK_ELEMENTS, &mut out).unwrap();
        dequantize_q6_k(&q6k, &mut all).unwrap();
        assert_eq!(out, all[..K_BLOCK_ELEMENTS]);

        // A 640-wide Gemma row straddles Q4_K blocks; the helper must still
        // decode only the overlapping blocks and fill the row in order.
        let wide_values: Vec<f32> = (0..1280)
            .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
            .collect();
        let wide_q4k = quantize_f32_to_q4_k(&wide_values).unwrap();
        let mut wide_out = [0.0f32; 640];
        embedding_lookup(&wide_q4k, DType::Q4_K, 1, 640, &mut wide_out).unwrap();
        let mut wide_all = [0.0f32; 1280];
        dequantize_q4_k(&wide_q4k, &mut wide_all).unwrap();
        assert_eq!(wide_out, wide_all[640..]);

        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        embedding_lookup(&f32_bytes, DType::F32, 1, K_BLOCK_ELEMENTS, &mut out).unwrap();
        assert_eq!(out[0], values[K_BLOCK_ELEMENTS]);

        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        embedding_lookup(&f16_bytes, DType::F16, 0, K_BLOCK_ELEMENTS, &mut out).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));

        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        embedding_lookup(&bf16_bytes, DType::BF16, 1, K_BLOCK_ELEMENTS, &mut out).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));

        let mut small = [0.0f32; BLOCK_ELEMENTS];
        let q8 = quantize_f32_to_q8_0(&values).unwrap();
        embedding_lookup(&q8, DType::Q8_0, 0, BLOCK_ELEMENTS, &mut small).unwrap();
        assert!(small.iter().all(|value| value.is_finite()));
        let q4 = q4_block(f16::from_f32(0.25), &[1i8; BLOCK_ELEMENTS]);
        embedding_lookup(&q4, DType::Q4_0, 0, BLOCK_ELEMENTS, &mut small).unwrap();
        assert_eq!(small[0], 0.25);
        let q5 = q5_block(f16::from_f32(0.25), &[1i16; BLOCK_ELEMENTS]);
        embedding_lookup(&q5, DType::Q5_0, 0, BLOCK_ELEMENTS, &mut small).unwrap();
        assert_eq!(small[0], 0.25);

        let mut q2 = vec![0x23u8; 16];
        q2.extend_from_slice(&[0xe4u8; 64]);
        q2.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        q2.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
        embedding_lookup(&q2, DType::Q2_K, 0, K_BLOCK_ELEMENTS, &mut out).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));

        let q5k =
            crate::matvec::matvec_q5_k as fn(&[u8], &[f32], &mut [f32]) -> Result<(), GgufError>;
        assert!(q5k(&[0u8; 176], &input, &mut [0.0; 1]).is_ok());

        assert!(embedding_lookup(&q4k, DType::Q4_K, 0, 0, &mut out).is_err());
        assert!(embedding_lookup(&q4k, DType::Unknown(99), 0, 32, &mut out).is_err());
        assert!(embedding_lookup(&q4k[..143], DType::Q4_K, 0, 32, &mut out).is_err());
        assert!(embedding_lookup(&q4k, DType::Q4_K, 0, 100, &mut out).is_err());
        assert!(embedding_lookup(&q4k, DType::Q4_K, 2, K_BLOCK_ELEMENTS, &mut out).is_err());
        assert!(embedding_lookup(&q4k, DType::Q4_K, 0, K_BLOCK_ELEMENTS, &mut out[..8]).is_err());

        let q8_row = quantize_f32_to_q8_0(&[0.5f32; 64]).unwrap();
        let mut row = [0.0f32; 32];
        embedding_lookup_q8_0(&q8_row, 1, 32, &mut row).unwrap();
        assert!(row.iter().all(|value| *value > 0.0));
        assert!(embedding_lookup_q8_0(&q8_row, 0, 0, &mut row).is_err());
        assert!(embedding_lookup_q8_0(&q8_row, 0, 31, &mut row).is_err());
        assert!(embedding_lookup_q8_0(&q8_row[..67], 0, 32, &mut row).is_err());
        assert!(embedding_lookup_q8_0(&q8_row, 2, 32, &mut row).is_err());
        assert!(embedding_lookup_q8_0(&q8_row, 0, 32, &mut row[..8]).is_err());
    }

    #[test]
    fn quantizers_cover_all_layouts_and_error_paths() {
        let q8_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let q8 = quantize_f32_to_q8_0(&q8_values).unwrap();
        assert_eq!(q8.len(), 2 * Q8_0_BLOCK_BYTES);
        let mut decoded = vec![0.0f32; q8_values.len()];
        dequantize_q8_0(&q8, &mut decoded).unwrap();
        for (actual, &expected) in decoded.iter().zip(&q8_values) {
            assert!((actual - expected).abs() < 0.05, "{actual} vs {expected}");
        }

        let mixed: Vec<f32> = (0..K_BLOCK_ELEMENTS)
            .map(|index| ((index % 29) as f32 - 14.0) / 16.0)
            .collect();
        for values in [
            mixed.clone(),
            mixed.iter().map(|value| value.abs()).collect(),
            vec![0.0f32; K_BLOCK_ELEMENTS],
        ] {
            let q4k = quantize_f32_to_q4_k(&values).unwrap();
            let mut out = vec![0.0f32; values.len()];
            dequantize_q4_k(&q4k, &mut out).unwrap();
            assert!(out.iter().all(|value| value.is_finite()));
            let q6k = quantize_f32_to_q6_k(&values).unwrap();
            dequantize_q6_k(&q6k, &mut out).unwrap();
            assert!(out.iter().all(|value| value.is_finite()));
        }

        for values in [
            Vec::new(),
            vec![1.0f32; 31],
            vec![f32::NAN; 32],
            vec![f32::INFINITY; 32],
        ] {
            assert!(quantize_f32_to_q8_0(&values).is_err());
        }
        for values in [
            vec![1.0f32; K_BLOCK_ELEMENTS - 1],
            vec![f32::NAN; K_BLOCK_ELEMENTS],
        ] {
            assert!(quantize_f32_to_q4_k(&values).is_err());
            assert!(quantize_f32_to_q6_k(&values).is_err());
        }
        assert!(quantize_f32_to_q4_k(&[f32::MAX; K_BLOCK_ELEMENTS]).is_err());
        assert!(quantize_f32_to_q6_k(&[f32::MAX; K_BLOCK_ELEMENTS]).is_err());
    }
}
