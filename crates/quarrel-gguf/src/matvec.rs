//! Allocation-free matrix–vector products over borrowed GGUF weight bytes.
//!
//! Every kernel interprets `weights` as a row-major matrix with `out.len()`
//! rows and `input.len()` columns, computes one dot product per row and writes
//! the result to the matching entry of `out`. Nothing is materialised between
//! the borrowed bytes and the caller's output buffer: unquantized rows are
//! decoded into a small stack chunk, and quantized blocks are decoded into a
//! fixed-size stack array (`[f32; 32]` or `[f32; 256]`) and dotted immediately
//! with the matching input slice.
//!
//! # Storage layouts (all little-endian, as required by GGUF)
//!
//! - **F32** (GGUF type 0): four bytes per element, IEEE-754 binary32.
//! - **F16** (type 1): two bytes per element, IEEE-754 binary16.
//! - **BF16** (type 28): two bytes per element, bfloat16 (binary32 with the
//!   low 16 mantissa bits dropped).
//! - **Q8_0** (type 6): 34-byte blocks. Bytes 0..2 hold an f16 scale, bytes
//!   2..34 hold 32 two's-complement `i8` values. Element `i` is
//!   `scale * (block[2 + i] as i8)`.
//! - **Q4_0** (type 2): 18-byte blocks. Bytes 0..2 hold an f16 scale, bytes
//!   2..18 hold 16 bytes of packed 4-bit signed nibbles with bias 8 (range
//!   -8..=7). The low nibble of byte `2 + j` is element `2 * j`, the high
//!   nibble is element `2 * j + 1`. Element `i` is `scale * nibble`.
//!
//! Q8_0/Q4_0 rows occupy `cols / 32 * block_bytes` bytes, so `cols` must be a
//! nonzero multiple of 32. K-quant rows occupy `cols / 256 * block_bytes`
//! bytes, so `cols` must be a nonzero multiple of 256. All K-quant layouts
//! below are the canonical ggml/llama.cpp little-endian super-block layouts
//! (`QK_K = 256`), and all indices refer to bytes inside one 256-element block.
//!
//! - **Q2_K** (type 8, 84 bytes): bytes `0..16` hold 16 scale/min pairs packed
//!   as `scale = b & 0xf`, `min = b >> 4`; bytes `16..80` hold 64 bytes with
//!   four unsigned 2-bit quants per byte, in four 16-byte planes (one byte per
//!   element of a 16-element group) selected by bits 0-1, 2-3, 4-5 and 6-7;
//!   bytes `80..82` and `82..84` hold the f16
//!   super-block scale `d` and minimum scale `dmin`. For element `n`, the
//!   scale/min group is `n / 16`, the quant byte is `qs[32 * (n / 128) +
//!   n % 32]`, the bit shift is `2 * ((n % 128) / 32)`, and the decoded value
//!   is `d * scale * q - dmin * min`.
//! - **Q3_K** (type 9, 110 bytes): bytes `0..32` hold the high (sign) bit of
//!   each quant; element `n` uses bit `(n % 128) / 32 + 4 * (n / 128)` of byte
//!   `n % 32` (the bit acts as a sign: clear means subtract 4); bytes `32..96`
//!   hold the low two bits in the same four-plane packing as Q2_K; bytes
//!   `96..108` hold 12
//!   bytes of 6-bit scales (four 4-bit low nibbles in bytes `0..8`, four more
//!   in bytes `4..8`, and two high bits per scale in bytes `8..12`); bytes
//!   `108..110` hold the f16 super-block scale `d`. Scale `g` in `0..16` takes
//!   its low four bits from byte `g % 4` (`g < 8`) or `4 + g % 4` (`g >= 8`),
//!   low nibble in the first case and high nibble in the second; its high two
//!   bits are `(scales[8 + g % 4] >> (2 * ((g / 4) % 4))) & 3`. The result is
//!   biased by -32. Element `n` uses scale `n / 16` and high-bit mask
//!   `1 << ((n % 128) / 32 + 4 * (n / 128))`.
//! - **Q4_K** (type 10, 144 bytes): bytes `0..2` and `2..4` hold the f16
//!   scales `d` and `dmin`; bytes `4..16` hold twelve 6-bit scale/min pairs
//!   for eight 32-element groups (pairs `0..4` are `scales[j]`/`scales[j+4]`;
//!   pairs `4..8` take the low four bits from `scales[j+4]` and the high two
//!   bits from `scales[j-4]`/`scales[j]`); bytes `16..144` hold 128 bytes with
//!   two unsigned 4-bit quants per byte, low nibble first. Element `n` uses
//!   group `n / 32` and quant byte `qs[32 * (n / 64) + n % 32]`; the value is
//!   `d * scale * q - dmin * min`.
//! - **Q5_K** (type 11, 176 bytes): the f16 scales, 6-bit scale/min pairs and
//!   low 4-bit quants occupy the same offsets as Q4_K, with 32 high-bit bytes
//!   at `16..48` (and the low nibbles at `48..176`). Element `n` adds 16 to
//!   its low nibble when bit `1 << (2 * (n / 64) + (n % 64) / 32)` of
//!   `qh[n % 32]` is set, giving an unsigned 5-bit quant.
//! - **Q6_K** (type 12, 210 bytes): bytes `0..128` hold the low four bits of
//!   each quant, bytes `128..192` hold the upper two bits of four quants per
//!   byte (within each 128-element half, for `l` in `0..32`, elements `l`,
//!   `l+32`, `l+64` and `l+96` share byte `l` in bits 0-1, 2-3, 4-5 and 6-7),
//!   bytes `192..208` hold 16 signed 8-bit scales (one per 16-element group),
//!   and bytes `208..210` hold the f16 scale `d`.
//!   Each quant is biased by -32 and element `n` is `d * scale[n / 16] * q`.
//!
//! Floating-point inputs, f16 scales and decoded weights must be finite; the
//! first non-finite value is reported as [`GgufError::NonFiniteValue`]. Dot
//! products use [`quarrel_simd::dot_f32`], so the summation order (and thus
//! the last few ULPs) depends on the SIMD backend selected for the process.
//! Unquantized rows are accumulated in 64-element chunks; every quantized
//! block is decoded into a stack array and dotted immediately, so no payload
//! sized allocation is needed.

use crate::DType;
use crate::error::GgufError;
use half::{bf16, f16};
use quarrel_simd::dot_f32;

/// Elements decoded into one stack buffer before calling `dot_f32`.
const CHUNK_ELEMENTS: usize = 64;
/// Logical elements in one Q8_0/Q4_0 block.
const BLOCK_ELEMENTS: usize = 32;
/// Logical elements in one K-quant block (`QK_K` in ggml).
const QK_K: usize = 256;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q2_K_BLOCK_BYTES: usize = 84;
const Q3_K_BLOCK_BYTES: usize = 110;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;

type DecodeFn = fn(&[u8], &mut [f32]);
type KDecodeFn = fn(&[u8], usize, &mut [f32; QK_K]) -> Result<(), GgufError>;

/// Dispatch a matvec to the kernel for `dtype`.
///
/// F32, F16, BF16, Q8_0, Q4_0 and all five K-quants are supported. Only
/// genuinely unknown types return [`GgufError::UnsupportedDType`] without
/// reading the buffers.
pub fn matvec(
    dtype: DType,
    weights: &[u8],
    input: &[f32],
    out: &mut [f32],
) -> Result<(), GgufError> {
    match dtype {
        DType::F32 => matvec_f32(weights, input, out),
        DType::F16 => matvec_f16(weights, input, out),
        DType::BF16 => matvec_bf16(weights, input, out),
        DType::Q8_0 => matvec_q8_0(weights, input, out),
        DType::Q4_0 => matvec_q4_0(weights, input, out),
        DType::Q2_K => matvec_q2_k(weights, input, out),
        DType::Q3_K => matvec_q3_k(weights, input, out),
        DType::Q4_K => matvec_q4_k(weights, input, out),
        DType::Q5_K => matvec_q5_k(weights, input, out),
        DType::Q6_K => matvec_q6_k(weights, input, out),
        other => Err(GgufError::UnsupportedDType(other)),
    }
}

fn internal_dot_error() -> GgufError {
    GgufError::InvalidQuantArgument("internal dot operand length invariant violated".to_owned())
}

fn check_finite(values: &[f32], role: &'static str, base: usize) -> Result<(), GgufError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GgufError::NonFiniteValue {
                role,
                index: base.checked_add(index).ok_or(GgufError::Overflow)?,
                value,
            });
        }
    }
    Ok(())
}

fn element_index(row: usize, cols: usize, offset: usize) -> Result<usize, GgufError> {
    row.checked_mul(cols)
        .and_then(|base| base.checked_add(offset))
        .ok_or(GgufError::Overflow)
}

fn check_shape(
    weights: &[u8],
    input: &[f32],
    out: &[f32],
    row_bytes: usize,
) -> Result<(), GgufError> {
    if input.is_empty() || out.is_empty() {
        return Err(GgufError::InvalidQuantArgument(
            "matvec requires a non-empty input and output".to_owned(),
        ));
    }
    let expected = out
        .len()
        .checked_mul(row_bytes)
        .ok_or(GgufError::Overflow)?;
    if weights.len() != expected {
        return Err(GgufError::InvalidMatvecShape {
            expected,
            actual: weights.len(),
            rows: out.len(),
            cols: input.len(),
        });
    }
    Ok(())
}

fn check_uniform_shape(
    weights: &[u8],
    input: &[f32],
    out: &[f32],
    element_bytes: usize,
) -> Result<usize, GgufError> {
    let row_bytes = input
        .len()
        .checked_mul(element_bytes)
        .ok_or(GgufError::Overflow)?;
    check_shape(weights, input, out, row_bytes)?;
    Ok(row_bytes)
}

fn check_block_shape(
    weights: &[u8],
    input: &[f32],
    out: &[f32],
    block_bytes: usize,
) -> Result<usize, GgufError> {
    if !input.len().is_multiple_of(BLOCK_ELEMENTS) {
        return Err(GgufError::InvalidQuantArgument(format!(
            "matvec input length {} must be a multiple of {BLOCK_ELEMENTS} for block quant data",
            input.len()
        )));
    }
    let row_bytes = (input.len() / BLOCK_ELEMENTS)
        .checked_mul(block_bytes)
        .ok_or(GgufError::Overflow)?;
    check_shape(weights, input, out, row_bytes)?;
    Ok(row_bytes)
}

fn decode_f32(bytes: &[u8], dst: &mut [f32]) {
    for (pair, value) in bytes.as_chunks::<4>().0.iter().zip(dst) {
        *value = f32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]);
    }
}

fn decode_f16(bytes: &[u8], dst: &mut [f32]) {
    for (pair, value) in bytes.as_chunks::<2>().0.iter().zip(dst) {
        *value = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
    }
}

fn decode_bf16(bytes: &[u8], dst: &mut [f32]) {
    for (pair, value) in bytes.as_chunks::<2>().0.iter().zip(dst) {
        *value = bf16::from_le_bytes([pair[0], pair[1]]).to_f32();
    }
}

fn matvec_uniform(
    weights: &[u8],
    input: &[f32],
    out: &mut [f32],
    element_bytes: usize,
    decode: DecodeFn,
) -> Result<(), GgufError> {
    let row_bytes = check_uniform_shape(weights, input, out, element_bytes)?;
    check_finite(input, "input", 0)?;

    let cols = input.len();
    let mut decoded = [0.0f32; CHUNK_ELEMENTS];
    for (row_index, (row, dst)) in weights
        .chunks_exact(row_bytes)
        .zip(out.iter_mut())
        .enumerate()
    {
        let mut acc = 0.0f32;
        for (chunk_index, input_chunk) in input.chunks(CHUNK_ELEMENTS).enumerate() {
            let count = input_chunk.len();
            let byte_start = chunk_index
                .checked_mul(CHUNK_ELEMENTS)
                .and_then(|offset| offset.checked_mul(element_bytes))
                .ok_or(GgufError::Overflow)?;
            let byte_end = byte_start
                .checked_add(
                    count
                        .checked_mul(element_bytes)
                        .ok_or(GgufError::Overflow)?,
                )
                .ok_or(GgufError::Overflow)?;
            let offset = chunk_index
                .checked_mul(CHUNK_ELEMENTS)
                .ok_or(GgufError::Overflow)?;
            let base = element_index(row_index, cols, offset)?;
            decode(&row[byte_start..byte_end], &mut decoded[..count]);
            check_finite(&decoded[..count], "weights", base)?;
            acc += dot_f32(&decoded[..count], input_chunk).map_err(|_| internal_dot_error())?;
        }
        *dst = acc;
    }
    Ok(())
}

/// Matrix–vector product with little-endian binary32 weights.
pub fn matvec_f32(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_uniform(weights, input, out, 4, decode_f32)
}

/// Matrix–vector product with little-endian binary16 weights.
pub fn matvec_f16(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_uniform(weights, input, out, 2, decode_f16)
}

/// Matrix–vector product with little-endian bfloat16 weights.
pub fn matvec_bf16(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_uniform(weights, input, out, 2, decode_bf16)
}

/// Matrix–vector product with Q8_0 weights (34-byte blocks, 32 values each).
pub fn matvec_q8_0(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    let row_bytes = check_block_shape(weights, input, out, Q8_0_BLOCK_BYTES)?;
    check_finite(input, "input", 0)?;

    let cols = input.len();
    for (row_index, (row, dst)) in weights
        .chunks_exact(row_bytes)
        .zip(out.iter_mut())
        .enumerate()
    {
        let mut acc = 0.0f32;
        for (block_index, (block, lhs)) in row
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
            .iter()
            .zip(input.as_chunks::<BLOCK_ELEMENTS>().0)
            .enumerate()
        {
            let mut decoded = [0.0f32; BLOCK_ELEMENTS];
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for (&encoded, value) in block[2..].iter().zip(&mut decoded) {
                *value = f32::from(encoded as i8) * scale;
            }
            let offset = block_index
                .checked_mul(BLOCK_ELEMENTS)
                .ok_or(GgufError::Overflow)?;
            check_finite(&decoded, "weights", element_index(row_index, cols, offset)?)?;
            acc += dot_f32(&decoded, lhs).map_err(|_| internal_dot_error())?;
        }
        *dst = acc;
    }
    Ok(())
}

/// Matrix–vector product with Q4_0 weights (18-byte blocks, 32 values each).
pub fn matvec_q4_0(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    let row_bytes = check_block_shape(weights, input, out, Q4_0_BLOCK_BYTES)?;
    check_finite(input, "input", 0)?;

    let cols = input.len();
    for (row_index, (row, dst)) in weights
        .chunks_exact(row_bytes)
        .zip(out.iter_mut())
        .enumerate()
    {
        let mut acc = 0.0f32;
        for (block_index, (block, lhs)) in row
            .as_chunks::<Q4_0_BLOCK_BYTES>()
            .0
            .iter()
            .zip(input.as_chunks::<BLOCK_ELEMENTS>().0)
            .enumerate()
        {
            let mut decoded = [0.0f32; BLOCK_ELEMENTS];
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for (pair_index, &packed) in block[2..].iter().enumerate() {
                let low = (packed & 0x0f) as i8 - 8;
                let high = (packed >> 4) as i8 - 8;
                decoded[2 * pair_index] = f32::from(low) * scale;
                decoded[2 * pair_index + 1] = f32::from(high) * scale;
            }
            let offset = block_index
                .checked_mul(BLOCK_ELEMENTS)
                .ok_or(GgufError::Overflow)?;
            check_finite(&decoded, "weights", element_index(row_index, cols, offset)?)?;
            acc += dot_f32(&decoded, lhs).map_err(|_| internal_dot_error())?;
        }
        *dst = acc;
    }
    Ok(())
}

// ---- K-quants: canonical ggml super-blocks, decoded into stack arrays ----

fn check_k_shape(
    weights: &[u8],
    input: &[f32],
    out: &[f32],
    block_bytes: usize,
) -> Result<usize, GgufError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(GgufError::InvalidQuantArgument(format!(
            "matvec input length {} must be a multiple of {QK_K} for K-quant data",
            input.len()
        )));
    }
    let row_bytes = (input.len() / QK_K)
        .checked_mul(block_bytes)
        .ok_or(GgufError::Overflow)?;
    check_shape(weights, input, out, row_bytes)?;
    Ok(row_bytes)
}

/// Read one f16 super-block scale; `base` is the logical index of the block.
fn decode_f16_scale(pair: &[u8], base: usize) -> Result<f32, GgufError> {
    let value = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
    if !value.is_finite() {
        return Err(GgufError::NonFiniteValue {
            role: "weights",
            index: base,
            value,
        });
    }
    Ok(value)
}

fn matvec_k(
    weights: &[u8],
    input: &[f32],
    out: &mut [f32],
    block_bytes: usize,
    decode: KDecodeFn,
) -> Result<(), GgufError> {
    let row_bytes = check_k_shape(weights, input, out, block_bytes)?;
    check_finite(input, "input", 0)?;

    let cols = input.len();
    for (row_index, (row, dst)) in weights
        .chunks_exact(row_bytes)
        .zip(out.iter_mut())
        .enumerate()
    {
        let mut acc = 0.0f32;
        for (block_index, (block, lhs)) in row
            .chunks_exact(block_bytes)
            .zip(input.as_chunks::<QK_K>().0)
            .enumerate()
        {
            let mut decoded = [0.0f32; QK_K];
            let offset = block_index.checked_mul(QK_K).ok_or(GgufError::Overflow)?;
            let base = element_index(row_index, cols, offset)?;
            decode(block, base, &mut decoded)?;
            check_finite(&decoded, "weights", base)?;
            acc += dot_f32(&decoded, lhs).map_err(|_| internal_dot_error())?;
        }
        *dst = acc;
    }
    Ok(())
}

/// Q2_K: 16 packed scale/min pairs, 64 bytes of 2-bit quants, then `d`/`dmin`.
fn decode_q2_k(block: &[u8], base: usize, decoded: &mut [f32; QK_K]) -> Result<(), GgufError> {
    let scales = &block[..16];
    let qs = &block[16..80];
    let d = decode_f16_scale(&block[80..82], base)?;
    let dmin = decode_f16_scale(&block[82..84], base)?;

    for (n, value) in decoded.iter_mut().enumerate() {
        let packed = scales[n / 16];
        let dl = d * f32::from(packed & 0x0f);
        let ml = dmin * f32::from(packed >> 4);
        let shift = 2 * ((n % 128) / 32);
        let q = (qs[32 * (n / 128) + n % 32] >> shift) & 0x03;
        *value = dl * f32::from(q) - ml;
    }
    Ok(())
}

/// Unpack the twelve packed 6-bit Q3_K scales into their 16 signed values.
///
/// Bytes `0..4` and `4..8` contribute four low nibbles each, while bytes
/// `8..12` contribute two high bits per scale (little-endian bit order).
fn unpack_q3_scales(packed: &[u8]) -> [i16; 16] {
    let mut scales = [0i16; 16];
    for i in 0..4 {
        scales[i] = i16::from((packed[i] & 0x0f) | ((packed[8 + i] & 0x03) << 4)) - 32;
        scales[4 + i] =
            i16::from((packed[4 + i] & 0x0f) | (((packed[8 + i] >> 2) & 0x03) << 4)) - 32;
        scales[8 + i] =
            i16::from(((packed[i] >> 4) & 0x0f) | (((packed[8 + i] >> 4) & 0x03) << 4)) - 32;
        scales[12 + i] =
            i16::from(((packed[4 + i] >> 4) & 0x0f) | (((packed[8 + i] >> 6) & 0x03) << 4)) - 32;
    }
    scales
}

/// Q3_K: 32 sign bits, 64 bytes of low 2-bit quants, 12 packed scales, then `d`.
fn decode_q3_k(block: &[u8], base: usize, decoded: &mut [f32; QK_K]) -> Result<(), GgufError> {
    let hmask = &block[..32];
    let qs = &block[32..96];
    let scales = unpack_q3_scales(&block[96..108]);
    let d = decode_f16_scale(&block[108..110], base)?;

    for (n, value) in decoded.iter_mut().enumerate() {
        let half = n / 128;
        let shift = 2 * ((n % 128) / 32);
        let l = n % 32;
        let low = i16::from((qs[32 * half + l] >> shift) & 0x03);
        // ggml keeps the sign-bit plane fixed while the low-bit plane advances.
        let mask = 1u8 << (4 * half + (n % 128) / 32);
        let sign_adjust: i16 = if hmask[l] & mask != 0 { 0 } else { 4 };
        *value = d * f32::from(scales[n / 16]) * f32::from(low - sign_adjust);
    }
    Ok(())
}

/// Read scale/min pair `j` from the twelve packed 6-bit Q4_K/Q5_K bytes.
fn unpack_k4_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3f, scales[j + 4] & 0x3f)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Q4_K: `d`/`dmin`, twelve packed scale/min bytes, then 128 nibble pairs.
pub(crate) fn decode_q4_k(
    block: &[u8],
    base: usize,
    decoded: &mut [f32; QK_K],
) -> Result<(), GgufError> {
    let d = decode_f16_scale(&block[..2], base)?;
    let dmin = decode_f16_scale(&block[2..4], base)?;
    let packed = &block[4..16];
    let qs = &block[16..144];

    let mut params = [(0.0f32, 0.0f32); QK_K / 32];
    for (j, param) in params.iter_mut().enumerate() {
        let (sc, m) = unpack_k4_scale_min(j, packed);
        *param = (d * f32::from(sc), dmin * f32::from(m));
    }
    for (n, value) in decoded.iter_mut().enumerate() {
        let byte = qs[32 * (n / 64) + n % 32];
        let q = if n % 64 < 32 { byte & 0x0f } else { byte >> 4 };
        let (dl, ml) = params[n / 32];
        *value = dl * f32::from(q) - ml;
    }
    Ok(())
}

/// Q5_K: `d`/`dmin`, twelve packed scale/min bytes, 32 high bits, low nibbles.
fn decode_q5_k(block: &[u8], base: usize, decoded: &mut [f32; QK_K]) -> Result<(), GgufError> {
    let d = decode_f16_scale(&block[..2], base)?;
    let dmin = decode_f16_scale(&block[2..4], base)?;
    let packed = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut params = [(0.0f32, 0.0f32); QK_K / 32];
    for (j, param) in params.iter_mut().enumerate() {
        let (sc, m) = unpack_k4_scale_min(j, packed);
        *param = (d * f32::from(sc), dmin * f32::from(m));
    }
    for (n, value) in decoded.iter_mut().enumerate() {
        let half = n / 64;
        let l = n % 32;
        let bit = 1u8 << (2 * half + (n % 64) / 32);
        // ggml advances only the low-nibble plane; the high-bit plane stays put.
        let high = if qh[l] & bit != 0 { 16u8 } else { 0 };
        let byte = qs[32 * half + l];
        let q = if n % 64 < 32 {
            (byte & 0x0f) + high
        } else {
            (byte >> 4) + high
        };
        let (dl, ml) = params[n / 32];
        *value = dl * f32::from(q) - ml;
    }
    Ok(())
}

/// Q6_K: 128 low nibbles, 64 bytes of two high bits, 16 scales, then `d`.
pub(crate) fn decode_q6_k(
    block: &[u8],
    base: usize,
    decoded: &mut [f32; QK_K],
) -> Result<(), GgufError> {
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = decode_f16_scale(&block[208..210], base)?;

    for (n, value) in decoded.iter_mut().enumerate() {
        let half = n / 128;
        let lane = (n % 128) / 32;
        let l = n % 32;
        let (low, high_shift, scale_delta) = match lane {
            0 => (ql[64 * half + l] & 0x0f, 0u32, 0usize),
            1 => (ql[64 * half + 32 + l] & 0x0f, 2, 2),
            2 => (ql[64 * half + l] >> 4, 4, 4),
            _ => (ql[64 * half + 32 + l] >> 4, 6, 6),
        };
        let high = (qh[32 * half + l] >> high_shift) & 0x03;
        let q = i16::from(low | (high << 4)) - 32;
        let scale = f32::from(scales[8 * half + l / 16 + scale_delta]);
        *value = d * scale * f32::from(q);
    }
    Ok(())
}

/// Matrix–vector product with Q2_K weights (84-byte blocks, 256 values each).
///
/// See the module documentation for the exact byte layout and unpacking rules.
pub fn matvec_q2_k(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_k(
        weights,
        input,
        out,
        Q2_K_BLOCK_BYTES,
        decode_q2_k as KDecodeFn,
    )
}

/// Matrix–vector product with Q3_K weights (110-byte blocks, 256 values each).
///
/// See the module documentation for the exact byte layout and unpacking rules.
pub fn matvec_q3_k(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_k(
        weights,
        input,
        out,
        Q3_K_BLOCK_BYTES,
        decode_q3_k as KDecodeFn,
    )
}

/// Matrix–vector product with Q4_K weights (144-byte blocks, 256 values each).
///
/// See the module documentation for the exact byte layout and unpacking rules.
pub fn matvec_q4_k(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_k(
        weights,
        input,
        out,
        Q4_K_BLOCK_BYTES,
        decode_q4_k as KDecodeFn,
    )
}

/// Matrix–vector product with Q5_K weights (176-byte blocks, 256 values each).
///
/// See the module documentation for the exact byte layout and unpacking rules.
pub fn matvec_q5_k(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_k(
        weights,
        input,
        out,
        Q5_K_BLOCK_BYTES,
        decode_q5_k as KDecodeFn,
    )
}

/// Matrix–vector product with Q6_K weights (210-byte blocks, 256 values each).
///
/// See the module documentation for the exact byte layout and unpacking rules.
pub fn matvec_q6_k(weights: &[u8], input: &[f32], out: &mut [f32]) -> Result<(), GgufError> {
    matvec_k(
        weights,
        input,
        out,
        Q6_K_BLOCK_BYTES,
        decode_q6_k as KDecodeFn,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::hint::black_box;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Kernel = fn(&[u8], &[f32], &mut [f32]) -> Result<(), GgufError>;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn q8_bytes(scale: f16, values: &[i8]) -> Vec<u8> {
        assert_eq!(values.len() % BLOCK_ELEMENTS, 0);
        let mut bytes = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * Q8_0_BLOCK_BYTES);
        for block in values.chunks(BLOCK_ELEMENTS) {
            bytes.extend_from_slice(&scale.to_le_bytes());
            bytes.extend(block.iter().map(|v| *v as u8));
        }
        bytes
    }

    fn q4_bytes(scale: f16, values: &[i8]) -> Vec<u8> {
        assert_eq!(values.len() % BLOCK_ELEMENTS, 0);
        let mut bytes = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * Q4_0_BLOCK_BYTES);
        for block in values.chunks(BLOCK_ELEMENTS) {
            bytes.extend_from_slice(&scale.to_le_bytes());
            for pair in block.as_chunks::<2>().0 {
                let low = (pair[0] + 8) as u8 & 0x0f;
                let high = (pair[1] + 8) as u8 & 0x0f;
                bytes.push(low | (high << 4));
            }
        }
        bytes
    }

    // Each builder writes one canonical ggml K-quant block, in field order.
    fn q2_k_block(scales: &[u8; 16], qs: &[u8; 64], d: f16, dmin: f16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        bytes.extend_from_slice(scales);
        bytes.extend_from_slice(qs);
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes.extend_from_slice(&dmin.to_le_bytes());
        bytes
    }

    fn q3_k_block(hmask: &[u8; 32], qs: &[u8; 64], scales: &[u8; 12], d: f16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q3_K_BLOCK_BYTES);
        bytes.extend_from_slice(hmask);
        bytes.extend_from_slice(qs);
        bytes.extend_from_slice(scales);
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes
    }

    fn q4_k_block(d: f16, dmin: f16, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q4_K_BLOCK_BYTES);
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes.extend_from_slice(&dmin.to_le_bytes());
        bytes.extend_from_slice(scales);
        bytes.extend_from_slice(qs);
        bytes
    }

    fn q5_k_block(d: f16, dmin: f16, scales: &[u8; 12], qh: &[u8; 32], qs: &[u8; 128]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes.extend_from_slice(&dmin.to_le_bytes());
        bytes.extend_from_slice(scales);
        bytes.extend_from_slice(qh);
        bytes.extend_from_slice(qs);
        bytes
    }

    fn q6_k_block(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], d: f16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Q6_K_BLOCK_BYTES);
        bytes.extend_from_slice(ql);
        bytes.extend_from_slice(qh);
        bytes.extend(scales.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes
    }

    fn repeat_block(block: &[u8], rows: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * block.len());
        for _ in 0..rows {
            bytes.extend_from_slice(block);
        }
        bytes
    }

    fn check_k_block(
        decode: KDecodeFn,
        block: &[u8],
        expected: &[(usize, f32)],
        direct: Kernel,
        dtype: DType,
    ) {
        let mut decoded = [0.0f32; QK_K];
        decode(block, 0, &mut decoded).expect("valid K-quant block");
        let mut input = [0.0f32; QK_K];
        for &(index, value) in expected {
            assert_eq!(decoded[index], value, "decoded element {index}");
            input[index] = 1.0;
        }
        // Only the probed elements contribute; other groups can legitimately
        // decode to -min or -32*scale when their quant is zero.
        let sum: f32 = expected.iter().map(|&(_, value)| value).sum();
        let mut out = [f32::NAN; 1];
        direct(block, &input, &mut out).expect("direct K-quant kernel");
        assert_eq!(out, [sum], "direct dot product");
        let mut dispatched = [f32::NAN; 1];
        matvec(dtype, block, &input, &mut dispatched).expect("dispatched K-quant kernel");
        assert_eq!(dispatched, out, "dispatcher mismatch for {dtype:?}");
    }

    #[test]
    fn matvec_f32_hand_computed() {
        // row 0: 1*0.5 + 2*-1 + 3*2 + 4*1 = 8.5
        // row 1: -1*0.5 + 0*-1 + 0.5*2 + 2*1 = 2.5
        let weights = f32_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.0]);
        let input = [0.5, -1.0, 2.0, 1.0];
        let mut out = [f32::NAN; 2];
        matvec_f32(&weights, &input, &mut out).expect("valid f32 matvec");
        assert_eq!(out, [8.5, 2.5]);
    }

    #[test]
    fn matvec_f16_hand_computed() {
        let weights = f16_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.0]);
        let input = [0.5, -1.0, 2.0, 1.0];
        let mut out = [f32::NAN; 2];
        matvec_f16(&weights, &input, &mut out).expect("valid f16 matvec");
        assert_eq!(out, [8.5, 2.5]);
    }

    #[test]
    fn matvec_bf16_hand_computed() {
        let weights = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.0]);
        let input = [0.5, -1.0, 2.0, 1.0];
        let mut out = [f32::NAN; 2];
        matvec_bf16(&weights, &input, &mut out).expect("valid bf16 matvec");
        assert_eq!(out, [8.5, 2.5]);
    }

    #[test]
    fn matvec_q8_0_hand_computed() {
        let mut values = [0i8; 64];
        values[..4].copy_from_slice(&[2, -4, 0, 1]);
        values[32..36].copy_from_slice(&[1, 1, -2, 0]);
        let weights = q8_bytes(f16::from_f32(0.5), &values);
        let mut input = [0.0f32; 32];
        input[..4].copy_from_slice(&[1.0, 0.5, 3.0, -2.0]);
        // row 0 = 2*.5*1 + -4*.5*.5 + 1*.5*-2 = -1
        // row 1 = 1*.5*1 + 1*.5*.5 + -2*.5*3 = -2.25
        let mut out = [f32::NAN; 2];
        matvec_q8_0(&weights, &input, &mut out).expect("valid q8_0 matvec");
        assert_eq!(out, [-1.0, -2.25]);
    }

    #[test]
    fn matvec_q4_0_hand_computed() {
        let mut values = [-8i8; 64];
        values[..4].copy_from_slice(&[2, 4, -2, -8]);
        values[32..36].copy_from_slice(&[4, -8, -4, 2]);
        let weights = q4_bytes(f16::from_f32(2.0), &values);
        let mut input = [0.0f32; 32];
        input[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // row 0 = 4*1 + 8*2 + -4*3 + -16*4 = -56
        // row 1 = 8*1 + -16*2 + -8*3 + 4*4 = -32
        let mut out = [f32::NAN; 2];
        matvec_q4_0(&weights, &input, &mut out).expect("valid q4_0 matvec");
        assert_eq!(out, [-56.0, -32.0]);
    }

    #[test]
    fn matvec_q2_k_hand_computed() {
        // d = 0.5, dmin = 1.0. scales[0] = 0x23 -> dl = 1.5, ml = 2,
        // scales[1] = 0x41 -> dl = 0.5, ml = 4, scales[4] = 0x12 -> dl = 1,
        // ml = 1, scales[10] = 0x52 -> dl = 1, ml = 5.
        let mut scales = [0u8; 16];
        scales[0] = 0x23;
        scales[1] = 0x41;
        scales[4] = 0x12;
        scales[10] = 0x52;
        let mut qs = [0u8; 64];
        // Each 16-element group reads one byte per element; planes are selected
        // by bits 0-1, 2-3, 4-5 and 6-7 of that byte.
        qs[0] = 0b11_10_01_00; // element 0 -> 0, element 64 -> 2
        qs[1] = 0b0000_0001; // element 1 -> 1
        qs[2] = 0b0000_0010; // element 2 -> 2
        qs[3] = 0b0000_0011; // element 3 -> 3
        qs[16] = 0b0000_0001; // element 16 -> 1
        qs[32] = 0b0000_1100; // element 160 (second half) -> 3 at shift 2
        let block = q2_k_block(&scales, &qs, f16::from_f32(0.5), f16::from_f32(1.0));
        check_k_block(
            decode_q2_k,
            &block,
            &[
                (0, -2.0),
                (1, -0.5),
                (2, 1.0),
                (3, 2.5),
                (16, -3.5),
                (64, 1.0),
                (160, -2.0),
            ],
            matvec_q2_k as Kernel,
            DType::Q2_K,
        );
    }

    #[test]
    fn matvec_q3_k_hand_computed() {
        // Packed scales start at 0xAA (all 6-bit values = 32 -> scale 0).
        // s[0] = 33 -> 1, s[1] = 30 -> -2, s[4] = 35 -> 3, s[10] = 34 -> 2.
        let mut scales = [0xaa_u8; 12];
        scales[0] = 0x01; // low nibble 1 -> s[0]; high nibble 0 -> s[8]
        scales[1] = 0x0e;
        scales[2] = 0x20; // high nibble of s[10]
        scales[4] = 0x03;
        scales[8] = 0xaa; // high two bits = 2 for s[0], s[4], s[8], s[12]
        scales[9] = 0xa9; // bits 0-1 = 1 -> s[1] = 30
        assert_eq!(unpack_q3_scales(&scales)[0], 1);
        assert_eq!(unpack_q3_scales(&scales)[1], -2);
        assert_eq!(unpack_q3_scales(&scales)[4], 3);
        assert_eq!(unpack_q3_scales(&scales)[10], 2);
        assert_eq!(unpack_q3_scales(&scales)[8], 0);

        let mut qs = [0u8; 64];
        qs[0] = 0b0011_0110; // n = 0 -> 2 (bits 0-1), n = 64 -> 3 (bits 4-5)
        qs[1] = 0b0000_0001; // n = 1 -> 1
        qs[4] = 0b0000_0011; // bit 0 clear in hmask[4] -> q = -1
        qs[16] = 0b0000_0011;
        qs[32] = 0b0000_1100; // second half, shift 2 -> q = 3
        let mut hmask = [0u8; 32];
        // hmask[n % 32] holds sign bits for n, n+32, n+64, ... .
        hmask[0] = 0b10_0101; // n = 0, 64 and 160 (not n = 4)
        hmask[1] = 0b1; // n = 1
        hmask[16] = 0b1; // n = 16
        let block = q3_k_block(&hmask, &qs, &scales, f16::from_f32(0.5));
        check_k_block(
            decode_q3_k,
            &block,
            &[
                (0, 1.0),
                (1, 0.5),
                (4, -0.5),
                (16, -3.0),
                (64, 4.5),
                (160, 3.0),
            ],
            matvec_q3_k as Kernel,
            DType::Q3_K,
        );
    }

    #[test]
    fn matvec_q4_k_hand_computed() {
        // d = 0.25, dmin = 2.0. scales[0] = 2/scales[4] = 1 gives group 0
        // (sc = 2, m = 1); group 1 is (4, 3); the split 6-bit pairs for
        // groups 4 and 5 are (6, 2) and (5, 1).
        let mut scales = [0u8; 12];
        scales[0] = 2;
        scales[4] = 1;
        scales[1] = 4;
        scales[5] = 3;
        scales[8] = 0x26;
        scales[9] = 0x15;
        assert_eq!(unpack_k4_scale_min(0, &scales), (2, 1));
        assert_eq!(unpack_k4_scale_min(5, &scales), (5, 1));
        let mut qs = [0u8; 128];
        qs[0] = 0x21; // n = 0 -> low 1, n = 32 -> high 2
        qs[1] = 0x30; // n = 1 -> low 0, n = 33 -> high 3
        qs[64] = 0x45; // n = 128 -> low 5, n = 160 -> high 4
        let block = q4_k_block(f16::from_f32(0.25), f16::from_f32(2.0), &scales, &qs);
        check_k_block(
            decode_q4_k,
            &block,
            &[
                (0, -1.5),
                (1, -2.0),
                (32, -4.0),
                (33, -3.0),
                (128, 3.5),
                (160, 3.0),
            ],
            matvec_q4_k as Kernel,
            DType::Q4_K,
        );
    }

    #[test]
    fn matvec_q5_k_hand_computed() {
        // Same scale/min pairs as the Q4_K test, with d = 0.5 and dmin = 1.0.
        let mut scales = [0u8; 12];
        scales[0] = 2;
        scales[4] = 1;
        scales[1] = 4;
        scales[5] = 3;
        scales[8] = 0x26;
        scales[9] = 0x15;
        let mut qs = [0u8; 128];
        qs[0] = 0x21;
        qs[1] = 0x30;
        qs[64] = 0x45;
        let mut qh = [0u8; 32];
        // qh[n % 32] holds the high bits for all 32-element lanes; the bit
        // position also encodes which 64-element group the element is in.
        qh[0] = 0b0001_0001; // n = 0 and n = 128 set; n = 32/160 clear
        qh[1] = 0b10; // n = 33 set
        let block = q5_k_block(f16::from_f32(0.5), f16::from_f32(1.0), &scales, &qh, &qs);
        check_k_block(
            decode_q5_k,
            &block,
            &[
                (0, 16.0),
                (1, -1.0),
                (32, 1.0),
                (33, 35.0),
                (128, 61.0),
                (160, 9.0),
            ],
            matvec_q5_k as Kernel,
            DType::Q5_K,
        );
    }

    #[test]
    fn matvec_q6_k_hand_computed() {
        // d = 0.25; scales[0] = 2, [2] = 1, [4] = 2, [6] = 3 and [10] = 4.
        // qh[0] = 0xed supplies upper bits 1, 3, 2, 3 for n = 0, 32, 64, 96.
        let mut scales = [0i8; 16];
        scales[0] = 2;
        scales[2] = 1;
        scales[4] = 2;
        scales[6] = 3;
        scales[10] = 4;
        let mut ql = [0u8; 128];
        ql[0] = 0x21;
        ql[32] = 0x03;
        ql[96] = 0x05; // second 128-element half
        let mut qh = [0u8; 64];
        qh[0] = 0xed;
        qh[32] = 0x08; // upper bits 2 for n = 160 in the second half
        let block = q6_k_block(&ql, &qh, &scales, f16::from_f32(0.25));
        check_k_block(
            decode_q6_k,
            &block,
            &[(0, -7.5), (32, 4.75), (64, 1.0), (96, 12.0), (160, 5.0)],
            matvec_q6_k as Kernel,
            DType::Q6_K,
        );
    }

    #[test]
    fn dispatcher_matches_direct_kernels() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let input = [0.5f32, -1.0, 2.0, 1.0];
        for (dtype, weights, direct_kernel) in [
            (DType::F32, f32_bytes(&values), matvec_f32 as Kernel),
            (DType::F16, f16_bytes(&values), matvec_f16 as Kernel),
            (DType::BF16, bf16_bytes(&values), matvec_bf16 as Kernel),
        ] {
            let mut direct = [0.0f32; 1];
            let mut dispatched = [0.0f32; 1];
            direct_kernel(&weights, &input, &mut direct).expect("direct kernel");
            matvec(dtype, &weights, &input, &mut dispatched).expect("dispatched kernel");
            assert_eq!(direct, dispatched, "dtype {dtype:?}");
        }

        let q8 = q8_bytes(f16::from_f32(0.5), &[1i8; 32]);
        let q4 = q4_bytes(f16::from_f32(0.5), &[1i8; 32]);
        let q8_input = [1.0f32; 32];
        let mut direct = [0.0f32; 1];
        let mut dispatched = [0.0f32; 1];
        matvec_q8_0(&q8, &q8_input, &mut direct).expect("direct q8_0 kernel");
        matvec(DType::Q8_0, &q8, &q8_input, &mut dispatched).expect("dispatched q8_0");
        assert_eq!(direct, dispatched);
        matvec_q4_0(&q4, &q8_input, &mut direct).expect("direct q4_0 kernel");
        matvec(DType::Q4_0, &q4, &q8_input, &mut dispatched).expect("dispatched q4_0");
        assert_eq!(direct, dispatched);

        // Every K-quant is reachable through both the dispatcher and its
        // dedicated public kernel.
        let k_input = [1.0f32; QK_K];
        let one = f16::from_f32(1.0);
        let k_cases: [(DType, Vec<u8>, Kernel); 5] = [
            (
                DType::Q2_K,
                q2_k_block(&[0u8; 16], &[0u8; 64], one, f16::from_f32(0.0)),
                matvec_q2_k as Kernel,
            ),
            (
                DType::Q3_K,
                q3_k_block(&[0u8; 32], &[0u8; 64], &[0xaa_u8; 12], one),
                matvec_q3_k as Kernel,
            ),
            (
                DType::Q4_K,
                q4_k_block(one, f16::from_f32(0.0), &[0u8; 12], &[0u8; 128]),
                matvec_q4_k as Kernel,
            ),
            (
                DType::Q5_K,
                q5_k_block(one, f16::from_f32(0.0), &[0u8; 12], &[0u8; 32], &[0u8; 128]),
                matvec_q5_k as Kernel,
            ),
            (
                DType::Q6_K,
                q6_k_block(&[0u8; 128], &[0u8; 64], &[0i8; 16], one),
                matvec_q6_k as Kernel,
            ),
        ];
        for (dtype, weights, kernel) in k_cases {
            let mut direct = [f32::NAN; 1];
            let mut dispatched = [f32::NAN; 1];
            kernel(&weights, &k_input, &mut direct).expect("direct K-quant kernel");
            matvec(dtype, &weights, &k_input, &mut dispatched).expect("dispatched K-quant");
            assert_eq!(direct, dispatched, "dtype {dtype:?}");
        }
    }

    #[test]
    fn unknown_dtype_is_reported_cleanly() {
        let dtype = DType::Unknown(999);
        let result = matvec(dtype, &[0u8; 144], &[1.0f32; 256], &mut [0.0f32; 1]);
        assert!(
            matches!(result, Err(GgufError::UnsupportedDType(actual)) if actual == dtype),
            "unknown dtype {dtype:?} was not reported cleanly"
        );
    }

    #[test]
    fn matvec_error_paths() {
        let weights = f32_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.0]);
        let input = [0.5f32, -1.0, 2.0, 1.0];

        // Wrong number of columns changes the expected byte count.
        assert!(matches!(
            matvec_f32(&weights, &input[..3], &mut [0.0; 2]),
            Err(GgufError::InvalidMatvecShape { .. })
        ));
        // Truncated and over-long payloads are both rejected, and out is intact.
        let mut out = [7.0f32; 2];
        assert!(matches!(
            matvec_f32(&weights[..weights.len() - 1], &input, &mut out),
            Err(GgufError::InvalidMatvecShape { .. })
        ));
        assert!(out == [7.0, 7.0]);
        let mut too_long = weights.clone();
        too_long.push(0);
        assert!(matches!(
            matvec_f32(&too_long, &input, &mut out),
            Err(GgufError::InvalidMatvecShape { .. })
        ));
        assert!(out == [7.0, 7.0]);
        assert!(matches!(
            matvec_f32(&weights, &input, &mut [0.0; 3]),
            Err(GgufError::InvalidMatvecShape { .. })
        ));
        assert!(matches!(
            matvec_f32(&weights, &[], &mut [0.0; 2]),
            Err(GgufError::InvalidQuantArgument(_))
        ));

        // Block quant kernels additionally require a whole number of blocks.
        let q8 = q8_bytes(f16::from_f32(0.5), &[1i8; 32]);
        let q4 = q4_bytes(f16::from_f32(0.5), &[1i8; 32]);
        assert!(matches!(
            matvec_q8_0(&q8, &[1.0f32; 31], &mut [0.0; 1]),
            Err(GgufError::InvalidQuantArgument(_))
        ));
        assert!(matches!(
            matvec_q4_0(&q4, &[1.0f32; 33], &mut [0.0; 1]),
            Err(GgufError::InvalidQuantArgument(_))
        ));
        assert!(matches!(
            matvec_q8_0(&q8[..q8.len() - 1], &[1.0f32; 32], &mut [0.0; 1]),
            Err(GgufError::InvalidMatvecShape { .. })
        ));
        assert!(matches!(
            matvec_q4_0(&q4[..q4.len() - 1], &[1.0f32; 32], &mut [0.0; 1]),
            Err(GgufError::InvalidMatvecShape { .. })
        ));

        // Non-finite inputs are rejected before any output is written.
        let mut nan_input = input;
        nan_input[2] = f32::NAN;
        assert!(matches!(
            matvec_f32(&weights, &nan_input, &mut out),
            Err(GgufError::NonFiniteValue { role: "input", index: 2, value }) if value.is_nan()
        ));
        let mut inf_input = [1.0f32; 32];
        inf_input[7] = f32::INFINITY;
        assert!(matches!(
            matvec_q8_0(&q8, &inf_input, &mut [0.0; 1]),
            Err(GgufError::NonFiniteValue {
                role: "input",
                index: 7,
                ..
            })
        ));
        assert!(matches!(
            matvec_q4_0(&q4, &inf_input, &mut [0.0; 1]),
            Err(GgufError::NonFiniteValue {
                role: "input",
                index: 7,
                ..
            })
        ));
        let mut nan_f16 = input;
        nan_f16[1] = f32::NEG_INFINITY;
        let two_rows = vec![1.0f32; 8];
        assert!(matches!(
            matvec_f16(&f16_bytes(&two_rows), &nan_f16, &mut out),
            Err(GgufError::NonFiniteValue {
                role: "input",
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            matvec_bf16(&bf16_bytes(&two_rows), &nan_f16, &mut out),
            Err(GgufError::NonFiniteValue {
                role: "input",
                index: 1,
                ..
            })
        ));
        assert!(out == [7.0, 7.0]);

        // Non-finite decoded weights are reported with their logical index.
        let mut bad_weights = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        bad_weights[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            matvec_f32(&bad_weights, &input[..2], &mut [0.0; 2]),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 1,
                ..
            })
        ));
        let bad_scale = q8_bytes(f16::INFINITY, &[1i8; 32]);
        assert!(matches!(
            matvec_q8_0(&bad_scale, &[1.0f32; 32], &mut [0.0; 1]),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                ..
            })
        ));
    }

    #[test]
    fn k_quant_error_paths() {
        let one = f16::from_f32(1.0);
        let zero = f16::from_f32(0.0);
        let cases: [(DType, Kernel, Vec<u8>, usize); 5] = [
            (
                DType::Q2_K,
                matvec_q2_k as Kernel,
                q2_k_block(&[0u8; 16], &[0u8; 64], one, zero),
                Q2_K_BLOCK_BYTES,
            ),
            (
                DType::Q3_K,
                matvec_q3_k as Kernel,
                q3_k_block(&[0u8; 32], &[0u8; 64], &[0xaa_u8; 12], one),
                Q3_K_BLOCK_BYTES,
            ),
            (
                DType::Q4_K,
                matvec_q4_k as Kernel,
                q4_k_block(one, zero, &[0u8; 12], &[0u8; 128]),
                Q4_K_BLOCK_BYTES,
            ),
            (
                DType::Q5_K,
                matvec_q5_k as Kernel,
                q5_k_block(one, zero, &[0u8; 12], &[0u8; 32], &[0u8; 128]),
                Q5_K_BLOCK_BYTES,
            ),
            (
                DType::Q6_K,
                matvec_q6_k as Kernel,
                q6_k_block(&[0u8; 128], &[0u8; 64], &[0i8; 16], one),
                Q6_K_BLOCK_BYTES,
            ),
        ];

        for (dtype, kernel, block, block_bytes) in cases {
            let input = [1.0f32; QK_K];
            let mut out = [7.0f32; 1];
            assert!(matches!(
                kernel(&block, &input[..QK_K - 1], &mut out),
                Err(GgufError::InvalidQuantArgument(_))
            ));
            let mut truncated = block.clone();
            truncated.pop();
            assert!(matches!(
                kernel(&truncated, &input, &mut out),
                Err(GgufError::InvalidMatvecShape { .. })
            ));
            let mut too_long = block.clone();
            too_long.push(0);
            assert!(matches!(
                kernel(&too_long, &input, &mut out),
                Err(GgufError::InvalidMatvecShape { .. })
            ));
            assert!(matches!(
                kernel(&block, &input, &mut [0.0f32; 2]),
                Err(GgufError::InvalidMatvecShape { .. })
            ));
            assert!(matches!(
                kernel(&block, &[], &mut out),
                Err(GgufError::InvalidQuantArgument(_))
            ));
            assert!(matches!(
                kernel(&block, &input, &mut []),
                Err(GgufError::InvalidQuantArgument(_))
            ));
            assert_eq!(block.len(), block_bytes);
            assert_eq!(out, [7.0], "failed kernels must not write output");

            // The dispatcher reports exactly the same shape errors.
            assert!(matches!(
                matvec(dtype, &truncated, &input, &mut out),
                Err(GgufError::InvalidMatvecShape { .. })
            ));
            let mut bad_input = [1.0f32; QK_K];
            bad_input[7] = f32::INFINITY;
            assert!(matches!(
                kernel(&block, &bad_input, &mut out),
                Err(GgufError::NonFiniteValue {
                    role: "input",
                    index: 7,
                    ..
                })
            ));
        }

        // Non-finite f16 super-block scales are rejected before decoding.
        let scales = [0u8; 16];
        let qs = [0u8; 64];
        assert!(matches!(
            matvec_q2_k(
                &q2_k_block(&scales, &qs, f16::INFINITY, zero),
                &[1.0f32; QK_K],
                &mut [0.0f32; 1]
            ),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 0,
                value
            }) if value.is_infinite()
        ));
        assert!(matches!(
            matvec_q3_k(
                &q3_k_block(&[0u8; 32], &qs, &[0xaa_u8; 12], f16::NAN),
                &[1.0f32; QK_K],
                &mut [0.0f32; 1]
            ),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 0,
                value
            }) if value.is_nan()
        ));
        assert!(matches!(
            matvec_q4_k(
                &q4_k_block(f16::NEG_INFINITY, zero, &[0u8; 12], &[0u8; 128]),
                &[1.0f32; QK_K],
                &mut [0.0f32; 1]
            ),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            matvec_q5_k(
                &q5_k_block(one, f16::INFINITY, &[0u8; 12], &[0u8; 32], &[0u8; 128]),
                &[1.0f32; QK_K],
                &mut [0.0f32; 1]
            ),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            matvec_q6_k(
                &q6_k_block(&[0u8; 128], &[0u8; 64], &[0i8; 16], f16::INFINITY),
                &[1.0f32; QK_K],
                &mut [0.0f32; 1]
            ),
            Err(GgufError::NonFiniteValue {
                role: "weights",
                index: 0,
                ..
            })
        ));
    }

    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    static LARGE_ALLOCS: AtomicUsize = AtomicUsize::new(0);
    static MAX_ALLOC: AtomicUsize = AtomicUsize::new(0);
    const LARGE_ALLOC_BYTES: usize = 64 * 1024;

    struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: delegated unchanged to the system allocator.
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                record(layout.size());
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: delegated unchanged to the system allocator.
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                record(layout.size());
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: caller supplies a pointer previously allocated here.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: delegated unchanged to the system allocator.
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                record(new_size);
            }
            new_ptr
        }
    }

    fn record(size: usize) {
        ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
        if size >= LARGE_ALLOC_BYTES {
            LARGE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        MAX_ALLOC.fetch_max(size, Ordering::Relaxed);
    }

    fn reset_counters() {
        ALLOC_BYTES.store(0, Ordering::Relaxed);
        LARGE_ALLOCS.store(0, Ordering::Relaxed);
        MAX_ALLOC.store(0, Ordering::Relaxed);
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    #[test]
    fn warm_matvec_does_not_allocate_payload_sized_temporary() {
        // 1024x256: the F32 payload is 1 MiB, so even one payload-sized stack
        // replacement (a Vec<f32> row/block cache) would be unmistakable.
        const ROWS: usize = 1024;
        const COLS: usize = 256;
        let input = vec![0.25f32; COLS];
        let q8_input = vec![0.25f32; COLS];

        let f32_weights = f32_bytes(&vec![0.5f32; ROWS * COLS]);
        let f16_weights = f16_bytes(&vec![0.5f32; ROWS * COLS]);
        let bf16_weights = bf16_bytes(&vec![0.5f32; ROWS * COLS]);
        let q8_weights = q8_bytes(f16::from_f32(0.5), &vec![1i8; ROWS * COLS]);
        let q4_values = vec![-8i8; ROWS * COLS];
        let q4_weights = q4_bytes(f16::from_f32(0.5), &q4_values);

        // One canonical 256-element block per row for each K-quant. The
        // payloads are 84..210 KiB, so a Vec<f32> block cache would be large.
        let k_one = f16::from_f32(1.0);
        let k_zero = f16::from_f32(0.0);
        let mut q3_scales = [0xaa_u8; 12];
        q3_scales[0] = 0x21;
        q3_scales[8] = 0x8a;
        let mut q4_scales = [0u8; 12];
        q4_scales[0] = 0x10;
        let q2_weights = repeat_block(
            &q2_k_block(&[0x11u8; 16], &[0xe4u8; 64], k_one, k_zero),
            ROWS,
        );
        let q3_weights = repeat_block(
            &q3_k_block(&[0xffu8; 32], &[0xe4u8; 64], &q3_scales, k_one),
            ROWS,
        );
        let q4_k_weights =
            repeat_block(&q4_k_block(k_one, k_zero, &q4_scales, &[0x21u8; 128]), ROWS);
        let q5_weights = repeat_block(
            &q5_k_block(k_one, k_zero, &q4_scales, &[0x55u8; 32], &[0x21u8; 128]),
            ROWS,
        );
        let q6_weights = repeat_block(
            &q6_k_block(&[0x21u8; 128], &[0xedu8; 64], &[4i8; 16], k_one),
            ROWS,
        );
        let mut out = vec![0.0f32; ROWS];

        // Warm up SIMD dispatch and all kernels before taking counter readings.
        for _ in 0..2 {
            matvec_f32(&f32_weights, &input, &mut out).expect("warm f32");
            matvec_f16(&f16_weights, &input, &mut out).expect("warm f16");
            matvec_bf16(&bf16_weights, &input, &mut out).expect("warm bf16");
            matvec_q8_0(&q8_weights, &q8_input, &mut out).expect("warm q8_0");
            matvec_q4_0(&q4_weights, &q8_input, &mut out).expect("warm q4_0");
            matvec_q2_k(&q2_weights, &input, &mut out).expect("warm q2_k");
            matvec_q3_k(&q3_weights, &input, &mut out).expect("warm q3_k");
            matvec_q4_k(&q4_k_weights, &input, &mut out).expect("warm q4_k");
            matvec_q5_k(&q5_weights, &input, &mut out).expect("warm q5_k");
            matvec_q6_k(&q6_weights, &input, &mut out).expect("warm q6_k");
        }

        reset_counters();
        for _ in 0..4 {
            assert!(matvec_f32(black_box(&f32_weights), &input, &mut out).is_ok());
            assert!(matvec_f16(black_box(&f16_weights), &input, &mut out).is_ok());
            assert!(matvec_bf16(black_box(&bf16_weights), &input, &mut out).is_ok());
            assert!(matvec_q8_0(black_box(&q8_weights), &q8_input, &mut out).is_ok());
            assert!(matvec_q4_0(black_box(&q4_weights), &q8_input, &mut out).is_ok());
            assert!(matvec_q2_k(black_box(&q2_weights), &input, &mut out).is_ok());
            assert!(matvec_q3_k(black_box(&q3_weights), &input, &mut out).is_ok());
            assert!(matvec_q4_k(black_box(&q4_k_weights), &input, &mut out).is_ok());
            assert!(matvec_q5_k(black_box(&q5_weights), &input, &mut out).is_ok());
            assert!(matvec_q6_k(black_box(&q6_weights), &input, &mut out).is_ok());
        }

        let allocated = ALLOC_BYTES.load(Ordering::Relaxed);
        let large = LARGE_ALLOCS.load(Ordering::Relaxed);
        let largest = MAX_ALLOC.load(Ordering::Relaxed);
        let payload_bytes = ROWS * COLS * 4;
        assert!(
            large == 0,
            "warm matvec made {large} allocation(s) of >= {LARGE_ALLOC_BYTES} bytes"
        );
        assert!(
            largest < payload_bytes / 4,
            "largest warm allocation was {largest} bytes"
        );
        assert!(
            allocated < payload_bytes / 2,
            "warm matvec allocated {allocated} bytes across 40 calls"
        );
    }
}
