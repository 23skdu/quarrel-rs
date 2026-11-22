//! Allocation-free matrix–vector products over borrowed GGUF weight bytes.
//!
//! Every kernel interprets `weights` as a row-major matrix with `out.len()`
//! rows and `input.len()` columns, computes one dot product per row and writes
//! the result to the matching entry of `out`. Nothing is materialised between
//! the borrowed bytes and the caller's output buffer: unquantized rows are
//! decoded into a small stack chunk, and quantized blocks are decoded into a
//! fixed-size `[f32; 32]` and dotted immediately with the matching input slice.
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
//! A quantized row therefore occupies `cols / 32 * block_bytes` bytes and
//! `cols` must be a nonzero multiple of 32. K-quants (Q2_K, Q3_K, Q4_K, Q5_K,
//! Q6_K) are deliberately *not* guessed at: [`matvec`] reports them through
//! [`GgufError::UnsupportedDType`].
//!
//! Floating-point inputs and decoded weights must be finite; the first
//! non-finite value is reported as [`GgufError::NonFiniteValue`]. Dot products
//! use [`quarrel_simd::dot_f32`], so the summation order (and thus the last few
//! ULPs) depends on the SIMD backend selected for the process. Unquantized
//! rows are accumulated in 64-element chunks.

use crate::DType;
use crate::error::GgufError;
use half::{bf16, f16};
use quarrel_simd::dot_f32;

/// Elements decoded into one stack buffer before calling `dot_f32`.
const CHUNK_ELEMENTS: usize = 64;
/// Logical elements in one Q8_0/Q4_0 block.
const BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_0_BLOCK_BYTES: usize = 18;

type DecodeFn = fn(&[u8], &mut [f32]);

/// Dispatch a matvec to the kernel for `dtype`.
///
/// F32, F16, BF16, Q8_0 and Q4_0 are supported. K-quants and unknown types
/// return [`GgufError::UnsupportedDType`] without reading the buffers.
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
    }

    #[test]
    fn unsupported_k_quant_dtypes_are_reported() {
        for dtype in [
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Unknown(999),
        ] {
            let result = matvec(dtype, &[0u8; 144], &[1.0f32; 256], &mut [0.0f32; 1]);
            assert!(
                matches!(result, Err(GgufError::UnsupportedDType(actual)) if actual == dtype),
                "unsupported dtype {dtype:?} was not reported cleanly"
            );
        }
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
        let mut out = vec![0.0f32; ROWS];

        // Warm up SIMD dispatch and all kernels before taking counter readings.
        for _ in 0..2 {
            matvec_f32(&f32_weights, &input, &mut out).expect("warm f32");
            matvec_f16(&f16_weights, &input, &mut out).expect("warm f16");
            matvec_bf16(&bf16_weights, &input, &mut out).expect("warm bf16");
            matvec_q8_0(&q8_weights, &q8_input, &mut out).expect("warm q8_0");
            matvec_q4_0(&q4_weights, &q8_input, &mut out).expect("warm q4_0");
        }

        reset_counters();
        for _ in 0..4 {
            assert!(matvec_f32(black_box(&f32_weights), &input, &mut out).is_ok());
            assert!(matvec_f16(black_box(&f16_weights), &input, &mut out).is_ok());
            assert!(matvec_bf16(black_box(&bf16_weights), &input, &mut out).is_ok());
            assert!(matvec_q8_0(black_box(&q8_weights), &q8_input, &mut out).is_ok());
            assert!(matvec_q4_0(black_box(&q4_weights), &q8_input, &mut out).is_ok());
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
            "warm matvec allocated {allocated} bytes across 20 calls"
        );
    }
}
