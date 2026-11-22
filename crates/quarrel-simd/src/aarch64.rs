//! AArch64 NEON kernels. All unsafe operations are confined here.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::{
    vaddq_f32, vaddvq_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vmulq_f32,
    vst1q_f32,
};

use crate::Kernels;

/// Exact vector maximum over four-element chunks plus scalar tail.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn max_neon(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    for chunk in chunks {
        // SAFETY: every chunk has four initialised floats and NEON is enabled.
        let v = unsafe { vld1q_f32(chunk.as_ptr()) };
        acc = vmaxq_f32(acc, v);
    }
    let mut best = vmaxvq_f32(acc);
    for &value in tail {
        if value > best {
            best = value;
        }
    }
    best
}

/// NEON dot product; inputs are assumed to have equal length.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
    let (chunks, tail) = lhs.as_chunks::<4>();
    let (rhs_chunks, _) = rhs.as_chunks::<4>();
    let mut acc = vdupq_n_f32(0.0);
    for (a, b) in chunks.iter().zip(rhs_chunks) {
        // SAFETY: both chunks hold four initialised floats; NEON is enabled.
        let av = unsafe { vld1q_f32(a.as_ptr()) };
        // SAFETY: `b` holds four initialised floats; NEON is enabled.
        let bv = unsafe { vld1q_f32(b.as_ptr()) };
        acc = vaddq_f32(acc, vmulq_f32(av, bv));
    }
    let mut sum = vaddvq_f32(acc);
    let rhs_tail = &rhs[rhs.len() - tail.len()..];
    for (&a, &b) in tail.iter().zip(rhs_tail) {
        sum += a * b;
    }
    sum
}

/// NEON elementwise multiply.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mul_into_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let (src_chunks, src_tail) = lhs.as_chunks::<4>();
    let (rhs_chunks, rhs_tail) = rhs.as_chunks::<4>();
    let (dst_chunks, dst_tail) = out.as_chunks_mut::<4>();
    for ((a, b), d) in src_chunks.iter().zip(rhs_chunks).zip(dst_chunks.iter_mut()) {
        // SAFETY: each chunk holds four floats and they are disjoint; NEON is enabled.
        let av = unsafe { vld1q_f32(a.as_ptr()) };
        // SAFETY: `b` holds four initialised floats; NEON is enabled.
        let bv = unsafe { vld1q_f32(b.as_ptr()) };
        let product = vmulq_f32(av, bv);
        // SAFETY: `d` is a writable four-float chunk, disjoint from the inputs.
        unsafe { vst1q_f32(d.as_mut_ptr(), product) };
    }
    for (slot, (&a, &b)) in dst_tail.iter_mut().zip(src_tail.iter().zip(rhs_tail)) {
        *slot = a * b;
    }
}

/// NEON softmax. Inputs are validated by the public API.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn softmax_into_neon(logits: &[f32], out: &mut [f32]) {
    // SAFETY: NEON is enabled and the input is finite, non-empty and matches `out`.
    let shift = unsafe { max_neon(logits) };
    let mut total = 0.0_f32;
    for (&logit, slot) in logits.iter().zip(out.iter_mut()) {
        let exp = (logit - shift).exp();
        *slot = exp;
        total += exp;
    }
    let scale = vdupq_n_f32(1.0 / total);
    let (chunks, tail) = out.as_chunks_mut::<4>();
    for chunk in chunks {
        // SAFETY: the chunk is writable and holds four elements; NEON is enabled.
        let v = unsafe { vld1q_f32(chunk.as_ptr()) };
        let scaled = vdivq_f32(v, scale);
        // SAFETY: the same writable four-float chunk; no overlapping loads remain.
        unsafe { vst1q_f32(chunk.as_mut_ptr(), scaled) };
    }
    for slot in tail {
        *slot /= total;
    }
}

/// NEON argmax; ties pick the lowest index.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn argmax_neon(values: &[f32]) -> usize {
    // SAFETY: NEON is enabled and the input is finite and non-empty.
    let best = unsafe { max_neon(values) };
    values.iter().position(|&value| value == best).unwrap_or(0)
}

pub(crate) fn neon_kernels() -> Kernels {
    Kernels {
        dot: |lhs, rhs| {
            // SAFETY: selected only after NEON feature detection.
            unsafe { dot_neon(lhs, rhs) }
        },
        mul_into: |lhs, rhs, out| {
            // SAFETY: selected only after NEON feature detection.
            unsafe { mul_into_neon(lhs, rhs, out) }
        },
        softmax_into: |logits, out| {
            // SAFETY: selected only after NEON feature detection.
            unsafe { softmax_into_neon(logits, out) }
        },
        argmax: |values| {
            // SAFETY: selected only after NEON feature detection.
            unsafe { argmax_neon(values) }
        },
    }
}
