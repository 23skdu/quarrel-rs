//! x86-64 SSE2 and AVX2/FMA kernels. All unsafe operations are confined here.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
    _mm_storeu_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
};

use crate::Kernels;

/// Horizontal sum of four lanes.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn sum_sse2(lanes: __m128) -> f32 {
    let mut stored = [0.0_f32; 4];
    // SAFETY: `stored` is a writable, 16-byte-sized array and SSE2 is enabled.
    unsafe { _mm_storeu_ps(stored.as_mut_ptr(), lanes) };
    stored.iter().sum()
}

/// Exact vector maximum over four-element chunks plus scalar tail.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn max_sse2(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut acc = _mm_set1_ps(f32::NEG_INFINITY);
    for chunk in chunks {
        // SAFETY: every chunk has four initialised floats and SSE2 is enabled.
        let v = unsafe { _mm_loadu_ps(chunk.as_ptr()) };
        acc = _mm_max_ps(acc, v);
    }
    let mut stored = [f32::NEG_INFINITY; 4];
    // SAFETY: `stored` has room for four floats; SSE2 is enabled here.
    unsafe { _mm_storeu_ps(stored.as_mut_ptr(), acc) };
    let mut best = f32::NEG_INFINITY;
    for lane in stored {
        if lane > best {
            best = lane;
        }
    }
    for &value in tail {
        if value > best {
            best = value;
        }
    }
    best
}

/// SSE2 dot product; inputs are assumed to have equal length.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn dot_sse2(lhs: &[f32], rhs: &[f32]) -> f32 {
    let (chunks, tail) = lhs.as_chunks::<4>();
    let (rhs_chunks, _) = rhs.as_chunks::<4>();
    let mut acc = _mm_setzero_ps();
    for (a, b) in chunks.iter().zip(rhs_chunks) {
        // SAFETY: both chunks hold four initialised floats; SSE2 is enabled.
        let av = unsafe { _mm_loadu_ps(a.as_ptr()) };
        // SAFETY: `b` holds four initialised floats; SSE2 is enabled.
        let bv = unsafe { _mm_loadu_ps(b.as_ptr()) };
        acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
    }
    // SAFETY: SSE2 is enabled by this function's target feature.
    let mut sum = unsafe { sum_sse2(acc) };
    let rhs_tail = &rhs[rhs.len() - tail.len()..];
    for (&a, &b) in tail.iter().zip(rhs_tail) {
        sum += a * b;
    }
    sum
}

/// SSE2 elementwise multiply.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn mul_into_sse2(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let (src_chunks, src_tail) = lhs.as_chunks::<4>();
    let (rhs_chunks, rhs_tail) = rhs.as_chunks::<4>();
    let (dst_chunks, dst_tail) = out.as_chunks_mut::<4>();
    for ((a, b), d) in src_chunks.iter().zip(rhs_chunks).zip(dst_chunks.iter_mut()) {
        // SAFETY: each chunk holds four floats and they are disjoint; SSE2 is enabled.
        let av = unsafe { _mm_loadu_ps(a.as_ptr()) };
        // SAFETY: `b` holds four initialised floats; SSE2 is enabled.
        let bv = unsafe { _mm_loadu_ps(b.as_ptr()) };
        let product = _mm_mul_ps(av, bv);
        // SAFETY: `d` is a writable four-float chunk, disjoint from the inputs.
        unsafe { _mm_storeu_ps(d.as_mut_ptr(), product) };
    }
    for (slot, (&a, &b)) in dst_tail.iter_mut().zip(src_tail.iter().zip(rhs_tail)) {
        *slot = a * b;
    }
}

/// SSE2 softmax. Inputs are validated (finite, equal lengths, non-empty) by
/// the public API before this kernel runs.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn softmax_into_sse2(logits: &[f32], out: &mut [f32]) {
    // SAFETY: SSE2 is enabled and the input preconditions hold.
    let shift = unsafe { max_sse2(logits) };
    let mut total = 0.0_f32;
    for (&logit, slot) in logits.iter().zip(out.iter_mut()) {
        let exp = (logit - shift).exp();
        *slot = exp;
        total += exp;
    }
    let scale = _mm_set1_ps(1.0 / total);
    let (chunks, tail) = out.as_chunks_mut::<4>();
    for chunk in chunks {
        // SAFETY: the chunk is writable and holds four elements; SSE2 is enabled.
        let v = unsafe { _mm_loadu_ps(chunk.as_ptr()) };
        let scaled = _mm_mul_ps(v, scale);
        // SAFETY: the same writable four-float chunk; no overlapping loads remain.
        unsafe { _mm_storeu_ps(chunk.as_mut_ptr(), scaled) };
    }
    for slot in tail {
        *slot /= total;
    }
}

/// SSE2 argmax; ties pick the lowest index.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn argmax_sse2(values: &[f32]) -> usize {
    // SAFETY: SSE2 is enabled and the input is finite and non-empty.
    let best = unsafe { max_sse2(values) };
    values.iter().position(|&value| value == best).unwrap_or(0)
}

/// Horizontal sum of eight lanes.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sum_avx2(lanes: __m256) -> f32 {
    let mut stored = [0.0_f32; 8];
    // SAFETY: `stored` has room for eight floats and AVX2 is enabled.
    unsafe { _mm256_storeu_ps(stored.as_mut_ptr(), lanes) };
    stored.iter().sum()
}

/// Exact vector maximum over eight-element chunks plus scalar tail.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn max_avx2(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<8>();
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    for chunk in chunks {
        // SAFETY: every chunk has eight initialised floats and AVX2 is enabled.
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        acc = _mm256_max_ps(acc, v);
    }
    let mut stored = [f32::NEG_INFINITY; 8];
    // SAFETY: `stored` has room for eight floats; AVX2 is enabled here.
    unsafe { _mm256_storeu_ps(stored.as_mut_ptr(), acc) };
    let mut best = f32::NEG_INFINITY;
    for lane in stored {
        if lane > best {
            best = lane;
        }
    }
    for &value in tail {
        if value > best {
            best = value;
        }
    }
    best
}

/// AVX2/FMA dot product; inputs are assumed to have equal length.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    let (chunks, tail) = lhs.as_chunks::<8>();
    let (rhs_chunks, _) = rhs.as_chunks::<8>();
    let mut acc = _mm256_setzero_ps();
    for (a, b) in chunks.iter().zip(rhs_chunks) {
        // SAFETY: both chunks hold eight initialised floats; AVX2 is enabled.
        let av = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        // SAFETY: `b` holds eight initialised floats; AVX2 is enabled.
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        // FMA is enabled by this function's target feature.
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    // SAFETY: AVX2/FMA are enabled by this function's target feature.
    let mut sum = unsafe { sum_avx2(acc) };
    let rhs_tail = &rhs[rhs.len() - tail.len()..];
    for (&a, &b) in tail.iter().zip(rhs_tail) {
        sum += a * b;
    }
    sum
}

/// AVX2 elementwise multiply.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn mul_into_avx2(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let (src_chunks, src_tail) = lhs.as_chunks::<8>();
    let (rhs_chunks, rhs_tail) = rhs.as_chunks::<8>();
    let (dst_chunks, dst_tail) = out.as_chunks_mut::<8>();
    for ((a, b), d) in src_chunks.iter().zip(rhs_chunks).zip(dst_chunks.iter_mut()) {
        // SAFETY: each chunk holds eight floats and they are disjoint; AVX2 is enabled.
        let av = unsafe { _mm256_loadu_ps(a.as_ptr()) };
        // SAFETY: `b` holds eight initialised floats; AVX2 is enabled.
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let product = _mm256_mul_ps(av, bv);
        // SAFETY: `d` is a writable eight-float chunk, disjoint from the inputs.
        unsafe { _mm256_storeu_ps(d.as_mut_ptr(), product) };
    }
    for (slot, (&a, &b)) in dst_tail.iter_mut().zip(src_tail.iter().zip(rhs_tail)) {
        *slot = a * b;
    }
}

/// AVX2 softmax. Inputs are validated by the public API.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn softmax_into_avx2(logits: &[f32], out: &mut [f32]) {
    // SAFETY: AVX2 is enabled and the input is finite, non-empty and matches `out`.
    let shift = unsafe { max_avx2(logits) };
    let mut total = 0.0_f32;
    for (&logit, slot) in logits.iter().zip(out.iter_mut()) {
        let exp = (logit - shift).exp();
        *slot = exp;
        total += exp;
    }
    let scale = _mm256_set1_ps(1.0 / total);
    let (chunks, tail) = out.as_chunks_mut::<8>();
    for chunk in chunks {
        // SAFETY: the chunk is writable and holds eight elements; AVX2 is enabled.
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        let scaled = _mm256_mul_ps(v, scale);
        // SAFETY: the same writable eight-float chunk; no overlapping loads remain.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), scaled) };
    }
    for slot in tail {
        *slot /= total;
    }
}

/// AVX2 argmax; ties pick the lowest index.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn argmax_avx2(values: &[f32]) -> usize {
    // SAFETY: AVX2 is enabled and the input is finite and non-empty.
    let best = unsafe { max_avx2(values) };
    values.iter().position(|&value| value == best).unwrap_or(0)
}

pub(crate) fn sse2_kernels() -> Kernels {
    Kernels {
        dot: |lhs, rhs| {
            // SAFETY: selected only after SSE2 feature detection (baseline on x86-64).
            unsafe { dot_sse2(lhs, rhs) }
        },
        mul_into: |lhs, rhs, out| {
            // SAFETY: selected only after SSE2 feature detection (baseline on x86-64).
            unsafe { mul_into_sse2(lhs, rhs, out) }
        },
        softmax_into: |logits, out| {
            // SAFETY: selected only after SSE2 feature detection (baseline on x86-64).
            unsafe { softmax_into_sse2(logits, out) }
        },
        argmax: |values| {
            // SAFETY: selected only after SSE2 feature detection (baseline on x86-64).
            unsafe { argmax_sse2(values) }
        },
    }
}

pub(crate) fn avx2_kernels() -> Kernels {
    Kernels {
        dot: |lhs, rhs| {
            // SAFETY: selected only after AVX2 and FMA feature detection.
            unsafe { dot_avx2(lhs, rhs) }
        },
        mul_into: |lhs, rhs, out| {
            // SAFETY: selected only after AVX2 and FMA feature detection.
            unsafe { mul_into_avx2(lhs, rhs, out) }
        },
        softmax_into: |logits, out| {
            // SAFETY: selected only after AVX2 and FMA feature detection.
            unsafe { softmax_into_avx2(logits, out) }
        },
        argmax: |values| {
            // SAFETY: selected only after AVX2 and FMA feature detection.
            unsafe { argmax_avx2(values) }
        },
    }
}
