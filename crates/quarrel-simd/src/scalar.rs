//! Portable scalar kernels; also the independent oracle used by the tests.

use crate::Kernels;

/// Naive left-to-right dot product: the reference accumulation order.
pub(crate) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (&a, &b) in lhs.iter().zip(rhs) {
        sum += a * b;
    }
    sum
}

/// Naive elementwise multiply, used as the reference for `dot_f32_into`.
pub(crate) fn mul_into(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    for (slot, (&a, &b)) in out.iter_mut().zip(lhs.iter().zip(rhs)) {
        *slot = a * b;
    }
}

/// Exact scalar maximum. Called only with finite, non-empty input.
pub(crate) fn max(values: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for &value in values {
        if value > best {
            best = value;
        }
    }
    best
}

/// Stable softmax reference: shift by the maximum, exponentiate, normalise.
pub(crate) fn softmax_into(logits: &[f32], out: &mut [f32]) {
    let shift = max(logits);
    let mut total = 0.0_f32;
    for (&logit, slot) in logits.iter().zip(out.iter_mut()) {
        let exp = (logit - shift).exp();
        *slot = exp;
        total += exp;
    }
    for slot in out.iter_mut() {
        *slot /= total;
    }
}

/// Reference argmax using strict comparisons, hence lowest-index ties.
pub(crate) fn argmax(values: &[f32]) -> usize {
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0;
    for (index, &value) in values.iter().enumerate() {
        if value > best {
            best = value;
            best_index = index;
        }
    }
    best_index
}

pub(crate) fn kernels() -> Kernels {
    Kernels {
        dot,
        mul_into,
        softmax_into,
        argmax,
    }
}
