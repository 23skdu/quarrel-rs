//! Unit tests, including a scalar oracle comparison for every dispatched kernel.

use proptest::prelude::*;

use super::{
    Backend, SimdError, argmax, detected_backend, dot_f32, dot_f32_into, scalar, softmax_into,
};

pub(crate) fn ok<T>(result: Result<T, SimdError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("unexpected SimdError: {error}"),
    }
}

pub(crate) fn assert_close(got: f32, expected: f32, scale: f32) {
    let tolerance = 1.0e-5 * (1.0 + scale);
    assert!(
        (got - expected).abs() <= tolerance,
        "got {got}, expected {expected}, tolerance {tolerance}"
    );
}

pub(crate) fn check_dot(lhs: &[f32], rhs: &[f32], got: f32) {
    let expected = scalar::dot(lhs, rhs);
    let scale: f32 = lhs.iter().zip(rhs).map(|(&a, &b)| (a * b).abs()).sum();
    assert_close(got, expected, scale);
}

pub(crate) fn check_mul(lhs: &[f32], rhs: &[f32], got: &[f32]) {
    assert_eq!(got.len(), lhs.len());
    let mut expected = vec![0.0_f32; lhs.len()];
    scalar::mul_into(lhs, rhs, &mut expected);
    for (index, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close(a, b, (lhs[index] * rhs[index]).abs());
    }
}

pub(crate) fn check_softmax(logits: &[f32], got: &[f32]) {
    assert_eq!(got.len(), logits.len());
    let mut expected = vec![0.0_f32; logits.len()];
    scalar::softmax_into(logits, &mut expected);
    for (index, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite() && a >= 0.0,
            "invalid probability at {index}: {a}"
        );
        assert_close(a, b, 1.0);
    }
    let total: f32 = got.iter().sum();
    assert_close(total, 1.0, 1.0);
}

pub(crate) fn check_argmax(values: &[f32], got: usize) {
    assert!(got < values.len(), "index {got} out of bounds");
    assert_eq!(got, scalar::argmax(values));
    let best = values[got];
    assert!(values.iter().all(|&value| value <= best));
    assert!(
        values[..got].iter().all(|&value| value < best),
        "an earlier element ties the maximum"
    );
}

fn cases() -> Vec<(Vec<f32>, Vec<f32>)> {
    let lengths = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 129,
    ];
    lengths
        .iter()
        .map(|&len| {
            let lhs: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25 - 17.0).collect();
            let rhs: Vec<f32> = (0..len).map(|i| (i as f32) * -0.5 + 9.0).collect();
            (lhs, rhs)
        })
        .collect()
}

#[test]
fn scalar_kernels_match_oracle() {
    for (lhs, rhs) in cases() {
        check_dot(&lhs, &rhs, scalar::dot(&lhs, &rhs));
        let mut product = vec![0.0_f32; lhs.len()];
        scalar::mul_into(&lhs, &rhs, &mut product);
        check_mul(&lhs, &rhs, &product);
        if !lhs.is_empty() {
            let mut probs = vec![0.0_f32; lhs.len()];
            scalar::softmax_into(&lhs, &mut probs);
            check_softmax(&lhs, &probs);
            check_argmax(&lhs, scalar::argmax(&lhs));
        }
    }
}

#[test]
fn public_api_happy_paths() {
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [5.0_f32, 6.0, 7.0, 8.0];
    assert_close(ok(dot_f32(&lhs, &rhs)), 70.0, 70.0);
    assert_eq!(ok(dot_f32(&[], &[])), 0.0);

    let mut product = [0.0_f32; 4];
    ok(dot_f32_into(&lhs, &rhs, &mut product));
    assert_eq!(product, [5.0, 12.0, 21.0, 32.0]);
    ok(dot_f32_into(&[], &[], &mut []));

    let mut probs = [0.0_f32; 2];
    ok(softmax_into(&[-1.0, 1.0], &mut probs));
    assert_close(probs[0], 1.0 / (1.0 + std::f32::consts::E.powi(2)), 1.0);
    assert_close(probs[0] + probs[1], 1.0, 1.0);
    ok(softmax_into(&[], &mut []));

    assert_eq!(ok(argmax(&[1.0, 3.0, 3.0, 2.0])), 1);
}

#[test]
fn error_paths() {
    assert_eq!(
        dot_f32(&[1.0], &[1.0, 2.0]),
        Err(SimdError::LengthMismatch { lhs: 1, rhs: 2 })
    );
    let mut product = [0.0_f32; 2];
    assert_eq!(
        dot_f32_into(&[1.0], &[1.0, 2.0], &mut product),
        Err(SimdError::LengthMismatch { lhs: 1, rhs: 2 })
    );
    let mut short = [0.0_f32; 1];
    assert_eq!(
        dot_f32_into(&[1.0, 2.0], &[1.0, 2.0], &mut short),
        Err(SimdError::OutputLengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        softmax_into(&[1.0, 2.0], &mut short),
        Err(SimdError::OutputLengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    for (index, value) in [(0, f32::NAN), (1, f32::INFINITY), (2, f32::NEG_INFINITY)] {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        let mut with_nan = logits;
        with_nan[index] = value;
        for result in [
            softmax_into(&with_nan, &mut out).err(),
            argmax(&with_nan).err(),
        ] {
            match result {
                Some(SimdError::NonFinite {
                    index: found,
                    value: found_value,
                }) => {
                    assert_eq!(found, index);
                    if value.is_nan() {
                        assert!(found_value.is_nan());
                    } else {
                        assert_eq!(found_value, value);
                    }
                }
                other => panic!("expected NonFinite, got {other:?}"),
            }
        }
    }
    assert_eq!(argmax(&[]), Err(SimdError::EmptyInput));
}

#[test]
fn softmax_is_stable_for_large_finite_inputs() {
    let logits = [f32::MAX, f32::MAX, 0.0];
    let mut out = [0.0_f32; 3];
    ok(softmax_into(&logits, &mut out));
    assert!(out.iter().all(|value| value.is_finite() && *value >= 0.0));
    assert_close(out[0], 0.5, 1.0);
    assert_close(out[1], 0.5, 1.0);
    assert_eq!(out[2], 0.0);
}

#[test]
fn backend_is_stable_and_matches_platform() {
    let backend = detected_backend();
    assert_eq!(backend, detected_backend());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            assert_eq!(backend, Backend::Avx2);
        } else if std::arch::is_x86_feature_detected!("sse2") {
            assert_eq!(backend, Backend::Sse2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            assert_eq!(backend, Backend::Neon);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    assert_eq!(backend, Backend::Scalar);
}

#[cfg(target_arch = "x86_64")]
mod arch_x86 {
    use super::*;

    #[test]
    fn sse2_kernels_match_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        for (lhs, rhs) in cases() {
            // SAFETY: SSE2 was detected above and the slices are valid.
            let got = unsafe { crate::x86::dot_sse2(&lhs, &rhs) };
            check_dot(&lhs, &rhs, got);

            let mut product = vec![0.0_f32; lhs.len()];
            // SAFETY: SSE2 was detected; input lengths match and chunks are in bounds.
            unsafe { crate::x86::mul_into_sse2(&lhs, &rhs, &mut product) };
            check_mul(&lhs, &rhs, &product);

            if !lhs.is_empty() {
                let mut probs = vec![0.0_f32; lhs.len()];
                // SAFETY: SSE2 was detected; inputs are finite, non-empty and equal-length.
                unsafe { crate::x86::softmax_into_sse2(&lhs, &mut probs) };
                check_softmax(&lhs, &probs);

                // SAFETY: SSE2 was detected; the input is finite and non-empty.
                let index = unsafe { crate::x86::argmax_sse2(&lhs) };
                check_argmax(&lhs, index);
            }
        }
    }

    #[test]
    fn avx2_fma_kernels_match_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        for (lhs, rhs) in cases() {
            // SAFETY: AVX2 and FMA were detected above and the slices are valid.
            let got = unsafe { crate::x86::dot_avx2(&lhs, &rhs) };
            check_dot(&lhs, &rhs, got);

            let mut product = vec![0.0_f32; lhs.len()];
            // SAFETY: AVX2/FMA were detected; lengths match and chunks are in bounds.
            unsafe { crate::x86::mul_into_avx2(&lhs, &rhs, &mut product) };
            check_mul(&lhs, &rhs, &product);

            if !lhs.is_empty() {
                let mut probs = vec![0.0_f32; lhs.len()];
                // SAFETY: AVX2/FMA were detected; inputs are finite, non-empty, equal-length.
                unsafe { crate::x86::softmax_into_avx2(&lhs, &mut probs) };
                check_softmax(&lhs, &probs);

                // SAFETY: AVX2/FMA were detected; the input is finite and non-empty.
                let index = unsafe { crate::x86::argmax_avx2(&lhs) };
                check_argmax(&lhs, index);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arch_neon {
    use super::*;

    #[test]
    fn neon_kernels_match_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for (lhs, rhs) in cases() {
            // SAFETY: NEON was detected above and the slices are valid.
            let got = unsafe { crate::aarch64::dot_neon(&lhs, &rhs) };
            check_dot(&lhs, &rhs, got);

            let mut product = vec![0.0_f32; lhs.len()];
            // SAFETY: NEON was detected; lengths match and chunks are in bounds.
            unsafe { crate::aarch64::mul_into_neon(&lhs, &rhs, &mut product) };
            check_mul(&lhs, &rhs, &product);

            if !lhs.is_empty() {
                let mut probs = vec![0.0_f32; lhs.len()];
                // SAFETY: NEON was detected; inputs are finite, non-empty, equal-length.
                unsafe { crate::aarch64::softmax_into_neon(&lhs, &mut probs) };
                check_softmax(&lhs, &probs);

                // SAFETY: NEON was detected; the input is finite and non-empty.
                let index = unsafe { crate::aarch64::argmax_neon(&lhs) };
                check_argmax(&lhs, index);
            }
        }
    }
}

fn finite_slice() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0e3_f32..1.0e3_f32, 0..129)
}

proptest! {
    #[test]
    fn dot_matches_scalar_for_arbitrary_finite_slices(
        lhs in finite_slice(), rhs in finite_slice()
    ) {
        let result = dot_f32(&lhs, &rhs);
        if lhs.len() != rhs.len() {
            prop_assert!(
                matches!(result, Err(SimdError::LengthMismatch { .. })),
                "unexpected result: {result:?}"
            );
        } else {
            let got = ok(result);
            prop_assert!(got.is_finite());
            check_dot(&lhs, &rhs, got);
            let expected = scalar::dot(&lhs, &rhs);
            let tolerance = 1.0e-4 * (1.0 + lhs.len() as f32 * 1.0e6);
            prop_assert!(
                (got - expected).abs() <= tolerance,
                "got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn dot_into_matches_scalar_for_arbitrary_finite_slices(
        lhs in finite_slice(), rhs in finite_slice()
    ) {
        let len = lhs.len().min(rhs.len());
        let mut out = vec![0.0_f32; len];
        if lhs.len() == rhs.len() {
            ok(dot_f32_into(&lhs, &rhs, &mut out));
            prop_assert_eq!(out.len(), lhs.len());
            prop_assert!(out.iter().all(|value| value.is_finite()));
            check_mul(&lhs, &rhs, &out);
        } else {
            let result = dot_f32_into(&lhs, &rhs, &mut out);
            prop_assert!(
                matches!(result, Err(SimdError::LengthMismatch { .. })),
                "unexpected result: {result:?}"
            );
        }
    }

    #[test]
    fn softmax_matches_scalar_for_arbitrary_finite_slices(logits in finite_slice()) {
        let len = logits.len();
        let mut out = vec![f32::NAN; len];
        ok(softmax_into(&logits, &mut out));
        prop_assert_eq!(out.len(), len);
        prop_assert!(out.iter().all(|value| value.is_finite() && *value >= 0.0));
        if len > 0 {
            let total: f32 = out.iter().sum();
            prop_assert!((total - 1.0).abs() < 1.0e-4);
            check_softmax(&logits, &out);
        }
    }

    #[test]
    fn argmax_matches_scalar_for_arbitrary_finite_slices(values in finite_slice()) {
        if values.is_empty() {
            prop_assert!(matches!(argmax(&values), Err(SimdError::EmptyInput)));
        } else {
            let index = ok(argmax(&values));
            prop_assert!(index < values.len());
            check_argmax(&values, index);
        }
    }

    #[test]
    fn length_errors_for_arbitrary_slices(
        lhs in finite_slice(), rhs in finite_slice()
    ) {
        if lhs.len() != rhs.len() {
            let result = dot_f32(&lhs, &rhs);
            prop_assert!(
                matches!(result, Err(SimdError::LengthMismatch { .. })),
                "unexpected result: {result:?}"
            );
        }
    }
}
