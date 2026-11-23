//! Unit tests, including a scalar oracle comparison for every dispatched kernel.

use proptest::prelude::*;

use super::{
    Backend, KernelMode, KernelSet, SimdError, argmax, detected_backend, dot_f32, dot_f32_into,
    scalar, softmax_into,
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

/// Finite inputs that stress chunk tails, signed zero, subnormals, extremes
/// and cancellation without overflowing the dot-product accumulator.
fn edge_cases() -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (vec![], vec![]),
        (vec![0.0_f32; 17], vec![-0.0_f32; 17]),
        (vec![1.0e-40_f32; 19], vec![1.0e-40_f32; 19]),
        (
            vec![
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                0.0,
                -0.0,
                f32::MIN_POSITIVE,
            ],
            vec![
                f32::MIN_POSITIVE,
                f32::MIN_POSITIVE,
                1.0,
                1.0,
                -f32::MIN_POSITIVE,
            ],
        ),
        (
            vec![1.0e18, -1.0e18, 1.0e18, -1.0e18, 1.0e18],
            vec![1.0e18; 5],
        ),
        (vec![f32::MAX, -f32::MAX, 1.0, -1.0, f32::MAX], vec![1.0; 5]),
    ]
}

/// Finite logits that stress stable softmax, ties and argmax edge cases.
fn softmax_edge_logits() -> Vec<Vec<f32>> {
    vec![
        vec![f32::MAX, f32::MAX, 0.0, -f32::MAX],
        vec![
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            0.0,
            -0.0,
        ],
        vec![0.0_f32; 17],
        vec![5.0, 5.0, 5.0, 5.0, 4.999_999],
        vec![1.0, f32::MAX, 1.0 - f32::EPSILON, 0.0, -f32::MAX],
        vec![-1.0e30, -1.0e30, -2.0e30, -1.0e30 + 1.0e22],
    ]
}

fn all_modes() -> [KernelMode; 6] {
    [
        KernelMode::Auto,
        KernelMode::Scalar,
        KernelMode::Sse2,
        KernelMode::Avx2,
        KernelMode::Avx512,
        KernelMode::Neon,
    ]
}

fn check_kernels(set: KernelSet, lhs: &[f32], rhs: &[f32]) {
    check_dot(lhs, rhs, ok(set.dot(lhs, rhs)));

    let mut product = vec![0.0_f32; lhs.len()];
    ok(set.dot_into(lhs, rhs, &mut product));
    check_mul(lhs, rhs, &product);

    if !lhs.is_empty() {
        let mut probs = vec![0.0_f32; lhs.len()];
        ok(set.softmax_into(lhs, &mut probs));
        check_softmax(lhs, &probs);
        check_argmax(lhs, ok(set.argmax(lhs)));
    }
}

fn check_softmax_edges(set: KernelSet) {
    for logits in softmax_edge_logits() {
        let mut probs = vec![0.0_f32; logits.len()];
        ok(set.softmax_into(&logits, &mut probs));
        check_softmax(&logits, &probs);
        check_argmax(&logits, ok(set.argmax(&logits)));
    }
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
        if std::arch::is_x86_feature_detected!("avx512f") {
            assert_eq!(backend, Backend::Avx512);
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
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

#[test]
fn all_available_kernels_match_scalar_oracle() {
    for mode in all_modes() {
        let Some(set) = KernelSet::for_mode(mode) else {
            continue;
        };
        for (lhs, rhs) in cases().into_iter().chain(edge_cases()) {
            check_kernels(set, &lhs, &rhs);
        }
        check_softmax_edges(set);
    }
}

#[test]
fn dispatch_forces_each_mode_and_is_stable() {
    let lhs: Vec<f32> = (0..19).map(|i| (i as f32) * 0.375 - 3.0).collect();
    let rhs: Vec<f32> = (0..19).map(|i| (i as f32) * -0.125 + 1.5).collect();

    assert_eq!(KernelSet::auto().backend(), detected_backend());
    let scalar = KernelSet::for_mode(KernelMode::Scalar).expect("scalar is always available");
    let expected_product = {
        let mut out = vec![0.0_f32; lhs.len()];
        ok(scalar.dot_into(&lhs, &rhs, &mut out));
        out
    };
    let expected_dot = ok(scalar.dot(&lhs, &rhs));
    let expected_argmax = ok(scalar.argmax(&lhs));

    for mode in all_modes() {
        let Some(set) = KernelSet::for_mode(mode) else {
            assert_eq!(mode.resolve(), None, "unexpected unavailable mode {mode:?}");
            continue;
        };
        assert_eq!(Some(set.backend()), mode.resolve());

        let dot = ok(set.dot(&lhs, &rhs));
        assert_eq!(dot, ok(set.dot(&lhs, &rhs)), "unstable dot in {mode:?}");
        check_dot(&lhs, &rhs, dot);
        assert_close(dot, expected_dot, expected_dot.abs());

        let mut product = vec![0.0_f32; lhs.len()];
        ok(set.dot_into(&lhs, &rhs, &mut product));
        let mut again = vec![0.0_f32; lhs.len()];
        ok(set.dot_into(&lhs, &rhs, &mut again));
        assert_eq!(product, again, "unstable dot_into in {mode:?}");
        assert_eq!(product, expected_product);

        let mut probs = vec![0.0_f32; lhs.len()];
        ok(set.softmax_into(&lhs, &mut probs));
        let mut probs_again = vec![0.0_f32; lhs.len()];
        ok(set.softmax_into(&lhs, &mut probs_again));
        assert_eq!(probs, probs_again, "unstable softmax in {mode:?}");
        check_softmax(&lhs, &probs);

        let index = ok(set.argmax(&lhs));
        assert_eq!(index, ok(set.argmax(&lhs)), "unstable argmax in {mode:?}");
        assert_eq!(index, expected_argmax);
    }
}

#[test]
fn forced_modes_report_typed_errors() {
    for mode in all_modes() {
        let Some(set) = KernelSet::for_mode(mode) else {
            continue;
        };
        assert_eq!(
            set.dot(&[1.0], &[1.0, 2.0]),
            Err(SimdError::LengthMismatch { lhs: 1, rhs: 2 })
        );
        let mut out = [0.0_f32; 1];
        assert_eq!(
            set.dot_into(&[1.0, 2.0], &[1.0, 2.0], &mut out),
            Err(SimdError::OutputLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(set.argmax(&[]), Err(SimdError::EmptyInput));
        let mut probs = [0.0_f32; 1];
        assert!(matches!(
            set.softmax_into(&[f32::NAN], &mut probs),
            Err(SimdError::NonFinite { index: 0, .. })
        ));
    }
}

#[cfg(target_arch = "x86_64")]
mod arch_x86 {
    use super::*;

    #[test]
    fn sse2_kernels_match_scalar_reference() {
        let Some(set) = KernelSet::for_mode(KernelMode::Sse2) else {
            return;
        };
        for (lhs, rhs) in cases().into_iter().chain(edge_cases()) {
            check_kernels(set, &lhs, &rhs);
        }
        check_softmax_edges(set);
    }

    #[test]
    fn avx2_fma_kernels_match_scalar_reference() {
        let Some(set) = KernelSet::for_mode(KernelMode::Avx2) else {
            return;
        };
        for (lhs, rhs) in cases().into_iter().chain(edge_cases()) {
            check_kernels(set, &lhs, &rhs);
        }
        check_softmax_edges(set);
    }

    #[test]
    fn avx512_kernels_match_scalar_reference() {
        let Some(set) = KernelSet::for_mode(KernelMode::Avx512) else {
            return;
        };
        for (lhs, rhs) in cases().into_iter().chain(edge_cases()) {
            check_kernels(set, &lhs, &rhs);
        }
        check_softmax_edges(set);
    }
}

#[cfg(target_arch = "aarch64")]
mod arch_neon {
    use super::*;

    #[test]
    fn neon_kernels_match_scalar_reference() {
        let Some(set) = KernelSet::for_mode(KernelMode::Neon) else {
            return;
        };
        for (lhs, rhs) in cases().into_iter().chain(edge_cases()) {
            check_kernels(set, &lhs, &rhs);
        }
        check_softmax_edges(set);
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
