//! Scalar oracle and runtime-dispatched numeric kernels for `quarrel-rs`.
//!
//! Every public function validates its arguments and then delegates to a
//! backend chosen once, on first use, by [`detected_backend`]. The selected
//! backend never changes during the process: detection is cached in a
//! [`std::sync::OnceLock`], so a call cannot observe a different CPU feature
//! set halfway through a computation.
//!
//! All hand-written SIMD code lives in this crate. The portable scalar
//! implementations double as the reference oracle used by the tests; the
//! architecture-specific kernels must agree with them up to floating-point
//! rounding.
//!
//! ```
//! use quarrel_simd::{argmax, dot_f32, dot_f32_into, softmax_into};
//!
//! let lhs = [1.0_f32, 2.0, 3.0];
//! let rhs = [4.0_f32, 5.0, 6.0];
//! let mut product = [0.0_f32; 3];
//! let mut probs = [0.0_f32; 3];
//!
//! assert!(dot_f32(&lhs, &rhs).is_ok());
//! assert!(dot_f32_into(&lhs, &rhs, &mut product).is_ok());
//! assert!(softmax_into(&lhs, &mut probs).is_ok());
//! assert_eq!(argmax(&lhs).ok(), Some(2));
//! ```

#![deny(missing_docs)]

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use thiserror::Error;

/// SIMD backend selected for this process.
///
/// The variant reflects the *dispatch decision*, not merely what the CPU
/// happens to support: [`Backend::Scalar`] is also reported on targets with
/// no SIMD implementation in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar implementation (also the fallback on unknown CPUs).
    Scalar,
    /// x86-64 SSE2 (two-wide double, four-wide single precision).
    Sse2,
    /// x86-64 AVX2 with FMA.
    Avx2,
    /// AArch64 Advanced SIMD (NEON).
    Neon,
}

/// Returns the backend used by all kernels in this process.
///
/// Feature detection and the resulting choice are made exactly once, on the
/// first call (which may happen implicitly through any kernel). Later calls
/// return the same value.
pub fn detected_backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(detect_backend)
}

fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Backend::Avx2;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return Backend::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Backend::Neon;
        }
    }
    Backend::Scalar
}

/// Typed failure modes of the numeric kernels.
///
/// This enum is non-exhaustive; match with a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum SimdError {
    /// The two input slices have different lengths.
    #[error("length mismatch: lhs has {lhs} elements, rhs has {rhs}")]
    LengthMismatch {
        /// Length of the left-hand-side slice.
        lhs: usize,
        /// Length of the right-hand-side slice.
        rhs: usize,
    },

    /// The output slice has a different length from the input.
    #[error("output length mismatch: expected {expected}, got {actual}")]
    OutputLengthMismatch {
        /// Required number of output elements.
        expected: usize,
        /// Number of elements actually supplied.
        actual: usize,
    },

    /// The operation requires at least one element but received an empty slice.
    #[error("empty input is not allowed")]
    EmptyInput,

    /// A NaN or infinity was found where only finite values are accepted.
    #[error("non-finite value at index {index}: {value}")]
    NonFinite {
        /// Index of the offending element.
        index: usize,
        /// The offending value.
        value: f32,
    },
}

/// Bundle of safe wrappers around one backend's kernels.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) dot: fn(&[f32], &[f32]) -> f32,
    pub(crate) mul_into: fn(&[f32], &[f32], &mut [f32]),
    pub(crate) softmax_into: fn(&[f32], &mut [f32]),
    pub(crate) argmax: fn(&[f32]) -> usize,
}

pub(crate) fn kernels() -> Kernels {
    match detected_backend() {
        Backend::Scalar => scalar::kernels(),
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => x86::sse2_kernels(),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => x86::avx2_kernels(),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => aarch64::neon_kernels(),
        _ => scalar::kernels(),
    }
}

/// Computes the dot product of two equally sized `f32` slices.
///
/// Returns `0.0` for empty inputs. Non-finite inputs are not rejected; IEEE
/// arithmetic (including NaN propagation) applies. The summation order is
/// backend-dependent, so results may differ by a few ULPs.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] if `lhs.len() != rhs.len()`.
pub fn dot_f32(lhs: &[f32], rhs: &[f32]) -> Result<f32, SimdError> {
    if lhs.len() != rhs.len() {
        return Err(SimdError::LengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
        });
    }
    Ok((kernels().dot)(lhs, rhs))
}

/// Elementwise multiplies two equally sized slices into `out`.
///
/// This is the vector-producing counterpart of [`dot_f32`]. Empty inputs are
/// accepted and leave `out` untouched.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] if the inputs differ in length, or
/// [`SimdError::OutputLengthMismatch`] if `out` has the wrong length.
pub fn dot_f32_into(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
    if lhs.len() != rhs.len() {
        return Err(SimdError::LengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
        });
    }
    if out.len() != lhs.len() {
        return Err(SimdError::OutputLengthMismatch {
            expected: lhs.len(),
            actual: out.len(),
        });
    }
    (kernels().mul_into)(lhs, rhs, out);
    Ok(())
}

/// Computes a numerically stable softmax of `logits` into `out`.
///
/// Every output is in `[0, 1]` and the outputs sum to approximately `1`.
/// The maximum logit is subtracted before exponentiation, so large finite
/// inputs cannot overflow. Empty inputs are accepted and leave `out` empty.
/// Rust's borrowing rules guarantee that `logits` and `out` do not overlap.
///
/// # Errors
///
/// Returns [`SimdError::OutputLengthMismatch`] if the lengths differ and
/// [`SimdError::NonFinite`] for the first NaN or infinity in `logits`.
pub fn softmax_into(logits: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
    if logits.len() != out.len() {
        return Err(SimdError::OutputLengthMismatch {
            expected: logits.len(),
            actual: out.len(),
        });
    }
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(SimdError::NonFinite {
                index,
                value: *value,
            });
        }
    }
    if !logits.is_empty() {
        (kernels().softmax_into)(logits, out);
    }
    Ok(())
}

/// Returns the index of the largest finite value in `values`.
///
/// Ties are resolved in favour of the lowest index.
///
/// # Errors
///
/// Returns [`SimdError::EmptyInput`] if `values` is empty and
/// [`SimdError::NonFinite`] for the first NaN or infinity.
pub fn argmax(values: &[f32]) -> Result<usize, SimdError> {
    if values.is_empty() {
        return Err(SimdError::EmptyInput);
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SimdError::NonFinite {
                index,
                value: *value,
            });
        }
    }
    Ok((kernels().argmax)(values))
}
