//! Scalar oracle and runtime-dispatched numeric kernels for `quarrel-rs`.
//!
//! Every public function validates its arguments and then delegates to a
//! backend chosen once, on first use, by [`detected_backend`]. The selected
//! backend never changes during the process: detection is cached in a
//! [`std::sync::OnceLock`], so a call cannot observe a different CPU feature
//! set halfway through a computation.
//!
//! The stable facade is [`KernelSet`]: [`KernelSet::auto`] always picks the
//! best available kernel (AVX-512F, AVX2/FMA, SSE2, NEON, then scalar), while
//! [`KernelSet::for_mode`] can force a specific [`KernelMode`] for parity and
//! dispatch testing. A forced mode that this machine cannot run returns
//! `None` instead of dispatching to unsupported instructions.
//!
//! All hand-written SIMD code lives in the architecture-specific kernel
//! modules. The portable scalar implementations double as the reference
//! oracle used by the tests; the SIMD kernels must agree with them up to
//! floating-point rounding.
//!
//! ```
//! use quarrel_simd::{argmax, dot_f32, dot_f32_into, softmax_into};
//! use quarrel_simd::{Backend, KernelMode, KernelSet};
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
//!
//! let kernels = KernelSet::auto();
//! assert_eq!(kernels.backend(), quarrel_simd::detected_backend());
//! let scalar = KernelSet::for_mode(KernelMode::Scalar).unwrap();
//! assert_eq!(scalar.backend(), Backend::Scalar);
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
    /// x86-64 SSE2 (four-wide single precision).
    Sse2,
    /// x86-64 AVX2 with FMA (eight-wide single precision).
    Avx2,
    /// x86-64 AVX-512F (sixteen-wide single precision, with FMA).
    Avx512,
    /// AArch64 Advanced SIMD (NEON, four-wide single precision).
    Neon,
}

/// Explicit kernel selection for the [`KernelSet`] facade.
///
/// This is used to force a backend in tests; normal callers should use
/// [`KernelMode::Auto`] (the default) so the fastest supported kernel is
/// selected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelMode {
    /// Choose the best backend available on this machine.
    Auto,
    /// Force the portable scalar kernels.
    Scalar,
    /// Force the x86-64 SSE2 kernels.
    Sse2,
    /// Force the x86-64 AVX2/FMA kernels.
    Avx2,
    /// Force the x86-64 AVX-512F kernels.
    Avx512,
    /// Force the AArch64 NEON kernels.
    Neon,
}

impl KernelMode {
    /// Resolves this mode to the concrete backend it would use on this
    /// machine.
    ///
    /// Returns `None` when the requested ISA extension is unavailable on this
    /// target or CPU. [`KernelMode::Auto`] always resolves to
    /// [`detected_backend`].
    pub fn resolve(self) -> Option<Backend> {
        match self {
            Self::Auto => Some(detected_backend()),
            Self::Scalar => Some(Backend::Scalar),
            Self::Sse2 => backend_available(Backend::Sse2).then_some(Backend::Sse2),
            Self::Avx2 => backend_available(Backend::Avx2).then_some(Backend::Avx2),
            Self::Avx512 => backend_available(Backend::Avx512).then_some(Backend::Avx512),
            Self::Neon => backend_available(Backend::Neon).then_some(Backend::Neon),
        }
    }
}

/// Returns the backend used by all default kernels in this process.
///
/// Feature detection and the resulting choice are made exactly once, on the
/// first call (which may happen implicitly through any kernel). Later calls
/// return the same value. AVX-512F takes precedence over AVX2/FMA, which takes
/// precedence over SSE2; NEON is the only SIMD backend on AArch64. Everything
/// else falls back to [`Backend::Scalar`].
pub fn detected_backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(detect_backend)
}

fn backend_available(backend: Backend) -> bool {
    match backend {
        Backend::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => std::arch::is_x86_feature_detected!("sse2"),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
        // Variants for other architectures, or the scalar fallback elsewhere.
        _ => false,
    }
}

fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if backend_available(Backend::Avx512) {
            return Backend::Avx512;
        }
        if backend_available(Backend::Avx2) {
            return Backend::Avx2;
        }
        if backend_available(Backend::Sse2) {
            return Backend::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if backend_available(Backend::Neon) {
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

/// Bundle of safe wrappers around one backend's raw kernels.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) dot: fn(&[f32], &[f32]) -> f32,
    pub(crate) mul_into: fn(&[f32], &[f32], &mut [f32]),
    pub(crate) softmax_into: fn(&[f32], &mut [f32]),
    pub(crate) argmax: fn(&[f32]) -> usize,
}

/// A validated, safe handle to one set of numeric kernels.
///
/// Use [`KernelSet::auto`] for normal work, or [`KernelSet::for_mode`] to force
/// a concrete backend. Every method performs the same validation and returns
/// the same typed errors as the corresponding crate-level function, so no
/// kernel can see a malformed slice.
#[derive(Clone, Copy)]
pub struct KernelSet {
    backend: Backend,
    ops: Kernels,
}

impl KernelSet {
    /// Returns kernels for the best backend available on this target.
    pub fn auto() -> Self {
        Self::for_backend(detected_backend())
    }

    /// Returns kernels for `mode`, or `None` if the mode is not available on
    /// this machine (for example AVX-512F on an AVX2-only CPU).
    pub fn for_mode(mode: KernelMode) -> Option<Self> {
        mode.resolve().map(Self::for_backend)
    }

    /// Returns the concrete backend backing this kernel set.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Computes the dot product of two equally sized `f32` slices.
    ///
    /// Returns `0.0` for empty inputs. Non-finite inputs are not rejected by
    /// this kernel; IEEE arithmetic (including NaN propagation) applies. The
    /// summation order is backend-dependent, so results may differ by a few
    /// ULPs.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if `lhs.len() != rhs.len()`.
    pub fn dot(&self, lhs: &[f32], rhs: &[f32]) -> Result<f32, SimdError> {
        if lhs.len() != rhs.len() {
            return Err(SimdError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }
        Ok((self.ops.dot)(lhs, rhs))
    }

    /// Elementwise multiplies two equally sized slices into `out`.
    ///
    /// Empty inputs are accepted and leave `out` untouched.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if the inputs differ in length,
    /// or [`SimdError::OutputLengthMismatch`] if `out` has the wrong length.
    pub fn dot_into(&self, lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
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
        (self.ops.mul_into)(lhs, rhs, out);
        Ok(())
    }

    /// Computes a numerically stable softmax of `logits` into `out`.
    ///
    /// Every output is in `[0, 1]` and the outputs sum to approximately `1`.
    /// Empty inputs are accepted and leave `out` empty. Rust's borrowing rules
    /// guarantee that `logits` and `out` do not overlap.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::OutputLengthMismatch`] if the lengths differ and
    /// [`SimdError::NonFinite`] for the first NaN or infinity in `logits`.
    pub fn softmax_into(&self, logits: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
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
            (self.ops.softmax_into)(logits, out);
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
    pub fn argmax(&self, values: &[f32]) -> Result<usize, SimdError> {
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
        Ok((self.ops.argmax)(values))
    }

    fn for_backend(backend: Backend) -> Self {
        let ops = match backend {
            Backend::Scalar => scalar::kernels(),
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => x86::sse2_kernels(),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => x86::avx2_kernels(),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => x86::avx512_kernels(),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => aarch64::neon_kernels(),
            _ => scalar::kernels(),
        };
        Self { backend, ops }
    }
}

fn default_kernels() -> &'static KernelSet {
    static KERNELS: OnceLock<KernelSet> = OnceLock::new();
    KERNELS.get_or_init(KernelSet::auto)
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
    default_kernels().dot(lhs, rhs)
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
    default_kernels().dot_into(lhs, rhs, out)
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
    default_kernels().softmax_into(logits, out)
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
    default_kernels().argmax(values)
}
