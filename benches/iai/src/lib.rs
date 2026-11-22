//! Harness-only crate for the pinned iai-callgrind hot-kernel benchmarks.
//!
//! The benchmark binary is defined in `benches/hot_kernels.rs`. This library
//! target exists only so Cargo has a package target to attach that benchmark
//! to; no runtime code is exported.
