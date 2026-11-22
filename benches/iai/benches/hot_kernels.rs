//! Deterministic iai-callgrind benchmarks for the dispatched SIMD kernels.
//!
//! Each case has a stable `group::function::id` name so local reference
//! baselines can be compared across refreshes. Setup work (input generation)
//! is not measured by iai-callgrind. Sampler cases remain in the Criterion
//! suite because they already have stable wall-clock coverage there.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

fn dot_inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..len).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let rhs = (0..len).map(|i| (i as f32) * -0.02 + 2.0).collect();
    (lhs, rhs)
}

#[library_benchmark]
#[bench::dot_256(dot_inputs(256))]
#[bench::dot_4096(dot_inputs(4096))]
fn dot_f32((lhs, rhs): (Vec<f32>, Vec<f32>)) -> f32 {
    black_box(quarrel_simd::dot_f32(black_box(&lhs), black_box(&rhs)).expect("valid dot inputs"))
}

fn softmax_inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let logits = (0..len).map(|i| ((i as f32) * 0.37).sin()).collect();
    let out = vec![0.0_f32; len];
    (logits, out)
}

#[library_benchmark]
#[bench::softmax_256(softmax_inputs(256))]
#[bench::softmax_4096(softmax_inputs(4096))]
fn softmax_into((logits, mut out): (Vec<f32>, Vec<f32>)) {
    let out_slice: &mut [f32] = black_box(&mut out);
    quarrel_simd::softmax_into(black_box(&logits), out_slice).expect("valid softmax inputs");
}

library_benchmark_group!(
    name = simd_kernels;
    benchmarks = dot_f32, softmax_into
);

main!(library_benchmark_groups = simd_kernels);
