//! Criterion micro-benchmarks for the dispatched dot and softmax kernels.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..len).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let rhs = (0..len).map(|i| (i as f32) * -0.02 + 2.0).collect();
    (lhs, rhs)
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");
    for &len in &[16_usize, 256, 4096] {
        let (lhs, rhs) = inputs(len);
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(
                || match quarrel_simd::dot_f32(black_box(&lhs), black_box(&rhs)) {
                    Ok(value) => black_box(value),
                    Err(_) => 0.0,
                },
            );
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_into");
    for &len in &[16_usize, 256, 4096] {
        let (logits, _) = inputs(len);
        let mut out = vec![0.0_f32; len];
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                let _ = quarrel_simd::softmax_into(black_box(&logits), black_box(&mut out));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dot, bench_softmax);
criterion_main!(benches);
