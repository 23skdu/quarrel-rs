//! Criterion benchmarks for allocation-free quant dot products.

use criterion::{Criterion, criterion_group, criterion_main};
use half::f16;
use quarrel_gguf::quant::{dot_q4_0, dot_q8_0};
use std::hint::black_box;

const ELEMENTS: usize = 4096;
const BLOCK_ELEMENTS: usize = 32;

fn q8_data() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ELEMENTS / BLOCK_ELEMENTS * 34);
    for block in 0..ELEMENTS / BLOCK_ELEMENTS {
        bytes.extend_from_slice(&f16::from_f32(0.125).to_le_bytes());
        for i in 0..BLOCK_ELEMENTS {
            let value = ((block * BLOCK_ELEMENTS + i) % 17) as i8 - 8;
            bytes.push(value as u8);
        }
    }
    bytes
}

fn q4_data() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ELEMENTS / BLOCK_ELEMENTS * 18);
    for block in 0..ELEMENTS / BLOCK_ELEMENTS {
        bytes.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
        for pair in 0..BLOCK_ELEMENTS / 2 {
            let low = ((block * BLOCK_ELEMENTS + 2 * pair) % 16) as u8;
            let high = ((block * BLOCK_ELEMENTS + 2 * pair + 1) % 16) as u8;
            bytes.push(low | (high << 4));
        }
    }
    bytes
}

fn bench_dot(c: &mut Criterion) {
    let lhs = (0..ELEMENTS)
        .map(|i| (i % 13) as f32 / 13.0)
        .collect::<Vec<_>>();
    let q8 = q8_data();
    let q4 = q4_data();

    let mut group = c.benchmark_group("quant_dot");
    group.bench_function("dot_q8_0", |b| {
        b.iter(|| dot_q8_0(black_box(&q8), black_box(&lhs)).expect("valid q8 data"));
    });
    group.bench_function("dot_q4_0", |b| {
        b.iter(|| dot_q4_0(black_box(&q4), black_box(&lhs)).expect("valid q4 data"));
    });
    group.finish();
}

criterion_group!(benches, bench_dot);
criterion_main!(benches);
