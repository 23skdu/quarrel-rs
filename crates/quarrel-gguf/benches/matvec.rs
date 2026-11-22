//! Criterion benchmarks for the allocation-free GGUF matvec kernels.

use criterion::{Criterion, criterion_group, criterion_main};
use half::f16;
use quarrel_gguf::matvec::{matvec_f32, matvec_q4_0, matvec_q8_0};
use std::hint::black_box;

const ROWS: usize = 256;
const COLS: usize = 256;
const BLOCK_ELEMENTS: usize = 32;

fn f32_weights() -> Vec<u8> {
    (0..ROWS * COLS)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn q8_weights() -> Vec<u8> {
    let blocks_per_row = COLS / BLOCK_ELEMENTS;
    let mut bytes = Vec::with_capacity(ROWS * blocks_per_row * 34);
    for block in 0..ROWS * blocks_per_row {
        bytes.extend_from_slice(&f16::from_f32(0.125).to_le_bytes());
        for i in 0..BLOCK_ELEMENTS {
            let value = ((block + i) % 17) as i8 - 8;
            bytes.push(value as u8);
        }
    }
    bytes
}

fn q4_weights() -> Vec<u8> {
    let blocks_per_row = COLS / BLOCK_ELEMENTS;
    let mut bytes = Vec::with_capacity(ROWS * blocks_per_row * 18);
    for block in 0..ROWS * blocks_per_row {
        bytes.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
        for pair in 0..BLOCK_ELEMENTS / 2 {
            let low = ((block + pair) % 16) as u8;
            let high = ((block + pair + 1) % 16) as u8;
            bytes.push(low | (high << 4));
        }
    }
    bytes
}

fn bench_matvec(c: &mut Criterion) {
    let input = (0..COLS)
        .map(|i| (i % 13) as f32 / 13.0)
        .collect::<Vec<_>>();
    let f32_data = f32_weights();
    let q8_data = q8_weights();
    let q4_data = q4_weights();

    let mut group = c.benchmark_group("gguf_matvec");
    group.bench_function("f32", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_f32(black_box(&f32_data), black_box(&input), black_box(&mut out))
                .expect("valid f32 weights")
        });
    });
    group.bench_function("q8_0", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q8_0(black_box(&q8_data), black_box(&input), black_box(&mut out))
                .expect("valid q8_0 weights")
        });
    });
    group.bench_function("q4_0", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q4_0(black_box(&q4_data), black_box(&input), black_box(&mut out))
                .expect("valid q4_0 weights")
        });
    });
    group.finish();
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);
