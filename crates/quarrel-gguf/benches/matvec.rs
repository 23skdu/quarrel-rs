//! Criterion benchmarks for the allocation-free GGUF matvec kernels.

use criterion::{Criterion, criterion_group, criterion_main};
use half::{bf16, f16};
use quarrel_gguf::matvec::{
    matvec_bf16, matvec_f16, matvec_f32, matvec_q2_k, matvec_q3_k, matvec_q4_0, matvec_q4_k,
    matvec_q5_k, matvec_q6_k, matvec_q8_0,
};
use std::hint::black_box;

const ROWS: usize = 256;
const COLS: usize = 256;
const BLOCK_ELEMENTS: usize = 32;
const QK_K: usize = 256;

fn f32_weights() -> Vec<u8> {
    (0..ROWS * COLS)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn f16_weights() -> Vec<u8> {
    (0..ROWS * COLS)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .flat_map(|value| f16::from_f32(value).to_le_bytes())
        .collect()
}

fn bf16_weights() -> Vec<u8> {
    (0..ROWS * COLS)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .flat_map(|value| bf16::from_f32(value).to_le_bytes())
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

// Repeat one canonical K-quant block for every row (one block per 256 cols).
fn k_weights(fields: &[&[u8]]) -> Vec<u8> {
    let block_bytes: usize = fields.iter().map(|field| field.len()).sum();
    let mut block = Vec::with_capacity(block_bytes);
    for field in fields {
        block.extend_from_slice(field);
    }
    let blocks = ROWS * (COLS / QK_K);
    let mut bytes = Vec::with_capacity(blocks * block_bytes);
    for _ in 0..blocks {
        bytes.extend_from_slice(&block);
    }
    bytes
}

fn q2_k_weights() -> Vec<u8> {
    let scales = [0x11u8; 16];
    let qs = [0xe4u8; 64];
    let d = f16::from_f32(0.125).to_le_bytes();
    let dmin = f16::from_f32(0.0625).to_le_bytes();
    k_weights(&[&scales[..], &qs[..], &d[..], &dmin[..]])
}

fn q3_k_weights() -> Vec<u8> {
    let hmask = [0xffu8; 32];
    let qs = [0xe4u8; 64];
    let mut scales = [0xaau8; 12]; // all scales biased to zero
    scales[0] = 0x21; // 6-bit value 33
    let d = f16::from_f32(0.125).to_le_bytes();
    k_weights(&[&hmask[..], &qs[..], &scales[..], &d[..]])
}

fn q4_k_weights() -> Vec<u8> {
    let d = f16::from_f32(0.125).to_le_bytes();
    let dmin = f16::from_f32(0.0625).to_le_bytes();
    let mut scales = [0u8; 12];
    scales[0] = 0x10; // scale 16, min 0 for group 0
    scales[4] = 0x08;
    let qs = [0x21u8; 128];
    k_weights(&[&d[..], &dmin[..], &scales[..], &qs[..]])
}

fn q5_k_weights() -> Vec<u8> {
    let d = f16::from_f32(0.125).to_le_bytes();
    let dmin = f16::from_f32(0.0625).to_le_bytes();
    let mut scales = [0u8; 12];
    scales[0] = 0x10;
    scales[4] = 0x08;
    let qh = [0x55u8; 32];
    let qs = [0x21u8; 128];
    k_weights(&[&d[..], &dmin[..], &scales[..], &qh[..], &qs[..]])
}

fn q6_k_weights() -> Vec<u8> {
    let ql = [0x21u8; 128];
    let qh = [0xedu8; 64];
    let scales = [4u8; 16]; // signed i8 scale +4
    let d = f16::from_f32(0.125).to_le_bytes();
    k_weights(&[&ql[..], &qh[..], &scales[..], &d[..]])
}

fn bench_matvec(c: &mut Criterion) {
    let input = (0..COLS)
        .map(|i| (i % 13) as f32 / 13.0)
        .collect::<Vec<_>>();
    let f32_data = f32_weights();
    let f16_data = f16_weights();
    let bf16_data = bf16_weights();
    let q8_data = q8_weights();
    let q4_data = q4_weights();
    let q2_k_data = q2_k_weights();
    let q3_k_data = q3_k_weights();
    let q4_k_data = q4_k_weights();
    let q5_k_data = q5_k_weights();
    let q6_k_data = q6_k_weights();

    let mut group = c.benchmark_group("gguf_matvec");
    group.bench_function("f32", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_f32(black_box(&f32_data), black_box(&input), black_box(&mut out))
                .expect("valid f32 weights")
        });
    });
    group.bench_function("f16", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_f16(black_box(&f16_data), black_box(&input), black_box(&mut out))
                .expect("valid f16 weights")
        });
    });
    group.bench_function("bf16", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_bf16(
                black_box(&bf16_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid bf16 weights")
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
    group.bench_function("q2_k", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q2_k(
                black_box(&q2_k_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid q2_k weights")
        });
    });
    group.bench_function("q3_k", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q3_k(
                black_box(&q3_k_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid q3_k weights")
        });
    });
    group.bench_function("q4_k", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q4_k(
                black_box(&q4_k_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid q4_k weights")
        });
    });
    group.bench_function("q5_k", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q5_k(
                black_box(&q5_k_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid q5_k weights")
        });
    });
    group.bench_function("q6_k", |b| {
        let mut out = vec![0.0f32; ROWS];
        b.iter(|| {
            matvec_q6_k(
                black_box(&q6_k_data),
                black_box(&input),
                black_box(&mut out),
            )
            .expect("valid q6_k weights")
        });
    });
    group.finish();
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);
