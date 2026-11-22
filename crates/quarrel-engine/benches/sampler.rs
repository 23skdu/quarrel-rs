use criterion::{Criterion, criterion_group, criterion_main};
use quarrel_core::SamplingParams;
use quarrel_engine::sampler::{Sampler, argmax, softmax};
use std::hint::black_box;

fn sampling_params() -> SamplingParams {
    let mut params = SamplingParams::new();
    params.temperature = 0.8;
    params.top_p = 0.9;
    params.top_k = 16;
    params.repetition_penalty = 1.05;
    params.max_tokens = 32;
    params.stop_token_ids.clear();
    params
}

fn benchmark_sampler(criterion: &mut Criterion) {
    let logits: Vec<f32> = (0..128)
        .map(|index| ((index as f32) * 0.37).sin())
        .collect();
    let previous: Vec<i32> = (0..16).collect();
    let params = sampling_params();

    let mut group = criterion.benchmark_group("sampler");
    group.bench_function("argmax", |bencher| {
        bencher.iter(|| black_box(argmax(black_box(&logits))));
    });
    group.bench_function("softmax", |bencher| {
        bencher.iter(|| black_box(softmax(black_box(&logits))));
    });
    group.bench_function("sample", |bencher| {
        let mut sampler = Sampler::new(42);
        bencher
            .iter(|| black_box(sampler.sample(black_box(&logits), &params, black_box(&previous))));
    });
    group.finish();
}

criterion_group!(benches, benchmark_sampler);
criterion_main!(benches);
