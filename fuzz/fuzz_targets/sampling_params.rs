#![no_main]

use libfuzzer_sys::fuzz_target;

use quarrel_core::{SamplingParams, Token};
use quarrel_engine::sampler::Sampler;

fn u32_at(data: &[u8], offset: usize) -> u32 {
    data.get(offset..offset + 4)
        .and_then(|chunk| chunk.try_into().ok())
        .map(u32::from_le_bytes)
        .unwrap_or_default()
}

fn finite_unit(bits: u32) -> f32 {
    let value = f32::from_bits(bits);
    if value.is_finite() {
        value.clamp(-4.0, 4.0)
    } else {
        0.0
    }
}

// Sampling parameters and logits come from clients. Validation and sampling must return typed
// errors rather than panicking, including for NaN/Inf bit patterns and nonsense stop tokens.
fuzz_target!(|data: &[u8]| {
    let mut params = SamplingParams::new();
    params.temperature = finite_unit(u32_at(data, 0)).abs();
    params.top_p = finite_unit(u32_at(data, 4)).abs();
    params.top_k = u32_at(data, 8) % 65;
    params.repetition_penalty = 0.5 + finite_unit(u32_at(data, 12)).abs() * 0.25;
    params.max_tokens = 1 + u32_at(data, 16) % 128;

    params.stop_token_ids.clear();
    let mut offset = 24;
    while offset + 4 <= data.len() && params.stop_token_ids.len() < 16 {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[offset..offset + 4]);
        params.stop_token_ids.push(i32::from_le_bytes(bytes) % 64);
        offset += 4;
    }

    if params.validate().is_err() {
        return;
    }

    let logits: Vec<f32> = data
        .iter()
        .skip(64)
        .take(32)
        .map(|&byte| f32::from(byte) - 127.5)
        .collect();
    let previous: Vec<i32> = data
        .iter()
        .skip(24)
        .take(8)
        .map(|&byte| i32::from(byte) % 64)
        .collect();

    let seed = u64::from(u32_at(data, 20));
    let mut sampler = Sampler::new(seed);
    if let Ok(token) = sampler.sample(&logits, &params, &previous) {
        let _ = Token::new(token.id, token.logprob);
    }
});
