//! Deterministic generation checks against real public Q4_K_M GGUFs.
//!
//! The model files are deliberately not stored in this repository. Set
//! `QUARREL_MODEL_DIR` to a directory containing the two files below to enable
//! the checks; without that variable both tests are skipped so the ordinary
//! workspace test suite stays fast and offline.

use quarrel_runtime::{ChatMessage, GenerationOptions, InferenceSession, Model, Tokenizer};
use std::path::{Path, PathBuf};

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("QUARREL_MODEL_DIR").map(PathBuf::from)
}

fn generate(path: &Path, max_new_tokens: usize) -> (Vec<u32>, String) {
    let model = Model::open(path).expect("open GGUF model");
    let tokenizer = Tokenizer::from_header(model.header()).expect("build tokenizer");
    let prompt = tokenizer
        .encode_chat(&[ChatMessage::new("user", "What is the capital of France?")])
        .expect("encode chat prompt");
    let mut session = InferenceSession::new(&model);
    let options = GenerationOptions {
        max_new_tokens,
        ..GenerationOptions::default()
    };
    let ids = quarrel_runtime::generate(&mut session, &prompt, &options).expect("generate");
    let text = tokenizer.decode(&ids);
    (ids, text)
}

#[test]
fn smollm2_360m_chat_greedy_tokens() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping SmolLM2 golden test: QUARREL_MODEL_DIR is not set");
        return;
    };
    let path = dir.join("smollm2-360m-q4_k_m.gguf");
    if !path.exists() {
        eprintln!("skipping SmolLM2 golden test: {} is absent", path.display());
        return;
    }
    let (ids, text) = generate(&path, 8);
    assert_eq!(
        ids,
        [504, 3575, 282, 4649, 314, 7042, 30],
        "greedy SmolLM2 token ids"
    );
    assert_eq!(text, "The capital of France is Paris.");
}

#[test]
fn gemma_3_270m_chat_greedy_tokens() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping Gemma 3 golden test: QUARREL_MODEL_DIR is not set");
        return;
    };
    let path = dir.join("gemma-3-270m-it-q4_k_m.gguf");
    if !path.exists() {
        eprintln!("skipping Gemma 3 golden test: {} is absent", path.display());
        return;
    }
    let (ids, text) = generate(&path, 12);
    assert_eq!(
        ids,
        [818, 5279, 529, 7001, 563, 9079, 236761],
        "greedy Gemma 3 token ids"
    );
    assert_eq!(text, "The capital of France is Paris.");
}
