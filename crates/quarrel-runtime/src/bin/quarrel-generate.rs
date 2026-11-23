//! Generate tokens from a real GGUF model using the native Rust runtime.

use clap::Parser;
use quarrel_runtime::{
    ChatMessage, GenerationOptions, InferenceSession, Model, Tokenizer, generate,
};
use std::error::Error;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "quarrel-generate",
    version,
    about = "Generate text from a Gemma 3 or SmolLM2 Q4_K_M GGUF in pure Rust"
)]
struct Args {
    /// Path to a GGUF model file.
    model: PathBuf,

    /// Prompt text.
    prompt: String,

    /// Apply the model's chat template around the prompt.
    #[arg(long)]
    chat: bool,

    /// Maximum number of new tokens.
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// Sampling temperature; zero selects greedy decoding.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Deterministic sampling seed.
    #[arg(long, default_value_t = 0x9E37_79B9_7F4A_7C15)]
    seed: u64,

    /// Print generated token ids to stderr.
    #[arg(long)]
    ids: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let model = Model::open(&args.model)?;
    let tokenizer = Tokenizer::from_header(model.header())?;
    let prompt = if args.chat {
        tokenizer.encode_chat(&[ChatMessage::new("user", &args.prompt)])?
    } else {
        let mut tokens = Vec::new();
        if tokenizer.adds_bos() || tokenizer.bos_id() != 0 {
            tokens.push(tokenizer.bos_id());
        }
        tokens.extend_from_slice(&tokenizer.encode(&args.prompt, true)?);
        tokens
    };
    let options = GenerationOptions {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        ..GenerationOptions::default()
    };
    let mut session = InferenceSession::new(&model);
    let tokens = generate(&mut session, &prompt, &options)?;
    if args.ids {
        eprintln!("generated ids: {tokens:?}");
    }
    println!("{}", tokenizer.decode(&tokens));
    Ok(())
}
