//! Inspect GGUF-native tokenization for prompts and generated text.

use clap::Parser;
use quarrel_runtime::{ChatMessage, Model, Tokenizer};
use std::error::Error;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "quarrel-tokens",
    version,
    about = "Print GGUF prompt token ids"
)]
struct Args {
    /// Path to a GGUF model file.
    model: PathBuf,

    /// Text to tokenize.
    text: String,

    /// Apply the model's chat template around the text.
    #[arg(long)]
    chat: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let model = Model::open(&args.model)?;
    let tokenizer = Tokenizer::from_header(model.header())?;
    let tokens = if args.chat {
        tokenizer.encode_chat(&[ChatMessage::new("user", &args.text)])?
    } else {
        tokenizer.encode(&args.text, true)?
    };
    for (index, token) in tokens.iter().enumerate() {
        println!("{index:4}: {token:6} -> {:?}", tokenizer.decode(&[*token]));
    }
    println!("total: {}", tokens.len());
    Ok(())
}
