//! CLI front end for [`quarrel_gguf::writer::requantize_gguf`].

use clap::{Parser, ValueEnum};
use quarrel_gguf::writer::requantize_gguf;
use quarrel_gguf::{DType, GgufError};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetDType {
    #[value(name = "f32")]
    F32,
    #[value(name = "f16")]
    F16,
    #[value(name = "q8_0")]
    Q8Zero,
    #[value(name = "q4_k")]
    Q4K,
    #[value(name = "q6_k")]
    Q6K,
}

impl From<TargetDType> for DType {
    fn from(value: TargetDType) -> Self {
        match value {
            TargetDType::F32 => Self::F32,
            TargetDType::F16 => Self::F16,
            TargetDType::Q8Zero => Self::Q8_0,
            TargetDType::Q4K => Self::Q4_K,
            TargetDType::Q6K => Self::Q6_K,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "quarrel-quantize",
    version,
    about = "Requantize F32/F16 tensors in a GGUF v1-v3 file"
)]
struct Args {
    /// Input GGUF file with F32 or F16 tensors.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Destination GGUF file, overwritten if it already exists.
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Target tensor dtype: f32, f16, q8_0, q4_k or q6_k.
    #[arg(value_name = "DTYPE")]
    dtype: TargetDType,
}

fn run(args: &Args) -> Result<(), GgufError> {
    let bytes = std::fs::read(&args.input)?;
    let output = requantize_gguf(&bytes, args.dtype.into())?;
    std::fs::write(&args.output, output)?;
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(error) = run(&args) {
        eprintln!("quarrel-quantize: {error}");
        std::process::exit(1);
    }
}
