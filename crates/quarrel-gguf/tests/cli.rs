//! End-to-end tests for the `quarrel-quantize` binary.

use quarrel_gguf::writer::GgufWriter;
use quarrel_gguf::{DType, GgufFile, GgufVersion, MetadataValue};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "quarrel-quantize-{label}-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn source_fixture() -> Vec<u8> {
    let values: Vec<f32> = (0..256)
        .map(|index| (index as f32 - 127.5) / 127.5)
        .collect();
    let data: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut writer = GgufWriter::new(GgufVersion::V3);
    writer
        .set_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        )
        .unwrap();
    writer
        .set_metadata("general.file_type", MetadataValue::Uint32(0))
        .unwrap();
    writer
        .add_tensor("w0", vec![16, 16], DType::F32, data)
        .unwrap();
    writer.to_bytes().unwrap()
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quarrel-quantize"))
}

#[test]
fn cli_requantizes_generated_fixture() {
    let dir = temp_dir("success");
    let input = dir.join("source.gguf");
    let output = dir.join("q4_k.gguf");
    fs::write(&input, source_fixture()).unwrap();

    let status = binary()
        .arg(&input)
        .arg(&output)
        .arg("q4_k")
        .status()
        .unwrap();
    assert!(status.success(), "CLI failed with {status}");

    let bytes = fs::read(&output).unwrap();
    let file = GgufFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.header().version, GgufVersion::V3);
    assert_eq!(file.header().tensors[0].dtype, DType::Q4_K);
    assert_eq!(file.tensor_bytes("w0").unwrap().len(), 144);

    let second = dir.join("q4_k-again.gguf");
    assert!(
        binary()
            .arg(&input)
            .arg(&second)
            .arg("q4_k")
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(fs::read(&second).unwrap(), bytes);

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cli_rejects_bad_arguments() {
    let dir = temp_dir("bad-args");
    let input = dir.join("source.gguf");
    fs::write(&input, source_fixture()).unwrap();

    let missing = binary().output().unwrap();
    assert!(!missing.status.success());

    let invalid_dtype = binary()
        .arg(&input)
        .arg(dir.join("out.gguf"))
        .arg("q5_k")
        .output()
        .unwrap();
    assert!(!invalid_dtype.status.success());
    let stderr = String::from_utf8_lossy(&invalid_dtype.stderr);
    assert!(
        stderr.contains("invalid value"),
        "unexpected stderr: {stderr}"
    );

    let missing_input = binary()
        .arg(dir.join("absent.gguf"))
        .arg(dir.join("out.gguf"))
        .arg("f16")
        .output()
        .unwrap();
    assert!(!missing_input.status.success());
    assert!(!dir.join("out.gguf").exists());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cli_rejects_unwritable_output() {
    let dir = temp_dir("bad-output");
    let input = dir.join("source.gguf");
    fs::write(&input, source_fixture()).unwrap();

    let result = binary()
        .arg(&input)
        .arg(&dir) // a directory is not a valid output file
        .arg("f16")
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("I/O error"), "unexpected stderr: {stderr}");

    let blocked = dir.join("missing-parent").join("out.gguf");
    let result = binary()
        .arg(&input)
        .arg(&blocked)
        .arg("q6_k")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!Path::new(&blocked).exists());

    fs::remove_dir_all(&dir).unwrap();
}
