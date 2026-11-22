# Fuzz targets

This crate is excluded from the main workspace and is driven by `cargo-fuzz` on nightly
(`fuzz/rust-toolchain.toml`). It fuzzes production entry points, never copies of them.

| Target | Entry point | What it must prove |
|--------|-------------|--------------------|
| `gguf_parse` | `quarrel_gguf::GgufFile::from_bytes` | Arbitrary bytes are rejected with a typed error; parsed descriptors never panic on lookup |
| `sampling_params` | `quarrel_core::SamplingParams::validate` + `quarrel_engine::sampler::Sampler::sample` | Client-controlled floats, stop tokens and logits produce values or typed errors, never panics |
| `arrow_ipc` | `quarrel_flight::ipc_to_batch` | Arbitrary IPC bytes stay inside `FlightError` for each canonical schema |

Run from the repository root:

```bash
make fuzz-fmt-check   # rustfmt, pinned stable toolchain
make fuzz-smoke       # 5 s per target, used in CI
make fuzz             # 60 s per target, release gate
make fuzz-coverage    # corpus coverage per target
```

Crash inputs are written under `fuzz/artifacts/<target>/`; corpora live under `fuzz/corpus/`.
Both directories are ignored by git. A reproducible crash should be reduced with
`cargo fuzz tmin <target> <artifact>` and then added as a regression unit test in the owning crate.
