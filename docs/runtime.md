# Native GGUF runtime

`quarrel-runtime` is the real-model execution path. It is pure Rust: GGUF bytes are memory-mapped by
`quarrel-gguf`, weights are read as slices, and matrix–vector products run through the SIMD facade in
`quarrel-simd`. There is no llama.cpp FFI, downloaded tokenizer or payload-sized dequantization
buffer.

> **Boundary:** the OpenAI/HTTP and Arrow Flight planes still serve the deterministic
> `ReferenceBackend`. `quarrel-runtime` is consumed directly by its library API and CLIs. See
> [architecture](architecture.md) for the package graph.

## Supported models

| Family identifier | Runtime architecture | Verified GGUF |
|-------------------|----------------------|---------------|
| `llama` / `smollm2` | `Architecture::Llama` | `SmolLM2-360M-Instruct-Q4_K_M.gguf` |
| `gemma3` | `Architecture::Gemma3` | `gemma-3-270m-it-Q4_K_M.gguf` |

The engine's architecture router additionally recognises Qwen2, Qwen3.5, Mistral, Gemma4, Phi-3 and
Granite for KV/layout planning. Kernel dispatch covers every quantization used by common releases of
those families; see [SIMD coverage](architecture.md#33-simd-kernel-coverage-by-family).

## Run generation

Model files are not committed. Download the two verification files into a directory outside the
repository, then:

```bash
# SmolLM2 360M instruct, Q4_K_M
cargo run --release -p quarrel-runtime --bin quarrel-generate -- \
  /path/to/models/smollm2-360m-q4_k_m.gguf \
  "What is the capital of France?" --chat --ids

# Gemma 3 270M instruct, Q4_K_M
cargo run --release -p quarrel-runtime --bin quarrel-generate -- \
  /path/to/models/gemma-3-270m-it-q4_k_m.gguf \
  "What is the capital of France?" --chat --ids
```

Both prompts produce `The capital of France is Paris.`. With `--ids`, the first command reports
`[504, 3575, 282, 4649, 314, 7042, 30]` and the second reports
`[818, 5279, 529, 7001, 563, 9079, 236761]`.

Inspect prompt tokenization without running inference:

```bash
cargo run --release -p quarrel-runtime --bin quarrel-tokens -- \
  /path/to/models/smollm2-360m-q4_k_m.gguf \
  "What is the capital of France?" --chat
```

## Golden-token checks

The integration test is opt-in so normal `make test` stays offline:

```bash
QUARREL_MODEL_DIR=/path/to/models \
  cargo test -p quarrel-runtime --test real_models -- --nocapture
```

It asserts the exact greedy ids and decoded text for both models. Missing files skip the test with a
message rather than failing. Use release mode (`--release`) for faster local iteration.

## Implementation notes

- GGUF tensor dimensions are `[n_in, n_out]`; matvec validates `[input.len(), out.len()]`.
- Canonical ggml dtype ids are authoritative; `general.file_type` is kept separate.
- K-quant blocks are decoded over the flattened tensor, so blocks may straddle row boundaries.
- Gemma 3 uses NEOX RoPE (local base `10000`, global base `1e6`), `Q/sqrt(head_dim)`, score scale
  `1.0`, head-wise QK normalization and GELU-tanh. Llama/SmolLM2 use interleaved RoPE, score scale
  `1/sqrt(head_dim)` and SwiGLU with `x * sigmoid(x) * gate`.
- GGUF Gemma normalization tensors already include the `1 + w` shift, so stored weights are
  multiplied directly.
- Tokenizers are built from GGUF metadata arrays: BPE with byte fallback for SmolLM2 and BPE/SPM
  merges for Gemma 3. The chat renderer supplies the model-specific turn tokens.

## Known limits

- Gemma 3 chat rendering covers user turns and the standard `<start_of_turn>`/`<end_of_turn>`
  markers; arbitrary Jinja templates and system-message variants are not yet interpreted.
- GQA is materialised per head during attention; there is no paged-KV integration with
  `quarrel-engine` yet.
- Go oracle fixtures are still required before any parity claim (roadmap P1-03/P2-06).
