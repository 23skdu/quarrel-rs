# Local reference baselines

These files are **local reference baselines, not cross-machine gates**. They record one
idle-machine run so a future developer can see the shape of the numbers and compare stable
benchmark case names. Do not turn the absolute values into CI assertions: wall-clock timings and
Callgrind event counts both depend on CPU, microcode, kernel, compiler and valgrind version.

| File | Producer | Contents |
|------|----------|----------|
| `iai-callgrind-local.txt` | `make bench-iai-baseline` | Tool versions, CPU/memory, exact commands and normalized Callgrind event counts for the pinned iai-callgrind hot-kernel cases |
| `criterion-local.txt` | `make bench-criterion-baseline` | Tool versions, CPU/memory, exact command and Criterion output for every workspace Criterion case |

Neither benchmark family is part of `make ci`, `make test` or the default `make bench` criteria:
`make ci` must stay fast and hardware-independent, and the iai runner is environment-sensitive.

## iai-callgrind setup

The harness lives in the standalone crate `benches/iai` (its own workspace so the root
`Cargo.lock` and root `cargo bench --workspace` are untouched). It pins `iai-callgrind =0.16.1`,
which supports Rust 1.100.0. Install the matching runner once:

```bash
cargo install iai-callgrind-runner --version 0.16.1 --locked
valgrind --version   # 3.26.0 was used for the committed baseline
```

Then:

```bash
make bench-iai             # run the four deterministic kernels once
make bench-iai-baseline    # refresh benches/baselines/iai-callgrind-local.txt and save target/iai baseline
make bench-iai-compare     # compare a fresh run against the saved `committed` baseline
```

`--save-baseline=committed` writes machine-specific Callgrind files under
`benches/iai/target/iai/` (gitignored). The committed text artifact is the readable, normalized
copy: for each case it records `Instructions`, `L1 Hits`, `LL Hits`, `RAM Hits`,
`Total read+write` and `Estimated Cycles` as `current|baseline`. When a baseline file is present,
`|N/A` in the right column means the runner had no earlier value for that case.

## Criterion setup

The five existing Criterion bench binaries are unchanged in scope. Refresh with:

```bash
make bench-criterion-baseline   # saves `committed` in target/criterion and rewrites the text artifact
make bench-criterion-compare    # compares against the locally saved `committed` baseline
make bench                      # ordinary Criterion run, no baseline flags
```

The Arrow round-trip case allocates a 250k x 384 fixture, so the refresh target is intentionally
explicit and slow. Run both refresh targets only on an idle, documented machine.

## Exported benchmark case names

Criterion (`group/case`):

- `gguf_matvec/f32`, `gguf_matvec/f16`, `gguf_matvec/bf16`, `gguf_matvec/q8_0`,
  `gguf_matvec/q4_0`, `gguf_matvec/q2_k`, `gguf_matvec/q3_k`, `gguf_matvec/q4_k`,
  `gguf_matvec/q5_k`, `gguf_matvec/q6_k`
- `quant_dot/dot_q8_0`, `quant_dot/dot_q4_0`
- `dot_f32/16`, `dot_f32/256`, `dot_f32/4096`
- `softmax_into/16`, `softmax_into/256`, `softmax_into/4096`
- `sampler/argmax`, `sampler/softmax`, `sampler/sample`
- `arrow_f32_matrix/encode_ipc_250k_x_384`,
  `arrow_f32_matrix/decode_ipc_and_borrow_slices_250k_x_384`

iai-callgrind (`group::function id`, from the harness in `benches/iai/benches/hot_kernels.rs`):

- `hot_kernels::simd_kernels::dot_f32 dot_256`
- `hot_kernels::simd_kernels::dot_f32 dot_4096`
- `hot_kernels::simd_kernels::softmax_into softmax_256`
- `hot_kernels::simd_kernels::softmax_into softmax_4096`

These names are part of the baseline contract: rename a case only when you intentionally create a
new baseline series.
