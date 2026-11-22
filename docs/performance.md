# Performance testing and baselines

This document defines how quarrel-rs measures performance, stores baselines and detects regressions.
The implementation is the Go [`cmd/benchtool`](../cmd/benchtool/main.go) plus the Python driver
[`scripts/bench.py`](../scripts/bench.py). Gate policy lives in [gates](gates.md); the overall test
strategy lives in [the test plan](testplan.md).

## 1. Principles

1. **One runner, many commands.** benchtool never implements a benchmark itself; it invokes the
   repository's Makefile/cargo commands so CI and a laptop run the same thing.
2. **Wall clock is necessary but not sufficient.** Criterion provides statistically meaningful
   per-kernel numbers; benchtool records its own wall time to detect setup/teardown regressions.
3. **Baselines are committed artefacts.** A change in performance must be visible in a diff, not
   just in a developer's terminal.
4. **Quality and performance share a harness.** A faster build that fails clippy is not faster.
5. **No silent tolerance changes.** A regression threshold change requires an explanation in the PR.

## 2. Toolchain layout

```text
cmd/benchtool/main.go          Go runner: config → execute → compare → JSON/Markdown
config/benchtool/quality.json  blocking quality commands
config/benchtool/performance.json  timed cargo benchmarks
scripts/bench.py               Python profile selector and CI entry point
artifacts/bench/results.json   latest raw results (gitignored)
artifacts/bench/results.md     latest human-readable report (gitignored)
artifacts/bench/baseline.json  promoted baseline (committed when intentionally updated)
```

The tooling is tested like any other code: `make test-benchtool` runs `gofmt`, `go vet`, Go unit
tests (90 %+ statement coverage), Python bytecode compilation, Python unit tests and a profile
dry-run. GitHub Actions runs the same target in the `tooling` job.

Build and run directly:

```bash
go build -o target/benchtool ./cmd/benchtool
./target/benchtool \
  -config config/benchtool/performance.json \
  -root . \
  -out artifacts/bench/results.json \
  -markdown artifacts/bench/results.md \
  -baseline artifacts/bench/baseline.json
```

Or use the Python driver, which is the preferred interface:

```bash
python3 scripts/bench.py --profile performance
python3 scripts/bench.py --profile quality
python3 scripts/bench.py --profile regression --baseline artifacts/bench/baseline.json
```

`make` wrappers are available as `make bench-quality`, `make bench-performance` and
`make bench-regression`.

## 3. Profiles

| Profile | Config | Purpose |
|---------|--------|---------|
| `quality` | `quality.json` | rustfmt, clippy, unit/integration tests, security, contract checks, shellcheck, Helm, compose |
| `performance` | `performance.json` | release build, GGUF quant kernels, engine sampler, OpenAPI generation |
| `regression` / `all` | merged quality + performance | full gate replay for release candidates |

Quality checks run once and must exit with the expected code (normally `0`). Performance checks
run a configured number of iterations after optional warm-up runs. The mean of those iterations is
the measured value used for baseline comparison.

### 3.1 Config schema

```json
{
  "name": "performance",
  "warmup": 0,
  "iterations": 3,
  "timeout_seconds": 900,
  "tolerance": 0.1,
  "commands": [
    {
      "name": "gguf-quant-kernels",
      "kind": "performance",
      "command": "cargo",
      "args": ["bench", "-p", "quarrel-gguf", "--bench", "quant", "--", "--warm-up-time", "1"],
      "iterations": 3
    }
  ]
}
```

- `kind`: `quality` (exit-code gate) or `performance` (timed gate).
- `expect_exit`: expected exit code; defaults to `0`.
- `workdir`: relative to `--root`; defaults to the repository root.
- `timeout_seconds`: per-command timeout; overrides the config default.
- `tolerance`: allowed positive delta versus baseline, e.g. `0.1` means 10 %.
- `iterations` / `warmup`: per-command overrides of the config defaults.

Flags on the Go tool (`-iterations`, `-warmup`, `-tolerance`, `-timeout`, `-filter`, `-fail-fast`)
override the config for quick experiments. The Python driver exposes the same overrides.

## 4. Baselines

Create or refresh a baseline only from a clean checkout with all other gates green:

```bash
python3 scripts/bench.py --profile performance --update-baseline
```

This copies `artifacts/bench/results.json` to `artifacts/bench/baseline.json`. The diff must be
included in the PR with a short explanation of why the new numbers are acceptable.

Comparison rules:

- A check without a baseline is reported but not treated as a regression.
- A performance check fails when `(mean - baseline_mean) / baseline_mean > tolerance`.
- A quality check fails when its exit code differs from `expect_exit`, even if it got faster.
- Missing baseline entries are informational; renamed checks are new baselines.

## 5. What is measured today

| Area | Command | Unit | Notes |
|------|---------|------|-------|
| GGUF quant kernels | `cargo bench -p quarrel-gguf --bench quant` | ns/element, ns/dot | Q8_0 and Q4_0 scalar kernels |
| Sampler | `cargo bench -p quarrel-engine --bench sampler` | µs/sample | top-k/top-p/repetition paths |
| Release build | `make build-release` | seconds | Catches dependency and LTO regressions |
| OpenAPI generation | `quarrel-server --print-openapi` | milliseconds | Catches DTO/schema bloat |

Planned additions as the engine lands:

- paged KV append/rollback throughput and allocation counts;
- prefix-cache lookup latency at 1k/10k/100k tokens;
- Arrow Flight 250k×384 vector round-trip with allocation assertions;
- end-to-end TTFT and decode tokens/s for the reference backend;
- first-token latency and steady-state RSS for a 4B Q8_0 model.

## 6. Environment discipline

Measurements are only comparable when collected under similar conditions:

- run on an idle machine, with the same OS/kernel and CPU governor;
- use the pinned toolchain from `rust-toolchain.toml`;
- use `--locked` builds so the dependency graph cannot drift;
- record `rustc -V`, `go version`, `python3 --version`, CPU model and available memory in the PR;
- run the release profile (`lto = "thin"`, `codegen-units = 1`, `panic = "abort"`).

CI runners are shared and noisy, so CI uses a 10 % tolerance for performance checks and treats
quality checks as exact. Local baselines should be generated on a dedicated runner when precision
matters.

## 7. Artifacts and CI

Every run writes:

- `artifacts/bench/results.json` — machine-readable timings, exit codes, output tails and deltas;
- `artifacts/bench/results.md` — Markdown table suitable for PR comments;
- command stdout/stderr tails retained in JSON for failed checks.

CI archives both files as build artifacts. When `make ci` runs, quality commands are also executed
natively by GitHub Actions; benchtool is the reproducible replay mechanism, not a replacement for
the blocking workflow.

## 8. Interpreting a failure

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Quality check FAIL | Lint/test/contract drift | Read `output_tail` in `results.json`, fix code, re-run profile |
| Performance check FAIL, delta > tolerance | Real regression or noisy runner | Re-run twice; inspect Criterion output; update baseline only with justification |
| Timeout | Deadlock or excessive benchmark scope | Lower scope, add timeout override to config, investigate |
| Missing baseline entry | Check renamed or added | Run with `--update-baseline` and review the diff |

## 9. Relationship to zero-copy claims

Criterion timings alone do not prove zero-copy. The allocation gate (`make allocation-gate`, Phase
8) must run with a global allocation counter around the Flight payload path. A faster timings run
that allocates a per-token `Vec<f32>` is a regression even if benchtool reports PASS.
