# Test plan

This is the master test plan for quarrel-rs. It documents what is tested, how, with which tooling,
and what must be true before a change can be merged or a release can be tagged. It complements the
[roadmap](roadmap.md) (what is being built), [architecture](architecture.md) (how it fits together),
[gates](gates.md) (what CI enforces) and [performance](performance.md) (how benchmarks are run).

## 1. Purpose and scope

Verify that the Rust rewrite:

1. preserves the memory-efficient inference semantics of the upstream Go engine;
2. exposes a zero-copy Arrow Flight plane for Agents;
3. remains compatible with unmodified OpenAI SDKs over HTTP;
4. is safe, secure, observable and deployable;
5. has performance numbers that are reproducible and regression-detectable.

In scope: all workspace crates, the fuzz harness, the Go benchtool, the Python driver and Flight
SDK, container
images, Helm chart and monitoring assets.

Out of scope for v1.0: multi-GPU tensor parallelism, GPTQ/AWQ/Vulkan/ROCm/WebGPU backends and the
Python inference rewrite listed in roadmap §1.2.

## 2. Test levels

| Level | What it covers | Primary command | Blocking |
|-------|----------------|-----------------|----------|
| Unit | Pure logic in every crate, including parsers and samplers | `make test-unit` / `cargo test --workspace --lib --bins` | Yes |
| Integration | axum `oneshot` routes, in-process Flight server, config file round-trips | `make test` / `cargo nextest run --workspace` | Yes |
| Property | Invariants for arbitrary inputs to parsers, sampler and template renderer | `cargo test -p <crate>` (proptest) | Yes |
| Fuzz | Crash-free parsing/decoding of arbitrary bytes | `make fuzz-smoke` (PR), `make fuzz` (nightly) | Yes before release |
| Oracle | Go-recorded logits/tokens/KV bytes | `make oracle-gate` | Yes before release |
| Mutation | Test-suite strength on core/SIMD/sampler | `make mutants` | Yes before release |
| Contract | OpenAPI drift, metric-name drift, Arrow IPC golden files | `make openapi-check metrics-check` + crate tests | Yes |
| Python SDK | Schema specs, request builders and `pyarrow.flight` round trips | `make test-pythonsdk`; opt-in `QUARREL_FLIGHT_URI=...` integration suite | Yes |
| Allocation / leak | Payload-copy counters in matvec and Flight paths; 16-cycle server leak gate | `cargo test -p quarrel-gguf -p quarrel-flight -p quarrel-server` | Yes |
| Security | Advisories, licenses, duplicate versions, shell scripts | `make security shellcheck` | Yes |
| Deployment | Helm render/lint, compose config, image healthcheck | `make helm-lint helm-template`, `docker compose config` | Yes |
| Performance | Criterion benches and benchtool baselines | `make bench`, `python3 scripts/bench.py --profile performance` | Yes before release |
| Safety | Miri on unsafe/FFI-adjacent crates | `make miri` | Yes before release |

### 2.1 Coverage floors

Enforced by `make coverage-floors`:

| Area | Minimum line coverage |
|------|-----------------------|
| `quarrel-core`, `quarrel-gguf`, `quarrel-simd` | 85 % |
| `quarrel-engine`, `quarrel-api`, `quarrel-server` | 80 % |

Coverage is a floor, not a substitute for property/fuzz/oracle tests. A parser with 100 % line
coverage can still panic on an input the tests never generated.

## 3. Test environment

| Component | Required version | Source |
|-----------|------------------|--------|
| Rust | 1.100.0 | `rust-toolchain.toml` |
| Go | ≥ 1.24 | benchtool only |
| Python | ≥ 3.11 | `scripts/bench.py`, stdlib only |
| Helm | current stable | `make tools-brew` |
| Docker + Compose | current stable | local stack validation |
| cargo-fuzz + nightly | current nightly | `make tools-cargo`, `make tools-miri` |

Set up a clean environment with:

```bash
make tools
make makehooks
make check-tools
```

## 4. Test data

| Fixture set | Location | Used by |
|-------------|----------|---------|
| Synthetic GGUF v1/v2/v3 files | `quarrel-gguf` tests | header/metadata/tensor parsing |
| Truncated/mutated GGUF bytes | proptest + fuzz corpus | parser robustness |
| Arrow IPC golden batches | `quarrel-flight/tests/fixtures` | schema drift, zero-copy adapters |
| Recorded Go logits/tokens/KV blocks | `tests/oracle/` (Phase 1) | oracle gate |
| OpenAI SDK request/response pairs | `quarrel-api` contract tests | HTTP compatibility |
| Performance baselines | `artifacts/bench/baseline.json` | regression detection |

Test data must be small, deterministic and checked in. Never download fixtures during a test run;
network access makes tests non-reproducible.

## 5. Tooling: Go benchtool and Python driver

`cmd/benchtool` is a dependency-free Go CLI that executes commands from a JSON config, records wall
time and exit codes, compares performance checks against a baseline and emits JSON/Markdown
artifacts. `scripts/bench.py` selects profiles, builds the tool, merges configs and prints a summary.

### 5.1 Commands

```bash
# Quality gates only
python3 scripts/bench.py --profile quality

# Timed benchmarks only
python3 scripts/bench.py --profile performance

# Full release-candidate replay
python3 scripts/bench.py --profile regression

# Promote a new baseline (requires review)
python3 scripts/bench.py --profile performance --update-baseline

# Investigate one check without rebuilding
python3 scripts/bench.py --profile regression --filter sampler --rebuild
```

Makefile wrappers:

```bash
make bench-quality
make bench-performance
make bench-regression
```

The tooling itself is covered by `make test-benchtool`: Go unit tests, `go vet`, `gofmt`, Python
unit tests and a `--dry-run` profile resolution check. This target is blocking in the CI `tooling`
job so a broken benchmark harness cannot silently invalidate performance evidence.

### 5.2 Quality checks performed

The `quality` profile runs: `make fmt-check`, `make lint`, `make test-unit`, `make test`,
`make security`, `make openapi-check`, `make metrics-check`, `make shellcheck`, `make helm-lint`
and `docker compose config --quiet`.

### 5.3 Performance checks performed

The `performance` profile runs the release build, `quarrel-gguf` quant-kernel Criterion bench,
`quarrel-engine` sampler Criterion bench and OpenAPI generation. See [performance](performance.md)
for config schema, tolerance rules and baseline workflow.

### 5.4 Accuracy checks

Accuracy is not inferred from quality or performance results. The oracle gate replays recorded Go
fixtures and asserts:

- FP32 logits within `1e-3` absolute / `1e-4` relative tolerance;
- sampled token IDs exactly equal for deterministic seeds;
- quantized integer formats within 1 ULP;
- KV block bytes byte-for-byte equal where the codec is deterministic.

Until the recorded Go fixtures land (roadmap P1-03), reference-backend snapshots prove determinism
but **not** Go parity, and P1-03 stays open.

## 6. Phase traceability

| Roadmap phase | Test focus | Evidence |
|---------------|------------|----------|
| P0 | Bootstrap HTTP/metrics/chart | 12 unit tests, `make ci`, smoke endpoints |
| P1 | Domain types, config round-trip, proptest/fuzz infra | core unit + property tests, fuzz targets |
| P2 | GGUF parsing, mmap, quant kernels, resolver | fixture tests, truncated-input tests, quant oracle vectors |
| P3 | Sampler, KV cache, prefix cache, template renderer, health audit | property tests, snapshots, release tests |
| P4 | Arrow schemas, adapters, Flight service | round-trips, IPC goldens, in-process client test |
| P5 | OpenAI DTOs/SSE/OpenAPI/auth | SDK replay, contract tests, SSE cancellation test |
| P6 | CPU/CUDA/Metal kernel parity | CPU vs GPU parity tests, element-size audit |
| P7 | Metrics/dashboards/chart | `metrics-check`, `helm unittest`, dashboard smoke |
| P8 | Correctness/performance gates | oracle, mutation, allocation, leak, CI matrix |
| P9 | Release | parity report, SBOM, signed images |

## 7. Entry and exit criteria

### 7.1 Entry criteria for a PR

- branch is based on current `main`;
- `make precommit` passes locally;
- new behaviour has unit tests and, for parsers/decoders, property tests;
- public API changes update OpenAPI/Arrow schema docs and regenerate artifacts;
- performance-sensitive changes include a benchmark or an explanation of why not.

### 7.2 Exit criteria for merge

- all blocking gates in [gates](gates.md) are green;
- no surviving mutants in changed core/SIMD/sampler code;
- coverage floors hold;
- contract drift checks pass;
- review sign-off from one maintainer.

### 7.3 Exit criteria for release

- all PR gates plus deep quality gates green on the tagged commit;
- baseline diff reviewed and committed;
- parity report and performance numbers reproducible from committed commands;
- Helm chart and container image validated from the release artefact, not a developer tree.

## 8. Defect classification

| Severity | Definition | Response |
|----------|------------|----------|
| S0 | Data loss, UB, security breach, wrong tokens served silently | Stop release, hotfix, regression test mandatory |
| S1 | Core feature broken or > 2× performance regression | Block merge until fixed |
| S2 | Non-critical feature degraded, flaky test, minor drift | Fix in current sprint |
| S3 | Cosmetic/docs/tooling issue | Backlog |

Every fixed defect needs a regression test that fails without the fix. Security defects follow the
RustSec advisory process and must be recorded in `deny.toml` only as a temporary, linked waiver.

## 9. Flakiness policy

- A flaky test is a bug, not a retry policy.
- nextest retries are disabled by default; a retry may be used only to diagnose, never to make CI
  green.
- Time-based tests must use tokio's paused time or injectable clocks.
- Benchtool regressions are re-run twice on the same runner before a baseline is changed; the PR
  must record both runs.

## 10. Reporting and sign-off

CI archives `artifacts/bench/results.json` and `results.md`, coverage HTML and nextest output. The
release checklist links those artifacts plus `docs/performance.md` and the parity report. Sign-off
requires the release owner and one maintainer. Any waived gate must be listed explicitly with an
expiry date and tracking issue.
