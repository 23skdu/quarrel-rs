# CI/CD gates

This document is the operational companion to [the roadmap](roadmap.md). It describes every gate
that protects `main`, how to run it locally, and what evidence a green CI run produces. Benchmark
methodology is in [performance](performance.md); the full strategy is in [the test plan](testplan.md).

> **Rule:** a gate is either automated and blocking, or it is not a gate. Manual "we checked it"
> steps belong in review checklists, not in this document.

## 1. Gate layers

| Layer | When | Commands | Blocking? |
|-------|------|----------|-----------|
| Commit | every `git commit` | `scripts/pre-commit` via `make makehooks` | Yes, locally |
| Pull request | every push/PR to `main` | `.github/workflows/ci.yml` (including `make precommit` and fuzz smoke) | Yes, on GitHub |
| Security | every PR | `make security` (`cargo audit` + `cargo deny`) | Yes |
| Coverage | every PR once floors are enabled | `make coverage-floors`, `make coverage` | Yes |
| Contract | every PR | `make openapi-check`, `make metrics-check`, `make helm-lint` | Yes |
| Tooling | every PR | `make test-benchtool`, `python3 scripts/bench.py --profile quality --dry-run` | Yes |
| Deep quality | nightly / release candidate | `make fuzz`, `make mutants`, `make miri`, `make bench-regression` | Yes before release |
| Release | tag `v*` | SBOM, image signing, parity report | Yes before publish |

The pre-commit hook is intentionally fast: rustfmt and unit tests only. Everything else runs in
CI so commits stay quick while `main` remains protected.

## 2. Commit-time gate

`make makehooks` installs `scripts/pre-commit` through `git rev-parse --git-path hooks`, so it
honours both git worktrees and a custom `core.hooksPath`. The hook:

1. refuses to run if `cargo` is not on `PATH`;
2. runs `cargo fmt --all -- --check`;
3. runs `cargo test --workspace --locked --lib --bins` — **unit tests are mandatory**;
4. exits non-zero on the first failure.

Run it manually with `make precommit`. Never use `--no-verify` as a normal workflow; if the hook is
wrong, fix the hook.

## 3. Pull-request gates

`.github/workflows/ci.yml` currently has four jobs:

| Job | Steps | Rationale |
|-----|-------|-----------|
| `rust` | `make fmt-check`, `make fuzz-fmt-check`, `make precommit`, `make lint`, `make test`, `make security`, `make openapi-check`, `make metrics-check` | Source, tests, supply chain and API contract |
| `charts` | `make helm-lint`, `make helm-template`, `docker compose config --quiet` | Deployment artefacts are valid YAML and render cleanly |
| `tooling` | `make test-benchtool`, benchtool dry-run | The Go/Python test tooling itself is tested like any other code |
| `fuzz` | `make fuzz-smoke` on nightly with `cargo-fuzz` | Parser/sampler/Arrow decoders survive malformed input |

Dependabot PRs use exactly the same workflow. Major Cargo upgrades are deliberately excluded from
bot automation because the MSRV is pinned in `rust-toolchain.toml`.

### 3.1 Source and style

- `make fmt-check` — `cargo fmt --all -- --check`; no unformatted Rust enters `main`.
- `make lint` — `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`.
  Any warning is an error, including redundant documentation attributes and needless returns.
- `make test` — `cargo nextest run --workspace --locked` when `cargo-nextest` is installed, with a
  `cargo test --workspace --locked` fallback for minimal environments.

### 3.2 Supply chain

- `make audit` — `cargo audit` against the RustSec advisory database.
- `make deny` — `cargo deny check` with the policy in `deny.toml`: advisories denied, yanked crates
  denied, unknown registries denied, duplicate versions reported, explicit license allowlist.
- `make security` — both of the above. A vulnerability must be fixed or explicitly waived in
  `deny.toml` with an issue link; blanket ignores are not allowed.
- `make shellcheck` — shell scripts (including the pre-commit hook) are checked as shell code.

### 3.3 Tooling checks

The Go benchtool and its Python driver are part of the quality infrastructure, so they are tested
in CI too:

- `go test ./cmd/benchtool` — unit tests for config parsing, statistics, execution, baselines and
  artifact writing;
- `go vet ./cmd/benchtool` and `gofmt -l cmd/benchtool` — Go source hygiene;
- `python3 -m unittest discover -s scripts` — profile merging, argument construction and CLI
  behaviour of `scripts/bench.py`;
- `python3 scripts/bench.py --profile quality --dry-run` — verifies the committed profiles resolve
  without running the expensive commands.

Run all of them with `make test-benchtool`.

### 3.4 Contract and drift

| Gate | What it compares | Failure means |
|------|------------------|---------------|
| `openapi-check` | `api/openapi.json` vs `quarrel-server --print-openapi` | Regenerate with `make openapi` |
| `metrics-check` | metric constants in code vs Prometheus rules vs Grafana dashboard | A panel or alert silently broke |
| `helm-lint` / `helm-template` | chart metadata and rendered manifests | Deployment broke |
| `docker compose config` | compose file and mounted config paths | Local stack broke |

Arrow IPC golden files are checked by `quarrel-flight` tests. Schema changes require a major
contract bump and regenerated fixtures, not a test edit that blindly accepts new bytes.

### 3.5 Coverage floors

`make coverage-floors` enforces the floors from roadmap §6; `make coverage` writes the HTML report:

| Area | Floor |
|------|-------|
| `quarrel-core`, `quarrel-gguf`, `quarrel-simd` | 85 % lines |
| `quarrel-engine` | 80 % lines |
| `quarrel-api` / `quarrel-server` | 80 % lines |

Coverage is a floor, not a goal: a well-tested parser still needs property and fuzz tests. The HTML
report is written to `target/llvm-cov/html/index.html`.

## 4. Deep quality gates

These are not required for every commit, but are required before a release tag.

| Gate | Command | Purpose |
|------|---------|---------|
| Fuzz | `make fuzz` | 60 s per target: `gguf_parse`, `sampling_params`, `arrow_ipc` |
| Fuzz smoke | `make fuzz-smoke` | Short 5 s per target for PRs when enabled |
| Fuzz coverage | `make fuzz-coverage` | Corpus coverage for fuzz targets, reported with `cargo fuzz coverage` |
| Mutation | `make mutants` | `cargo mutants` on core, SIMD and sampler; no surviving mutants |
| Miri | `make miri` | `cargo +nightly miri test` for unsafe/FFI-adjacent crates |
| Benchmarks | `make bench` | Criterion baselines under `benches/baselines/` |
| Bench replay | `make bench-regression` | Go benchtool replays quality + performance profiles with baseline comparison |
| Allocation | `make allocation-gate` | Zero payload-copy allocations in decode/Flight paths after warm-up |
| Leak | `make leak-gate` | 16 load/search/close cycles with < 5 MB heap growth |
| Oracle | `make oracle-gate` | Recorded Go logits/tokens/KV bytes within tolerance |
| Accuracy | `make accuracy-gate` | Golden generation suite parity report |

The fuzz targets live in the root `fuzz/` crate and are excluded from the main workspace. Each
target exposes a small safe parsing/decoding entry point from its library crate, so fuzzing exercises
production code rather than a copy of it.

## 5. Release gates

A `v*` tag is eligible for release only when:

1. every PR gate above is green on that commit;
2. all deep quality gates have run against the release candidate and been archived as CI artifacts;
3. `docs/performance.md` numbers are reproducible with the committed make commands;
4. the parity report in `docs/parity.md` is current;
5. the container image has an SBOM and is signed (cosign), with `SOURCE_DATE_EPOCH` set for
   reproducible layers;
6. `helm lint` and `docker compose config` pass against the tagged artefacts, not a developer tree.

## 6. Local quick reference

```bash
make precommit          # same checks as the git hook
make ci                 # full PR gate set, including charts and compose
make test-benchtool     # Go + Python tooling tests
make bench-quality      # replay quality gates through benchtool
make bench-performance  # timed checks with baseline comparison
make bench-regression   # quality + performance replay for a release candidate
make coverage-floors    # enforce roadmap coverage floors
make coverage           # HTML coverage report
make fuzz               # 60 s per fuzz target
make fuzz-smoke         # 5 s per fuzz target
make security           # audit + deny
make openapi-check      # API contract drift
make metrics-check      # metric-name drift
make helm-lint helm-template
docker compose config --quiet
```

## 7. Adding a new gate

1. Add a self-documenting target to the root `Makefile` with a `##` comment so `make help` lists it.
2. Add the command to `.github/workflows/ci.yml` (PR gate) or the nightly/release workflow.
3. Document the gate in the tables above, including what it catches and how to reproduce failures.
4. If it is a contract gate, add a drift test so CI fails on mismatch rather than on silence.
5. Update the relevant roadmap checkbox only after the gate passes on a clean checkout.
