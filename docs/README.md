# Documentation index

This directory is the project's design and operations reference. The repository root
[`README.md`](../README.md) is the quick-start entry point; [`roadmap.md`](roadmap.md) is the
source of truth for what is implemented and what remains open.

> **Validation model:** local checks are authoritative in this environment. Run `make ci` (plus
> `make precommit`, `make coverage-floors` and `make fuzz` for release candidates) before claiming a
> gate is green. The committed GitHub Actions workflow is a convenience for repositories where
> remote CI is enabled; it is not required for local validation here.

## Start here

| Document | Answers |
|----------|---------|
| [roadmap.md](roadmap.md) | Phased plan, per-item status, parity matrix and acceptance criteria. Tick a row only when its criteria pass locally. |
| [architecture.md](architecture.md) | Rust package graph, data flow, native runtime and the HTTP/Flight planes. |
| [gates.md](gates.md) | What each gate checks and the exact local command that runs it. |
| [runtime.md](runtime.md) | Native GGUF runtime, supported families/kernels and real-model golden checks. |

## Build and operate

| Document | Answers |
|----------|---------|
| [containers.md](containers.md) | CPU/CUDA/distroless images, build arguments and reproducible builds. |
| [metrics.md](metrics.md) | Every dashboard panel and alert, with emitted-vs-reserved metric status. |
| [performance.md](performance.md) | Benchmark methodology, baselines and regression workflow. |
| [testplan.md](testplan.md) | Test layers, fixtures, fuzzing and the Go/Python benchtool driver. |
| [`pythonsdk/`](../pythonsdk/README.md) | `pyarrow.flight` Agent client, schema specs and the opt-in end-to-end Flight suite. |

## Local gate sequence

```bash
make ci                # fmt, fuzz formatting, precommit, clippy, tests, security, contracts, charts, containers
make precommit         # fast unit-test + rustfmt hook, mandatory before every commit
make coverage-floors   # per-area line/region floors
make fuzz              # 60 s per libFuzzer target
make test-benchtool    # Go benchtool and Python driver tests
make test-pythonsdk    # pyarrow Flight SDK tests
make shellcheck        # shell scripts
QUARREL_MODEL_DIR=/path/to/models cargo test -p quarrel-runtime --test real_models
```

A roadmap row may be ticked only when the relevant local gates above pass. Gates that need
external evidence (recorded Go oracle fixtures, GPU hardware, two consecutive remote PR runs)
stay open until that evidence exists.
