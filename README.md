# quarrel-rs

Apache Arrow based inference engine in Rust — a clean-room rewrite of
[longbow-quarrel](https://github.com/23skdu/longbow-quarrel).

> **Status: Phase 0 complete; Phases 1–5 and the deployment plane landing incrementally.** The
> OpenAI-compatible HTTP plane, embedded WebUI, Ollama shims, generated OpenAPI contract, Prometheus
> registry, Arrow Flight service, container variants, compose stack and Helm chart are implemented
> and tested. `quarrel-core`, `quarrel-simd`, `quarrel-gguf`, `quarrel-engine` and `quarrel-flight`
> provide the domain model, kernels, GGUF reader, sampler and vector plane. Inference currently uses
> a deterministic reference backend, not Go parity; see [docs/roadmap.md](docs/roadmap.md).

## What works today

- `quarrel-server`: axum API with `/healthz`, `/readyz`, `/v1/models`, chat/completions/embeddings,
  SSE streaming, grammar-constrained tool calls, `/metrics`, OpenAI-shaped errors, bearer-token auth,
  rate limiting and graceful shutdown. Inference is served by the deterministic `ReferenceBackend`
  fixture.
- Generated OpenAPI 3.1 document at `/openapi.json` with Swagger UI at `/docs`; contract replay
  tests and fuzz targets (`gguf_parse`, `sampling_params`, `arrow_ipc`) are green.
- Prometheus metrics on the same port; compose stack with Prometheus + Grafana provisioning and a
  documented dashboard ([metrics](docs/metrics.md)).
- Helm chart with deployment, service, model PVC, GPU resources, probes, NetworkPolicy, HPA,
  ServiceMonitor and PrometheusRule.
- Reproducible CPU, CUDA-base and distroless container images ([containers](docs/containers.md)).
- `quarrel-core` (typed IDs, config, universal resolver, chat templates), `quarrel-simd`
  (runtime-dispatched kernels), `quarrel-gguf` (GGUF v1–v3 read/write, mmap-backed tensor views,
  quant kernels and `quarrel-quantize` CLI),
  `quarrel-engine` (scheduler, batch backend trait, block-table KV, radix prefix cache, sampler,
  reference backend), `quarrel-backend-cpu` (deterministic surrogate batch forward pass) and
  `quarrel-flight` (Arrow schemas, adapters, tonic Flight service and generate/embed paths).
- `pythonsdk/`: `pyarrow.flight` client with generate/embed round trips, health action and a notebook
  example, verified against the local Flight example server.
- Developer tooling: `make tools`, `make makehooks` (unit-test pre-commit hook), `make ci`.
  Local `make ci` is authoritative in this environment; remote GitHub CI is disabled.
- Reproducible quality/performance checks via the Go `benchtool` and `scripts/bench.py`
  (see [performance](docs/performance.md) and [test plan](docs/testplan.md)).

## Quick start

```bash
make tools            # install Homebrew dev tools (idempotent)
make makehooks        # install the pre-commit hook that runs unit tests
make check-tools      # verify the toolchain
make fmt lint test    # local developer loop
make run              # http://localhost:8080/docs
```

Containerised monitoring stack:

```bash
make docker-up        # quarrel :8080, Prometheus :9090, Grafana :3000 (admin/admin)
make docker-down
make helm-lint helm-template
```

Place `.gguf` models in `./models/` (mounted read-only into the container and chart).

## Interfaces

| Interface | Endpoint | Status |
|-----------|----------|--------|
| OpenAI-compatible HTTP | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` | Implemented on the reference backend, including grammar-constrained tool calls (no Go parity claim) |
| OpenAPI 3.1 / Swagger UI | `/openapi.json`, `/docs` | Implemented from handlers |
| Embedded WebUI | `/` | Implemented (self-contained HTML, no CDN) |
| Ollama compatibility | `/api/generate`, `/api/chat` | Implemented with NDJSON streaming on the reference backend |
| Prometheus | `/metrics` | Implemented (control-plane metrics; §7 names contract-checked) |
| Arrow Flight (gRPC) | `/quarrel/v1/generate`, `/quarrel/v1/embed`, `/quarrel/v1/health` | Implemented; schemas in [api/flight-schema.md](api/flight-schema.md), client in [pythonsdk](pythonsdk/README.md) |

## Layout

```text
crates/quarrel-server/   bootstrap control plane (later split into the crates in the roadmap)
api/                     Arrow Flight schema + generated OpenAPI document
pythonsdk/               pyarrow.flight client, schema specs, examples and tests
cmd/benchtool/           Go runner for quality/performance checks
config/benchtool/        JSON profiles consumed by benchtool
deploy/                  Prometheus rules and Grafana provisioning/dashboard
helm/quarrel/            Kubernetes chart
scripts/                 pre-commit hook, bench.py driver and helpers
docs/README.md           documentation index and local validation model
docs/roadmap.md          phased rewrite plan, parity matrix and quality gates
docs/architecture.md     Rust package graph and model/inference/switching flows
docs/gates.md            CI/CD gate reference
docs/containers.md       CPU/CUDA/distroless image variants and reproducible builds
docs/metrics.md          dashboard panels, alert rules and emitted vs reserved metrics
docs/performance.md      benchmark methodology and baseline workflow
docs/testplan.md         master test plan and benchtool usage
```

## Links

- [Documentation index](docs/README.md)
- [Technical roadmap](docs/roadmap.md)
- [Architecture and package graph](docs/architecture.md)
- [CI/CD gates](docs/gates.md)
- [Container images](docs/containers.md)
- [Metrics and dashboards](docs/metrics.md)
- [Performance testing](docs/performance.md)
- [Test plan](docs/testplan.md)
- [Arrow Flight contract](api/flight-schema.md)
- [Python Flight SDK](pythonsdk/README.md)
- [Upstream Go engine](https://github.com/23skdu/longbow-quarrel)
- [Rust API guidelines](https://rust-lang.github.io/api-guidelines/)
