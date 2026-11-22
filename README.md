# quarrel-rs

Apache Arrow based inference engine in Rust — a clean-room rewrite of
[longbow-quarrel](https://github.com/23skdu/longbow-quarrel).

> **Status: Phase 0 complete; Phases 1–5 landing incrementally.** The OpenAI-compatible HTTP plane,
> generated OpenAPI contract, Prometheus registry, Arrow Flight service, Dockerfile, compose stack
> and Helm chart are implemented and tested. `quarrel-core`, `quarrel-simd`, `quarrel-gguf`,
> `quarrel-engine` and `quarrel-flight` provide the domain model, kernels, GGUF reader, sampler and
> vector plane. Inference currently uses a deterministic reference backend, not Go parity; see
> [docs/roadmap.md](docs/roadmap.md).

## What works today

- `quarrel-server`: axum API with `/healthz`, `/readyz`, `/v1/models`, chat/completions/embeddings,
  SSE streaming, `/metrics`, OpenAI-shaped errors, bearer-token auth, rate limiting and graceful
  shutdown. Inference is served by the deterministic `ReferenceBackend` fixture.
- Generated OpenAPI 3.1 document at `/openapi.json` with Swagger UI at `/docs`; contract replay
  tests and fuzz targets (`gguf_parse`, `sampling_params`, `arrow_ipc`) are green.
- Prometheus metrics on the same port; compose stack with Prometheus + Grafana provisioning.
- Helm chart with deployment, service, model PVC, ingress, HPA, ServiceMonitor and PrometheusRule.
- `quarrel-core` (typed IDs, config, resolver, chat templates), `quarrel-simd` (runtime-dispatched
  kernels), `quarrel-gguf` (GGUF v1–v3, mmap-backed tensor views, quant kernels), `quarrel-engine`
  (sampler, KV/prefix caches, reference backend) and `quarrel-flight` (Arrow schemas, adapters and
  tonic Flight service).
- Developer tooling: `make tools`, `make makehooks` (unit-test pre-commit hook), `make ci`.
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
| OpenAI-compatible HTTP | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` | Implemented on the reference backend (no Go parity claim) |
| OpenAPI 3.1 / Swagger UI | `/openapi.json`, `/docs` | Implemented from handlers |
| Prometheus | `/metrics` | Implemented (control-plane metrics) |
| Arrow Flight (gRPC) | `/quarrel/v1/*` | Implemented; schemas in [api/flight-schema.md](api/flight-schema.md) |

## Layout

```text
crates/quarrel-server/   bootstrap control plane (later split into the crates in the roadmap)
api/                     Arrow Flight schema + generated OpenAPI document
cmd/benchtool/           Go runner for quality/performance checks
config/benchtool/        JSON profiles consumed by benchtool
deploy/                  Prometheus rules and Grafana provisioning/dashboard
helm/quarrel/            Kubernetes chart
scripts/                 pre-commit hook, bench.py driver and helpers
docs/roadmap.md          phased rewrite plan, parity matrix and quality gates
docs/architecture.md     Rust package graph and model/inference/switching flows
docs/gates.md            CI/CD gate reference
docs/performance.md      benchmark methodology and baseline workflow
docs/testplan.md         master test plan and benchtool usage
```

## Links

- [Technical roadmap](docs/roadmap.md)
- [Architecture and package graph](docs/architecture.md)
- [CI/CD gates](docs/gates.md)
- [Performance testing](docs/performance.md)
- [Test plan](docs/testplan.md)
- [Arrow Flight contract](api/flight-schema.md)
- [Upstream Go engine](https://github.com/23skdu/longbow-quarrel)
- [Rust API guidelines](https://rust-lang.github.io/api-guidelines/)
