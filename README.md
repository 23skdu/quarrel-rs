# quarrel-rs

Apache Arrow based inference engine in Rust — a clean-room rewrite of
[longbow-quarrel](https://github.com/23skdu/longbow-quarrel).

> **Status: Phase 0 (bootstrap).** The HTTP control plane, generated OpenAPI contract, Prometheus
> registry, Dockerfile, compose stack and Helm chart are implemented and tested. Inference and the
> Arrow Flight vector plane are planned; see [docs/roadmap.md](docs/roadmap.md).

## What works today

- `quarrel-server`: axum API with `/healthz`, `/readyz`, `/v1/models`, `/metrics`, OpenAI-shaped
  errors for the not-yet-implemented inference routes, bearer-token auth and graceful shutdown.
- Generated OpenAPI 3.1 document at `/openapi.json` with Swagger UI at `/docs`.
- Prometheus metrics on the same port; compose stack with Prometheus + Grafana provisioning.
- Helm chart with deployment, service, model PVC, ingress, HPA, ServiceMonitor and PrometheusRule.
- Developer tooling: `make tools`, `make makehooks` (unit-test pre-commit hook), `make ci`.

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
| OpenAI-compatible HTTP | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` | Contract + skeleton; inference in Phase 5 |
| OpenAPI 3.1 / Swagger UI | `/openapi.json`, `/docs` | Implemented |
| Prometheus | `/metrics` | Implemented (control-plane metrics) |
| Arrow Flight (gRPC) | `/quarrel/v1/*` | Schema defined in [api/flight-schema.md](api/flight-schema.md), implementation in Phase 4 |

## Layout

```text
crates/quarrel-server/   bootstrap control plane (later split into the crates in the roadmap)
api/                     Arrow Flight schema + generated OpenAPI document
deploy/                  Prometheus rules and Grafana provisioning/dashboard
helm/quarrel/            Kubernetes chart
scripts/                 pre-commit hook and helpers
docs/roadmap.md          phased rewrite plan, parity matrix and quality gates
```

## Links

- [Technical roadmap](docs/roadmap.md)
- [Arrow Flight contract](api/flight-schema.md)
- [Upstream Go engine](https://github.com/23skdu/longbow-quarrel)
- [Rust API guidelines](https://rust-lang.github.io/api-guidelines/)
