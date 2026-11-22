# Container images

Three Dockerfiles share one pinned Rust toolchain (`1.100.0`) and the committed `Cargo.lock`.

| Dockerfile | Base | Use |
|------------|------|-----|
| `Dockerfile` | `rust:1.100.0-bookworm` → `debian:bookworm-slim` | Default CPU image used by `docker compose` and `make docker-build` |
| `Dockerfile.cuda` | `nvidia/cuda:12.8.1-devel-ubuntu22.04` → `nvidia/cuda:12.8.1-runtime-ubuntu22.04` | Deployment target prepared for the Phase 6 GPU backend |
| `Dockerfile.distroless` | `rust:1.100.0-bookworm` → `gcr.io/distroless/cc-debian12:nonroot` | Minimal runtime with no shell or package manager |

> The CUDA image currently ships the same deterministic CPU reference backend as the other
> variants. It is a deployment target, not a GPU-parity claim; the P6 parity gates remain open.

## Build

```bash
make container-check          # Dockerfile lint without building
make docker-build             # alias for docker-build-cpu
make docker-build-cuda
make docker-build-distroless
make docker-build-all
```

`CUDA_VERSION`, `IMAGE`, `VERSION` and `REVISION` can be overridden on the command line, e.g.

```bash
make docker-build-cuda CUDA_VERSION=13.0.0 IMAGE=myregistry/quarrel:cuda
```

## Reproducibility

Every image receives `VERSION`, `REVISION` and `SOURCE_DATE_EPOCH` build arguments. By default the
Makefile derives them from `git describe`, `git rev-parse HEAD` and the commit timestamp, and the
builder sets:

- `CARGO_INCREMENTAL=0`
- `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1`
- `SOURCE_DATE_EPOCH` in the environment

The OCI labels record the same values. Pinning the compiler, lockfile and build arguments makes
rebuilds byte-stable for a given source revision and base image digest.

## Runtime notes

- All images run as a non-root user and listen on `0.0.0.0:8080`.
- The default and CUDA images include `curl` for a Docker `HEALTHCHECK`; the distroless image has
  no shell, so Kubernetes probes (or compose `depends_on`) own health gating.
- Mount models at `/models` and set `QUARREL_MODELS_DIR` if your layout differs.
- The CUDA image sets `NVIDIA_VISIBLE_DEVICES=all` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`; restrict these in production.
