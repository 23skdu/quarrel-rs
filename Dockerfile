# syntax=docker/dockerfile:1.7

# ---- builder -----------------------------------------------------------------
ARG RUST_VERSION=1.100.0
ARG CUDA_VERSION=12.8.1
FROM rust:${RUST_VERSION}-bookworm AS builder

# aws-lc-sys (pulled in by rustls through the vendored Swagger UI) needs cmake.
RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake pkg-config perl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates

ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    CARGO_INCREMENTAL=0 \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

RUN cargo build --release --locked -p quarrel-server

# ---- runtime -----------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

ARG VERSION=dev
ARG REVISION=unknown
ARG SOURCE_DATE_EPOCH=0

LABEL org.opencontainers.image.title="quarrel-rs" \
      org.opencontainers.image.description="Arrow-based inference control plane (CPU image)" \
      org.opencontainers.image.source="https://github.com/23skdu/quarrel-rs" \
      org.opencontainers.image.licenses="Unlicense" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${SOURCE_DATE_EPOCH}"

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd --system --gid 1000 quarrel \
  && useradd --system --uid 1000 --gid quarrel --no-create-home --home-dir /app quarrel \
  && mkdir -p /models /app \
  && chown -R quarrel:quarrel /models /app

COPY --from=builder /app/target/release/quarrel-server /usr/local/bin/quarrel

USER quarrel:quarrel
ENV QUARREL_LISTEN=0.0.0.0:8080 \
    QUARREL_MODELS_DIR=/models \
    RUST_LOG=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8080/healthz >/dev/null || exit 1

ENTRYPOINT ["quarrel"]
