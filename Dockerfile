# syntax=docker/dockerfile:1.7

# ---- builder -----------------------------------------------------------------
ARG RUST_VERSION=1.100.0
FROM rust:${RUST_VERSION}-bookworm AS builder

# aws-lc-sys (pulled in by rustls through the vendored Swagger UI) needs cmake.
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake pkg-config perl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates

RUN cargo build --release --locked -p quarrel-server

# ---- runtime -----------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

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
