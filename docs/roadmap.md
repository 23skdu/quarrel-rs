# quarrel-rs — Technical Roadmap

> **Status:** Phase 0 is implemented; the Phase 1–4 library crates are complete with unit and
> property tests. Fuzz targets (`gguf_parse`, `sampling_params`, `arrow_ipc`) pass 60 s runs. The
> OpenAI/OpenAPI HTTP plane, embedded WebUI, Ollama shims, Python Flight SDK with generate/embed,
> paged KV/batching/radix prefix engine primitives, architecture routing, speculative decoding,
> all quant matvec kernels with a unified AVX2/AVX-512/NEON facade, Criterion/iai-callgrind
> baselines, metric contract, optional OTLP tracing and container/Helm/deployment variants are
> live. A native mmap-backed GGUF runtime now runs real Gemma 3 270M and SmolLM2 360M Q4_K_M
> models through the CLI/library. Go oracle parity, CUDA/Metal backends, LoRA and the remaining
> integration gates stay open until their acceptance criteria pass.
> **Goal:** a clean-room Rust rewrite of [`23skdu/longbow-quarrel`](https://github.com/23skdu/longbow-quarrel)
> that preserves its memory-efficient inference semantics while adding two first-class client planes:
> an **Arrow Flight (gRPC) zero-copy vector plane for Agents**, and an **OpenAI-compatible HTTP API
> with a generated OpenAPI contract and browser UI**.
> **Reference:** upstream v0.3.0 architecture, feature matrix and gates ([upstream docs](https://github.com/23skdu/longbow-quarrel/tree/main/docs)).
> This document is the single source of truth for *what is not finished yet*. When an item lands,
> tick it and move the detail to the commit that closed it. Do not delete completed rows.

---

## 1. Goals, non-goals and success criteria

### 1.1 Goals

| ID | Goal | Measurable outcome |
|----|------|--------------------|
| G1 | Byte-for-byte semantic parity for inference | Golden-vector suite (logits, sampled token IDs, KV blocks) agrees with the Go engine within `1e-3` absolute / `1e-4` relative tolerance on FP32 and within 1 ULP on integer quant formats |
| G2 | Zero-copy Agent plane | An Arrow `RecordBatch` produced by an Agent is passed to the engine and back **without serde-JSON, without a Rust `Vec<f32>` copy of the payload, and without a second IPC encode**; verified with allocation counters and `cargo bench` |
| G3 | OpenAI-compatible HTTP plane | Existing OpenAI SDKs work unchanged against `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/v1/models`, including SSE `stream: true` |
| G4 | Machine-readable API contract | `openapi.json` is generated from the axum handlers, served at `/openapi.json`, and checked for drift in CI |
| G5 | Memory advantage preserved | A 4B Q8_0 model keeps steady-state heap/RSS below 20 MB above the mapped weights, matching the Go engine's headline result |
| G6 | Production operations | Docker image, compose monitoring stack, Helm chart, Prometheus rules and Grafana dashboard all ship in this repository and are linted in CI |
| G7 | Rust-engineering quality | Zero `clippy -D warnings`, zero `cargo deny` advisories, zero unsafe blocks outside `quarrel-simd`/FFI shims, 100% of public items documented |

### 1.2 Non-goals for v1.0

- Multi-GPU tensor/pipeline parallelism (the Go engine explicitly removed unused managers).
- GPTQ, AWQ, Vulkan, ROCm, WebGPU and TPU backends — tracked in Phase 9 only.
- A Python engine rewrite. `pythonsdk/` is reserved for a thin Arrow/HTTP client, not inference.
- Replacing llama.cpp kernels wholesale; we port the *algorithms* and benchmark against the Go oracle.

### 1.3 Definition of done for v1.0.0

Every P0–P2 item is complete; every gate in §8 is green on a clean checkout; every number published
in `docs/performance.md` is reproducible from a committed `make` command; `helm lint` and
`docker compose config` succeed; and a fresh `make ci` passes on a machine with only `brew` installed.

---

## 2. Target architecture

```text
                       Agents / Python SDK / other engines
                                     │
                    Arrow Flight over gRPC (tonic + arrow-flight)
                    FlightData stream: Dictionary + RecordBatch, LZ4/ZSTD
                                     │
    Browser UI ── HTTP/1.1 + SSE ──► axum ── OpenAPI 3.1 / Swagger UI
    OpenAI SDK ── JSON / SSE ───────►  │
                                      ▼
                          quarrel-api (auth, limits, routing)
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
             request adapter    engine service      observability
             (JSON ⇄ Arrow)     (tokio tasks)     (tracing + metrics)
                    │                 │
                    └────────► Arrow record batches (zero-copy)
                                      │
                                      ▼
                            quarrel-engine
                 scheduler · paged KV · prefix radix · sampler
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              quarrel-gguf      quarrel-simd       backend trait
              mmap + dequant    AVX2/AVX512/NEON   CPU │ CUDA │ Metal
                    │
              memory-mapped .gguf weights (never fully materialised)
```

### 2.1 Workspace crate map

| Crate | Responsibility | Depends on | Upstream Go package |
|-------|----------------|------------|---------------------|
| `crates/quarrel-core` | Domain traits, error types, sampling params, request/response models | `thiserror`, `serde` | `internal/models`, `internal/config` |
| `crates/quarrel-gguf` | GGUF header/KV reader, mmap-backed tensors, zero-copy quant dot products | `memmap2`, `bytes`, `quarrel-core` | `internal/gguf` |
| `crates/quarrel-simd` | Runtime-dispatched SIMD kernels + scalar reference oracle | `wide`/`std::arch` | `internal/simd`, `internal/mathx` |
| `crates/quarrel-engine` | Batching, paged KV, radix prefix cache, speculative decoding, LoRA | `tokio`, `quarrel-core` | `internal/engine`, `internal/sampler`, `internal/vector` |
| `crates/quarrel-backend-cpu` | CPU execution of the backend trait | `quarrel-engine`, `quarrel-simd` | `internal/device` (CPU) |
| `crates/quarrel-backend-cuda` | CUDA kernels via `cudarc`, half-precision weights | `quarrel-engine` | `internal/device` (CUDA) |
| `crates/quarrel-backend-metal` | MSL kernels via `metal` (macOS only) | `quarrel-engine` | `internal/device` (Metal) |
| `crates/quarrel-flight` | Arrow schemas, Flight service, zero-copy encode/decode | `arrow`, `arrow-flight`, `tonic` | `internal/api` (Flight server) |
| `crates/quarrel-api` | axum routes, OpenAI DTOs, OpenAPI generation, SSE, WebUI assets | `axum`, `utoipa`, `quarrel-core` | `internal/api` (REST), `cmd/webui` |
| `crates/quarrel-observability` | `tracing` layers, OTLP export, Prometheus registry and metric names | `tracing`, `metrics` | `internal/metrics`, `internal/telemetry` |
| `crates/quarrel-cli` | `clap` binary that wires everything together | all of the above | `cmd/quarrel`, `cmd/simple`, `cmd/quantize` |

Shkeleton rule: **dependencies point inward only**. `quarrel-core` must never depend on a transport,
a backend or `tokio`; this keeps unit tests fast and the domain model reusable from the Python SDK.

### 2.2 Arrow Flight (gRPC) contract for Agents

The Flight service is the preferred Agent interface. HTTP remains a compatibility plane.

**Flight descriptors (paths):**

| Path | Direction | Payload |
|------|-----------|---------|
| `/quarrel/v1/generate` | `DoPut` (agent → engine) | prompt token IDs + sampling parameters |
| `/quarrel/v1/generate` | `DoGet` (engine → agent) | generated tokens, logprobs and finish state |
| `/quarrel/v1/embed` | `DoPut`/`DoGet` | input IDs → pooled/per-token embeddings |
| `/quarrel/v1/kv/blocks` | `DoPut`/`DoGet` | paged KV cache blocks for distributed/offloaded decode |
| `/quarrel/v1/health` | `DoAction` | readiness/Liveness + loaded model descriptor |

**Canonical schemas** (owned by `quarrel-flight`, versioned in `api/flight-schema.md`, asserted by golden IPC fixture tests):

```text
request schema  (FlightDescriptor path = /quarrel/v1/generate)
  request_id      Utf8
  sequence        UInt64
  token_ids       List<Int32>              # dictionary-encoded when repeated
  max_tokens      UInt32
  temperature     Float32
  top_p           Float32
  top_k           UInt32
  stop_token_ids  List<Int32>
  stream          Boolean

response schema
  request_id      Utf8
  sequence        UInt64
  token_id        Int32
  logprob         Float32
  top_logprobs    List<Struct<token_id: Int32, logprob: Float32>>
  finish_reason   Dictionary<UInt16, Utf8> # null until terminal
  finished        Boolean

embedding schema
  request_id      Utf8
  token_id        Int32
  embedding       FixedSizeList<Float16 | Float32, D>

kv block schema
  request_id      Utf8
  block_id        UInt32
  layer_id        UInt16
  dtype           Dictionary<UInt8, Utf8>  # f32 | f16 | turboquant
  blocks          FixedSizeList<UInt8, N>  # raw bytes; no reinterpretation copy
```

**Zero-copy rules (enforced by review and by `cargo bench` allocation assertions):**

1. Buffers arriving from Flight are `arrow::buffer::Buffer` backed by `bytes::Bytes`; keep them as
   `Buffer` slices. Never call `Array::to_data`, `iter().collect::<Vec<_>>()` or `serde_json::to_vec`
   on the hot path.
2. The engine consumes `&[u8]` / `&[f16]` slices into mapped weights and Arrow buffers with
   explicit lifetimes; no `Arc<Vec<T>>` clones per token.
3. Output arrays are built with `arrow::array::PrimitiveBuilder` using pre-sized capacity from a
   reusable `ArrayBuilder` pool; finish with `Freeze` and wrap in `RecordBatch` + `Arc`.
4. Stream with `FlightDataEncoder`/`FlightDataDecoder`; enable `LZ4_FRAME` for token streams and
   `ZSTD` for KV blocks. Do not re-encode a batch received from another Flight peer — forward the
   original `FlightData` frames.
5. Backpressure is gRPC flow control plus a bounded `mpsc::channel` sized from `--flight-queue-depth`;
   never accumulate an unbounded batch queue.
6. Every batch carries `request_id` and monotonic `sequence`; Agents may retry `DoGet` and de-duplicate.
7. Schema compatibility is a SemVer contract: additive fields only within a major version; the
   schema hash is reported in `FlightAction` metadata and in the `quarrel_build_info` metric.

### 2.3 OpenAI-compatible / OpenAPI HTTP plane

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/v1/models` | GET | OpenAI `ModelList`; populated from the model resolver |
| `/v1/chat/completions` | POST | Chat DTOs, tool/function calling, SSE `text/event-stream` when `stream: true` |
| `/v1/completions` | POST | Legacy prompt-completion DTO |
| `/v1/embeddings` | POST | Returns base64 or array embeddings; Arrow path used internally |
| `/v1/rerank` | POST | Optional, uses the vector crate |
| `/api/generate`, `/api/chat` | POST | Ollama-compatible shims (Phase 5) |
| `/openapi.json` | GET | utoipa-generated OpenAPI 3.1 document |
| `/docs` | GET | Swagger UI for the generated contract |
| `/healthz`, `/readyz` | GET | Kubernetes liveness/readiness; `/readyz` checks model load + KV capacity |
| `/metrics` | GET | Prometheus exposition |

Contract rules:

- Request/response DTOs live **only** in `quarrel-api` and derive `Serialize`, `Deserialize`,
  `utoipa::ToSchema`, `schemars::JsonSchema` (when needed) and `PartialEq` for snapshot tests.
- Unknown JSON fields are ignored for forward compatibility; unknown `model` values return an
  OpenAI-shaped `invalid_request_error`, never a Rust panic.
- SSE chunks are `chat.completion.chunk` objects terminated by `data: [DONE]`, with `retry` and
  heartbeat comments for idle streams. A dropped client cancels the tokio task via `CancellationToken`.
- `openapi.json` is generated in a unit test and compared with `api/openapi.json`; CI fails on drift.
- Auth is optional bearer/API-key middleware, off by default for local use, on by default in Helm.

---

## 3. Rust engineering standards (apply to every phase)

### 3.1 Toolchain and dependencies

- Pin the compiler in `rust-toolchain.toml` and install components `rustfmt` and `clippy`.
- Edition 2024, workspace `resolver = "3"`, MSRV documented in the README and asserted in CI.
- Commit `Cargo.lock`; build with `--locked` in CI and in the Docker image.
- Put all version pins in `[workspace.dependencies]`; member crates use `workspace = true`.
- `cargo deny` governs licenses, duplicate versions, bans and advisories; `cargo audit` runs on every PR.
- Prefer pure-Rust crates. Anything needing a C toolchain (`cudarc`, `metal`) gets a feature gate and
  a documented system-dependency section.

### 3.2 Error handling and API design

- Library crates return `thiserror`-derived enums with `#[non_exhaustive]`; `anyhow` is allowed only
  in `quarrel-cli`'s `main` and in `#[cfg(test)]`.
- No `unwrap`/`expect`/`panic!` in library code paths reachable from a request; use `Result` and
  `tracing::warn!`. Panics are confined to `#[cfg(test)]` and invariant assertions with messages.
- Use `#[must_use]` on pure constructors and accessors; `new()` returns `Self` or `Result<Self, Error>`
  with no side effects.
- Typed IDs (`RequestId`, `ModelId`, `BlockId`) via newtypes instead of raw strings/integers.

### 3.3 Ownership, zero-copy and performance

- Model weights are `memmap2::Mmap` owned by a single `Model` struct; kernels take `&[u8]` slices and
  return values, never owned dequantized buffers.
- Arrow buffers are `Arc`-shared `Buffer`s; conversions use `array.data().buffers()` and
  `Buffer::slice_with_length` rather than copying values.
- Async tasks never hold a `std::sync::MutexGuard` across `.await`; use `tokio::sync` primitives,
  `parking_lot` only for short synchronous critical sections.
- Per-token scratch memory comes from reusable pools (`Vec::with_capacity` retained in a `Pool`),
  so the decode loop is allocation-free after warm-up. Assert this with a global allocator counter.
- CPU SIMD is runtime-dispatched once via `is_x86_feature_detected!` / `std::arch::aarch64`, with a
  scalar oracle. `unsafe` is allowed only in `quarrel-simd` and FFI shims, always with a preceding
  `// SAFETY:` comment, `#[inline]`, and coverage in `cargo miri` / `cargo fuzz`.

### 3.4 Observability and security

- Structured `tracing` spans with `request_id`, `model`, `seq` and `operation`; never log prompts or
  token text at default level. OTLP export is optional via feature `otel`.
- Metrics are declared once in `quarrel-observability` (§7); metric names are constants, never inline
  string literals scattered across crates.
- `cargo fmt --check`, `clippy --all-targets --all-features -- -D warnings`, `cargo test`, `cargo nextest`,
  `cargo llvm-cov`, `cargo deny`, `cargo audit`, `cargo mutants`, shellcheck on hooks, `helm lint`,
  `docker compose config` and OpenAPI drift are all local `make` targets. Remote GitHub Actions is
  disabled for this repository; a green local `make ci` is the required evidence.
- Pre-commit hook (`scripts/pre-commit`, installed by `make makehooks`) always runs **unit tests**
  plus formatting; heavier gates run in local `make ci`. Never use `--no-verify` as a normal workflow.

---

## 4. Phased work breakdown

Each subtask is independently reviewable, has a checkbox, and ends with an acceptance criterion.
Phases are ordered by dependency, not by calendar.

### Phase 0 — Bootstrap and developer experience *(this commit)*

- [x] **P0-01** Create Cargo workspace with `resolver = "3"`, edition 2024 and pinned toolchain.
- [x] **P0-02** Add `quarrel-server` skeleton: config from env/CLI, `/healthz`, `/readyz`, `/metrics`,
      `/v1/models`, OpenAI-shaped errors, graceful shutdown, structured logging.
- [x] **P0-03** Commit a static OpenAPI 3.1 document served at `/openapi.json` with Swagger UI at
      `/docs`; generation from handlers is P5-04.
- [x] **P0-04** Add root `Makefile` with `help`, `build`, `test`, `lint`, `fmt`, `tools`, `makehooks`,
      `docker-*`, `helm-*`, `ci`, `clean` and `distclean` targets.
- [x] **P0-05** Add `tools` / `tools-brew` targets that install the Homebrew formulas listed in §8.2
      and a `check-tools` target that reports missing binaries.
- [x] **P0-06** Add `scripts/pre-commit` (unit tests + formatting) and `make makehooks` installer that
      honours `core.hooksPath`.
- [x] **P0-07** Add multi-stage `Dockerfile`, `.dockerignore` and `docker-compose.yml` wiring the
      server to Prometheus and Grafana with provisioned datasources/dashboards.
- [x] **P0-08** Add `helm/quarrel` chart (deployment, service, config, secret, ingress, HPA,
      ServiceMonitor, PrometheusRule, PVC) and `helm lint` clean.
- [x] **P0-09** Add Prometheus scrape config + alert rules and a Grafana overview dashboard whose
      queries reference only the metric names in §7.
- [x] **P0-10** Document the Arrow schemas and the OpenAI/OpenAPI contract under `api/`.
- [x] **P0-11** Add `.github/dependabot.yml` with weekly Cargo updates, grouped Arrow/gRPC/axum
      upgrades (those crates must move in lockstep), monthly Actions/container updates and a
      policy that major Rust upgrades are deliberate MSRV changes.
- [x] **P0-12** Add `.github/workflows/ci.yml` so every Dependabot PR runs fmt, clippy, unit tests,
      `cargo audit`, `cargo deny`, contract checks and chart validation.
- [x] **P0-13** Add `deny.toml` with the license allowlist, duplicate-version policy and registry
      allowlist; keep advisories at `deny` with no blanket ignores.

**Acceptance:** `make ci`, `make docker-build`, `docker compose config`, and `helm lint helm/quarrel`
all pass on a clean checkout; `make makehooks && git commit` runs the unit tests.

### Phase 1 — Domain model and parity harness

- [x] **P1-01** Create `quarrel-core`: `RequestId`, `ModelId`, `SamplingParams`, `FinishReason`,
      `GenerateRequest`, `Token`, `Embedding`, `KvBlock`, non-exhaustive error enum.
- [x] **P1-02** Port the config surface of Go `internal/config` to `serde` + `clap` with a single
      `Config` struct, env prefix `QUARREL_`, and a config-file round-trip test.
- [ ] **P1-03** Build the oracle harness: load recorded GGUF fixtures from the Go engine, capture
      logits/tokens/KV bytes, and expose `make oracle-gate`.
- [x] **P1-04** Add `proptest` + `insta` snapshot infrastructure and `cargo-fuzz` targets for GGUF
      parsing, sampling params and Arrow schema decoding.
- [x] **P1-05** Establish `criterion` benchmarks and `iai-callgrind` for hot kernels with baselines
      committed under `benches/baselines/`.

**Acceptance:** P1-03 reproduces Go logits within tolerance on three models (Llama 3.2, Qwen 3.5,
Mistral); fuzz targets run for 60 s without a crash.

### Phase 2 — GGUF reader and zero-copy quantized kernels

- [x] **P2-01** Parse GGUF v1–v3 headers, KV metadata and tensor descriptors with `zerocopy`-style
      typed views over `&[u8]`; reject truncated/malformed files with typed errors.
- [x] **P2-02** Memory-map weights with `memmap2`; expose typed tensor views without copying.
- [x] **P2-03** Implement zero-copy matvec for FP32, FP16, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q4_0,
      Q3_K, Q2_K; prove no intermediate `Vec<f32>` with allocator counters.
- [x] **P2-04** Port GGUF writer and `quarrel-quantize` CLI (`q4_k`, `q6_k`, `q8_0`, `f16`, `f32`).
- [x] **P2-05** Port on-demand token embedding lookup (never materialise the vocabulary table).
- [ ] **P2-06** Validate TurboQuant KV codec (PolarQuant + QJL) against the Go oracle; 8× compression.
- [x] **P2-07** Universal model resolver across `~/.cache/llmfit`, llama.cpp, Hugging Face, Ollama
      tags and explicit paths, with fuzzy matching and a resolver unit test per source.

**Acceptance:** 4B Q8_0 model loads with < 20 MB steady-state heap above mapped weights; oracle gate
green for every supported quant; quantizer output reloads identically.

### Phase 3 — Engine core

- [x] **P3-01** Backend trait: `fn forward(&self, batch: &Batch, kv: &mut KvCache) -> Result<Logits>`;
      CPU implementation behind `quarrel-backend-cpu`.
- [x] **P3-02** Continuous batching with iteration-level preemption and bounded queue depth.
- [x] **P3-03** Paged KV cache with block tables, per-sequence isolation, rollback and prefix sharing.
- [x] **P3-04** Radix-tree prompt prefix cache with LRU eviction and zero-copy prefix reuse.
- [x] **P3-05** Sampler: temperature, top-K, top-P, repetition penalty, grammar-constrained PDA.
- [x] **P3-06** Dynamic architecture routing for llama/qwen2/qwen35/mistral/gemma4/phi3/smollm2/granite,
      including Qwen 3.5 hybrid SSM + full-attention layers.
- [x] **P3-07** Jinja2-subset chat template renderer (`{% for %}`, `{% if %}`, `{% set %}`, `{{ var }}`)
      with a fallback template.
- [x] **P3-08** Speculative decoding with `min(1, p_target/p_draft)` acceptance and residual correction.
- [ ] **P3-09** LoRA adapters: merge-on-load on CPU, hot-swappable fused projection on accelerators.
- [x] **P3-10** Engine health audit (NaN/Inf/flat logits) surfaced as metrics and warnings.
- [x] **P3-11** Native mmap-backed GGUF runtime with Gemma 3 and SmolLM2 tokenizer, forward pass and
      deterministic greedy generation; validated against real Q4_K_M GGUFs.

**Acceptance:** end-to-end generation matches Go on the golden suite; 100k-token sliding-window
context holds constant VRAM; no data races under `cargo test --release` + `loom` for the scheduler.

### Phase 4 — Arrow Flight Agent plane

- [x] **P4-01** `quarrel-flight` crate with schemas from §2.2 as Rust constants + generated IPC fixtures.
- [x] **P4-02** tonic Flight server: `DoPut`, `DoGet`, `DoAction`, `ListFlights`, `GetFlightInfo`.
- [x] **P4-03** Zero-copy request adapter: Arrow `RecordBatch` → engine requests with no JSON hop.
- [x] **P4-04** Zero-copy response adapter: pre-sized builders, pooled buffers, `RecordBatchStream`.
- [x] **P4-05** KV block transfer path with dtype metadata and endianness assertions.
- [x] **P4-06** Schema-compatibility and drift tests; IPC golden files checked in.
- [x] **P4-07** Python client in `pythonsdk/` using `pyarrow.flight` with a notebook-style example.
- [x] **P4-08** Smoke benchmark: 250k×384 float vectors round-trip with zero payload copies.

**Acceptance:** an external `pyarrow.flight` client generates tokens and embeddings through the exact
schemas in `api/flight-schema.md`; allocator assertions show zero payload-copy allocations.

### Phase 5 — OpenAI / OpenAPI web plane

- [x] **P5-01** Move skeleton DTOs into `quarrel-api`; full `/v1/models`, chat, completions, embeddings.
- [x] **P5-02** SSE streaming with `chat.completion.chunk`, `[DONE]`, heartbeats and cancellation.
- [x] **P5-03** Function/tool calling and grammar-constrained decoding wired to P3-05.
- [x] **P5-04** Generate `openapi.json` from handlers with utoipa; keep `/docs` Swagger UI; drift test.
- [x] **P5-05** Embedded WebUI served from the binary (`rust-embed`), replacing the second Go binary.
- [x] **P5-06** Ollama-compatible `/api/generate` and `/api/chat` shims.
- [x] **P5-07** Auth (API key/bearer), CORS, request-size limits and per-key rate limiting.
- [x] **P5-08** Contract tests replaying recorded OpenAI SDK request/response pairs.

**Acceptance:** an unmodified OpenAI Python SDK completes chat, streaming and embeddings calls;
`make openapi-check` passes; WebUI is reachable at `/` from the single binary.

### Phase 6 — Hardware backends

- [ ] **P6-01** CUDA backend via `cudarc`: dynamic `-ngl` layer split, host/device activation transfer,
      paged KV in VRAM, FP16 weights / FP32 activations at the boundary.
- [ ] **P6-02** CUDA kernel parity tests (CPU vs GPU) and VRAM ceiling assertions.
- [ ] **P6-03** Metal backend via `metal`, gated to `target_os = "macos"`, MSL kernels + tensor pool.
- [x] **P6-04** AVX2 / AVX-512 / NEON kernels behind one runtime-dispatch facade; scalar fallback.
- [ ] **P6-05** Element-size audit table enforced by tests: CPU f32, CUDA weights f16/activations f32,
      Metal f16; positions and block tables f32 everywhere.

**Acceptance:** CPU/CUDA/Metal parity gates green; a 24-layer offload stays within the configured VRAM
budget; unsafe code confined to P6-04 and FFI shims.

### Phase 7 — Operations and observability

- [x] **P7-01** Wire all §7 metrics into the code; add metric-name unit tests to prevent drift with
      `deploy/prometheus/alerts/quarrel.rules.yml` and the Grafana dashboard.
- [x] **P7-02** OTLP trace export behind the `otel` feature; propagators for HTTP and gRPC.
- [x] **P7-03** Dockerfile variants: CPU (default), CUDA (`nvidia/cuda` base), distroless runtime option.
- [x] **P7-04** Helm values for model PVC, GPU resources, probes, NetworkPolicy, ServiceMonitor and
      PrometheusRule; add `helm unittest` golden templates.
- [x] **P7-05** Dashboard panels for TTFT, decode throughput, KV saturation, Flight traffic, offload
      and memory; document every panel in `docs/metrics.md`.
- [ ] **P7-06** SBOM and reproducible image builds (`SOURCE_DATE_EPOCH`, cosign signing in CI).

**Acceptance:** `make docker-up` brings up server + Prometheus + Grafana with the dashboard populated
after one generated request; `helm template` renders valid Kubernetes YAML with monitoring enabled.

### Phase 8 — Correctness and performance gates

- [ ] **P8-01** `make accuracy-gate`, `make oracle-gate`, `make fuzz-coverage` (from Go roadmap) ported.
- [ ] **P8-02** `cargo mutants` on `quarrel-core`, `quarrel-simd` and the sampler: no surviving mutants.
- [x] **P8-03** Allocation gate: zero allocations in decode and Flight payload paths after warm-up.
- [x] **P8-04** Leak gate: 16 load/search/close cycles with < 5 MB heap growth.
- [ ] **P8-05** CI matrix: stable + MSRV, linux x86_64, linux aarch64, macOS arm64, CUDA job.
- [x] **P8-06** Pre-commit hook enforced in CI by running `make precommit`.

**Acceptance:** all gates in §8.1 green for two consecutive PRs; performance numbers committed with
the commands that produced them.

### Phase 9 — v1.0 release and later

- [ ] **P9-01** Publish v1.0.0 with parity report against upstream v0.3.0.
- [ ] **P9-02** Optional follow-ons: AMD ROCm, GPTQ/AWQ, Vulkan, multi-LoRA batch fusion,
      persistent grouped GEMM for MoE, AMX kernels, tree-style speculative decoding, sliding-window
      attention with sinks, adaptive `nprobe` and FP16 IVF for vector search.

---

## 5. Go → Rust parity matrix

| Upstream (Go) | Rust target | Strategy |
|---------------|-------------|----------|
| `cmd/quarrel`, `cmd/simple` | `quarrel-cli` | `clap` derive subcommands; one binary for serve/quantize/bench |
| `cmd/webui` (separate module) | `quarrel-api` static assets | embed with `rust-embed`, delete second module |
| `internal/api` REST + Flight | `quarrel-api` + `quarrel-flight` | split transport, share domain DTOs |
| `internal/config` | `quarrel-core::Config` | `serde` + env + clap, no global mutable config |
| `internal/cpu` | `quarrel-simd` | `is_x86_feature_detected!`, atomic dispatch once |
| `internal/device` | backend crates | trait objects with static dispatch where possible |
| `internal/engine` | `quarrel-engine` | tokio tasks + bounded mpsc; keep scheduler deterministic for tests |
| `internal/gguf` | `quarrel-gguf` | `memmap2` + `bytes` typed views; remove Go's allocator pressure entirely |
| `internal/mathx` | `quarrel-simd::math` | feature-swappable scalar/fastmath with bit-exact default |
| `internal/metrics`, `telemetry` | `quarrel-observability` | `metrics` facade + tracing + OTLP |
| `internal/models`, `ollama` | `quarrel-core::resolver` | same cache search order and tag parsing |
| `internal/sampler` | `quarrel-engine::sampler` | grammar PDA port + oracle tests |
| `internal/tokenizer` | `quarrel-engine::tokenizer` | BPE + SentencePiece; keep token IDs stable |
| `internal/vector` | `quarrel-engine::vector` | Arrow-native, mmap store, flat/IVF/HNSW |
| `internal/vlm` | Phase 9 | defer until text parity is green |
| `scripts/*` gates | `make` targets + CI | shellcheck every script |
| Helm, Grafana, Docker | this repository | port and validate with `helm lint`/`compose config` |

---

## 6. Testing strategy

The executable test plan is [docs/testplan.md](testplan.md), benchmark methodology is
[docs/performance.md](performance.md), and CI enforcement is documented in [docs/gates.md](gates.md).

1. **Unit** — `cargo test --workspace --lib --bins`; mandatory in the pre-commit hook.
2. **Integration** — `cargo nextest run --workspace`; axum `oneshot` tests and Flight in-process servers.
3. **Oracle** — recorded Go logits/tokens/KV bytes; `cargo test -p quarrel-oracle` + `make oracle-gate`.
4. **Property/fuzz** — `proptest` and `cargo fuzz` for parsers, samplers and Arrow decoding.
5. **Mutation** — `cargo mutants` to prove tests can fail.
6. **Benchmarks** — `criterion` for throughput, `iai-callgrind` for instruction counts, allocation
   assertions for the zero-copy claim; baselines committed and compared in CI.
7. **Safety** — `cargo miri` on unsafe SIMD/FFI crates, `cargo audit`, `cargo deny`, `cargo geiger`.
8. **Coverage** — `cargo llvm-cov --workspace --html`, ratchet floors: core/gguf/simd ≥ 85 %,
   engine ≥ 80 %, api ≥ 80 %.
9. **Contract** — OpenAPI drift, Arrow IPC golden files, OpenAI SDK replay, `helm unittest`.

---

## 7. Metrics, alerts and dashboard contract

All names start with `quarrel_`. They are declared as constants in `quarrel-server/src/metrics.rs`
and used verbatim in `deploy/prometheus/alerts/quarrel.rules.yml` and the canonical dashboard at
`grafana/dashboards/quarrel-overview.json`. The Helm chart carries a byte-identical copy at
`helm/quarrel/files/quarrel-overview.json`; `make metrics-check` checks all four locations for
drift.

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `quarrel_build_info` | gauge | `version`, `commit`, `arrow_schema` | Build/schema identity, always 1 |
| `quarrel_requests_total` | counter | `endpoint`, `method`, `status` | HTTP requests |
| `quarrel_request_duration_seconds` | histogram | `endpoint`, `status` | HTTP latency |
| `quarrel_inference_tokens_total` | counter | `model`, `operation` | `prefill` / `decode` tokens |
| `quarrel_inference_duration_seconds` | histogram | `model`, `operation` | Inference time |
| `quarrel_time_to_first_token_seconds` | histogram | `model` | TTFT |
| `quarrel_decode_tokens_per_second` | gauge | `model` | Latest decode rate |
| `quarrel_model_load_duration_seconds` | histogram | `model` | Model load time |
| `quarrel_model_weights_bytes` | gauge | `model` | Mapped weight bytes |
| `quarrel_gpu_layers_active` | gauge | `model` | Layers resident on GPU |
| `quarrel_cpu_layers_active` | gauge | `model` | Layers resident on CPU |
| `quarrel_layer_offload_transfers_total` | counter | `direction` | `host_to_device` / `device_to_host` |
| `quarrel_layer_offload_duration_seconds` | histogram | `direction` | Transfer latency |
| `quarrel_kv_cache_blocks_used` | gauge | `model`, `pool` | Used KV blocks |
| `quarrel_kv_cache_blocks_capacity` | gauge | `model`, `pool` | Capacity |
| `quarrel_kv_cache_lookups_total` | counter | `model` | Prefix/KV lookups |
| `quarrel_kv_cache_hits_total` | counter | `model` | Hits |
| `quarrel_kv_cache_evictions_total` | counter | `model` | Evictions |
| `quarrel_speculative_tokens_total` | counter | `model`, `outcome` | `accepted` / `rejected` |
| `quarrel_sampling_rejections_total` | counter | `model` | Grammar/vocab rejections |
| `quarrel_flight_batches_total` | counter | `method`, `status` | Flight RPC batches |
| `quarrel_flight_bytes_total` | counter | `method`, `direction` | Arrow payload bytes |
| `quarrel_flight_batch_duration_seconds` | histogram | `method` | Flight batch latency |
| `quarrel_sse_streams_active` | gauge | — | Open SSE streams |
| `quarrel_process_resident_bytes` | gauge | — | Process RSS |
| `quarrel_engine_errors_total` | counter | `code` | Typed engine errors |
| `quarrel_health_checks_total` | counter | `check`, `status` | Probe outcomes |

Alert rules: high 5xx rate, high TTFT p95, KV-cache saturation, Flight error Rate, model load
failures and sustained RSS growth. The dashboard has panels for request rate/errors/latency, TTFT,
decode throughput, KV saturation, Flight bytes, layer offload, model load and RSS.

---

## 8. Local developer experience

### 8.1 Make targets

`make help` (default) lists all targets. Highlights:

| Target | Purpose |
|--------|---------|
| `tools` / `tools-brew` | Install Homebrew formulas (§8.2); `tools-cargo` installs cargo-only tools; `tools-miri` adds nightly Miri |
| `check-tools` | Report missing required binaries with the command to fix them |
| `makehooks` / `unmakehooks` | Install/remove `.git/hooks/pre-commit` (respects `core.hooksPath`) |
| `precommit` | Run the pre-commit hook manually |
| `build`, `build-release` | `cargo build [--release] --workspace --locked` |
| `test` | `cargo nextest run --workspace` (falls back to `cargo test --workspace`) |
| `test-benchtool` | Go/Python tooling tests: `gofmt`, `go vet`, `go test`, `unittest`, profile dry-run |
| `bench-quality`, `bench-performance`, `bench-regression` | Replay quality/performance profiles through the Go benchtool |
| `lint`, `fmt`, `fmt-check` | clippy with `-D warnings`, rustfmt apply/check |
| `shellcheck` | Shellcheck every tracked shell script, including the pre-commit hook |
| `audit`, `deny`, `security` | `cargo audit`, `cargo deny check`, both |
| `coverage`, `bench`, `fuzz`, `miri`, `mutants` | quality and performance deep dives |
| `openapi-check`, `metrics-check` | contract/drift gates |
| `docker-build`, `docker-up`, `docker-down`, `docker-logs` | local monitoring stack |
| `helm-lint`, `helm-template` | chart validation |
| `ci` | the full local gate sequence; remote GitHub Actions is disabled here |
| `clean` | remove build artefacts (`cargo clean`, generated docs, caches) |
| `distclean` | `clean` plus tool-built caches and local secrets (never tracked files) |
| `links` | Print upstream/API/monitoring URLs |

### 8.2 Tools installed by `make tools`

Homebrew formulas: `cargo-nextest`, `cargo-deny`, `cargo-audit`, `bacon` (the maintained
`cargo-watch` replacement), `cargo-llvm-cov`, `cargo-edit`, `cargo-mutants`, `cargo-bloat`,
`cargo-outdated`, `taplo`, `yq`, `jq`, `shellcheck`, `helm`. `make tools-cargo` adds `cargo-fuzz`,
`cargo-udeps` and `cargo-geiger`, which Homebrew does not package. `make tools-miri` installs the
nightly toolchain with the `miri` and `rust-src` components. The benchmark tooling additionally
requires Go ≥ 1.24 and Python ≥ 3.11 (both use their standard libraries only).
Dependabot opens weekly Cargo PRs; `arrow`/`arrow-flight`, `tonic`/`prost`, the axum/tower/hyper
stack, `metrics`/`tracing` and `utoipa` are grouped so compatible versions land together.

### 8.3 Pre-commit hook

`scripts/pre-commit` is a dependency-free Bash script (shellchecked by local `make shellcheck`). On every `git commit` it:

1. refuses to run if `cargo` is missing, pointing at `make tools`;
2. runs `cargo fmt --all -- --check` (fast, deterministic);
3. runs `cargo test --workspace --lib --bins` — **unit tests are mandatory**;
4. prints a clear summary and exits non-zero on the first failure.

`make makehooks` installs it through `git rev-parse --git-path hooks`, so it also works with git
worktrees and a custom `core.hooksPath`. Heavier checks (clippy, audit, fuzz, Helm, OpenAPI drift)
run in local `make ci` so commits stay fast.

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Arrow version churn between `arrow` and `arrow-flight` | Build breaks on upgrade | Single workspace version pin; ABI test in P4-06; Dependabot groups the pair |
| `unsafe` SIMD/FFI defects | UB in hot path | Confine to `quarrel-simd`, `// SAFETY` comments, miri + fuzz + scalar oracle |
| Async cancellation loses KV state | Wrong tokens after client disconnect | `CancellationToken`, sequence IDs, per-request KV ownership and rollback tests |
| JSON path reintroduces copies | Zero-copy claim invalid | Allocator-count assertion in P4-08 and P8-03 |
| Numerical drift vs Go | Silent accuracy regression | Oracle gate P1-03 with pinned fixtures and tolerance policy in P3 |
| Crate MSRV drift | CI breaks on stable bumps | Exact toolchain pin + MSRV job in P8-05 |
| Model files with untrusted metadata | Panic / OOM | Checked arithmetic, bounded allocations, fuzz target for GGUF parser |
| Dashboard/metric drift | Blind spots in production | `make metrics-check` compares code, alerts and dashboard |

---

## 10. Quick reference

```bash
# First-time setup
make tools                 # Homebrew dev tools
make makehooks             # install the unit-test pre-commit hook
make check-tools           # verify the toolchain

# Everyday loop
make fmt lint test         # formatting, clippy, unit/integration tests
make build-release         # optimised binary
make precommit             # run the same checks the hook runs

# Containers and cluster
make docker-build docker-up
make helm-lint helm-template

# Gate tooling and performance checks
make test-benchtool
make bench-quality
make bench-performance

# Full gate
make ci
```

**Links:** [upstream repository](https://github.com/23skdu/longbow-quarrel) ·
[upstream architecture](https://github.com/23skdu/longbow-quarrel/blob/main/docs/architecture.md) ·
[Arrow Flight RPC](https://arrow.apache.org/docs/format/Flight.html) ·
[OpenAI API reference](https://platform.openai.com/docs/api-reference) ·
[OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) ·
[Rust API guidelines](https://rust-lang.github.io/api-guidelines/) ·
[Cargo book](https://doc.rust-lang.org/cargo/)
