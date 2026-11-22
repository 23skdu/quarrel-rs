# Architecture

This document describes the Rust package graph and the runtime flows for loading a model, serving
inference and switching models. It is the implementation view of the goals in
[the roadmap](roadmap.md); CI enforcement lives in [gates](gates.md).

> **Status legend:** ✅ implemented in the current tree · 🚧 in progress in the current roadmap
> slice · ⏳ planned. The diagrams show the target shape; the table marks what exists today.

## 1. Component packages

Dependencies point **inward only**: transports depend on the engine, the engine depends on
GGUF/SIMD/core, and `quarrel-core` depends on no transport, backend or async runtime.

```mermaid
flowchart TB
    Agent["Agent / Python SDK"] -->|"Arrow Flight over gRPC"| Flight
    Browser["Browser UI"] -->|"HTTP + SSE"| Api
    Sdk["OpenAI-compatible SDK"] -->|"JSON / SSE"| Api

    subgraph Transport["Transport plane"]
        Flight["quarrel-flight ✅<br/>schemas, adapters, tonic service"]
        Api["quarrel-api ✅<br/>axum routes, DTOs, OpenAPI, SSE"]
    end

    subgraph Domain["Domain and orchestration"]
        Engine["quarrel-engine ✅<br/>scheduler, KV, sampler, reference backend"]
        Core["quarrel-core ✅<br/>ids, requests, config, errors, resolver, templates"]
    end

    subgraph Storage["Weights and kernels"]
        Gguf["quarrel-gguf ✅<br/>mmap, GGUF v1-v3, quant views"]
        Simd["quarrel-simd ✅<br/>scalar oracle + dispatched kernels"]
        Weights[("GGUF files<br/>memory-mapped, never fully loaded")]
    end

    subgraph Backends["Execution backends"]
        Cpu["quarrel-backend-cpu ⏳<br/>CPU forward pass"]
        Cuda["quarrel-backend-cuda ⏳<br/>feature-gated"]
        Metal["quarrel-backend-metal ⏳<br/>macOS only"]
    end

    subgraph Operations["Process and observability"]
        Server["quarrel-server ✅<br/>thin binary / wiring"]
        Obs["quarrel-observability ⏳<br/>metrics + tracing overlap (today: server)"]
    end

    Api --> Core
    Api --> Engine
    Flight --> Core
    Flight --> Engine
    Engine --> Core
    Engine --> Gguf
    Engine --> Simd
    Gguf --> Core
    Gguf --> Simd
    Gguf --> Weights
    Engine --> Cpu
    Engine -. "future feature" .-> Cuda
    Engine -. "future feature" .-> Metal
    Cpu --> Simd
    Server --> Api
    Server --> Flight
    Server --> Core
    Obs -. "instrumentation" .-> Api
    Obs -. "instrumentation" .-> Flight
    Obs -. "instrumentation" .-> Engine
    Obs -. "instrumentation" .-> Gguf
```

### 1.1 Package responsibilities

| Package | Responsibility | Status |
|---------|----------------|--------|
| `quarrel-core` | Typed IDs, request/response models, sampling params, config, chat templates, model resolver, non-exhaustive errors | ✅ |
| `quarrel-simd` | Scalar reference math and runtime-dispatched SIMD; the only home for hot-path `unsafe` | ✅ |
| `quarrel-gguf` | GGUF v1–v3 parser, `memmap2` ownership, typed tensor views, quant dequant/dot kernels | ✅ |
| `quarrel-engine` | Scheduler, paged KV cache, radix prefix cache, sampler, health audit, backend trait, reference backend | ✅ |
| `quarrel-flight` | Canonical Arrow schemas, zero-copy adapters, Flight service and IPC golden fixtures | ✅ |
| `quarrel-api` | OpenAI/Ollama DTOs, axum handlers, SSE/NDJSON streaming, generated OpenAPI document, embedded WebUI | ✅ |
| `quarrel-server` | CLI/config parsing and process lifecycle; delegates to the libraries above | ✅ |
| `quarrel-observability` | Metric constants, recorder, tracing layers and OTLP feature (currently living in `quarrel-server`) | ⏳ |
| `quarrel-backend-cpu` | Default execution backend | ⏳ |
| `quarrel-backend-cuda` / `quarrel-backend-metal` | Partial offload behind feature flags | ⏳ |

`quarrel-server` is intentionally thin: all behaviour lives in libraries so integration tests and
future binaries can reuse it without starting a process.

## 2. Model loading

A model is an immutable `Arc<Model>` published into a registry. Loading never copies weights into
the heap: `quarrel-gguf` owns a `memmap2::Mmap`, and `TensorView`s are byte slices into that mapping.

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Operator / CLI
    participant Server as quarrel-server
    participant Resolver as quarrel-core::resolver
    participant Gguf as quarrel-gguf
    participant Registry as ModelRegistry
    participant Obs as Observability

    Admin->>Server: load model tag or explicit path
    Server->>Resolver: resolve(tag)
    Resolver->>Resolver: scan config.models_dir roots for *.gguf
    Resolver-->>Server: PathBuf (exact or unique fuzzy match)
    Server->>Gguf: GgufMmap::open(path)
    Gguf->>Gguf: parse magic/version, KV metadata, tensor descriptors
    Gguf->>Gguf: align header to 32-byte boundary
    Gguf-->>Server: mapped model + header
    Server->>Server: build Arc<Model> with quantized TensorViews
    Server->>Registry: publish(model_id, Arc<Model>)
    Registry-->>Server: published model descriptor
    Server->>Obs: model_load_duration_seconds, model_weights_bytes, build_info
    Server-->>Admin: 200 + model descriptor
```

Loading failures are typed (`GgufError`, `CoreError::Resolver`) and mapped to OpenAI-shaped errors
on the HTTP plane. `/readyz` reports `model_loaded = true` only after the registry publish step.

### 2.1 Model discovery order

`quarrel-core::resolver` searches, in order:

1. explicit paths supplied by the caller;
2. `QUARREL_MODELS_DIR` (Helm mounts a PVC here);
3. `~/.cache/llmfit`, llama.cpp model directories, Hugging Face cache and Ollama tags.

Exact file-stem matches win; a case-insensitive substring match is accepted only when unique.

## 3. Inference

Two client planes converge on the same `GenerateRequest`. The Flight path stays Arrow-native; the
HTTP path converts JSON to Arrow once at the edge, then follows the same engine path.

```mermaid
sequenceDiagram
    autonumber
    participant Client as SDK / Agent
    participant Edge as quarrel-api or quarrel-flight
    participant Adapter as Request adapter
    participant Engine as quarrel-engine
    participant KV as PagedKvCache + RadixPrefixCache
    participant Backend as Backend (CPU now, GPU later)
    participant Weights as quarrel-gguf TensorViews
    participant Sampler as Sampler
    participant Obs as Observability

    Client->>Edge: JSON / SSE request or Flight RecordBatch
    Edge->>Adapter: authenticate, validate, resolve model
    Adapter->>Engine: GenerateRequest (Arrow-backed on Flight)
    Engine->>KV: longest_prefix(token_ids)
    KV-->>Engine: cached KV blocks or miss
    Engine->>Backend: forward(batch, kv)
    Backend->>Weights: &[u8] tensor slices
    Weights-->>Backend: weights without allocation
    Backend-->>Engine: logits
    Engine->>Sampler: sample(logits, params, previous_tokens)
    Sampler-->>Engine: Token { id, logprob }
    Engine->>KV: append token, commit block table
    Engine-->>Adapter: token stream
    Adapter-->>Edge: SSE chunk or Arrow RecordBatch
    Edge-->>Client: data: {...} / [DONE] or Flight batches
    Engine->>Obs: tokens_total, TTFT, decode throughput, KV saturation
```

Cancellation is propagated end to end: a dropped HTTP client or cancelled gRPC stream triggers a
`CancellationToken`, which rolls back the per-request KV state and releases its blocks. Every token
carries `request_id` and a monotonic `sequence`, so an Agent can retry `DoGet` and de-duplicate.

### 3.1 Zero-copy boundary

```mermaid
flowchart LR
    A["Agent RecordBatch"] -->|"FlightData frames, no re-encode"| B["Arrow Buffer / bytes::Bytes"]
    B -->|"Buffer::slice_with_length"| C["TensorView / KV bytes"]
    C -->|"&[u8] kernels"| D["logits"]
    D -->|"pre-sized PrimitiveBuilder"| E["response RecordBatch"]
    E -->|"FlightData stream"| F["Agent"]
```

Rules enforced by review, fuzzing and allocation benchmarks:

- no `serde_json` in the Flight hot path;
- no `to_data`, `iter().collect::<Vec<_>>()` or payload-sized allocations;
- output builders are pre-sized and pooled, then frozen into `Arc<RecordBatch>`;
- schema hash is reported in `quarrel_build_info{arrow_schema=...}` and Flight action metadata.

## 4. Switching models

Switching is an **atomic registry swap**, not an in-place mutation. In-flight requests finish or are
cancelled against the old immutable model; new requests see only the new one.

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Operator / CLI
    participant Server as quarrel-server
    participant Queue as Bounded request queue
    participant Registry as ModelRegistry
    participant Engine as quarrel-engine
    participant Caches as KV + prefix caches
    participant Gguf as quarrel-gguf
    participant Obs as Observability

    Admin->>Server: switch to model B
    Server->>Queue: stop admission, drain in-flight requests
    Queue-->>Server: in-flight finished or cancelled
    Server->>Registry: unpublish model A
    Registry->>Engine: drop Arc<Model> references
    Engine->>Caches: invalidate KV + prefix entries owned by A
    Registry->>Gguf: drop GgufMmap A (munmap on last reference)
    Server->>Gguf: load model B (model-loading flow)
    Gguf-->>Server: Arc<Model B>
    Server->>Caches: optional prefix warm-up for B
    Server->>Registry: publish model B atomically
    Server->>Queue: resume admission
    Server->>Obs: weights_bytes, load_duration, health checks
    Server-->>Admin: switched: B
```

Why this shape:

- `Arc<Model>` means a long-running decode cannot observe a half-unloaded mapping.
- Bounded queues make drain time predictable; `--queue-depth` caps memory and latency.
- Per-model cache namespaces prevent a prefix hit on model A from being reused by model B.
- A failed load of B leaves A published and returns a typed error; there is no empty intermediate
  state.

## 5. Observability cross-cut

Metrics are declared once as constants and consumed by alerts and dashboards; `make metrics-check`
fails when code, rules or panels drift. Tracing spans carry `request_id`, `model`, `seq` and
`operation`, never prompt text at the default level. See [gates](gates.md) for the enforced checks
and roadmap §7 for the canonical metric names.

## 6. Relationship to the roadmap

| Roadmap phase | Packages touched |
|---------------|------------------|
| P1 | `quarrel-core`, fuzz harness |
| P2 | `quarrel-gguf`, `quarrel-simd` |
| P3 | `quarrel-engine` |
| P4 | `quarrel-flight` |
| P5 | `quarrel-api`, `quarrel-server` |
| P6 | backend crates, `quarrel-simd` |
| P7 | `quarrel-observability`, chart/dashboard assets |
| P8/P9 | gates, release artefacts |
