# Arrow Flight contract for Agents

> Status: **implemented in Phase 4**. This file is the normative schema contract; implementation
> must not drift from it without a major version bump. The Rust service lives in
> `crates/quarrel-flight` and the Python client lives in `pythonsdk/quarrel_flight`.

The Flight service is the primary Agent-facing interface. It carries Arrow `RecordBatch`es over
gRPC with no JSON conversion and no payload copies. HTTP/OpenAI remains the compatibility plane.

## Transport

- Protocol: Arrow Flight over gRPC/HTTP2 (`arrow-flight` + `tonic`), default port `50051`.
- Compression: `LZ4_FRAME` for token streams, `ZSTD` for KV blocks.
- Batches are `FlightData` frames; a batch received from a peer is forwarded unchanged.
- Every batch includes `request_id` and a monotonically increasing `sequence`.
- Cancellation is propagated with gRPC status `CANCELLED` and a token drop.

## Descriptor paths

| Path | RPC | Purpose |
|------|-----|---------|
| `/quarrel/v1/generate` | `DoPut` | Submit prompt token IDs and sampling parameters |
| `/quarrel/v1/generate` | `DoGet` | Receive generated tokens, logprobs and finish state |
| `/quarrel/v1/embed` | `DoPut` / `DoGet` | Embeddings for token IDs |
| `/quarrel/v1/kv/blocks` | `DoPut` / `DoGet` | Paged KV cache block transfer |
| `/quarrel/v1/health` | `DoAction` | Readiness and loaded-model descriptor |

## Schemas

### `generate.request`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Client-generated UUID |
| `sequence` | `UInt64` | Monotonic within a request |
| `token_ids` | `List<Int32>` | Dictionary-encode when prompts repeat |
| `max_tokens` | `UInt32` | Generation cap |
| `temperature` | `Float32` | 0 = greedy |
| `top_p` | `Float32` | Nucleus sampling |
| `top_k` | `UInt32` | 0 = disabled |
| `stop_token_ids` | `List<Int32>` | Optional stop set |
| `stream` | `Boolean` | Always true for Flight |

### `generate.response`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Echoed from request |
| `sequence` | `UInt64` | Echoed + 1 per token |
| `token_id` | `Int32` | Sampled token |
| `logprob` | `Float32` | Log probability |
| `top_logprobs` | `List<Struct<token_id: Int32, logprob: Float32>>` | Optional |
| `finish_reason` | `Dictionary<UInt16, Utf8>` | `null` until terminal |
| `finished` | `Boolean` | Terminal batch flag |

### `embed.request`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Client-generated id used as the `DoGet` ticket |
| `token_id` | `Int32` | Token to embed; one row per token |

A batch may carry several rows. The server groups response batches by
`request_id`, so one request id per `DoPut` call is recommended.

### `embed.response`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Echoed |
| `token_id` | `Int32` | Source token |
| `embedding` | `FixedSizeList<Float32, D>` | Backed by a mapped buffer slice; `D = 4` for the reference backend |

Embeddings are deterministic: the first four logits produced by
`Backend::forward(&[token_id])` are L2-normalised (`f64` accumulator, stored as
`Float32`). A non-finite or zero-norm result falls back to a token-derived unit
vector. `GetFlightInfo` and `DoGet` report the stored embed response schema,
not the generate response schema.

### `kv.blocks`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Echoed |
| `block_id` | `UInt32` | Paged KV block |
| `layer_id` | `UInt16` | Transformer layer |
| `dtype` | `Dictionary<UInt8, Utf8>` | `f32`, `f16` or `turboquant` |
| `blocks` | `FixedSizeList<UInt8, N>` | Raw bytes, no reinterpretation copy |

## Zero-copy rules

1. Keep incoming payloads as `arrow::buffer::Buffer` backed by `bytes::Bytes`; slice them, never
   collect them into `Vec<T>`.
2. Engine kernels take `&[u8]` views into mapped weights and Arrow buffers.
3. Build output arrays with pre-sized builders from a pool; freeze into `RecordBatch` + `Arc`.
4. Do not serialize payloads to JSON anywhere in the Flight path.
5. Backpressure is gRPC flow control plus a bounded `mpsc` queue (`--flight-queue-depth`).
6. Schema changes are additive within a major version; the schema hash is reported in
   `quarrel_build_info{arrow_schema=...}`.

## Round-trip usage (Python Agent)

The SDK in `pythonsdk/` performs the `DoPut` → `DoGet` round trip for both
paths. See `pythonsdk/examples/agent_example.ipynb` for a notebook walkthrough.

```python
from quarrel_flight import QuarrelFlightClient, schema_hash, GENERATE_REQUEST_SCHEMA

with QuarrelFlightClient("grpc://localhost:50051") as client:
    health = client.health()                 # DoAction /quarrel/v1/health
    assert health["schemas"]["generate_request"] == schema_hash(GENERATE_REQUEST_SCHEMA)

    # DoPut /quarrel/v1/generate, then DoGet with ticket b"agent-1".
    for row in client.generate("agent-1", [128000, 271], max_tokens=32):
        print(row["sequence"], row["token_id"], row["logprob"], row["finished"])

    # DoPut /quarrel/v1/embed, then DoGet with ticket b"agent-1-vectors".
    for row in client.embed("agent-1-vectors", [128000, 271, 42]):
        print(row["token_id"], row["embedding"])  # 4 floats, unit length
```

The raw wire exchange is the same one the SDK performs:

```python
import pyarrow as pa
from pyarrow import flight
from quarrel_flight import EMBED_REQUEST_SCHEMA, embed_request_columns, embed_response_schema

batch = EMBED_REQUEST_SCHEMA.record_batch(embed_request_columns("agent-1-vectors", [271]))
with flight.FlightClient("grpc://localhost:50051") as client:
    writer, _ = client.do_put(
        flight.FlightDescriptor.for_path("/quarrel/v1/embed"),
        EMBED_REQUEST_SCHEMA.to_pyarrow(),
    )
    writer.write_batch(batch)
    writer.close()

    reader = client.do_get(flight.Ticket(b"agent-1-vectors"))
    table = reader.read_all()  # schema equals embed_response_schema().to_pyarrow()
```

`DoPut` returns `PutResult.app_metadata` with the comma-separated request ids
that were enqueued. `GetFlightInfo` on a descriptor path of
`/quarrel/v1/{generate,embed}/<request_id>` returns the matching response
schema, endpoint ticket and queued batch count.
