# Arrow Flight contract for Agents

> Status: **planned for Phase 4** of [the roadmap](../docs/roadmap.md). This file is the normative
> schema contract; implementation must not drift from it without a major version bump.

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

### `embed.response`

| Field | Arrow type | Notes |
|-------|------------|-------|
| `request_id` | `Utf8` | Echoed |
| `token_id` | `Int32` | Source token |
| `embedding` | `FixedSizeList<Float16 or Float32, D>` | Backed by a mapped buffer slice |

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

## Example (Python Agent, Phase 4)

```python
import pyarrow as pa
from pyarrow import flight

schema = pa.schema([
    ("request_id", pa.string()),
    ("sequence", pa.uint64()),
    ("token_ids", pa.list_(pa.int32())),
    ("max_tokens", pa.uint32()),
    ("temperature", pa.float32()),
    ("top_p", pa.float32()),
    ("top_k", pa.uint32()),
    ("stop_token_ids", pa.list_(pa.int32())),
    ("stream", pa.bool_()),
])

batch = pa.record_batch(
    [["agent-1"], [0], [[128000, 271]], [64], [0.8], [0.95], [40], [[]], [True]],
    schema=schema,
)

with flight.FlightClient("grpc://localhost:50051") as client:
    writer, _ = client.do_put(
        flight.FlightDescriptor.for_path("/quarrel/v1/generate"), schema
    )
    writer.write_batch(batch)
    writer.close()
```
