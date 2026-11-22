# quarrel-flight Python SDK

A thin Arrow Flight client for the quarrel Agent plane. It sends Arrow
`RecordBatch`es over gRPC with `pyarrow.flight` — no JSON conversion of request
or response payloads.

The package also ships pure-Python schema specifications that mirror
[`api/flight-schema.md`](../api/flight-schema.md) and the Rust `quarrel-flight`
crate. Importing `quarrel_flight` does **not** import pyarrow, so request
payloads and schema hashes can be inspected in lightweight environments.

## Install

```bash
python3 -m pip install -r pythonsdk/requirements.txt
```

`pyarrow==21.0.0` is the only runtime dependency. Export
`PYTHONPATH=pythonsdk` (or install the directory with your packaging tool) to
import `quarrel_flight`.

## Quick start

```python
from quarrel_flight import QuarrelFlightClient

with QuarrelFlightClient("grpc://localhost:50051") as client:
    print(client.health())  # DoAction /quarrel/v1/health

    # DoPut /quarrel/v1/generate, then DoGet by ticket.
    generated = client.generate(
        "agent-1",
        [128000, 271],
        max_tokens=32,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        stop_token_ids=[],
    )
    for row in generated:
        if row["finished"]:
            print("done at sequence", row["sequence"])

    # DoPut /quarrel/v1/embed, then DoGet by ticket.
    embedded = client.embed("agent-1-vectors", [128000, 271])
    for row in embedded:
        print(row["token_id"], row["embedding"])  # L2-normalised, 4 dims
```

`generate()` returns one row per generated token following `generate.response`:
`request_id`, `sequence`, `token_id`, `logprob`, `top_logprobs`,
`finish_reason`, `finished`. `embed()` returns `request_id`, `token_id` and a
4-element `embedding` per requested token. The server derives embeddings from
the backend logits deterministically, so the same token id always yields the
same vector on the reference backend.

## Schema specs and hashes

```python
from quarrel_flight import (
    EMBED_REQUEST_SCHEMA,
    GENERATE_REQUEST_SCHEMA,
    embed_request_columns,
    schema_hash,
)

print(GENERATE_REQUEST_SCHEMA.names())
print(schema_hash(EMBED_REQUEST_SCHEMA))  # f8ea32cf6498f9e7
print(embed_request_columns("agent-1", [11, 22]))
```

`schema_hash()` reproduces the Rust FNV-1a canonical-field hash reported in the
health action, so version skew between client and server is detectable. To
construct a real Arrow batch, call `schema_spec.record_batch(columns)`; this is
where pyarrow is imported.

## Notebook

[`examples/agent_example.ipynb`](examples/agent_example.ipynb) is a short
walkthrough covering health, generation, embeddings and the wire schemas.

## Tests

```bash
python3 -m unittest discover -s pythonsdk/tests -v
python3 -m json.tool pythonsdk/examples/agent_example.ipynb > /dev/null
```

Pyarrow-dependent tests skip automatically when pyarrow is not installed; all
schema and request-construction tests still run.
