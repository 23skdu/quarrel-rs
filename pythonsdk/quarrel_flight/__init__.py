"""Python client for the quarrel Arrow Flight Agent plane.

The package exports pure-Python schema specs (no pyarrow required to import)
and a thin :class:`QuarrelFlightClient` that lazily imports pyarrow/flight.

Example
-------
::

    from quarrel_flight import QuarrelFlightClient

    with QuarrelFlightClient("grpc://localhost:50051") as client:
        tokens = client.generate("agent-1", [128000, 271], max_tokens=32)
        vectors = client.embed("agent-1-vectors", [128000, 271])
        health = client.health()
"""

from .client import (
    DEFAULT_URI,
    EMBED_PATH,
    GENERATE_PATH,
    HEALTH_ACTION,
    QuarrelFlightClient,
    QuarrelFlightError,
    embed_request_columns,
    generate_request_columns,
    ticket_for,
)
from .schemas import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_KV_BLOCK_BYTES,
    EMBED_REQUEST_SCHEMA,
    EMBED_RESPONSE_SCHEMA,
    GENERATE_REQUEST_SCHEMA,
    GENERATE_RESPONSE_SCHEMA,
    KV_BLOCKS_SCHEMA,
    FieldSpec,
    MissingPyArrowError,
    SchemaSpec,
    schema_hash,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_KV_BLOCK_BYTES",
    "DEFAULT_URI",
    "EMBED_PATH",
    "EMBED_REQUEST_SCHEMA",
    "EMBED_RESPONSE_SCHEMA",
    "GENERATE_PATH",
    "GENERATE_REQUEST_SCHEMA",
    "GENERATE_RESPONSE_SCHEMA",
    "HEALTH_ACTION",
    "KV_BLOCKS_SCHEMA",
    "FieldSpec",
    "MissingPyArrowError",
    "QuarrelFlightClient",
    "QuarrelFlightError",
    "SchemaSpec",
    "embed_request_columns",
    "generate_request_columns",
    "schema_hash",
    "ticket_for",
]

__version__ = "0.1.0"
