"""A thin :mod:`pyarrow.flight` client for the quarrel Agent plane.

Importing this module does not import pyarrow. The transport dependencies are
loaded lazily when a request is made, so schema specs and request payloads can
be inspected in environments where pyarrow is not installed.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    EMBED_REQUEST_SCHEMA,
    GENERATE_REQUEST_SCHEMA,
    MissingPyArrowError,
    require_pyarrow,
)

__all__ = [
    "DEFAULT_URI",
    "EMBED_PATH",
    "GENERATE_PATH",
    "HEALTH_ACTION",
    "MissingPyArrowError",
    "QuarrelFlightClient",
    "QuarrelFlightError",
    "embed_request_columns",
    "generate_request_columns",
    "ticket_for",
]

DEFAULT_URI = "grpc://127.0.0.1:50051"
GENERATE_PATH = "/quarrel/v1/generate"
EMBED_PATH = "/quarrel/v1/embed"
HEALTH_ACTION = "/quarrel/v1/health"


class QuarrelFlightError(RuntimeError):
    """Raised when a Flight operation cannot be completed."""


def ticket_for(request_id: str) -> bytes:
    """Encode a request id as the UTF-8 ticket used by ``DoGet``."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    return request_id.encode("utf-8")


def _validate_token_ids(token_ids: Iterable[int]) -> list[int]:
    ids = list(token_ids)
    if not ids:
        raise ValueError("token_ids must not be empty")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
        raise ValueError("token_ids must contain integers")
    if any(token < -(2**31) or token > 2**31 - 1 for token in ids):
        raise ValueError("token_ids must fit in Arrow int32")
    return ids


def generate_request_columns(
    request_id: str,
    token_ids: Sequence[int],
    *,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    stop_token_ids: Sequence[int] | None = None,
    stream: bool = True,
) -> dict[str, list[Any]]:
    """Build the exact ``generate.request`` column mapping (no pyarrow needed)."""
    ids = _validate_token_ids(token_ids)
    stops = _validate_token_ids(stop_token_ids) if stop_token_ids else []
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2**32 - 1:
        raise ValueError("max_tokens must be a positive uint32")
    if not 0 <= top_k <= 2**32 - 1:
        raise ValueError("top_k must be a uint32")
    if not 0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if not 0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1")
    return GENERATE_REQUEST_SCHEMA.columns(
        {
            "request_id": [request_id],
            "sequence": [0],
            "token_ids": [ids],
            "max_tokens": [max_tokens],
            "temperature": [temperature],
            "top_p": [top_p],
            "top_k": [top_k],
            "stop_token_ids": [stops],
            "stream": [stream],
        }
    )


def embed_request_columns(request_id: str, token_ids: Sequence[int]) -> dict[str, list[Any]]:
    """Build the exact ``embed.request`` column mapping (no pyarrow needed)."""
    return EMBED_REQUEST_SCHEMA.columns(
        {
            "request_id": [request_id] * len(token_ids),
            "token_id": _validate_token_ids(token_ids),
        }
    )


class QuarrelFlightClient:
    """Client for ``DoPut``/``DoGet`` generation and embedding round trips.

    Parameters
    ----------
    uri:
        Flight endpoint, for example ``grpc://localhost:50051``.
    """

    def __init__(self, uri: str = DEFAULT_URI) -> None:
        if not isinstance(uri, str) or not uri:
            raise ValueError("uri must be a non-empty string")
        self.uri = uri
        self._pa: Any = None
        self._flight: Any = None
        self._client: Any = None

    @property
    def pa(self) -> Any:
        """The lazily imported ``pyarrow`` module."""
        if self._pa is None:
            self._load_pyarrow()
        return self._pa

    @property
    def flight(self) -> Any:
        """The lazily imported ``pyarrow.flight`` module."""
        if self._flight is None:
            self._load_pyarrow()
        return self._flight

    def _load_pyarrow(self) -> None:
        require_pyarrow()
        try:
            import pyarrow
            import pyarrow.flight
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MissingPyArrowError(
                "pyarrow is required for Flight transport; install pythonsdk/requirements.txt"
            ) from exc
        self._pa = pyarrow
        self._flight = pyarrow.flight

    def _connection(self) -> Any:
        if self._client is None:
            self._client = self.flight.FlightClient(self.uri)
        return self._client

    def close(self) -> None:
        """Close the underlying Flight client, if one has been created."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> QuarrelFlightClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _round_trip(
        self, path: str, schema_spec: Any, columns: Mapping[str, Sequence[Any]], request_id: str
    ) -> list[dict[str, Any]]:
        batch = schema_spec.record_batch(columns)
        descriptor = self.flight.FlightDescriptor.for_path(path)
        writer, _metadata = self._connection().do_put(descriptor, schema_spec.to_pyarrow())
        writer.write_batch(batch)
        writer.close()
        stream = self._connection().do_get(self.flight.Ticket(ticket_for(request_id)))
        return stream.read_all().to_pylist()

    def generate(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_token_ids: Sequence[int] | None = None,
        stream: bool = True,
    ) -> list[dict[str, Any]]:
        """Submit a prompt with ``DoPut`` and stream generated rows with ``DoGet``.

        Each returned row follows the ``generate.response`` schema:
        ``request_id``, ``sequence``, ``token_id``, ``logprob``,
        ``top_logprobs``, ``finish_reason`` and ``finished``.
        """
        columns = generate_request_columns(
            request_id,
            token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
            stream=stream,
        )
        return self._round_trip(
            GENERATE_PATH, GENERATE_REQUEST_SCHEMA, columns, request_id
        )

    def embed(self, request_id: str, token_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Embed one or more token ids and return their ``embed.response`` rows.

        The server derives a deterministic, L2-normalised 4-dimensional vector
        per token and keys all response batches by ``request_id``.
        """
        columns = embed_request_columns(request_id, token_ids)
        return self._round_trip(EMBED_PATH, EMBED_REQUEST_SCHEMA, columns, request_id)

    def health(self) -> dict[str, Any]:
        """Read the server health action as a decoded JSON object."""
        results = self._connection().do_action(self.flight.Action(HEALTH_ACTION, b""))
        for result in results:
            payload = bytes(result.body).decode("utf-8")
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                raise QuarrelFlightError("health action did not return a JSON object")
            return decoded
        raise QuarrelFlightError("health action returned no results")
