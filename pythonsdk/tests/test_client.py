"""Unit tests for the pure-Python schema specs and client request construction.

The pyarrow-dependent tests skip cleanly when pyarrow is not installed; every
other test exercises the exact schemas and payloads without it.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SDK_ROOT = Path(__file__).resolve().parents[1]
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from quarrel_flight import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    EMBED_PATH,
    EMBED_REQUEST_SCHEMA,
    EMBED_RESPONSE_SCHEMA,
    GENERATE_PATH,
    GENERATE_REQUEST_SCHEMA,
    GENERATE_RESPONSE_SCHEMA,
    HEALTH_ACTION,
    KV_BLOCKS_SCHEMA,
    MissingPyArrowError,
    QuarrelFlightClient,
    SchemaSpec,
    embed_request_columns,
    generate_request_columns,
    schema_hash,
    ticket_for,
)

try:
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - depends on environment
    HAS_PYARROW = False
else:
    HAS_PYARROW = True


class _Writer:
    def __init__(self) -> None:
        self.batches: list[object] = []
        self.closed = False

    def write_batch(self, batch: object) -> None:
        self.batches.append(batch)

    def close(self) -> None:
        self.closed = True


class _Descriptor:
    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    def for_path(cls, path: str) -> _Descriptor:
        return cls(path)


class _Ticket:
    def __init__(self, ticket: bytes) -> None:
        self.ticket = ticket


class _Action:
    def __init__(self, action_type: str, body: bytes) -> None:
        self.type = action_type
        self.body = body


class _Stream:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def read_all(self) -> SimpleNamespace:
        return SimpleNamespace(to_pylist=lambda: self.rows)


class _Connection:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.writer = _Writer()
        self.descriptor: str | None = None
        self.schema: object = None
        self.ticket: bytes | None = None
        self.action: _Action | None = None
        self.rows = rows or []
        self.closed = False

    def do_put(self, descriptor: _Descriptor, schema: object) -> tuple[_Writer, None]:
        self.descriptor = descriptor.path
        self.schema = schema
        return self.writer, None

    def do_get(self, ticket: _Ticket) -> _Stream:
        self.ticket = ticket.ticket
        return _Stream(self.rows)

    def do_action(self, action: _Action) -> list[SimpleNamespace]:
        self.action = action
        return [SimpleNamespace(body=json.dumps({"ready": True}).encode("utf-8"))]

    def close(self) -> None:
        self.closed = True


def _client_with_fakes(connection: _Connection) -> QuarrelFlightClient:
    client = QuarrelFlightClient("grpc://fake:50051")
    client._flight = SimpleNamespace(
        FlightDescriptor=_Descriptor, Ticket=_Ticket, Action=_Action
    )
    client._client = connection
    return client


class SchemaSpecTests(unittest.TestCase):
    def test_schema_specs_mirror_rust_fields(self) -> None:
        self.assertEqual(
            GENERATE_REQUEST_SCHEMA.names(),
            (
                "request_id",
                "sequence",
                "token_ids",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stop_token_ids",
                "stream",
            ),
        )
        self.assertEqual(
            GENERATE_RESPONSE_SCHEMA.names(),
            (
                "request_id",
                "sequence",
                "token_id",
                "logprob",
                "top_logprobs",
                "finish_reason",
                "finished",
            ),
        )
        self.assertEqual(EMBED_REQUEST_SCHEMA.names(), ("request_id", "token_id"))
        self.assertEqual(
            EMBED_RESPONSE_SCHEMA.names(), ("request_id", "token_id", "embedding")
        )
        self.assertEqual(
            KV_BLOCKS_SCHEMA.names(),
            ("request_id", "block_id", "layer_id", "dtype", "blocks"),
        )
        self.assertEqual(
            EMBED_RESPONSE_SCHEMA.field("embedding").type,
            f"fixed_size_list<float32, {DEFAULT_EMBEDDING_DIM}>",
        )
        self.assertFalse(EMBED_REQUEST_SCHEMA.field("token_id").nullable)
        self.assertTrue(GENERATE_RESPONSE_SCHEMA.field("finish_reason").nullable)

    def test_schema_hashes_match_rust_contract(self) -> None:
        self.assertEqual(schema_hash(GENERATE_REQUEST_SCHEMA), "4e648477ea810831")
        self.assertEqual(schema_hash(GENERATE_RESPONSE_SCHEMA), "e10101ceca033e3e")
        self.assertEqual(schema_hash(EMBED_REQUEST_SCHEMA), "f8ea32cf6498f9e7")
        self.assertEqual(schema_hash(EMBED_RESPONSE_SCHEMA), "1e2c5be4abd217ba")
        self.assertEqual(schema_hash(KV_BLOCKS_SCHEMA), "7b33a401c93c62d1")

    def test_columns_are_validated_and_ordered_without_pyarrow(self) -> None:
        columns = EMBED_REQUEST_SCHEMA.columns(
            {"token_id": [1, 2], "request_id": ["a", "a"]}
        )
        self.assertEqual(list(columns), ["request_id", "token_id"])
        self.assertEqual(columns, {"request_id": ["a", "a"], "token_id": [1, 2]})
        with self.assertRaises(ValueError):
            EMBED_REQUEST_SCHEMA.columns({"request_id": ["a"]})
        with self.assertRaises(ValueError):
            EMBED_REQUEST_SCHEMA.columns({"request_id": ["a"], "token_id": [1], "x": [2]})
        with self.assertRaises(ValueError):
            EMBED_REQUEST_SCHEMA.columns({"request_id": ["a", "b"], "token_id": [1]})

    def test_schema_field_lookup_is_typed(self) -> None:
        with self.assertRaises(KeyError):
            EMBED_RESPONSE_SCHEMA.field("missing")

    @unittest.skipUnless(HAS_PYARROW, "pyarrow is not installed")
    def test_all_pyarrow_schemas_are_constructible(self) -> None:
        rows = {
            GENERATE_REQUEST_SCHEMA: generate_request_columns("agent-1", [11, 22]),
            GENERATE_RESPONSE_SCHEMA: {
                "request_id": ["agent-1"],
                "sequence": [1],
                "token_id": [271],
                "logprob": [-0.125],
                "top_logprobs": [[{"token_id": 7, "logprob": -1.0}]],
                "finish_reason": ["length"],
                "finished": [True],
            },
            EMBED_REQUEST_SCHEMA: embed_request_columns("agent-1", [11, 22]),
            EMBED_RESPONSE_SCHEMA: {
                "request_id": ["agent-1"],
                "token_id": [11],
                "embedding": [[0.0, 0.0, 0.0, 1.0]],
            },
            KV_BLOCKS_SCHEMA: {
                "request_id": ["agent-1"],
                "block_id": [1],
                "layer_id": [2],
                "dtype": ["f32"],
                "blocks": [[0] * 16],
            },
        }
        for spec in [
            GENERATE_REQUEST_SCHEMA,
            GENERATE_RESPONSE_SCHEMA,
            EMBED_REQUEST_SCHEMA,
            EMBED_RESPONSE_SCHEMA,
            KV_BLOCKS_SCHEMA,
        ]:
            with self.subTest(schema=spec.name):
                batch = spec.record_batch(rows[spec])
                self.assertEqual([field.name for field in batch.schema], list(spec.names()))
                self.assertEqual(str(batch.schema), str(spec.to_pyarrow()))


class RequestBuilderTests(unittest.TestCase):
    def test_generate_request_columns_are_exact(self) -> None:
        columns = generate_request_columns(
            "agent-1",
            [128000, 271],
            max_tokens=32,
            temperature=0.0,
            top_p=0.5,
            top_k=7,
            stop_token_ids=[9],
            stream=False,
        )
        self.assertEqual(
            columns,
            {
                "request_id": ["agent-1"],
                "sequence": [0],
                "token_ids": [[128000, 271]],
                "max_tokens": [32],
                "temperature": [0.0],
                "top_p": [0.5],
                "top_k": [7],
                "stop_token_ids": [[9]],
                "stream": [False],
            },
        )

    def test_embed_request_columns_have_one_row_per_token(self) -> None:
        columns = embed_request_columns("agent-embed", [11, -7, 42])
        self.assertEqual(
            columns,
            {
                "request_id": ["agent-embed"] * 3,
                "token_id": [11, -7, 42],
            },
        )

    def test_invalid_requests_are_rejected(self) -> None:
        for bad_ids in ([], [1.5], [2**31], [True]):
            with self.subTest(token_ids=bad_ids):
                with self.assertRaises(ValueError):
                    embed_request_columns("agent", bad_ids)
        with self.assertRaises(ValueError):
            generate_request_columns("agent", [1], max_tokens=0)
        with self.assertRaises(ValueError):
            generate_request_columns("agent", [1], top_p=1.5)
        with self.assertRaises(ValueError):
            generate_request_columns("agent", [1], temperature=-0.1)
        with self.assertRaises(ValueError):
            generate_request_columns("agent", [1], top_k=-1)

    def test_ticket_and_descriptor_paths(self) -> None:
        self.assertEqual(ticket_for("agent-1"), b"agent-1")
        self.assertEqual(GENERATE_PATH, "/quarrel/v1/generate")
        self.assertEqual(EMBED_PATH, "/quarrel/v1/embed")
        self.assertEqual(HEALTH_ACTION, "/quarrel/v1/health")
        with self.assertRaises(ValueError):
            ticket_for("")


class ClientTests(unittest.TestCase):
    def test_client_does_not_import_pyarrow_until_used(self) -> None:
        client = QuarrelFlightClient()
        self.assertEqual(client.uri, "grpc://127.0.0.1:50051")
        self.assertIsNone(client._pa)
        self.assertIsNone(client._flight)
        self.assertIsNone(client._client)
        client.close()
        self.assertIsNone(client._client)

    def test_missing_pyarrow_has_actionable_error(self) -> None:
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.flight": None}):
            with self.assertRaises(MissingPyArrowError) as caught:
                QuarrelFlightClient()._load_pyarrow()
        self.assertIn("requirements.txt", str(caught.exception))

    def test_generate_uses_do_put_then_do_get(self) -> None:
        rows = [{"token_id": 271, "finished": True}]
        connection = _Connection(rows)
        client = _client_with_fakes(connection)
        with patch.object(SchemaSpec, "record_batch", return_value="batch") as record_batch:
            with patch.object(SchemaSpec, "to_pyarrow", return_value="schema") as to_pyarrow:
                result = client.generate("agent-1", [11, 22], max_tokens=8)
        self.assertEqual(result, rows)
        self.assertEqual(connection.descriptor, GENERATE_PATH)
        self.assertEqual(connection.ticket, b"agent-1")
        self.assertEqual(connection.writer.batches, ["batch"])
        self.assertTrue(connection.writer.closed)
        self.assertEqual(connection.schema, "schema")
        record_batch.assert_called_once()
        to_pyarrow.assert_called_once()
        self.assertEqual(record_batch.call_args.args[0]["token_ids"], [[11, 22]])
        self.assertEqual(record_batch.call_args.args[0]["max_tokens"], [8])

    def test_embed_uses_do_put_then_do_get(self) -> None:
        rows = [{"token_id": 11, "embedding": [0.0, 0.0, 0.0, 1.0]}]
        connection = _Connection(rows)
        client = _client_with_fakes(connection)
        with patch.object(SchemaSpec, "record_batch", return_value="batch") as record_batch:
            with patch.object(SchemaSpec, "to_pyarrow", return_value="schema"):
                result = client.embed("agent-embed", [11])
        self.assertEqual(result, rows)
        self.assertEqual(connection.descriptor, EMBED_PATH)
        self.assertEqual(connection.ticket, b"agent-embed")
        self.assertEqual(record_batch.call_args.args[0]["token_id"], [11])

    def test_health_action_decodes_json(self) -> None:
        connection = _Connection()
        client = _client_with_fakes(connection)
        self.assertEqual(client.health(), {"ready": True})
        self.assertEqual(connection.action.type, HEALTH_ACTION)
        self.assertEqual(connection.action.body, b"")

    def test_context_manager_closes_connection(self) -> None:
        connection = _Connection()
        with _client_with_fakes(connection) as client:
            self.assertIs(client._client, connection)
        self.assertTrue(connection.closed)
        self.assertIsNone(client._client)


if __name__ == "__main__":
    unittest.main()
