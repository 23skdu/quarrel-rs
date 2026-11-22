"""Pure-Python Arrow schema specifications for the quarrel Flight plane.

The field names, order, types and nullability mirror the normative contract in
``api/flight-schema.md`` and the Rust helpers in ``quarrel-flight``. Importing
this module never imports :mod:`pyarrow`: the specs are plain data until
:meth:`SchemaSpec.to_pyarrow` or :meth:`SchemaSpec.record_batch` is called.

The :func:`schema_hash` helper reproduces the Rust FNV-1a canonical-field hash
so a client can verify which schema a server reports in its health action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_KV_BLOCK_BYTES",
    "EMBED_REQUEST_SCHEMA",
    "EMBED_RESPONSE_SCHEMA",
    "GENERATE_REQUEST_SCHEMA",
    "GENERATE_RESPONSE_SCHEMA",
    "KV_BLOCKS_SCHEMA",
    "FieldSpec",
    "SchemaSpec",
    "require_pyarrow",
    "schema_hash",
]

DEFAULT_EMBEDDING_DIM = 4
DEFAULT_KV_BLOCK_BYTES = 16


class MissingPyArrowError(ImportError):
    """Raised when a pyarrow-backed operation is used without pyarrow installed."""


@dataclass(frozen=True)
class FieldSpec:
    """One Arrow field of a Flight schema.

    ``type`` is a small declarative type string (``int32``, ``list<int32>``,
    ``struct<a:int32!, b:float32!>``, ``dictionary<uint16, utf8>`` or
    ``fixed_size_list<float32, 4>``). A trailing ``!`` marks a non-nullable
    nested struct field.
    """

    name: str
    type: str
    nullable: bool = True


@dataclass(frozen=True)
class SchemaSpec:
    """An ordered collection of :class:`FieldSpec` values."""

    name: str
    fields: tuple[FieldSpec, ...]

    def names(self) -> tuple[str, ...]:
        """Return the field names in wire order."""
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> FieldSpec:
        """Return the field called ``name`` or raise :class:`KeyError`."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def columns(self, rows: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
        """Validate a column mapping and return it in schema order.

        This is pure Python; it is used by request builders so callers can
        inspect (and test) the exact payload before pyarrow is involved.
        """
        if not isinstance(rows, Mapping):
            raise TypeError("rows must be a mapping of column name to values")
        missing = [field.name for field in self.fields if field.name not in rows]
        if missing:
            raise ValueError(f"missing columns for {self.name}: {', '.join(missing)}")
        extra = [name for name in rows if name not in self.names()]
        if extra:
            raise ValueError(f"unexpected columns for {self.name}: {', '.join(extra)}")
        ordered = {field.name: list(rows[field.name]) for field in self.fields}
        lengths = {len(values) for values in ordered.values()}
        if len(lengths) > 1:
            raise ValueError(f"columns for {self.name} have different lengths: {sorted(lengths)}")
        return ordered

    def to_pyarrow(self) -> Any:
        """Build a real ``pyarrow.Schema`` (imports pyarrow on first use)."""
        pa = require_pyarrow()
        return pa.schema(
            [
                pa.field(field.name, _arrow_type(field.type, pa), nullable=field.nullable)
                for field in self.fields
            ]
        )

    def record_batch(self, rows: Mapping[str, Sequence[Any]]) -> Any:
        """Build a ``pyarrow.RecordBatch`` from a validated column mapping."""
        pa = require_pyarrow()
        columns = self.columns(rows)
        return pa.record_batch(
            [columns[field.name] for field in self.fields], schema=self.to_pyarrow()
        )


GENERATE_REQUEST_SCHEMA = SchemaSpec(
    "generate.request",
    (
        FieldSpec("request_id", "utf8", False),
        FieldSpec("sequence", "uint64", False),
        FieldSpec("token_ids", "list<int32>", False),
        FieldSpec("max_tokens", "uint32", False),
        FieldSpec("temperature", "float32", False),
        FieldSpec("top_p", "float32", False),
        FieldSpec("top_k", "uint32", False),
        FieldSpec("stop_token_ids", "list<int32>", True),
        FieldSpec("stream", "bool", False),
    ),
)

GENERATE_RESPONSE_SCHEMA = SchemaSpec(
    "generate.response",
    (
        FieldSpec("request_id", "utf8", False),
        FieldSpec("sequence", "uint64", False),
        FieldSpec("token_id", "int32", False),
        FieldSpec("logprob", "float32", False),
        FieldSpec("top_logprobs", "list<struct<token_id:int32!, logprob:float32!>>", True),
        FieldSpec("finish_reason", "dictionary<uint16, utf8>", True),
        FieldSpec("finished", "bool", False),
    ),
)

EMBED_REQUEST_SCHEMA = SchemaSpec(
    "embed.request",
    (
        FieldSpec("request_id", "utf8", False),
        FieldSpec("token_id", "int32", False),
    ),
)

EMBED_RESPONSE_SCHEMA = SchemaSpec(
    "embed.response",
    (
        FieldSpec("request_id", "utf8", False),
        FieldSpec("token_id", "int32", False),
        FieldSpec("embedding", f"fixed_size_list<float32, {DEFAULT_EMBEDDING_DIM}>", False),
    ),
)

KV_BLOCKS_SCHEMA = SchemaSpec(
    "kv.blocks",
    (
        FieldSpec("request_id", "utf8", False),
        FieldSpec("block_id", "uint32", False),
        FieldSpec("layer_id", "uint16", False),
        FieldSpec("dtype", "dictionary<uint8, utf8>", False),
        FieldSpec("blocks", f"fixed_size_list<uint8, {DEFAULT_KV_BLOCK_BYTES}>", False),
    ),
)


def require_pyarrow() -> Any:
    """Import and return :mod:`pyarrow`, with a helpful error if absent."""
    try:
        import pyarrow  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingPyArrowError(
            "pyarrow is required for Arrow operations; install pythonsdk/requirements.txt"
        ) from exc
    return pyarrow


_SIMPLE_TYPES = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "utf8": "utf8",
}


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def _inner(spec: str, prefix: str) -> str:
    if not spec.startswith(prefix) or not spec.endswith(">"):
        raise ValueError(f"invalid type spec: {spec}")
    return spec[len(prefix) : -1]


def _arrow_type(spec: str, pa: Any) -> Any:
    spec = spec.strip()
    if spec.startswith("list<"):
        return pa.list_(_arrow_type(_inner(spec, "list<"), pa))
    if spec.startswith("fixed_size_list<"):
        parts = _split_top_level(_inner(spec, "fixed_size_list<"))
        if len(parts) != 2:
            raise ValueError(f"invalid fixed_size_list spec: {spec}")
        return pa.list_(_arrow_type(parts[0], pa), int(parts[1]))
    if spec.startswith("dictionary<"):
        parts = _split_top_level(_inner(spec, "dictionary<"))
        if len(parts) != 2:
            raise ValueError(f"invalid dictionary spec: {spec}")
        return pa.dictionary(_arrow_type(parts[0], pa), _arrow_type(parts[1], pa))
    if spec.startswith("struct<"):
        fields = []
        for part in _split_top_level(_inner(spec, "struct<")):
            name, _, nested = part.partition(":")
            if not _:
                raise ValueError(f"invalid struct field: {part}")
            nullable = not nested.endswith("!")
            nested_type = nested[:-1] if not nullable else nested
            fields.append(
                pa.field(name.strip(), _arrow_type(nested_type, pa), nullable=nullable)
            )
        return pa.struct(fields)
    name = _SIMPLE_TYPES.get(spec)
    if name is None:
        raise ValueError(f"unsupported Arrow type: {spec}")
    return getattr(pa, {"bool": "bool_", "utf8": "string"}.get(name, name))()


def _escape(value: str) -> str:
    out: list[str] = []
    for char in value:
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 32:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


def _canonical_field(field: FieldSpec) -> str:
    nullable = "true" if field.nullable else "false"
    return (
        f'{{"name":"{_escape(field.name)}","nullable":{nullable},'
        f'"type":{_canonical_type(field.type)}}}'
    )


def _canonical_type(spec: str) -> str:
    spec = spec.strip()
    if spec in _SIMPLE_TYPES:
        return f'{{"name":"{_SIMPLE_TYPES[spec]}"}}'
    if spec.startswith("list<"):
        item = FieldSpec("item", _inner(spec, "list<"), True)
        return f'{{"name":"list","item":{_canonical_field(item)}}}'
    if spec.startswith("fixed_size_list<"):
        parts = _split_top_level(_inner(spec, "fixed_size_list<"))
        if len(parts) != 2:
            raise ValueError(f"invalid fixed_size_list spec: {spec}")
        item = FieldSpec("item", parts[0], True)
        return f'{{"name":"fixed_size_list","item":{_canonical_field(item)},"size":{int(parts[1])}}}'
    if spec.startswith("dictionary<"):
        parts = _split_top_level(_inner(spec, "dictionary<"))
        if len(parts) != 2:
            raise ValueError(f"invalid dictionary spec: {spec}")
        return (
            '{"name":"dictionary","key":'
            f"{_canonical_type(parts[0])},\"value\":{_canonical_type(parts[1])}}}"
        )
    if spec.startswith("struct<"):
        fields = []
        for part in _split_top_level(_inner(spec, "struct<")):
            name, _, nested = part.partition(":")
            if not _:
                raise ValueError(f"invalid struct field: {part}")
            nullable = not nested.endswith("!")
            nested_type = nested[:-1] if not nullable else nested
            fields.append(_canonical_field(FieldSpec(name.strip(), nested_type, nullable)))
        return f'{{"name":"struct","fields":[{",".join(fields)}]}}'
    raise ValueError(f"unsupported Arrow type: {spec}")


def schema_hash(spec: SchemaSpec) -> str:
    """Return the stable 64-bit FNV-1a hash of the spec's canonical field JSON."""
    canonical = f'{{"fields":[{",".join(_canonical_field(f) for f in spec.fields)}]}}'
    digest = 0xCBF29CE484222325
    for byte in canonical.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"
