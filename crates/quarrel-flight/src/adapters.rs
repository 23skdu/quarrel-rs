//! Conversions between Arrow `RecordBatch`es and `quarrel_core` domain types.

use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, BooleanBuilder, DictionaryArray, FixedSizeListArray,
        Float32Array, Float32Builder, Int32Array, Int32Builder, ListArray, ListBuilder,
        RecordBatch, StringArray, StringBuilder, StringDictionaryBuilder, StructBuilder,
        UInt8Array, UInt8Builder, UInt16Array, UInt16Builder, UInt32Array, UInt32Builder,
        UInt64Builder,
    },
    datatypes::{DataType, Field, Fields, Schema, SchemaRef, UInt8Type, UInt16Type},
};
use quarrel_core::{GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams, Token};

use crate::{FlightError, generate_request_schema, generate_response_schema, schema_hash};

/// Model reported for decoded requests whose batch does not carry a model
/// column (the Flight contract has no such column).
pub const DEFAULT_MODEL_ID: &str = "reference";

/// Finish reason written for terminal generate responses.
pub const FINISH_REASON_LENGTH: &str = "length";

/// Return a generate-response schema with `top_logprobs` removed, for callers
/// that only need the flat token columns.
///
/// Not used by the service; kept as a convenience for schema introspection.
#[doc(hidden)]
pub fn generate_response_token_schema() -> SchemaRef {
    let schema = generate_response_schema();
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .filter(|field| field.name() != "top_logprobs")
            .cloned()
            .collect::<Vec<_>>(),
    ))
}

/// Encode a generate request as a single-row batch.
pub fn encode_generate_request(request: &GenerateRequest) -> Result<RecordBatch, FlightError> {
    let mut request_ids = StringBuilder::new();
    let mut sequences = UInt64Builder::new();
    let mut token_ids = ListBuilder::new(Int32Builder::new());
    let mut max_tokens = UInt32Builder::new();
    let mut temperatures = Float32Builder::new();
    let mut top_ps = Float32Builder::new();
    let mut top_ks = UInt32Builder::new();
    let mut stop_token_ids = ListBuilder::new(Int32Builder::new());
    let mut streams = BooleanBuilder::new();

    request_ids.append_value(request.request_id.as_str());
    sequences.append_value(0);
    for id in &request.token_ids {
        token_ids.values().append_value(*id);
    }
    token_ids.append(true);
    max_tokens.append_value(request.params.max_tokens);
    temperatures.append_value(request.params.temperature);
    top_ps.append_value(request.params.top_p);
    top_ks.append_value(request.params.top_k);
    for id in &request.params.stop_token_ids {
        stop_token_ids.values().append_value(*id);
    }
    stop_token_ids.append(true);
    streams.append_value(request.stream);

    Ok(RecordBatch::try_new(
        generate_request_schema(),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(sequences.finish()),
            Arc::new(token_ids.finish()),
            Arc::new(max_tokens.finish()),
            Arc::new(temperatures.finish()),
            Arc::new(top_ps.finish()),
            Arc::new(top_ks.finish()),
            Arc::new(stop_token_ids.finish()),
            Arc::new(streams.finish()),
        ],
    )?)
}

/// Decode a single-row generate request batch.
///
/// The Flight contract has no `model` column, so the returned request uses
/// [`DEFAULT_MODEL_ID`]. All other fields round-trip exactly.
pub fn decode_generate_request(batch: &RecordBatch) -> Result<GenerateRequest, FlightError> {
    check_row_count(batch)?;
    let request_id = string_value(batch, "request_id", 0)?;
    let request_id =
        RequestId::new(&request_id).map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let model = ModelId::new(DEFAULT_MODEL_ID)
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let token_ids = int_list_value(batch, "token_ids", 0)?;
    let max_tokens = scalar_u32(batch, "max_tokens", 0)?;
    let temperature = scalar_f32(batch, "temperature", 0)?;
    let top_p = scalar_f32(batch, "top_p", 0)?;
    let top_k = scalar_u32(batch, "top_k", 0)?;
    let stop_token_ids = int_list_value(batch, "stop_token_ids", 0)?;
    let stream = scalar_bool(batch, "stream", 0)?;

    let params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty: SamplingParams::DEFAULT_REPETITION_PENALTY,
        max_tokens,
        stop_token_ids,
    };

    GenerateRequest::new(request_id, model, token_ids, params, stream)
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))
}

/// Encode one generated token as a single-row response batch.
///
/// Equivalent to [`encode_generate_response_with_logprobs`] with no top-k
/// alternatives.
pub fn encode_generate_response(
    request_id: &RequestId,
    sequence: u64,
    token: &Token,
    finished: bool,
) -> Result<RecordBatch, FlightError> {
    encode_generate_response_with_logprobs(request_id, sequence, token, finished, &[])
}

/// Encode one generated token together with its alternative log-probabilities.
pub fn encode_generate_response_with_logprobs(
    request_id: &RequestId,
    sequence: u64,
    token: &Token,
    finished: bool,
    top_logprobs: &[Token],
) -> Result<RecordBatch, FlightError> {
    let mut request_ids = StringBuilder::new();
    let mut sequences = UInt64Builder::new();
    let mut token_ids = Int32Builder::new();
    let mut logprobs = Float32Builder::new();
    let mut finish_reasons = StringDictionaryBuilder::<UInt16Type>::new();
    let mut finisheds = BooleanBuilder::new();

    let response_schema = generate_response_schema();
    let DataType::List(top_item) = response_schema.field(4).data_type() else {
        return Err(FlightError::InvalidArgument(
            "top_logprobs must be a list".into(),
        ));
    };
    let top_item = top_item.clone();
    let struct_fields = Fields::from(vec![
        Field::new("token_id", DataType::Int32, false),
        Field::new("logprob", DataType::Float32, false),
    ]);
    let mut top_builder = ListBuilder::new(StructBuilder::new(
        struct_fields,
        vec![
            Box::new(Int32Builder::new()),
            Box::new(Float32Builder::new()),
        ],
    ))
    .with_field(top_item);

    request_ids.append_value(request_id.as_str());
    sequences.append_value(sequence);
    token_ids.append_value(token.id);
    logprobs.append_value(token.logprob);
    for alternative in top_logprobs {
        let struct_builder = top_builder.values();
        struct_builder
            .field_builder::<Int32Builder>(0)
            .expect("int32 field builder")
            .append_value(alternative.id);
        struct_builder
            .field_builder::<Float32Builder>(1)
            .expect("float32 field builder")
            .append_value(alternative.logprob);
        struct_builder.append(true);
    }
    top_builder.append(true);
    if finished {
        finish_reasons.append_value(FINISH_REASON_LENGTH);
    } else {
        finish_reasons.append_null();
    }
    finisheds.append_value(finished);

    Ok(RecordBatch::try_new(
        response_schema,
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(sequences.finish()),
            Arc::new(token_ids.finish()),
            Arc::new(logprobs.finish()),
            Arc::new(top_builder.finish()),
            Arc::new(finish_reasons.finish()),
            Arc::new(finisheds.finish()),
        ],
    )?)
}

/// Decode the tokens of one-or-more response batches, in row order.
pub fn decode_generate_response(batch: &RecordBatch) -> Result<Vec<Token>, FlightError> {
    check_nonempty(batch)?;
    let ids = typed_column::<Int32Array>(batch, "token_id", &DataType::Int32)?;
    let logprobs = typed_column::<Float32Array>(batch, "logprob", &DataType::Float32)?;
    // Flight clients may receive this column hydrated as plain Utf8, so only
    // require its presence; the token payload does not depend on its type.
    batch
        .schema()
        .column_with_name("finish_reason")
        .ok_or(FlightError::MissingField("finish_reason"))?;

    let mut tokens = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        // `finish_reason` is intentionally null until the terminal batch.
        if ids.is_null(row) || logprobs.is_null(row) {
            return Err(FlightError::InvalidArgument(format!(
                "null token column at row {row}"
            )));
        }
        tokens.push(Token::new(ids.value(row), logprobs.value(row)));
    }
    Ok(tokens)
}

/// Decode the nested `top_logprobs` list from a response batch.
pub fn decode_top_logprobs(batch: &RecordBatch) -> Result<Vec<Vec<Token>>, FlightError> {
    let field = generate_response_schema().field(4).data_type().clone();
    let lists = typed_column::<ListArray>(batch, "top_logprobs", &field)?;
    let mut result = Vec::with_capacity(lists.len());
    for row in 0..lists.len() {
        if lists.is_null(row) {
            result.push(Vec::new());
            continue;
        }
        let values = lists.value(row);
        let structs = values
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .ok_or(FlightError::MissingField("top_logprobs.item"))?;
        let ids = structs
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(FlightError::MissingField("top_logprobs.token_id"))?;
        let logprobs = structs
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(FlightError::MissingField("top_logprobs.logprob"))?;
        let mut alternatives = Vec::with_capacity(structs.len());
        for i in 0..structs.len() {
            if ids.is_null(i) || logprobs.is_null(i) {
                return Err(FlightError::InvalidArgument(format!(
                    "null top_logprobs entry at row {row}, item {i}"
                )));
            }
            alternatives.push(Token::new(ids.value(i), logprobs.value(i)));
        }
        result.push(alternatives);
    }
    Ok(result)
}

/// Return the terminal flags of a response batch.
pub fn decode_finished(batch: &RecordBatch) -> Result<Vec<bool>, FlightError> {
    let values = typed_column::<BooleanArray>(batch, "finished", &DataType::Boolean)?;
    Ok((0..values.len()).map(|i| values.value(i)).collect())
}

/// Encode an embedding as a single-row batch.
///
/// The `FixedSizeList` width is the actual embedding length, so any dimension
/// round-trips; the canonical helper uses [`crate::DEFAULT_EMBEDDING_DIM`].
pub fn encode_embedding(
    request_id: &RequestId,
    token_id: i32,
    embedding: &[f32],
) -> Result<RecordBatch, FlightError> {
    let dim = i32::try_from(embedding.len())
        .map_err(|_| FlightError::InvalidArgument("embedding too large".into()))?;
    let mut request_ids = StringBuilder::new();
    let mut token_ids = Int32Builder::new();
    let values = Float32Builder::new();
    let mut embeddings = arrow::array::FixedSizeListBuilder::new(values, dim);

    request_ids.append_value(request_id.as_str());
    token_ids.append_value(token_id);
    for value in embedding {
        embeddings.values().append_value(*value);
    }
    embeddings.append(true);

    // Arrow's builder collapses a zero-width list to zero rows; build the
    // one-row degenerate array explicitly so empty embeddings still round-trip.
    let embedding_column: ArrayRef = if dim == 0 {
        Arc::new(FixedSizeListArray::try_new_with_length(
            Arc::new(Field::new("item", DataType::Float32, true)),
            0,
            Arc::new(Float32Array::from(Vec::<f32>::new())),
            None,
            1,
        )?)
    } else {
        Arc::new(embeddings.finish())
    };

    Ok(RecordBatch::try_new(
        embedding_response_schema_for(embedding.len()),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(token_ids.finish()),
            embedding_column,
        ],
    )?)
}

/// Decode a single-row embedding batch into `(request_id, token_id, values)`.
pub fn decode_embedding(batch: &RecordBatch) -> Result<(String, i32, Vec<f32>), FlightError> {
    check_row_count(batch)?;
    let request_id = string_value(batch, "request_id", 0)?;
    let token_id = scalar_i32(batch, "token_id", 0)?;

    let schema = batch.schema();
    let (index, field) = schema
        .column_with_name("embedding")
        .ok_or(FlightError::MissingField("embedding"))?;
    let DataType::FixedSizeList(_, dim) = field.data_type() else {
        return Err(FlightError::InvalidArgument(
            "embedding must be a FixedSizeList".into(),
        ));
    };
    let array = batch.column(index);
    let list = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or(FlightError::MissingField("embedding"))?;
    if list.is_null(0) {
        return Err(FlightError::InvalidArgument("null embedding".into()));
    }
    let values = list.value(0);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or(FlightError::MissingField("embedding.item"))?;
    let expected = usize::try_from(*dim).unwrap_or(0);
    if floats.len() != expected {
        return Err(FlightError::InvalidArgument(format!(
            "embedding width mismatch: expected {expected}, got {}",
            floats.len()
        )));
    }
    Ok((
        request_id,
        token_id,
        (0..floats.len()).map(|i| floats.value(i)).collect(),
    ))
}

/// Return the embedding-response schema for a specific vector width.
pub fn embedding_response_schema_for(dim: usize) -> SchemaRef {
    let dim = i32::try_from(dim).unwrap_or(i32::MAX);
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("token_id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]))
}

/// Encode a raw KV-cache block as a single-row batch.
pub fn encode_kv_block(block: &KvBlock) -> Result<RecordBatch, FlightError> {
    let len = i32::try_from(block.bytes.len())
        .map_err(|_| FlightError::InvalidArgument("KV block too large".into()))?;
    let mut request_ids = StringBuilder::new();
    let mut block_ids = UInt32Builder::new();
    let mut layer_ids = UInt16Builder::new();
    let mut dtypes = StringDictionaryBuilder::<UInt8Type>::new();
    let byte_values = UInt8Builder::new();
    let mut blocks = arrow::array::FixedSizeListBuilder::new(byte_values, len);

    request_ids.append_value(block.request_id.as_str());
    block_ids.append_value(block.block_id.get());
    layer_ids.append_value(block.layer_id);
    dtypes.append_value(block.dtype.as_str());
    for byte in &block.bytes {
        blocks.values().append_value(*byte);
    }
    blocks.append(true);

    let blocks_column: ArrayRef = if len == 0 {
        Arc::new(FixedSizeListArray::try_new_with_length(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            0,
            Arc::new(UInt8Array::from(Vec::<u8>::new())),
            None,
            1,
        )?)
    } else {
        Arc::new(blocks.finish())
    };

    Ok(RecordBatch::try_new(
        kv_blocks_schema_for(block.bytes.len()),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(block_ids.finish()),
            Arc::new(layer_ids.finish()),
            Arc::new(dtypes.finish()),
            blocks_column,
        ],
    )?)
}

/// Decode a single-row KV-cache batch.
pub fn decode_kv_block(batch: &RecordBatch) -> Result<KvBlock, FlightError> {
    check_row_count(batch)?;
    let request_id = RequestId::new(&string_value(batch, "request_id", 0)?)
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let block_id = quarrel_core::BlockId::new(scalar_u32(batch, "block_id", 0)?);
    let layer_id = scalar_u16(batch, "layer_id", 0)?;

    let dtype_field = DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
    let dtype_array = typed_column::<DictionaryArray<UInt8Type>>(batch, "dtype", &dtype_field)?;
    if dtype_array.is_null(0) {
        return Err(FlightError::InvalidArgument("null dtype".into()));
    }
    let keys = dtype_array
        .keys()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or(FlightError::MissingField("dtype.key"))?;
    let values = dtype_array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(FlightError::MissingField("dtype.value"))?;
    let key = keys.value(0) as usize;
    if key >= values.len() {
        return Err(FlightError::InvalidArgument(
            "dtype dictionary key out of range".into(),
        ));
    }
    let dtype = KvDtype::from_str(values.value(key))
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;

    let schema = batch.schema();
    let (index, field) = schema
        .column_with_name("blocks")
        .ok_or(FlightError::MissingField("blocks"))?;
    let DataType::FixedSizeList(_, width) = field.data_type() else {
        return Err(FlightError::InvalidArgument(
            "blocks must be a FixedSizeList".into(),
        ));
    };
    let list = batch
        .column(index)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or(FlightError::MissingField("blocks"))?;
    if list.is_null(0) {
        return Err(FlightError::InvalidArgument("null blocks".into()));
    }
    let raw = list.value(0);
    let bytes = raw
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or(FlightError::MissingField("blocks.item"))?;
    let expected = usize::try_from(*width).unwrap_or(0);
    if bytes.len() != expected {
        return Err(FlightError::InvalidArgument(format!(
            "block width mismatch: expected {expected}, got {}",
            bytes.len()
        )));
    }

    Ok(KvBlock::new(
        request_id,
        block_id,
        layer_id,
        dtype,
        (0..bytes.len()).map(|i| bytes.value(i)).collect(),
    ))
}

/// Return the KV-block schema for a specific raw payload width.
pub fn kv_blocks_schema_for(width: usize) -> SchemaRef {
    let width = i32::try_from(width).unwrap_or(i32::MAX);
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        Field::new("block_id", DataType::UInt32, false),
        Field::new("layer_id", DataType::UInt16, false),
        Field::new_dictionary("dtype", DataType::UInt8, DataType::Utf8, false),
        Field::new(
            "blocks",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), width),
            false,
        ),
    ]))
}

// -- helpers ---------------------------------------------------------------

fn check_row_count(batch: &RecordBatch) -> Result<(), FlightError> {
    match batch.num_rows() {
        1 => Ok(()),
        n => Err(FlightError::ExpectedSingleRow(n)),
    }
}

fn check_nonempty(batch: &RecordBatch) -> Result<(), FlightError> {
    if batch.num_rows() == 0 {
        Err(FlightError::InvalidArgument("empty response batch".into()))
    } else {
        Ok(())
    }
}

fn typed_column<'a, A: Array + 'static>(
    batch: &'a RecordBatch,
    name: &'static str,
    expected: &DataType,
) -> Result<&'a A, FlightError> {
    let schema = batch.schema();
    let (index, field) = schema
        .column_with_name(name)
        .ok_or(FlightError::MissingField(name))?;
    if field.data_type() != expected {
        return Err(FlightError::SchemaMismatch {
            expected: schema_hash(&Schema::new(vec![field.as_ref().clone()])),
            actual: format!("{expected:?}"),
        });
    }
    let array = batch.column(index);
    array
        .as_any()
        .downcast_ref::<A>()
        .ok_or(FlightError::MissingField(name))
}

fn string_value(
    batch: &RecordBatch,
    name: &'static str,
    row: usize,
) -> Result<String, FlightError> {
    let values = typed_column::<StringArray>(batch, name, &DataType::Utf8)?;
    if row >= values.len() || values.is_null(row) {
        return Err(FlightError::InvalidArgument(format!(
            "missing value for {name} at row {row}"
        )));
    }
    Ok(values.value(row).to_owned())
}

fn array_column<'a, A: Array + 'static>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a A, FlightError> {
    let (index, _) = batch
        .schema()
        .column_with_name(name)
        .ok_or(FlightError::MissingField(name))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<A>()
        .ok_or(FlightError::MissingField(name))
}

macro_rules! scalar_getter {
    ($fn_name:ident, $array:ty, $native:ty) => {
        fn $fn_name(
            batch: &RecordBatch,
            name: &'static str,
            row: usize,
        ) -> Result<$native, FlightError> {
            let values = array_column::<$array>(batch, name)?;
            if row >= values.len() || values.is_null(row) {
                return Err(FlightError::InvalidArgument(format!(
                    "missing value for {name} at row {row}"
                )));
            }
            Ok(values.value(row))
        }
    };
}

scalar_getter!(scalar_i32, Int32Array, i32);
scalar_getter!(scalar_u16, UInt16Array, u16);
scalar_getter!(scalar_u32, UInt32Array, u32);
scalar_getter!(scalar_f32, Float32Array, f32);
scalar_getter!(scalar_bool, BooleanArray, bool);

fn int_list_value(
    batch: &RecordBatch,
    name: &'static str,
    row: usize,
) -> Result<Vec<i32>, FlightError> {
    let expected = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    let lists = typed_column::<ListArray>(batch, name, &expected)?;
    if row >= lists.len() || lists.is_null(row) {
        return Err(FlightError::InvalidArgument(format!(
            "missing value for {name} at row {row}"
        )));
    }
    let values = lists.value(row);
    let ints = values
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or(FlightError::MissingField("list.item"))?;
    let mut result = Vec::with_capacity(ints.len());
    for i in 0..ints.len() {
        if ints.is_null(i) {
            return Err(FlightError::InvalidArgument(format!(
                "null entry in {name} at index {i}"
            )));
        }
        result.push(ints.value(i));
    }
    Ok(result)
}
