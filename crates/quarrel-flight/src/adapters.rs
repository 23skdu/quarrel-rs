//! Conversions between Arrow `RecordBatch`es and `quarrel_core` domain types.

use std::sync::{Arc, OnceLock};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, BooleanBuilder, DictionaryArray, FixedSizeListArray,
        Float32Array, Float32Builder, Int32Array, Int32Builder, ListArray, ListBuilder,
        PrimitiveArray, RecordBatch, StringArray, StringBuilder, StringDictionaryBuilder,
        StructBuilder, UInt8Array, UInt8Builder, UInt16Array, UInt16Builder, UInt32Array,
        UInt32Builder, UInt64Array, UInt64Builder,
    },
    buffer::{BooleanBuffer, Buffer, OffsetBuffer, ScalarBuffer},
    datatypes::{
        ArrowPrimitiveType, DataType, Field, Fields, Float32Type, Int32Type, Schema, SchemaRef,
        UInt8Type, UInt16Type,
    },
};
use quarrel_core::{GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams, Token};

use crate::{FlightError, generate_request_schema, generate_response_schema, schema_hash};

/// Model reported for decoded requests whose batch does not carry a model
/// column (the Flight contract has no such column).
pub const DEFAULT_MODEL_ID: &str = "reference";

/// Finish reason written for terminal generate responses.
pub const FINISH_REASON_LENGTH: &str = "length";

/// A validated generate request whose prompt slices borrow Arrow buffers.
///
/// This is the engine-shaped view of a Flight request: the prompt is a
/// `&[i32]` pointing directly into the decoded `token_ids` array, so handing it
/// to a backend does not allocate or copy a token vector. Sampling controls are
/// owned because `quarrel_engine::generate` takes `&SamplingParams`.
pub struct GenerateRequestView<'a> {
    /// Identifier echoed on every response for this request.
    pub request_id: RequestId,
    /// Model selected for the request (always [`DEFAULT_MODEL_ID`] on Flight).
    pub model: ModelId,
    /// Prompt token ids borrowed from the Arrow `token_ids` buffer.
    pub token_ids: &'a [i32],
    /// Validated sampling controls.
    pub params: SamplingParams,
    /// Whether the engine should emit incremental token messages.
    pub stream: bool,
}

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

/// Decode a single-row generate request into a borrowed, engine-shaped view.
///
/// Unlike [`decode_generate_request`], the prompt and stop-token slices borrow
/// the Arrow buffers in `batch` instead of being copied into owned vectors.
/// The Flight contract has no `model` column, so the view uses
/// [`DEFAULT_MODEL_ID`]. Empty prompts and invalid sampling parameters are
/// rejected with [`FlightError::InvalidArgument`].
pub fn decode_generate_request_view(
    batch: &RecordBatch,
) -> Result<GenerateRequestView<'_>, FlightError> {
    check_row_count(batch)?;
    let request_id = RequestId::new(&string_value(batch, "request_id", 0)?)
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let model = ModelId::new(DEFAULT_MODEL_ID)
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let token_ids = request_token_ids(batch)?;
    if token_ids.is_empty() {
        return Err(FlightError::InvalidArgument(
            "empty prompt token_ids".into(),
        ));
    }
    let stop_token_ids = request_stop_token_ids(batch)?;
    let params = SamplingParams {
        temperature: scalar_f32(batch, "temperature", 0)?,
        top_p: scalar_f32(batch, "top_p", 0)?,
        top_k: scalar_u32(batch, "top_k", 0)?,
        repetition_penalty: SamplingParams::DEFAULT_REPETITION_PENALTY,
        max_tokens: scalar_u32(batch, "max_tokens", 0)?,
        // Stop ids are capped at 256 entries by `SamplingParams::validate`,
        // so this copy stays small and bounded.
        stop_token_ids: stop_token_ids.to_vec(),
    };
    params
        .validate()
        .map_err(|err| FlightError::InvalidArgument(err.to_string()))?;
    let stream = scalar_bool(batch, "stream", 0)?;

    Ok(GenerateRequestView {
        request_id,
        model,
        token_ids,
        params,
        stream,
    })
}

/// Decode a single-row generate request batch.
///
/// The Flight contract has no `model` column, so the returned request uses
/// [`DEFAULT_MODEL_ID`]. All other fields round-trip exactly. Callers that hand
/// the prompt straight to an engine should prefer
/// [`decode_generate_request_view`], which borrows the token buffer.
pub fn decode_generate_request(batch: &RecordBatch) -> Result<GenerateRequest, FlightError> {
    let view = decode_generate_request_view(batch)?;
    Ok(GenerateRequest {
        request_id: view.request_id,
        model: view.model,
        token_ids: view.token_ids.to_vec(),
        params: view.params,
        stream: view.stream,
    })
}

/// Borrow the prompt token ids from a generate-request batch.
///
/// The returned slice points into the Arrow `Int32` values buffer; no token
/// vector is allocated. Null list values or null entries are rejected.
pub fn request_token_ids(batch: &RecordBatch) -> Result<&[i32], FlightError> {
    check_row_count(batch)?;
    let lists = typed_column::<ListArray>(batch, "token_ids", token_list_datatype())?;
    int_list_slice(lists, "token_ids", 0)?
        .ok_or_else(|| FlightError::InvalidArgument("missing value for token_ids at row 0".into()))
}

/// Borrow the stop-token ids from a generate-request batch.
///
/// A null `stop_token_ids` list yields an empty slice. Non-null lists must not
/// contain null entries.
pub fn request_stop_token_ids(batch: &RecordBatch) -> Result<&[i32], FlightError> {
    check_row_count(batch)?;
    let lists = typed_column::<ListArray>(batch, "stop_token_ids", token_list_datatype())?;
    Ok(int_list_slice(lists, "stop_token_ids", 0)?.unwrap_or(&[]))
}

/// Borrow the token ids of a generate-response batch.
pub fn response_token_ids(batch: &RecordBatch) -> Result<&[i32], FlightError> {
    primitive_slice::<Int32Type>(batch, "token_id", &DataType::Int32)
}

/// Borrow the log-probabilities of a generate-response batch.
pub fn response_logprobs(batch: &RecordBatch) -> Result<&[f32], FlightError> {
    primitive_slice::<Float32Type>(batch, "logprob", &DataType::Float32)
}

/// Borrow the floating-point payload of a single-row embedding batch.
pub fn embedding_values(batch: &RecordBatch) -> Result<&[f32], FlightError> {
    check_row_count(batch)?;
    let (list, floats, width) = embedding_components(batch)?;
    if list.is_null(0) {
        return Err(FlightError::InvalidArgument("null embedding".into()));
    }
    let start = list.value_offset_at(0);
    let end = start.saturating_add(width);
    if end > floats.len() {
        return Err(FlightError::InvalidArgument(format!(
            "embedding width mismatch: expected {width}, got {}",
            end.saturating_sub(start)
        )));
    }
    Ok(&floats.values().as_ref()[start..end])
}

/// Borrow the full `rows x dim` float payload of an embedding batch.
///
/// Unlike [`embedding_values`], this helper accepts batches with any number of
/// rows and returns the underlying `FixedSizeList` values in row-major order.
/// It is the shape used by the 250k x 384 IPC benchmark.
pub fn embedding_matrix_values(batch: &RecordBatch) -> Result<&[f32], FlightError> {
    let (list, floats, width) = embedding_components(batch)?;
    let total = width
        .checked_mul(list.len())
        .ok_or_else(|| FlightError::InvalidArgument("embedding matrix is too large".into()))?;
    let start = list.value_offset_at(0);
    let end = start.saturating_add(total);
    if end > floats.len() {
        return Err(FlightError::InvalidArgument(format!(
            "embedding payload too short: expected {total}, got {}",
            end.saturating_sub(start)
        )));
    }
    Ok(&floats.values().as_ref()[start..end])
}

/// Borrow the raw bytes of a single-row KV-cache block batch.
///
/// Dtype parsing and host endianness checks happen in [`decode_kv_block`], not
/// here; this helper only exposes the Arrow byte buffer.
pub fn kv_block_payload(batch: &RecordBatch) -> Result<&[u8], FlightError> {
    check_row_count(batch)?;
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
    let expected = usize::try_from(*width).unwrap_or(0);
    let bytes = list
        .values()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or(FlightError::MissingField("blocks.item"))?;
    let start = list.value_offset_at(0);
    let end = start.saturating_add(expected);
    if end > bytes.len() {
        return Err(FlightError::InvalidArgument(format!(
            "block width mismatch: expected {expected}, got {}",
            end.saturating_sub(start)
        )));
    }
    if bytes.null_count() != 0 {
        return Err(FlightError::InvalidArgument(
            "null entry in KV block payload".into(),
        ));
    }
    Ok(&bytes.values().as_ref()[start..end])
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

/// Build a one-row response batch from existing Arrow token columns.
///
/// `token_ids`, `logprobs` and `finished` are cloned as `ArrayRef`s, i.e. their
/// value buffers are shared rather than copied. This is the zero-copy
/// counterpart of [`encode_generate_response`] for callers that already hold
/// Arrow arrays (for example after an IPC decode). Metadata columns and the
/// constant `top_logprobs`/`finish_reason` columns are still constructed here.
///
/// # Errors
///
/// Returns [`FlightError::InvalidArgument`] if any column is not exactly one
/// row or contains a null value.
pub fn encode_generate_response_from_arrays(
    request_id: &RequestId,
    sequence: u64,
    token_ids: &Int32Array,
    logprobs: &Float32Array,
    finished: &BooleanArray,
) -> Result<RecordBatch, FlightError> {
    if token_ids.len() != 1 || logprobs.len() != 1 || finished.len() != 1 {
        return Err(FlightError::InvalidArgument(
            "response token columns must have exactly one row".into(),
        ));
    }
    if token_ids.is_null(0) || logprobs.is_null(0) || finished.is_null(0) {
        return Err(FlightError::InvalidArgument(
            "null value in response token columns".into(),
        ));
    }
    let (request_ids, sequences) = scalar_metadata_columns(request_id, sequence);
    Ok(RecordBatch::try_new(
        generate_response_schema(),
        vec![
            Arc::new(request_ids),
            Arc::new(sequences),
            Arc::new(token_ids.clone()),
            Arc::new(logprobs.clone()),
            empty_top_logprobs()?,
            finish_reason_array(finished.value(0))?,
            Arc::new(finished.clone()),
        ],
    )?)
}

/// Reusable builder for generate-response batches.
///
/// The builder keeps small typed scratch vectors between calls so that the
/// repeated append/resize allocations of a fresh builder do not recur after
/// warm-up. Each returned [`RecordBatch`] owns immutable Arrow buffers, so
/// those output buffers are necessarily allocated per response; what is reused
/// is the builder's scratch capacity and its prebuilt, shared constant columns
/// (`top_logprobs` and `finish_reason`).
#[derive(Debug)]
pub struct GenerateResponseBuilder {
    request_id_values: Vec<u8>,
    request_id_offsets: Vec<i32>,
    sequences: Vec<u64>,
    token_ids: Vec<i32>,
    logprobs: Vec<f32>,
    finished: Vec<bool>,
    top_logprobs: ArrayRef,
    finish_reasons: [ArrayRef; 2],
    schema: SchemaRef,
}

impl Default for GenerateResponseBuilder {
    fn default() -> Self {
        Self::new().expect("canonical response schema is valid")
    }
}

impl GenerateResponseBuilder {
    /// Create a builder primed with the canonical response schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the canonical schema cannot produce the constant
    /// `top_logprobs` or `finish_reason` columns.
    pub fn new() -> Result<Self, FlightError> {
        Ok(Self {
            request_id_values: Vec::new(),
            request_id_offsets: Vec::new(),
            sequences: Vec::new(),
            token_ids: Vec::new(),
            logprobs: Vec::new(),
            finished: Vec::new(),
            top_logprobs: empty_top_logprobs()?,
            finish_reasons: [finish_reason_array(false)?, finish_reason_array(true)?],
            schema: generate_response_schema(),
        })
    }

    /// Encode one response token, reusing this builder's scratch capacity.
    pub fn encode(
        &mut self,
        request_id: &RequestId,
        sequence: u64,
        token: &Token,
        finished: bool,
    ) -> Result<RecordBatch, FlightError> {
        let id = request_id.as_str().as_bytes();
        let len = i32::try_from(id.len())
            .map_err(|_| FlightError::InvalidArgument("request id too long".into()))?;
        self.request_id_values.clear();
        self.request_id_values.extend_from_slice(id);
        self.request_id_offsets.clear();
        self.request_id_offsets.push(0);
        self.request_id_offsets.push(len);
        self.sequences.clear();
        self.sequences.push(sequence);
        self.token_ids.clear();
        self.token_ids.push(token.id);
        self.logprobs.clear();
        self.logprobs.push(token.logprob);
        self.finished.clear();
        self.finished.push(finished);

        let request_ids = StringArray::try_new(
            OffsetBuffer::new(ScalarBuffer::from(Buffer::from_slice_ref(
                &self.request_id_offsets,
            ))),
            Buffer::from_slice_ref(&self.request_id_values),
            None,
        )?;
        let sequences = UInt64Array::new(
            ScalarBuffer::from(Buffer::from_slice_ref(&self.sequences)),
            None,
        );
        let token_ids = Int32Array::new(
            ScalarBuffer::from(Buffer::from_slice_ref(&self.token_ids)),
            None,
        );
        let logprobs = Float32Array::new(
            ScalarBuffer::from(Buffer::from_slice_ref(&self.logprobs)),
            None,
        );
        let finished_array = BooleanArray::new(BooleanBuffer::from(self.finished.as_slice()), None);

        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(request_ids),
                Arc::new(sequences),
                Arc::new(token_ids),
                Arc::new(logprobs),
                Arc::clone(&self.top_logprobs),
                Arc::clone(&self.finish_reasons[usize::from(finished)]),
                Arc::new(finished_array),
            ],
        )?)
    }

    /// Total capacity, in bytes, of this builder's reusable scratch vectors.
    ///
    /// Tests use this to verify that warm-up stabilises scratch allocations.
    #[must_use]
    pub fn scratch_capacity(&self) -> usize {
        self.request_id_values.capacity()
            + self.request_id_offsets.capacity() * std::mem::size_of::<i32>()
            + self.sequences.capacity() * std::mem::size_of::<u64>()
            + self.token_ids.capacity() * std::mem::size_of::<i32>()
            + self.logprobs.capacity() * std::mem::size_of::<f32>()
            + self.finished.capacity() * std::mem::size_of::<bool>()
    }
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
    let values = embedding_values(batch)?;
    Ok((request_id, token_id, values.to_vec()))
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
    assert_kv_dtype_endianness(block.dtype)?;
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
    let dtype = parse_kv_dtype(values.value(key))?;
    let bytes = kv_block_payload(batch)?;

    Ok(KvBlock::new(
        request_id,
        block_id,
        layer_id,
        dtype,
        bytes.to_vec(),
    ))
}

/// Parse a canonical KV-cache dtype string and assert host byte order.
///
/// All Flight KV payloads are specified little-endian. On a big-endian host
/// this returns [`FlightError::EndiannessMismatch`] instead of silently
/// interpreting bytes with the wrong width order.
///
/// # Errors
///
/// Returns [`FlightError::UnsupportedKvDtype`] for unknown dtype strings or
/// [`FlightError::EndiannessMismatch`] on a big-endian host.
pub fn parse_kv_dtype(value: &str) -> Result<KvDtype, FlightError> {
    let dtype =
        KvDtype::from_str(value).map_err(|err| FlightError::UnsupportedKvDtype(err.to_string()))?;
    assert_kv_dtype_endianness(dtype)?;
    Ok(dtype)
}

/// Assert that this host can interpret the given KV-cache dtype.
///
/// # Errors
///
/// Returns [`FlightError::EndiannessMismatch`] when the host is big-endian.
pub fn assert_kv_dtype_endianness(dtype: KvDtype) -> Result<(), FlightError> {
    check_kv_dtype_endianness(dtype, cfg!(target_endian = "little"))
}

fn check_kv_dtype_endianness(
    dtype: KvDtype,
    host_is_little_endian: bool,
) -> Result<(), FlightError> {
    if host_is_little_endian {
        Ok(())
    } else {
        Err(FlightError::EndiannessMismatch {
            dtype: dtype.as_str(),
            expected: "little",
            actual: "big",
        })
    }
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

fn embedding_components(
    batch: &RecordBatch,
) -> Result<(&FixedSizeListArray, &Float32Array, usize), FlightError> {
    let schema = batch.schema();
    let (index, field) = schema
        .column_with_name("embedding")
        .ok_or(FlightError::MissingField("embedding"))?;
    let DataType::FixedSizeList(_, width) = field.data_type() else {
        return Err(FlightError::InvalidArgument(
            "embedding must be a FixedSizeList".into(),
        ));
    };
    let list = batch
        .column(index)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or(FlightError::MissingField("embedding"))?;
    let floats = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or(FlightError::MissingField("embedding.item"))?;
    if floats.null_count() != 0 {
        return Err(FlightError::InvalidArgument(
            "null entry in embedding".into(),
        ));
    }
    Ok((list, floats, usize::try_from(*width).unwrap_or(0)))
}

fn token_list_datatype() -> &'static DataType {
    static TOKEN_LIST: OnceLock<DataType> = OnceLock::new();
    TOKEN_LIST.get_or_init(|| DataType::List(Arc::new(Field::new("item", DataType::Int32, true))))
}

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

fn primitive_slice<'a, T: ArrowPrimitiveType>(
    batch: &'a RecordBatch,
    name: &'static str,
    expected: &DataType,
) -> Result<&'a [T::Native], FlightError> {
    let values = typed_column::<PrimitiveArray<T>>(batch, name, expected)?;
    if values.null_count() != 0 {
        return Err(FlightError::InvalidArgument(format!(
            "null entry in {name}"
        )));
    }
    Ok(values.values().as_ref())
}

fn int_list_slice<'a>(
    lists: &'a ListArray,
    name: &'static str,
    row: usize,
) -> Result<Option<&'a [i32]>, FlightError> {
    if row >= lists.len() {
        return Err(FlightError::InvalidArgument(format!(
            "missing value for {name} at row {row}"
        )));
    }
    if lists.is_null(row) {
        return Ok(None);
    }
    let offsets = lists.value_offsets();
    let start = usize::try_from(offsets[row]).map_err(|_| {
        FlightError::InvalidArgument(format!("invalid offset in {name} at row {row}"))
    })?;
    let end = usize::try_from(offsets[row + 1]).map_err(|_| {
        FlightError::InvalidArgument(format!("invalid offset in {name} at row {row}"))
    })?;
    let ints = lists
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or(FlightError::MissingField("list.item"))?;
    if end > ints.len() {
        return Err(FlightError::InvalidArgument(format!(
            "invalid list bounds in {name} at row {row}"
        )));
    }
    if ints.null_count() != 0 {
        return Err(FlightError::InvalidArgument(format!(
            "null entry in {name} at index 0"
        )));
    }
    Ok(Some(&ints.values().as_ref()[start..end]))
}

fn scalar_metadata_columns(request_id: &RequestId, sequence: u64) -> (StringArray, UInt64Array) {
    let mut request_ids = StringBuilder::new();
    let mut sequences = UInt64Builder::new();
    request_ids.append_value(request_id.as_str());
    sequences.append_value(sequence);
    (request_ids.finish(), sequences.finish())
}

fn empty_top_logprobs() -> Result<ArrayRef, FlightError> {
    let schema = generate_response_schema();
    let DataType::List(top_item) = schema.field(4).data_type() else {
        return Err(FlightError::InvalidArgument(
            "top_logprobs must be a list".into(),
        ));
    };
    let struct_fields = Fields::from(vec![
        Field::new("token_id", DataType::Int32, false),
        Field::new("logprob", DataType::Float32, false),
    ]);
    let mut builder = ListBuilder::new(StructBuilder::new(
        struct_fields,
        vec![
            Box::new(Int32Builder::new()),
            Box::new(Float32Builder::new()),
        ],
    ))
    .with_field(top_item.clone());
    builder.append(true);
    Ok(Arc::new(builder.finish()))
}

fn finish_reason_array(finished: bool) -> Result<ArrayRef, FlightError> {
    let keys = if finished {
        UInt16Array::from(vec![0])
    } else {
        UInt16Array::from(vec![None])
    };
    let values: ArrayRef = Arc::new(StringArray::from(vec![FINISH_REASON_LENGTH]));
    Ok(Arc::new(DictionaryArray::try_new(keys, values)?))
}

#[cfg(test)]
mod tests {
    use super::check_kv_dtype_endianness;
    use crate::FlightError;
    use quarrel_core::KvDtype;

    #[test]
    fn big_endian_host_yields_typed_endianness_error() {
        for dtype in [KvDtype::F32, KvDtype::F16, KvDtype::TurboQuant] {
            match check_kv_dtype_endianness(dtype, false) {
                Err(FlightError::EndiannessMismatch {
                    dtype: name,
                    expected,
                    actual,
                }) => {
                    assert_eq!(name, dtype.as_str());
                    assert_eq!(expected, "little");
                    assert_eq!(actual, "big");
                }
                other => panic!("unexpected endianness check result: {other:?}"),
            }
        }
    }
}
