use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, FixedSizeListBuilder, Float32Array, Int32Array,
        Int32Builder, ListBuilder, StringArray, StringBuilder, StringDictionaryBuilder,
        UInt8Builder,
    },
    datatypes::{DataType, Field, Schema, UInt8Type},
    record_batch::RecordBatch,
};
use quarrel_core::{
    BlockId, GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams, Token,
};
use quarrel_flight::{
    FlightError, GenerateResponseBuilder, assert_kv_dtype_endianness, decode_embedding,
    decode_finished, decode_generate_request, decode_generate_request_view,
    decode_generate_response, decode_kv_block, decode_top_logprobs, embed_response_schema,
    embedding_matrix_values, embedding_values, encode_embedding, encode_generate_request,
    encode_generate_response, encode_generate_response_from_arrays,
    encode_generate_response_with_logprobs, encode_kv_block, generate_request_schema,
    generate_response_schema, kv_block_payload, kv_blocks_schema, kv_blocks_schema_for,
    parse_kv_dtype, request_stop_token_ids, request_token_ids, response_logprobs,
    response_token_ids,
};

fn request_id() -> RequestId {
    RequestId::new("agent-0001").expect("valid request id")
}

fn params() -> SamplingParams {
    SamplingParams {
        temperature: 2.0,
        top_p: 1.0,
        top_k: 100_000,
        repetition_penalty: 1.0,
        max_tokens: SamplingParams::MAX_MAX_TOKENS,
        stop_token_ids: vec![i32::MIN, -1, i32::MAX],
    }
}

#[test]
fn generate_request_round_trip_with_extremes() {
    let request = GenerateRequest::new(
        request_id(),
        ModelId::new("fixture-model").unwrap(),
        vec![i32::MIN, -7, 0, i32::MAX],
        params(),
        true,
    )
    .unwrap();
    let batch = encode_generate_request(&request).unwrap();
    assert_eq!(batch.schema(), generate_request_schema());

    let decoded = decode_generate_request(&batch).unwrap();
    assert_eq!(decoded.request_id, request.request_id);
    assert_eq!(decoded.token_ids, request.token_ids);
    assert_eq!(decoded.stream, request.stream);
    // The Flight schema has no model column.
    assert_eq!(decoded.model.as_str(), quarrel_flight::DEFAULT_MODEL_ID);
    assert_eq!(decoded.params.max_tokens, SamplingParams::MAX_MAX_TOKENS);
    assert_eq!(decoded.params.temperature, 2.0);
    assert_eq!(decoded.params.top_p, 1.0);
    assert_eq!(decoded.params.top_k, 100_000);
    assert_eq!(decoded.params.stop_token_ids, request.params.stop_token_ids);
}

#[test]
fn generate_request_empty_optional_list_and_zero_temperature() {
    let mut settings = params();
    settings.temperature = 0.0;
    settings.top_p = 0.0;
    settings.top_k = 0;
    settings.max_tokens = 1;
    settings.stop_token_ids.clear();
    let request = GenerateRequest::new(
        request_id(),
        ModelId::new("m").unwrap(),
        vec![42],
        settings,
        false,
    )
    .unwrap();

    let decoded = decode_generate_request(&encode_generate_request(&request).unwrap()).unwrap();
    assert!(decoded.params.stop_token_ids.is_empty());
    assert_eq!(decoded.params.temperature, 0.0);
    assert_eq!(decoded.params.max_tokens, 1);
    assert!(!decoded.stream);
}

#[test]
fn generate_request_rejects_empty_prompt_on_decode() {
    let request = GenerateRequest {
        request_id: request_id(),
        model: ModelId::new("m").unwrap(),
        token_ids: Vec::new(),
        params: params(),
        stream: true,
    };
    let batch = encode_generate_request(&request).unwrap();
    assert!(matches!(
        decode_generate_request(&batch),
        Err(FlightError::InvalidArgument(_))
    ));
}

#[test]
fn generate_response_round_trip_with_nested_logprobs() {
    let token = Token::new(i32::MAX, f32::MIN);
    let alternatives = [
        Token::new(i32::MIN, f32::MAX),
        Token::new(-1, -0.0),
        Token::new(0, f32::MIN_POSITIVE),
    ];
    let batch = encode_generate_response_with_logprobs(
        &request_id(),
        u64::MAX,
        &token,
        true,
        &alternatives,
    )
    .unwrap();
    assert_eq!(batch.schema(), generate_response_schema());

    let decoded = decode_generate_response(&batch).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].id, token.id);
    assert_eq!(decoded[0].logprob.to_bits(), token.logprob.to_bits());
    assert_eq!(decode_finished(&batch).unwrap(), vec![true]);
    assert_eq!(
        decode_top_logprobs(&batch).unwrap(),
        vec![alternatives.to_vec()]
    );

    let dictionary = batch
        .column(5)
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt16Type>>()
        .unwrap();
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(values.value(0), "length");
}

#[test]
fn generate_response_empty_top_logprobs_and_nonterminal() {
    let token = Token::new(i32::MIN, f32::MIN_POSITIVE);
    let batch = encode_generate_response(&request_id(), 0, &token, false).unwrap();
    assert!(decode_top_logprobs(&batch).unwrap()[0].is_empty());
    assert_eq!(decode_finished(&batch).unwrap(), vec![false]);
    assert!(
        batch
            .column(5)
            .as_any()
            .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt16Type>>()
            .unwrap()
            .is_null(0)
    );
    assert_eq!(decode_generate_response(&batch).unwrap(), vec![token]);
}

#[test]
fn embedding_round_trip_including_empty_vector() {
    for values in [
        Vec::new(),
        vec![0.0],
        vec![f32::MIN, -0.0, f32::MAX, f32::MIN_POSITIVE],
    ] {
        let batch = encode_embedding(&request_id(), i32::MIN, &values).unwrap();
        let (id, token_id, decoded) = decode_embedding(&batch).unwrap();
        assert_eq!(id, "agent-0001");
        assert_eq!(token_id, i32::MIN);
        assert_eq!(decoded.len(), values.len());
        for (a, b) in decoded.iter().zip(&values) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    let canonical = encode_embedding(&request_id(), 1, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(canonical.schema(), embed_response_schema());
}

#[test]
fn kv_block_round_trip_for_every_dtype() {
    for dtype in [KvDtype::F32, KvDtype::F16, KvDtype::TurboQuant] {
        for bytes in [Vec::new(), vec![0_u8, 255, 1, 0, 128]] {
            let block = KvBlock::new(
                request_id(),
                BlockId::new(u32::MAX),
                u16::MAX,
                dtype,
                bytes.clone(),
            );
            let batch = encode_kv_block(&block).unwrap();
            let decoded = decode_kv_block(&batch).unwrap();
            assert_eq!(decoded, block);
        }
    }

    let canonical = KvBlock::new(request_id(), BlockId::new(3), 7, KvDtype::F32, vec![0; 16]);
    assert_eq!(
        encode_kv_block(&canonical).unwrap().schema(),
        kv_blocks_schema()
    );
}

#[test]
fn decoders_reject_wrong_types_and_missing_fields() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "token_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["not an int"]))],
    )
    .unwrap();
    assert!(decode_generate_response(&batch).is_err());
    assert!(matches!(
        decode_generate_request(&batch),
        Err(FlightError::MissingField(_))
    ));
    assert!(matches!(
        decode_embedding(&batch),
        Err(FlightError::MissingField(_))
    ));

    let id = Int32Array::from(vec![1, 2]);
    let two_rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "token_id",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(id)],
    )
    .unwrap();
    assert!(matches!(
        decode_generate_request(&two_rows),
        Err(FlightError::ExpectedSingleRow(2))
    ));

    let floats = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "logprob",
            DataType::Float32,
            false,
        )])),
        vec![Arc::new(Float32Array::from(vec![1.0]))],
    )
    .unwrap();
    assert!(decode_generate_response(&floats).is_err());
}

fn one_column_batch(field: Field, column: ArrayRef) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap()
}

fn kv_batch_with_dtype(raw: &str) -> RecordBatch {
    let mut request_ids = StringBuilder::new();
    let mut block_ids = arrow::array::UInt32Builder::new();
    let mut layer_ids = arrow::array::UInt16Builder::new();
    let mut dtypes = StringDictionaryBuilder::<UInt8Type>::new();
    let mut blocks = FixedSizeListBuilder::new(UInt8Builder::new(), 2);
    request_ids.append_value(request_id().as_str());
    block_ids.append_value(1);
    layer_ids.append_value(2);
    dtypes.append_value(raw);
    blocks.values().append_value(1);
    blocks.values().append_value(2);
    blocks.append(true);
    RecordBatch::try_new(
        kv_blocks_schema_for(2),
        vec![
            Arc::new(request_ids.finish()),
            Arc::new(block_ids.finish()),
            Arc::new(layer_ids.finish()),
            Arc::new(dtypes.finish()),
            Arc::new(blocks.finish()),
        ],
    )
    .unwrap()
}

#[test]
fn borrowed_request_and_payload_slices_point_into_arrow_buffers() {
    let tokens: Vec<i32> = (0..1024).collect();
    let stops = vec![-1, i32::MAX];
    let request = GenerateRequest::new(
        request_id(),
        ModelId::new("m").unwrap(),
        tokens.clone(),
        SamplingParams {
            stop_token_ids: stops.clone(),
            ..params()
        },
        true,
    )
    .unwrap();
    let batch = encode_generate_request(&request).unwrap();

    let view = decode_generate_request_view(&batch).unwrap();
    assert_eq!(view.token_ids, tokens);
    assert_eq!(view.params.stop_token_ids, stops);
    let ids = request_token_ids(&batch).unwrap();
    let stop_ids = request_stop_token_ids(&batch).unwrap();
    let raw_ids = batch
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .unwrap();
    let raw_ints = raw_ids
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.as_ptr(), raw_ints.values().as_ptr());
    assert_eq!(stop_ids, stops);

    let block = KvBlock::new(request_id(), BlockId::new(9), 3, KvDtype::F32, vec![7; 128]);
    let kv_batch = encode_kv_block(&block).unwrap();
    let payload = kv_block_payload(&kv_batch).unwrap();
    assert_eq!(payload, block.bytes);
    let raw_list = kv_batch
        .column(4)
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
        .unwrap();
    let raw_bytes = raw_list
        .values()
        .as_any()
        .downcast_ref::<arrow::array::UInt8Array>()
        .unwrap();
    assert_eq!(payload.as_ptr(), raw_bytes.values().as_ptr());
}

#[test]
fn kv_dtype_parsing_accepts_canonical_values_and_rejects_unknown() {
    for (raw, expected) in [
        ("f32", KvDtype::F32),
        ("f16", KvDtype::F16),
        ("turboquant", KvDtype::TurboQuant),
    ] {
        assert_eq!(parse_kv_dtype(raw).unwrap(), expected);
        assert_eq!(
            decode_kv_block(&kv_batch_with_dtype(raw)).unwrap().dtype,
            expected
        );
    }
    assert!(matches!(
        parse_kv_dtype("bf16"),
        Err(FlightError::UnsupportedKvDtype(_))
    ));
    assert!(matches!(
        decode_kv_block(&kv_batch_with_dtype("bf16")),
        Err(FlightError::UnsupportedKvDtype(_))
    ));
    assert_kv_dtype_endianness(KvDtype::F32).unwrap();
    if cfg!(target_endian = "big") {
        assert!(matches!(
            assert_kv_dtype_endianness(KvDtype::F16),
            Err(FlightError::EndiannessMismatch { .. })
        ));
    }
}

#[test]
fn borrowed_slice_helpers_reject_nulls_and_wrong_types() {
    let mut token_ids = ListBuilder::new(Int32Builder::new());
    token_ids.append(false);
    let batch = one_column_batch(
        Field::new(
            "token_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Arc::new(token_ids.finish()),
    );
    assert!(matches!(
        request_token_ids(&batch),
        Err(FlightError::InvalidArgument(_))
    ));

    let mut blocks = FixedSizeListBuilder::new(UInt8Builder::new(), 2);
    blocks.values().append_value(1);
    blocks.values().append_value(2);
    blocks.append(false);
    let batch = one_column_batch(
        Field::new(
            "blocks",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 2),
            true,
        ),
        Arc::new(blocks.finish()),
    );
    assert!(matches!(
        kv_block_payload(&batch),
        Err(FlightError::InvalidArgument(_))
    ));

    let embeddings = arrow::array::FixedSizeListArray::try_new_with_length(
        Arc::new(Field::new("item", DataType::Float32, true)),
        2,
        Arc::new(Float32Array::from(vec![None, Some(2.0)])),
        None,
        1,
    )
    .unwrap();
    let batch = one_column_batch(
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
        Arc::new(embeddings),
    );
    assert!(matches!(
        embedding_values(&batch),
        Err(FlightError::InvalidArgument(_))
    ));

    let batch = one_column_batch(
        Field::new("token_id", DataType::Utf8, false),
        Arc::new(StringArray::from(vec!["x"])),
    );
    assert!(matches!(
        response_token_ids(&batch),
        Err(FlightError::SchemaMismatch { .. } | FlightError::MissingField(_))
    ));

    let batch = one_column_batch(
        Field::new("embedding", DataType::Utf8, false),
        Arc::new(StringArray::from(vec!["x"])),
    );
    assert!(matches!(
        embedding_matrix_values(&batch),
        Err(FlightError::InvalidArgument(_) | FlightError::MissingField(_))
    ));
}

#[test]
fn response_from_arrays_shares_token_buffers_and_checks_inputs() {
    let ids = Int32Array::from(vec![42]);
    let logprobs = Float32Array::from(vec![-0.25]);
    let finished = BooleanArray::from(vec![true]);
    let batch =
        encode_generate_response_from_arrays(&request_id(), 7, &ids, &logprobs, &finished).unwrap();
    assert_eq!(batch.schema(), generate_response_schema());
    let raw_ids = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(raw_ids.values().as_ptr(), ids.values().as_ptr());
    assert_eq!(decode_generate_response(&batch).unwrap()[0].id, 42);
    assert_eq!(response_token_ids(&batch).unwrap(), &[42]);
    assert_eq!(response_logprobs(&batch).unwrap(), &[-0.25]);

    let short = Int32Array::from(Vec::<i32>::new());
    assert!(matches!(
        encode_generate_response_from_arrays(&request_id(), 7, &short, &logprobs, &finished),
        Err(FlightError::InvalidArgument(_))
    ));
    let null = Int32Array::from(vec![None]);
    assert!(matches!(
        encode_generate_response_from_arrays(&request_id(), 7, &null, &logprobs, &finished),
        Err(FlightError::InvalidArgument(_))
    ));
}

#[test]
fn generate_response_builder_stabilises_scratch_and_round_trips() {
    let mut builder = GenerateResponseBuilder::new().unwrap();
    let token = Token::new(11, -0.125);
    for index in 0..32 {
        builder
            .encode(&request_id(), index, &token, index == 31)
            .unwrap();
    }
    let capacity = builder.scratch_capacity();
    assert!(capacity > 0);
    for index in 0..64 {
        let batch = builder
            .encode(&request_id(), index, &token, index == 63)
            .unwrap();
        assert_eq!(batch.schema(), generate_response_schema());
        assert_eq!(
            decode_generate_response(&batch).unwrap(),
            vec![token.clone()]
        );
        assert_eq!(decode_finished(&batch).unwrap(), vec![index == 63]);
        assert!(decode_top_logprobs(&batch).unwrap()[0].is_empty());
        assert_eq!(builder.scratch_capacity(), capacity);
    }
}
