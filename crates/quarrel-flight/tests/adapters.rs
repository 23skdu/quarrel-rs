use std::sync::Arc;

use arrow::{
    array::{Array, Float32Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use quarrel_core::{
    BlockId, GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams, Token,
};
use quarrel_flight::{
    FlightError, decode_embedding, decode_finished, decode_generate_request,
    decode_generate_response, decode_kv_block, decode_top_logprobs, embed_response_schema,
    encode_embedding, encode_generate_request, encode_generate_response,
    encode_generate_response_with_logprobs, encode_kv_block, generate_request_schema,
    generate_response_schema, kv_blocks_schema,
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
