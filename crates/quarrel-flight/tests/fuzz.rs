//! Small proptests guarding the "malformed input never panics" contract.

use proptest::prelude::*;
use quarrel_core::{
    BlockId, GenerateRequest, KvBlock, KvDtype, ModelId, RequestId, SamplingParams, Token,
};
use quarrel_flight::{
    decode_embedding, decode_generate_request, decode_generate_response, decode_kv_block,
    decode_top_logprobs, embed_request_schema, embed_response_schema, encode_embedding,
    encode_generate_request, encode_generate_response_with_logprobs, encode_kv_block,
    generate_request_schema, generate_response_schema, ipc_to_batch, kv_blocks_schema,
};

fn id() -> RequestId {
    RequestId::new("fuzz-request").unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn arbitrary_ipc_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..128)) {
        for schema in [
            generate_request_schema(),
            generate_response_schema(),
            embed_request_schema(),
            embed_response_schema(),
            kv_blocks_schema(),
        ] {
            if let Ok(batch) = ipc_to_batch(&data, schema) {
                let _ = decode_generate_request(&batch);
                let _ = decode_generate_response(&batch);
                let _ = decode_top_logprobs(&batch);
                let _ = decode_embedding(&batch);
                let _ = decode_kv_block(&batch);
            }
        }
    }

    #[test]
    fn decode_generate_request_never_panics(
        token_ids in prop::collection::vec(any::<i32>(), 0..8),
        stop_ids in prop::collection::vec(any::<i32>(), 0..8),
        temperature in any::<f32>(),
        top_p in any::<f32>(),
        max_tokens in any::<u32>(),
    ) {
        let request = GenerateRequest {
            request_id: id(),
            model: ModelId::new("fuzz").unwrap(),
            token_ids,
            params: SamplingParams {
                temperature,
                top_p,
                top_k: 0,
                repetition_penalty: 1.0,
                max_tokens,
                stop_token_ids: stop_ids,
            },
            stream: true,
        };
        let batch = encode_generate_request(&request).unwrap();
        let _ = decode_generate_request(&batch);
    }

    #[test]
    fn decode_generate_response_never_panics(
        token_id in any::<i32>(),
        logprob in any::<f32>(),
        sequence in any::<u64>(),
        alternatives in prop::collection::vec((any::<i32>(), any::<f32>()), 0..4),
        finished in any::<bool>(),
    ) {
        let top = alternatives
            .into_iter()
            .map(|(id, logprob)| Token::new(id, logprob))
            .collect::<Vec<_>>();
        let batch = encode_generate_response_with_logprobs(
            &id(),
            sequence,
            &Token::new(token_id, logprob),
            finished,
            &top,
        )
        .unwrap();
        let _ = decode_generate_response(&batch);
        let _ = decode_top_logprobs(&batch);
    }

    #[test]
    fn decode_embedding_never_panics(values in prop::collection::vec(any::<f32>(), 0..16)) {
        let batch = encode_embedding(&id(), -1, &values).unwrap();
        let _ = decode_embedding(&batch);
    }

    #[test]
    fn decode_kv_block_never_panics(
        bytes in prop::collection::vec(any::<u8>(), 0..32),
        dtype_index in 0u8..3,
    ) {
        let dtype = match dtype_index {
            0 => KvDtype::F32,
            1 => KvDtype::F16,
            _ => KvDtype::TurboQuant,
        };
        let block = KvBlock::new(id(), BlockId::new(dtype_index as u32), 0, dtype, bytes);
        let batch = encode_kv_block(&block).unwrap();
        let _ = decode_kv_block(&batch);
    }
}
