use arrow::datatypes::{DataType, Field, Schema};
use quarrel_flight::{
    canonical_fields_json, embed_request_schema, embed_response_schema, generate_request_schema,
    generate_response_schema, kv_blocks_schema, schema_hash,
};

#[test]
fn generate_request_schema_matches_contract() {
    let schema = generate_request_schema();
    let expected = [
        ("request_id", DataType::Utf8),
        ("sequence", DataType::UInt64),
        (
            "token_ids",
            DataType::List(std::sync::Arc::new(Field::new(
                "item",
                DataType::Int32,
                true,
            ))),
        ),
        ("max_tokens", DataType::UInt32),
        ("temperature", DataType::Float32),
        ("top_p", DataType::Float32),
        ("top_k", DataType::UInt32),
        (
            "stop_token_ids",
            DataType::List(std::sync::Arc::new(Field::new(
                "item",
                DataType::Int32,
                true,
            ))),
        ),
        ("stream", DataType::Boolean),
    ];
    assert_eq!(schema.fields().len(), expected.len());
    for (field, (name, data_type)) in schema.fields().iter().zip(expected) {
        assert_eq!(field.name(), name);
        assert_eq!(field.data_type(), &data_type);
    }
}

#[test]
fn generate_response_schema_matches_contract() {
    let schema = generate_response_schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        [
            "request_id",
            "sequence",
            "token_id",
            "logprob",
            "top_logprobs",
            "finish_reason",
            "finished",
        ]
    );
    let DataType::List(item) = schema.field(4).data_type() else {
        panic!("top_logprobs must be a list");
    };
    let DataType::Struct(inner) = item.data_type() else {
        panic!("top_logprobs.item must be a struct");
    };
    assert_eq!(inner[0].name(), "token_id");
    assert_eq!(inner[0].data_type(), &DataType::Int32);
    assert_eq!(inner[1].name(), "logprob");
    assert_eq!(inner[1].data_type(), &DataType::Float32);
    assert_eq!(
        schema.field(5).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8))
    );
}

#[test]
fn embed_and_kv_schemas_match_contract() {
    let embed = embed_response_schema();
    assert_eq!(embed.field(0).data_type(), &DataType::Utf8);
    assert_eq!(embed.field(1).data_type(), &DataType::Int32);
    assert!(matches!(
        embed.field(2).data_type(),
        DataType::FixedSizeList(_, 4)
    ));

    let request = embed_request_schema();
    assert_eq!(request.fields().len(), 2);
    assert_eq!(request.field(0).name(), "request_id");
    assert_eq!(request.field(1).name(), "token_id");

    let kv = kv_blocks_schema();
    let names: Vec<&str> = kv.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        ["request_id", "block_id", "layer_id", "dtype", "blocks"]
    );
    assert_eq!(kv.field(1).data_type(), &DataType::UInt32);
    assert_eq!(kv.field(2).data_type(), &DataType::UInt16);
    assert_eq!(
        kv.field(3).data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
    );
    assert!(matches!(
        kv.field(4).data_type(),
        DataType::FixedSizeList(_, 16)
    ));
}

#[test]
#[allow(deprecated)] // deliberately exercising dict-id independence
fn schema_hash_is_stable_hex_and_sensitive_to_fields() {
    let hash = schema_hash(&generate_request_schema());
    assert_eq!(hash.len(), 16);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(hash, schema_hash(&generate_request_schema()));

    // Dictionary ids are IPC bookkeeping and must not affect the hash.
    let dict_a = Schema::new(vec![Field::new_dict(
        "finish_reason",
        DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
        true,
        1,
        false,
    )]);
    let dict_b = Schema::new(vec![Field::new_dict(
        "finish_reason",
        DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
        true,
        999,
        false,
    )]);
    assert_eq!(schema_hash(&dict_a), schema_hash(&dict_b));

    let renamed = Schema::new(vec![Field::new("renamed", DataType::Utf8, false)]);
    assert_ne!(schema_hash(&dict_a), schema_hash(&renamed));
}

#[test]
fn embed_schema_hashes_are_stable_and_distinct() {
    // Hashes are part of the health action contract; changing a field is a
    // wire-format change and must be reported by quarrel_build_info.
    assert_eq!(schema_hash(&embed_request_schema()), "f8ea32cf6498f9e7");
    assert_eq!(schema_hash(&embed_response_schema()), "1e2c5be4abd217ba");
    assert_ne!(
        schema_hash(&embed_request_schema()),
        schema_hash(&embed_response_schema())
    );
    assert_ne!(
        schema_hash(&embed_response_schema()),
        schema_hash(&generate_response_schema())
    );
}

#[test]
fn canonical_fields_json_is_deterministic() {
    let fields = generate_response_schema().fields().clone();
    assert_eq!(
        canonical_fields_json(&fields),
        canonical_fields_json(&fields)
    );
    assert!(canonical_fields_json(&fields).contains("top_logprobs"));
    assert!(canonical_fields_json(&fields).contains("token_id"));
}
