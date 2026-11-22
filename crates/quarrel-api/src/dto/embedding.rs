//! Embedding request and response types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::common::StringOrList;

/// `encoding_format` accepted by the embeddings endpoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    /// Return an array of floating point numbers.
    Float,
    /// Return IEEE-754 little-endian `f32` bytes as base64.
    Base64,
}

/// `POST /v1/embeddings` request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct EmbeddingRequest {
    /// Model id to embed with; must appear in `GET /v1/models`.
    pub model: String,
    /// Input strings, either one string or a list of strings.
    pub input: StringOrList,
    /// Output encoding; defaults to [`EncodingFormat::Float`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
}

impl EmbeddingRequest {
    /// Creates a request for one input string using the default float encoding.
    #[must_use]
    pub fn single(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: StringOrList::Single(input.into()),
            encoding_format: None,
        }
    }
}

/// Either the float array or the base64 form of an embedding.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingPayload {
    /// IEEE-754 `f32` values returned as a JSON array.
    Float(Vec<f32>),
    /// Base64-encoded little-endian `f32` bytes.
    Base64(String),
}

/// One embedding object in the response list.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct EmbeddingObject {
    /// Always `embedding`.
    pub object: String,
    /// Position of the input in the request.
    pub index: u32,
    /// The embedding vector in the requested encoding.
    pub embedding: EmbeddingPayload,
}

impl EmbeddingObject {
    /// Creates an embedding object with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(index: u32, embedding: EmbeddingPayload) -> Self {
        Self {
            object: "embedding".to_owned(),
            index,
            embedding,
        }
    }
}

/// Token accounting for an embeddings request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct EmbeddingUsage {
    /// Number of input tokens embedded.
    pub prompt_tokens: u32,
    /// Same as `prompt_tokens` for embeddings.
    pub total_tokens: u32,
}

impl EmbeddingUsage {
    /// Computes the total from the prompt token count.
    #[must_use]
    pub const fn new(prompt_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens,
        }
    }
}

/// Response for `POST /v1/embeddings`.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct EmbeddingList {
    /// Always `list`.
    pub object: String,
    /// One embedding per input string, in request order.
    pub data: Vec<EmbeddingObject>,
    /// Model used to produce the embeddings.
    pub model: String,
    /// Token accounting for the request.
    pub usage: EmbeddingUsage,
}

impl EmbeddingList {
    /// Creates a list with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(data: Vec<EmbeddingObject>, model: String, usage: EmbeddingUsage) -> Self {
        Self {
            object: "list".to_owned(),
            data,
            model,
            usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_accepts_both_input_forms_and_encodings() {
        let request: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":"hi","encoding_format":"base64"}"#)
                .unwrap();
        assert_eq!(request.input.as_slice(), ["hi"]);
        assert_eq!(request.encoding_format, Some(EncodingFormat::Base64));

        let request: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":["a","b"],"dimensions":8}"#).unwrap();
        assert_eq!(request.input.as_slice(), ["a", "b"]);
        assert_eq!(request.encoding_format, None);
        assert_eq!(EmbeddingRequest::single("m", "a").input.as_slice(), ["a"]);
    }

    #[test]
    fn unknown_encoding_is_rejected() {
        assert!(
            serde_json::from_str::<EmbeddingRequest>(
                r#"{"model":"m","input":"a","encoding_format":"hex"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn payload_and_list_round_trip() {
        let list = EmbeddingList::new(
            vec![EmbeddingObject::new(
                0,
                EmbeddingPayload::Float(vec![0.25, -0.5]),
            )],
            "m".to_owned(),
            EmbeddingUsage::new(3),
        );
        let json = serde_json::to_value(&list).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["embedding"][1], -0.5);
        assert_eq!(serde_json::from_value::<EmbeddingList>(json).unwrap(), list);

        let encoded = EmbeddingPayload::Base64("Zm9v".to_owned());
        let json = serde_json::to_string(&encoded).unwrap();
        assert_eq!(json, "\"Zm9v\"");
        assert_eq!(
            serde_json::from_str::<EmbeddingPayload>(&json).unwrap(),
            encoded
        );
    }
}
