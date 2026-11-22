//! `GET /v1/models` response types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// One model reported by the OpenAI model-list endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct Model {
    /// Model identifier used in generation requests (the GGUF file stem).
    pub id: String,
    /// Always `model`.
    pub object: String,
    /// Owner of the model; `quarrel-rs` for locally discovered files.
    pub owned_by: String,
}

impl Model {
    /// Creates a model entry with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_owned(),
            owned_by: "quarrel-rs".to_owned(),
        }
    }
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct ModelList {
    /// Always `list`.
    pub object: String,
    /// Models discovered from the configured models directory, sorted by id.
    pub data: Vec<Model>,
}

impl ModelList {
    /// Creates a model list with the OpenAI object discriminator filled in.
    #[must_use]
    pub fn new(data: Vec<Model>) -> Self {
        Self {
            object: "list".to_owned(),
            data,
        }
    }

    /// Creates an empty model list, useful when no GGUF files are present.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::{Model, ModelList};

    #[test]
    fn constructors_follow_openai_shape() {
        let model = Model::new("tiny");
        assert_eq!(model.object, "model");
        assert_eq!(model.owned_by, "quarrel-rs");
        let list = ModelList::new(vec![model.clone()]);
        assert_eq!(list.object, "list");
        assert_eq!(list.data, vec![model]);
        assert!(ModelList::empty().data.is_empty());

        let json = serde_json::to_value(&list).unwrap();
        assert_eq!(json["data"][0]["id"], "tiny");
        assert_eq!(
            serde_json::from_value::<ModelList>(json).unwrap(),
            ModelList::new(vec![Model::new("tiny")])
        );
    }
}
