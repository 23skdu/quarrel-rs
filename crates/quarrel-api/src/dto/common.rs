//! Wire type for fields OpenAI accepts as either one string or a list of strings.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// A JSON value that may be a single string or an array of strings.
///
/// Used by `input` on embedding requests, `prompt` on completion requests and `stop` on generation
/// requests. Keeping the untagged representation in the DTO means handlers never have to accept
/// `serde_json::Value` and lose compile-time checking.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(untagged)]
pub enum StringOrList {
    /// A single value, for example `"hello"`.
    Single(String),
    /// Several values, for example `["hello", "world"]`.
    List(Vec<String>),
}

impl StringOrList {
    /// Returns the values as a slice without cloning.
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::Single(value) => std::slice::from_ref(value),
            Self::List(values) => values,
        }
    }

    /// Consumes the value and returns its strings.
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(value) => vec![value],
            Self::List(values) => values,
        }
    }
}

impl From<&str> for StringOrList {
    fn from(value: &str) -> Self {
        Self::Single(value.to_owned())
    }
}

impl From<String> for StringOrList {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<Vec<String>> for StringOrList {
    fn from(values: Vec<String>) -> Self {
        Self::List(values)
    }
}

#[cfg(test)]
mod tests {
    use super::StringOrList;

    #[test]
    fn single_and_list_round_trip() {
        let single: StringOrList = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(single, StringOrList::Single("hello".to_owned()));
        assert_eq!(single.as_slice(), ["hello"]);
        assert_eq!(serde_json::to_string(&single).unwrap(), "\"hello\"");
        assert_eq!(single.into_vec(), vec!["hello"]);

        let list: StringOrList = serde_json::from_str("[\"a\", \"b\"]").unwrap();
        assert_eq!(list.as_slice(), ["a", "b"]);
        assert_eq!(serde_json::to_string(&list).unwrap(), "[\"a\",\"b\"]");
        assert_eq!(list.into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        assert!(serde_json::from_str::<StringOrList>("42").is_err());
        assert!(serde_json::from_str::<StringOrList>("[\"a\", 2]").is_err());
    }

    #[test]
    fn conversions_are_convenient() {
        assert_eq!(
            StringOrList::from("x"),
            StringOrList::Single("x".to_owned())
        );
        assert_eq!(
            StringOrList::from(String::from("x")),
            StringOrList::Single("x".to_owned())
        );
        assert_eq!(
            StringOrList::from(vec![String::from("x")]),
            StringOrList::List(vec!["x".to_owned()])
        );
    }
}
