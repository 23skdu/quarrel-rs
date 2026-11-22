//! Tiny deterministic byte-level tokenizer used by the reference HTTP handlers.
//!
//! This is deliberately *not* a real tokenizer and has no Go-parity claims. Each UTF-8 byte maps
//! to `byte % 32`, the vocabulary size of [`quarrel_engine::backend::ReferenceBackend`], so every
//! encoded id is in `0..32` and decoding is stable. It exists so HTTP requests can be turned into
//! reproducible engine inputs without shipping model weights.

/// Number of token ids produced and consumed by the reference tokenizer.
pub const VOCAB_SIZE: usize = 32;

/// Encodes UTF-8 bytes as token ids in `0..VOCAB_SIZE`.
///
/// Returns an empty vector for empty input; handlers reject empty prompts before calling the
/// engine.
#[must_use]
pub fn encode(text: &str) -> Vec<i32> {
    text.as_bytes()
        .iter()
        .map(|byte| i32::from(byte % VOCAB_SIZE as u8))
        .collect()
}

/// Decodes token ids back into printable ASCII characters.
///
/// Ids outside `0..VOCAB_SIZE` are normalised with `rem_euclid`, so arbitrary engine output cannot
/// panic. The mapping is the inverse of [`encode`] for lowercase ASCII letters, space and digits.
#[must_use]
pub fn decode(ids: &[i32]) -> String {
    ids.iter()
        .map(|&id| {
            let slot = id.rem_euclid(VOCAB_SIZE as i32) as u8;
            match slot {
                0 => b' ',
                1..=26 => b'a' + (slot - 1),
                _ => b'0' + (slot - 27),
            }
        })
        .map(char::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{VOCAB_SIZE, decode, encode};

    #[test]
    fn lowercase_ascii_round_trips() {
        let ids = encode("hello world");
        assert_eq!(ids, vec![8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4]);
        assert!(ids.iter().all(|&id| (0..VOCAB_SIZE as i32).contains(&id)));
        assert_eq!(decode(&ids), "hello world");
    }

    #[test]
    fn empty_and_non_ascii_inputs_are_handled() {
        assert!(encode("").is_empty());
        assert_eq!(decode(&[]), "");
        let ids = encode("héllo");
        assert_eq!(ids.len(), "héllo".len());
        assert!(ids.iter().all(|&id| (0..VOCAB_SIZE as i32).contains(&id)));
        assert_eq!(decode(&ids).len(), ids.len());
    }

    #[test]
    fn decode_normalises_out_of_range_ids() {
        assert_eq!(decode(&[-1, 32, 33, i32::MIN]), decode(&[31, 0, 1, 0]));
    }
}
