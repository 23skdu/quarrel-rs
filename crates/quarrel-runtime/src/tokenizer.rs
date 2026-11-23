//! GGUF-native tokenizers for SmolLM2 (BPE) and Gemma 3 (SentencePiece).

use crate::error::RuntimeError;
use crate::model::optional_metadata_string;
use quarrel_gguf::{GgufHeader, MetadataValue};
use std::collections::HashMap;

/// Tokenizer family stored in GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// GPT-2 byte-level BPE with `tokenizer.ggml.merges`.
    Bpe,
    /// SentencePiece-style model with `tokenizer.ggml.scores`.
    Spm,
}

/// One chat turn used by the built-in prompt templates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Role name, for example `user` or `assistant`.
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ChatMessage {
    /// Create a chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A tokenizer built entirely from GGUF metadata arrays.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    kind: TokenizerKind,
    tokens: Vec<String>,
    token_ids: HashMap<String, u32>,
    scores: Vec<f32>,
    merges: HashMap<(String, String), usize>,
    specials: Vec<(String, u32)>,
    bos: u32,
    eos: u32,
    unknown: u32,
    add_bos: bool,
    byte_to_char: HashMap<u8, char>,
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Build a tokenizer from a parsed GGUF header.
    ///
    /// # Errors
    ///
    /// Returns an error when token arrays are missing, inconsistent, or use an
    /// unsupported `tokenizer.ggml.model` value.
    pub fn from_header(header: &GgufHeader) -> Result<Self, RuntimeError> {
        let tokens = metadata_strings(header, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(RuntimeError::Tokenizer(
                "tokenizer.ggml.tokens is empty".to_owned(),
            ));
        }
        let mut token_ids = HashMap::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| {
                RuntimeError::Tokenizer(format!(
                    "vocabulary has more than u32::MAX tokens ({})",
                    tokens.len()
                ))
            })?;
            token_ids.insert(token.clone(), id);
        }
        let model = optional_metadata_string(header, "tokenizer.ggml.model")?
            .unwrap_or_else(|| "llama".to_owned());
        let kind = match model.as_str() {
            "gpt2" => TokenizerKind::Bpe,
            "llama" => TokenizerKind::Spm,
            other => {
                return Err(RuntimeError::Tokenizer(format!(
                    "unsupported tokenizer model {other:?}"
                )));
            }
        };
        let scores = match optional_floats(header, "tokenizer.ggml.scores")? {
            Some(scores) if scores.len() == tokens.len() => scores,
            Some(scores) => {
                return Err(RuntimeError::Tokenizer(format!(
                    "scores array has {} entries, expected {}",
                    scores.len(),
                    tokens.len()
                )));
            }
            None => vec![0.0; tokens.len()],
        };
        let types = match optional_u8s(header, "tokenizer.ggml.token_type")? {
            Some(types) if types.len() == tokens.len() => types,
            Some(types) => {
                return Err(RuntimeError::Tokenizer(format!(
                    "token_type array has {} entries, expected {}",
                    types.len(),
                    tokens.len()
                )));
            }
            None => vec![1; tokens.len()],
        };
        let merges = match kind {
            TokenizerKind::Bpe => load_merges(header)?,
            TokenizerKind::Spm => HashMap::new(),
        };
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(index, token)| {
                let index = *index;
                matches!(types.get(index), Some(3 | 4))
                    || token.starts_with("<|")
                    || token.starts_with("<start_of_turn>")
                    || token.starts_with("<end_of_turn>")
            })
            .map(|(index, token)| {
                (
                    token.clone(),
                    u32::try_from(index).expect("vocabulary fits u32"),
                )
            })
            .collect();
        specials.sort_by_key(|left| std::cmp::Reverse(left.0.len()));
        let (byte_to_char, char_to_byte) = gpt2_byte_alphabet();
        let bos = optional_u32(header, "tokenizer.ggml.bos_token_id")?
            .or_else(|| token_ids.get("<bos>").copied())
            .unwrap_or(0);
        let eos = optional_u32(header, "tokenizer.ggml.eos_token_id")?
            .or_else(|| token_ids.get("<eos>").copied())
            .or_else(|| token_ids.get("<|endoftext|>").copied())
            .unwrap_or(0);
        let unknown = optional_u32(header, "tokenizer.ggml.unknown_token_id")?
            .or_else(|| token_ids.get("<unk>").copied())
            .unwrap_or(0);
        let add_bos = optional_bool(header, "tokenizer.ggml.add_bos_token")?.unwrap_or(false);
        Ok(Self {
            kind,
            tokens,
            token_ids,
            scores,
            merges,
            specials,
            bos,
            eos,
            unknown,
            add_bos,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Tokenizer family inferred from metadata.
    #[must_use]
    pub const fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// Number of tokens in the vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty. Real GGUFs never produce an empty tokenizer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Beginning-of-sequence token id.
    #[must_use]
    pub const fn bos_id(&self) -> u32 {
        self.bos
    }

    /// End-of-sequence token id.
    #[must_use]
    pub const fn eos_id(&self) -> u32 {
        self.eos
    }

    /// Token id used for unencodable input.
    #[must_use]
    pub const fn unknown_id(&self) -> u32 {
        self.unknown
    }

    /// Whether plain text prompts should be prefixed with BOS.
    #[must_use]
    pub const fn adds_bos(&self) -> bool {
        self.add_bos
    }

    /// Convert token ids to text using the tokenizer's byte/unicode mapping.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            let Some(token) = self.tokens.get(id as usize) else {
                continue;
            };
            if token.starts_with("<|") || token.starts_with("<start_of_turn>") {
                out.extend_from_slice(token.as_bytes());
                continue;
            }
            match self.kind {
                TokenizerKind::Bpe => {
                    for ch in token.chars() {
                        if let Some(&byte) = self.char_to_byte.get(&ch) {
                            out.push(byte);
                        } else {
                            out.extend_from_slice(ch.to_string().as_bytes());
                        }
                    }
                }
                TokenizerKind::Spm => {
                    if let Some(byte) = byte_fallback(token) {
                        out.push(byte);
                    } else {
                        out.extend_from_slice(token.replace('▁', " ").as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Encode text, recognizing special-token literals in the input.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer metadata is internally inconsistent.
    pub fn encode(&self, text: &str, allow_special: bool) -> Result<Vec<u32>, RuntimeError> {
        let mut out = Vec::new();
        let mut pending = String::new();
        let mut pos = 0;
        while pos < text.len() {
            let matched = if allow_special {
                self.specials
                    .iter()
                    .find_map(|(token, id)| text[pos..].starts_with(token).then_some((token, *id)))
            } else {
                None
            };
            if let Some((token, id)) = matched {
                self.encode_ordinary(&pending, &mut out)?;
                pending.clear();
                out.push(id);
                pos += token.len();
            } else {
                let ch = text[pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| RuntimeError::Tokenizer("invalid UTF-8 position".to_owned()))?;
                pending.push(ch);
                pos += ch.len_utf8();
            }
        }
        self.encode_ordinary(&pending, &mut out)?;
        Ok(out)
    }

    /// Encode plain text without interpreting special-token literals.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata cannot represent the input.
    pub fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) -> Result<(), RuntimeError> {
        if text.is_empty() {
            return Ok(());
        }
        match self.kind {
            TokenizerKind::Bpe => {
                for piece in gpt2_pretokenize(text) {
                    let mapped: String = piece
                        .as_bytes()
                        .iter()
                        .map(|&byte| self.byte_to_char[&byte])
                        .collect();
                    for token in bpe_merge(&mapped, &self.merges) {
                        let id = self.token_ids.get(&token).copied().ok_or_else(|| {
                            RuntimeError::Tokenizer(format!(
                                "BPE piece {token:?} is absent from the vocabulary"
                            ))
                        })?;
                        out.push(id);
                    }
                }
            }
            TokenizerKind::Spm => {
                for piece in spm_pretokenize(text) {
                    let target = if piece.chars().all(char::is_whitespace) {
                        piece.replace(' ', "▁")
                    } else {
                        format!("▁{piece}")
                    };
                    if let Some(id) = self.longest_token(&target) {
                        out.push(id);
                        continue;
                    }
                    if let Some(id) = self.longest_token(&piece) {
                        out.push(id);
                        continue;
                    }
                    self.encode_byte_fallback(piece.as_bytes(), out)?;
                }
            }
        }
        Ok(())
    }

    fn longest_token(&self, text: &str) -> Option<u32> {
        let mut end = text.len();
        let mut best: Option<(usize, f32, u32)> = None;
        while end > 0 {
            if !text.is_char_boundary(end) {
                end -= 1;
                continue;
            }
            if let Some(&id) = self.token_ids.get(&text[..end]) {
                let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                if best.is_none_or(|(len, best_score, _)| {
                    end > len || (end == len && score > best_score)
                }) {
                    best = Some((end, score, id));
                }
            }
            end -= 1;
        }
        best.map(|(_, _, id)| id)
    }

    fn encode_byte_fallback(&self, bytes: &[u8], out: &mut Vec<u32>) -> Result<(), RuntimeError> {
        for &byte in bytes {
            let token = format!("<0x{byte:02X}>");
            let id = *self.token_ids.get(&token).ok_or_else(|| {
                RuntimeError::Tokenizer(format!("byte fallback token {token} is absent"))
            })?;
            out.push(id);
        }
        Ok(())
    }

    /// Encode a chat prompt using the model's built-in template convention.
    ///
    /// # Errors
    ///
    /// Returns an error when a message cannot be tokenized.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, RuntimeError> {
        let mut text = String::new();
        match self.kind {
            TokenizerKind::Bpe => {
                if messages
                    .first()
                    .is_none_or(|message| message.role != "system")
                {
                    text.push_str("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n");
                }
                for message in messages {
                    text.push_str("<|im_start|>");
                    text.push_str(&message.role);
                    text.push('\n');
                    text.push_str(&message.content);
                    text.push_str("<|im_end|>\n");
                }
                text.push_str("<|im_start|>assistant\n");
            }
            TokenizerKind::Spm => {
                text.push_str("<bos>");
                let mut start = 0;
                if messages
                    .first()
                    .is_some_and(|message| message.role == "system")
                    && let Some(message) = messages.first()
                {
                    text.push_str("<start_of_turn>system\n");
                    text.push_str(message.content.trim());
                    text.push_str("<end_of_turn>\n");
                    start = 1;
                }
                for message in &messages[start..] {
                    let role = if message.role == "assistant" {
                        "model"
                    } else {
                        &message.role
                    };
                    text.push_str("<start_of_turn>");
                    text.push_str(role);
                    text.push('\n');
                    text.push_str(message.content.trim());
                    text.push_str("<end_of_turn>\n");
                }
                text.push_str("<start_of_turn>model\n");
            }
        }
        self.encode(&text, true)
    }
}

fn gpt2_byte_alphabet() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bytes = Vec::new();
    bytes.extend(b'!'..=b'~');
    bytes.extend(0xA1..=0xAC);
    bytes.extend(0xAE..=0xFF);
    let mut byte_to_char = HashMap::with_capacity(256);
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut next = 256u32;
    for &byte in &bytes {
        let ch = char::from(byte);
        byte_to_char.insert(byte, ch);
        char_to_byte.insert(ch, byte);
    }
    for byte in 0..=255u8 {
        if byte_to_char.contains_key(&byte) {
            continue;
        }
        let ch = char::from_u32(next).expect("byte alphabet stays in Unicode range");
        next += 1;
        byte_to_char.insert(byte, ch);
        char_to_byte.insert(ch, byte);
    }
    (byte_to_char, char_to_byte)
}

fn gpt2_pretokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        let remaining: String = chars[pos..].iter().collect();
        if let Some(contraction) = CONTRACTIONS
            .iter()
            .find(|value| remaining.starts_with(**value))
        {
            pieces.push((*contraction).to_owned());
            pos += contraction.chars().count();
            continue;
        }
        let mut end = pos + 1;
        if chars[pos].is_whitespace() {
            if pos + 1 < chars.len() && !chars[pos + 1].is_whitespace() {
                // GPT-2's ` ?` prefix belongs to the following word or punctuation run.
                let next = char_class(chars[pos + 1]);
                end = pos + 2;
                while end < chars.len()
                    && !chars[end].is_whitespace()
                    && char_class(chars[end]) == next
                {
                    end += 1;
                }
            } else {
                while end < chars.len() && chars[end].is_whitespace() {
                    end += 1;
                }
            }
        } else {
            let class = char_class(chars[pos]);
            while end < chars.len()
                && !chars[end].is_whitespace()
                && char_class(chars[end]) == class
            {
                end += 1;
            }
        }
        pieces.push(chars[pos..end].iter().collect());
        pos = end;
    }
    pieces
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceClass {
    Letter,
    Number,
    Other,
}

fn char_class(ch: char) -> PieceClass {
    if ch.is_alphabetic() {
        PieceClass::Letter
    } else if ch.is_numeric() {
        PieceClass::Number
    } else {
        PieceClass::Other
    }
}

fn spm_pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        let mut end = pos + 1;
        if chars[pos].is_whitespace() {
            while end < chars.len() && chars[end].is_whitespace() {
                end += 1;
            }
        } else if chars[pos].is_alphabetic() {
            while end < chars.len() && chars[end].is_alphabetic() {
                end += 1;
            }
        } else if chars[pos].is_numeric() {
            // SentencePiece tokenizes numeric runs one digit at a time.
        } else {
            while end < chars.len() && !chars[end].is_whitespace() && !chars[end].is_alphanumeric()
            {
                end += 1;
            }
        }
        pieces.push(chars[pos..end].iter().collect());
        pos = end;
    }
    pieces
}

fn bpe_merge(piece: &str, merges: &HashMap<(String, String), usize>) -> Vec<String> {
    let mut parts: Vec<String> = piece.chars().map(|ch| ch.to_string()).collect();
    loop {
        let mut best: Option<(usize, usize)> = None;
        for index in 0..parts.len().saturating_sub(1) {
            if let Some(&rank) = merges.get(&(parts[index].clone(), parts[index + 1].clone()))
                && best.is_none_or(|(best_rank, _)| rank < best_rank)
            {
                best = Some((rank, index));
            }
        }
        let Some((_, index)) = best else {
            break;
        };
        let right = parts.remove(index + 1);
        parts[index].push_str(&right);
    }
    parts
}

fn load_merges(header: &GgufHeader) -> Result<HashMap<(String, String), usize>, RuntimeError> {
    let mut merges = HashMap::new();
    let Some(values) = optional_strings(header, "tokenizer.ggml.merges")? else {
        return Ok(merges);
    };
    for (rank, value) in values.into_iter().enumerate() {
        let (left, right) = value
            .split_once(' ')
            .ok_or_else(|| RuntimeError::Tokenizer(format!("invalid BPE merge entry {value:?}")))?;
        merges.insert((left.to_owned(), right.to_owned()), rank);
    }
    Ok(merges)
}

fn byte_fallback(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

fn metadata_strings(header: &GgufHeader, key: &str) -> Result<Vec<String>, RuntimeError> {
    optional_strings(header, key)?.ok_or_else(|| RuntimeError::Metadata {
        key: key.to_owned(),
        reason: "missing metadata key".to_owned(),
    })
}

fn optional_strings(header: &GgufHeader, key: &str) -> Result<Option<Vec<String>>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Array(values)) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    MetadataValue::String(text) => out.push(text.clone()),
                    other => {
                        return Err(RuntimeError::Metadata {
                            key: key.to_owned(),
                            reason: format!("expected string array element, found {other:?}"),
                        });
                    }
                }
            }
            Ok(Some(out))
        }
        Some(other) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected string array, found {other:?}"),
        }),
    }
}

fn optional_floats(header: &GgufHeader, key: &str) -> Result<Option<Vec<f32>>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                MetadataValue::Float32(number) => Ok(*number),
                MetadataValue::Float64(number) => Ok(*number as f32),
                other => Err(RuntimeError::Metadata {
                    key: key.to_owned(),
                    reason: format!("expected float array element, found {other:?}"),
                }),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(other) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected float array, found {other:?}"),
        }),
    }
}

fn optional_u8s(header: &GgufHeader, key: &str) -> Result<Option<Vec<u8>>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                MetadataValue::Uint8(number) => Ok(*number),
                MetadataValue::Int8(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Uint16(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Int16(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Uint32(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Int32(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Uint64(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                MetadataValue::Int64(number) => Ok(u8::try_from(*number).unwrap_or(0)),
                other => Err(RuntimeError::Metadata {
                    key: key.to_owned(),
                    reason: format!("expected u8 array element, found {other:?}"),
                }),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(other) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected u8 array, found {other:?}"),
        }),
    }
}

fn optional_u32(header: &GgufHeader, key: &str) -> Result<Option<u32>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Uint32(value)) => Ok(Some(*value)),
        Some(MetadataValue::Uint64(value)) => {
            Some(u32::try_from(*value).map_err(|_| RuntimeError::Metadata {
                key: key.to_owned(),
                reason: format!("value {value} does not fit u32"),
            }))
            .transpose()
        }
        Some(other) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected u32, found {other:?}"),
        }),
    }
}

fn optional_bool(header: &GgufHeader, key: &str) -> Result<Option<bool>, RuntimeError> {
    match header.metadata.get(key) {
        None => Ok(None),
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(other) => Err(RuntimeError::Metadata {
            key: key.to_owned(),
            reason: format!("expected bool, found {other:?}"),
        }),
    }
}
