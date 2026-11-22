//! Deterministic grammar-constrained decoding over token *strings*.
//!
//! A [`Grammar`] is a small pushdown automaton: every state maps a token-string
//! matcher to a list of states that are pushed on the stack when the transition
//! is taken. A [`GrammarState`] owns the stack and refuses any token whose
//! string does not match exactly one outgoing transition of the current state.
//! Matching is deterministic: literal transitions are checked before type
//! classes, and two transitions matching the same token produce a typed
//! [`EngineError::GrammarAmbiguous`] instead of a silent choice.
//!
//! [`Grammar::json_tool_call`] implements this exact JSON subset for tool calls:
//!
//! * one top-level object with the members `name` then `arguments`, in that
//!   order and exactly once each;
//! * `name` is a JSON string (the tool/function name);
//! * `arguments` is a flat JSON object: every key is a JSON string and every
//!   value is a string, finite-width JSON number, `true`, `false` or `null`;
//! * keys are unique after JSON escape decoding; nested objects, arrays, extra
//!   members, comments, trailing tokens and malformed strings or numbers are rejected;
//! * the automaton accepts only after the final closing `}` has been consumed.
//!
//! Token strings are supplied by a [`TokenAlphabet`], so no tokenizer or new
//! dependency is needed. [`Grammar::json_tool_call_with`] pins the function name
//! and argument tokens so structural literals cannot leak into the call.
//! Ids are interpreted as logit indices: only ids in `0..logits.len()` can be
//! masked. [`sampler::Sampler::sample_with_grammar`]
//! masks disallowed logits to `-inf` and advances the state atomically;
//! [`generate_with_grammar`] repeats that step until acceptance or a budget is
//! exhausted.

use std::collections::{BTreeMap, BTreeSet};

use quarrel_core::{SamplingParams, Token};

use crate::{EngineError, sampler::Sampler, to_usize};

/// Maps token ids to the strings the grammar sees, if any.
///
/// Ids with no string (or with `None`) are treated as unusable by the grammar:
/// they are masked out of every sampling step. Later entries for the same id
/// replace earlier ones, and iteration is ordered by id so masking is
/// deterministic regardless of insertion order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenAlphabet {
    strings: BTreeMap<i32, Option<String>>,
}

impl TokenAlphabet {
    /// Creates an empty alphabet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an alphabet from `(token_id, optional string)` entries.
    #[must_use]
    pub fn from_entries(entries: impl IntoIterator<Item = (i32, Option<String>)>) -> Self {
        let mut alphabet = Self::new();
        for (id, text) in entries {
            alphabet.insert(id, text);
        }
        alphabet
    }

    /// Builds an alphabet from `(token_id, string)` entries.
    #[must_use]
    pub fn from_strings(entries: impl IntoIterator<Item = (i32, String)>) -> Self {
        Self::from_entries(entries.into_iter().map(|(id, text)| (id, Some(text))))
    }

    /// Inserts or replaces the string for `token_id`, returning the old value.
    pub fn insert(&mut self, token_id: i32, text: Option<String>) -> Option<Option<String>> {
        self.strings.insert(token_id, text)
    }

    /// Returns the token string for `token_id`, if it has one.
    #[must_use]
    pub fn get(&self, token_id: i32) -> Option<&str> {
        self.strings.get(&token_id).and_then(Option::as_deref)
    }

    /// Returns the token string for `token_id` or a typed grammar error.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::GrammarTokenString`] when the id is unknown or
    /// explicitly mapped to `None`.
    pub fn string_for(&self, token_id: i32) -> Result<&str, EngineError> {
        self.get(token_id)
            .ok_or(EngineError::GrammarTokenString { token_id })
    }

    /// Iterates over all `(token_id, optional string)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (i32, Option<&str>)> + '_ {
        self.strings.iter().map(|(&id, text)| (id, text.as_deref()))
    }

    /// Number of registered ids, including ids mapped to `None`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether no ids are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Decides which token strings may leave a grammar state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenMatch {
    /// The token string must equal this literal exactly.
    Literal(String),
    /// The token string must be a complete JSON string literal, quotes included.
    JsonString,
    /// The token string must equal one pre-encoded JSON string literal exactly.
    ExactJsonString(String),
    /// The token string must be a non-empty JSON string usable as an object key.
    JsonKey,
    /// The token string must equal one pre-encoded JSON object key exactly.
    ExactJsonKey(String),
    /// The token string must be one flat JSON value: string, number or literal.
    JsonFlatValue,
    /// The token string must equal one pre-encoded flat JSON value exactly.
    ExactJsonFlatValue(String),
}

impl TokenMatch {
    /// Creates a literal matcher.
    #[must_use]
    pub fn literal(text: impl Into<String>) -> Self {
        Self::Literal(text.into())
    }

    /// Creates a matcher for one already JSON-encoded string literal.
    #[must_use]
    pub fn exact_json_string(token: impl Into<String>) -> Self {
        Self::ExactJsonString(token.into())
    }

    /// Creates a matcher for one already JSON-encoded object key.
    #[must_use]
    pub fn exact_json_key(token: impl Into<String>) -> Self {
        Self::ExactJsonKey(token.into())
    }

    /// Creates a matcher for one already JSON-encoded flat value.
    #[must_use]
    pub fn exact_json_flat_value(token: impl Into<String>) -> Self {
        Self::ExactJsonFlatValue(token.into())
    }
}

/// One deterministic transition out of a grammar state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    /// Token strings accepted by this transition.
    pub matching: TokenMatch,
    /// States pushed in consumption order; an empty list reaches acceptance.
    pub push: Vec<String>,
}

impl Transition {
    /// Creates a transition that consumes `text` and pushes `states`.
    #[must_use]
    pub fn literal(
        text: impl Into<String>,
        states: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            matching: TokenMatch::literal(text),
            push: states.into_iter().map(Into::into).collect(),
        }
    }

    /// Creates a transition that consumes a token matching `matching`.
    #[must_use]
    pub fn matching(
        matching: TokenMatch,
        states: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            matching,
            push: states.into_iter().map(Into::into).collect(),
        }
    }
}

/// A deterministic pushdown grammar over token strings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Grammar {
    start: String,
    states: BTreeMap<String, Vec<Transition>>,
}

#[derive(Debug, Clone)]
enum ArgumentTokens {
    Any,
    Exact(Vec<String>),
}

impl ArgumentTokens {
    fn exact(tokens: impl IntoIterator<Item = String>) -> Self {
        let mut unique = Vec::new();
        for token in tokens {
            if !unique.contains(&token) {
                unique.push(token);
            }
        }
        Self::Exact(unique)
    }

    fn key_transitions(&self, target: &str) -> Vec<Transition> {
        match self {
            Self::Any => vec![Transition::matching(TokenMatch::JsonKey, [target])],
            Self::Exact(tokens) => tokens
                .iter()
                .map(|token| Transition::matching(TokenMatch::exact_json_key(token), [target]))
                .collect(),
        }
    }

    fn value_transitions(&self, target: &str) -> Vec<Transition> {
        match self {
            Self::Any => vec![Transition::matching(TokenMatch::JsonFlatValue, [target])],
            Self::Exact(tokens) => tokens
                .iter()
                .map(|token| {
                    Transition::matching(TokenMatch::exact_json_flat_value(token), [target])
                })
                .collect(),
        }
    }
}

impl Grammar {
    /// Builds a grammar from a start state and named states.
    ///
    /// Duplicate state names are resolved by keeping the last definition so
    /// construction never panics. Call [`Grammar::validate`] before using a
    /// hand-built grammar; [`Grammar::initial_state`] does this for you.
    #[must_use]
    pub fn new(
        start: impl Into<String>,
        states: impl IntoIterator<Item = (String, Vec<Transition>)>,
    ) -> Self {
        Self {
            start: start.into(),
            states: states.into_iter().collect(),
        }
    }

    /// Returns the built-in flat-arguments tool-call grammar.
    ///
    /// The accepted subset is documented at the top of this module. Token
    /// strings are the JSON fragments `"{"`, `"\"name\""`, `":"`, any JSON
    /// string as the function name, `","`, `"\"arguments\""`, argument keys and
    /// values, and `"}"`. Use [`Grammar::json_tool_call_with`] when semantic
    /// tokens must be pinned to a specific tool definition.
    #[must_use]
    pub fn json_tool_call() -> Self {
        Self::tool_call(
            TokenMatch::JsonString,
            ArgumentTokens::Any,
            ArgumentTokens::Any,
        )
    }

    /// Returns the tool-call grammar with the function name pinned to `name_token`.
    ///
    /// `name_token` must be a complete JSON string literal such as
    /// `"\"calculate\""`. Only that exact token is accepted after `"name":`,
    /// which stops other string tokens in the alphabet from being emitted as
    /// the function name. Argument keys and values remain generic.
    #[must_use]
    pub fn json_tool_call_for(name_token: &str) -> Self {
        Self::tool_call(
            TokenMatch::exact_json_string(name_token),
            ArgumentTokens::Any,
            ArgumentTokens::Any,
        )
    }

    /// Returns the tool-call grammar pinned to one function and its flat arguments.
    ///
    /// `name_token`, `key_tokens` and `value_tokens` must be already JSON-encoded
    /// tokens. Only those tokens may be emitted at their respective positions,
    /// so structural literals such as `"arguments"` cannot leak into the
    /// generated argument object. Duplicate tokens are ignored.
    #[must_use]
    pub fn json_tool_call_with(
        name_token: &str,
        key_tokens: &[String],
        value_tokens: &[String],
    ) -> Self {
        Self::tool_call(
            TokenMatch::exact_json_string(name_token),
            ArgumentTokens::exact(key_tokens.iter().cloned()),
            ArgumentTokens::exact(value_tokens.iter().cloned()),
        )
    }

    fn tool_call(name_match: TokenMatch, keys: ArgumentTokens, values: ArgumentTokens) -> Self {
        const ROOT: &str = "root";
        const NAME_KEY: &str = "name_key";
        const NAME_COLON: &str = "name_colon";
        const NAME_VALUE: &str = "name_value";
        const COMMA: &str = "comma";
        const ARGS_KEY: &str = "arguments_key";
        const ARGS_COLON: &str = "arguments_colon";
        const ARGS_OPEN: &str = "arguments_open";
        const ARGS_FIRST: &str = "arguments_first";
        const ARG_KEY: &str = "argument_key";
        const ARG_COLON: &str = "argument_colon";
        const ARG_VALUE: &str = "argument_value";
        const ARG_NEXT: &str = "argument_next";
        const ROOT_BRACE: &str = "root_brace";

        let state =
            |name: &'static str, transitions: Vec<Transition>| (name.to_owned(), transitions);
        let mut first = vec![Transition::literal("}", [ROOT_BRACE])];
        first.extend(keys.key_transitions(ARG_COLON));
        Self::new(
            ROOT,
            [
                state(ROOT, vec![Transition::literal("{", [NAME_KEY])]),
                state(
                    NAME_KEY,
                    vec![Transition::literal(r#""name""#, [NAME_COLON])],
                ),
                state(NAME_COLON, vec![Transition::literal(":", [NAME_VALUE])]),
                state(NAME_VALUE, vec![Transition::matching(name_match, [COMMA])]),
                state(COMMA, vec![Transition::literal(",", [ARGS_KEY])]),
                state(
                    ARGS_KEY,
                    vec![Transition::literal(r#""arguments""#, [ARGS_COLON])],
                ),
                state(ARGS_COLON, vec![Transition::literal(":", [ARGS_OPEN])]),
                state(ARGS_OPEN, vec![Transition::literal("{", [ARGS_FIRST])]),
                state(ARGS_FIRST, first),
                state(ARG_KEY, keys.key_transitions(ARG_COLON)),
                state(ARG_COLON, vec![Transition::literal(":", [ARG_VALUE])]),
                state(ARG_VALUE, values.value_transitions(ARG_NEXT)),
                state(
                    ARG_NEXT,
                    vec![
                        Transition::literal(",", [ARG_KEY]),
                        Transition::literal("}", [ROOT_BRACE]),
                    ],
                ),
                state(
                    ROOT_BRACE,
                    vec![Transition::literal("}", std::iter::empty::<String>())],
                ),
            ],
        )
    }

    /// Name of the start state.
    #[must_use]
    pub fn start(&self) -> &str {
        &self.start
    }

    /// Names of all declared states in lexical order.
    pub fn state_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.states.keys().map(String::as_str)
    }

    /// Checks that the start state and every pushed state are declared.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::GrammarInvalidState`] for a missing start or
    /// target state.
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.states.contains_key(&self.start) {
            return Err(EngineError::GrammarInvalidState {
                state: self.start.clone(),
            });
        }
        for transitions in self.states.values() {
            for transition in transitions {
                for target in &transition.push {
                    if !self.states.contains_key(target) {
                        return Err(EngineError::GrammarInvalidState {
                            state: target.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates an initial state, validating the grammar first.
    ///
    /// # Errors
    ///
    /// Propagates grammar validation errors.
    pub fn initial_state(&self) -> Result<GrammarState<'_>, EngineError> {
        GrammarState::new(self)
    }
}

/// Mutable decoding position in a [`Grammar`].
#[derive(Debug, Clone)]
pub struct GrammarState<'a> {
    grammar: &'a Grammar,
    stack: Vec<String>,
    consumed: Vec<String>,
    keys: BTreeSet<String>,
}

impl<'a> GrammarState<'a> {
    /// Creates a state positioned at the grammar's start symbol.
    ///
    /// # Errors
    ///
    /// Propagates grammar validation errors from [`Grammar::validate`].
    pub fn new(grammar: &'a Grammar) -> Result<Self, EngineError> {
        grammar.validate()?;
        Ok(Self {
            grammar,
            stack: vec![grammar.start.clone()],
            consumed: Vec::new(),
            keys: BTreeSet::new(),
        })
    }

    /// Whether the complete grammar has been consumed.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        self.stack.is_empty()
    }

    /// Current state name, or `None` when the grammar is accepting.
    #[must_use]
    pub fn current_state(&self) -> Option<&str> {
        self.stack.last().map(String::as_str)
    }

    /// Token strings consumed so far, in order.
    #[must_use]
    pub fn consumed_tokens(&self) -> &[String] {
        &self.consumed
    }

    /// Number of states still on the stack.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.stack.len()
    }

    /// Consumes `token_id` if its string is the unique match for this state.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::GrammarTokenString`] for a missing token string,
    /// [`EngineError::GrammarAlreadyAccepting`] after completion,
    /// [`EngineError::GrammarDuplicateKey`] for a repeated object key,
    /// [`EngineError::GrammarRejected`] when no transition matches, and
    /// [`EngineError::GrammarAmbiguous`] when more than one transition matches.
    pub fn advance(&mut self, alphabet: &TokenAlphabet, token_id: i32) -> Result<(), EngineError> {
        let token = alphabet.string_for(token_id)?;
        let state = self
            .stack
            .last()
            .ok_or(EngineError::GrammarAlreadyAccepting {
                token: token.to_owned(),
            })?
            .clone();
        let transitions = self
            .grammar
            .states
            .get(&state)
            .map_or(&[][..], Vec::as_slice);
        let mut selected = None;
        for transition in transitions {
            if !matches_shape(transition, token) {
                continue;
            }
            if is_key_matcher(&transition.matching) && self.keys.contains(&json_key_body(token)) {
                return Err(EngineError::GrammarDuplicateKey {
                    key: json_key_body(token),
                });
            }
            if selected.is_some() {
                return Err(EngineError::GrammarAmbiguous {
                    state,
                    token: token.to_owned(),
                });
            }
            selected = Some(transition);
        }
        let Some(transition) = selected else {
            return Err(EngineError::GrammarRejected {
                state,
                token: token.to_owned(),
            });
        };
        if is_key_matcher(&transition.matching) {
            self.keys.insert(json_key_body(token));
        }
        self.stack.pop();
        for target in transition.push.iter().rev() {
            self.stack.push(target.clone());
        }
        self.consumed.push(token.to_owned());
        Ok(())
    }

    /// Token ids in `0..logits_len` that this state would currently accept.
    ///
    /// Ids are returned in ascending order. Ids mapped to `None` or absent from
    /// the alphabet are never included.
    #[must_use]
    pub fn allowed_token_ids(&self, alphabet: &TokenAlphabet, logits_len: usize) -> Vec<i32> {
        let Some(state) = self.stack.last() else {
            return Vec::new();
        };
        let Some(transitions) = self.grammar.states.get(state) else {
            return Vec::new();
        };
        let mut ids = Vec::new();
        for (id, text) in alphabet.iter() {
            if id < 0 || id as usize >= logits_len {
                continue;
            }
            if let Some(text) = text
                && transitions
                    .iter()
                    .any(|transition| self.matches(transition, text))
            {
                ids.push(id);
            }
        }
        ids
    }

    /// Boolean mask with `true` at every currently allowed logit index.
    #[must_use]
    pub fn allowed_mask(&self, alphabet: &TokenAlphabet, logits_len: usize) -> Vec<bool> {
        let mut mask = vec![false; logits_len];
        for id in self.allowed_token_ids(alphabet, logits_len) {
            if let Ok(index) = usize::try_from(id)
                && let Some(slot) = mask.get_mut(index)
            {
                *slot = true;
            }
        }
        mask
    }

    /// Rewrites every disallowed logit to `-inf` in place.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::GrammarNoAllowedTokens`] if the alphabet offers
    /// no acceptable id for the current state. Logits are left unchanged in
    /// that case.
    pub fn mask_logits(
        &self,
        alphabet: &TokenAlphabet,
        logits: &mut [f32],
    ) -> Result<(), EngineError> {
        let mask = self.allowed_mask(alphabet, logits.len());
        if !mask.iter().any(|&allowed| allowed) {
            return Err(EngineError::GrammarNoAllowedTokens {
                state: self.current_state().unwrap_or("<accepting>").to_owned(),
            });
        }
        for (score, &allowed) in logits.iter_mut().zip(&mask) {
            if !allowed {
                *score = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn matches(&self, transition: &Transition, token: &str) -> bool {
        if token == "," && self.current_state() == Some("argument_next") && !self.has_unused_key() {
            return false;
        }
        matches_shape(transition, token)
            && (!is_key_matcher(&transition.matching) || !self.keys.contains(&json_key_body(token)))
    }

    /// Guards the built-in grammar against emitting a comma after every declared key was used.
    fn has_unused_key(&self) -> bool {
        self.grammar
            .states
            .get("argument_key")
            .is_some_and(|transitions| {
                transitions
                    .iter()
                    .any(|transition| match &transition.matching {
                        TokenMatch::ExactJsonKey(expected) => {
                            !self.keys.contains(&json_key_body(expected))
                        }
                        TokenMatch::JsonKey => true,
                        _ => false,
                    })
            })
    }
}

/// Tokens, strings and acceptance produced by [`generate_with_grammar`].
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarOutput {
    /// Sampled tokens with their original log-probabilities.
    pub tokens: Vec<Token>,
    /// Token strings in the same order as `tokens`.
    pub token_strings: Vec<String>,
    /// Concatenation of `token_strings`.
    pub text: String,
    /// Whether the grammar reached an accepting state before the budget ran out.
    pub accepting: bool,
}

/// Drives grammar-constrained generation from a logit supplier.
///
/// `logits_for` is called with the token context built so far; the caller can
/// use a real backend, a fixed vector or a test double. Generation stops at
/// acceptance or when the smaller of `max_tokens` and `params.max_tokens` is
/// reached. Greedy parameters make the result deterministic; stochastic
/// parameters still only ever choose grammar-allowed token strings.
///
/// # Errors
///
/// Propagates parameter validation, logit supplier and sampling errors,
/// including [`EngineError::GrammarNoAllowedTokens`] when the alphabet cannot
/// satisfy the current state.
pub fn generate_with_grammar<F>(
    mut logits_for: F,
    params: &SamplingParams,
    seed: u64,
    max_tokens: usize,
    alphabet: &TokenAlphabet,
    grammar: &Grammar,
) -> Result<GrammarOutput, EngineError>
where
    F: FnMut(&[i32]) -> Result<Vec<f32>, EngineError>,
{
    params.validate().map_err(|_| EngineError::InvalidParams {
        reason: "quarrel-core rejected the sampling parameters",
    })?;
    let limit = max_tokens.min(to_usize(params.max_tokens)?);
    let mut state = grammar.initial_state()?;
    let mut sampler = Sampler::new(seed);
    let mut context = Vec::new();
    let mut output = GrammarOutput {
        tokens: Vec::with_capacity(limit),
        token_strings: Vec::with_capacity(limit),
        text: String::new(),
        accepting: false,
    };

    for _ in 0..limit {
        if state.is_accepting() {
            break;
        }
        let logits = logits_for(&context)?;
        let (token, next) =
            sampler.sample_with_grammar(&logits, params, &context, alphabet, &state)?;
        let text = alphabet.string_for(token.id)?.to_owned();
        context.push(token.id);
        output.text.push_str(&text);
        output.token_strings.push(text);
        output.tokens.push(token);
        state = next;
    }
    output.accepting = state.is_accepting();
    Ok(output)
}

fn matches_shape(transition: &Transition, token: &str) -> bool {
    match &transition.matching {
        TokenMatch::Literal(literal) => token == literal,
        TokenMatch::JsonString => is_json_string(token),
        TokenMatch::ExactJsonString(expected) => token == expected,
        TokenMatch::JsonKey => is_json_string(token),
        TokenMatch::ExactJsonKey(expected) => token == expected,
        TokenMatch::JsonFlatValue => is_flat_json_value(token),
        TokenMatch::ExactJsonFlatValue(expected) => token == expected,
    }
}

fn is_key_matcher(matching: &TokenMatch) -> bool {
    matches!(matching, TokenMatch::JsonKey | TokenMatch::ExactJsonKey(_))
}

fn is_json_string(token: &str) -> bool {
    let bytes = token.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'"' || bytes[bytes.len() - 1] != b'"' {
        return false;
    }
    let mut index = 1;
    let end = bytes.len() - 1;
    while index < end {
        let byte = bytes[index];
        match byte {
            b'"' | 0..=0x1f => return false,
            b'\\' => {
                index += 1;
                let Some(&escaped) = bytes.get(index) else {
                    return false;
                };
                match escaped {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => index += 1,
                    b'u' => {
                        if index + 4 >= bytes.len() {
                            return false;
                        }
                        for &hex in &bytes[index + 1..=index + 4] {
                            if !hex.is_ascii_hexdigit() {
                                return false;
                            }
                        }
                        index += 5;
                    }
                    _ => return false,
                }
            }
            _ => index += 1,
        }
    }
    true
}

/// Canonical spelling of a JSON string's body, used for duplicate-key checks.
///
/// Escapes are decoded so `"a"` and `"\u0061"` count as the same key. Lone or
/// malformed surrogates map to the replacement character; they are already
/// rejected by [`is_json_string`] unless encoded as `\u` escapes.
fn json_key_body(token: &str) -> String {
    if !is_json_string(token) || token.len() < 2 {
        return token.to_owned();
    }
    let mut chars = token[1..token.len() - 1].chars().peekable();
    let mut decoded = String::new();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }
        let Some(escaped) = chars.next() else {
            return token[1..token.len() - 1].to_owned();
        };
        match escaped {
            '"' => decoded.push('"'),
            '\\' => decoded.push('\\'),
            '/' => decoded.push('/'),
            'b' => decoded.push('\u{8}'),
            'f' => decoded.push('\u{c}'),
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            'u' => decoded.push(decode_unicode_escape(&mut chars)),
            _ => return token[1..token.len() - 1].to_owned(),
        }
    }
    decoded
}

fn decode_unicode_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> char {
    const REPLACEMENT: char = '\u{fffd}';
    let Some(high) = take_hex4(chars) else {
        return REPLACEMENT;
    };
    let scalar = if (0xd800..=0xdbff).contains(&high) {
        let low = if chars.next() == Some('\\') && chars.next() == Some('u') {
            take_hex4(chars)
        } else {
            None
        };
        match low {
            Some(low) if (0xdc00..=0xdfff).contains(&low) => {
                0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
            }
            _ => return REPLACEMENT,
        }
    } else if (0xdc00..=0xdfff).contains(&high) {
        return REPLACEMENT;
    } else {
        high
    };
    char::from_u32(scalar).unwrap_or(REPLACEMENT)
}

fn take_hex4(chars: &mut impl Iterator<Item = char>) -> Option<u32> {
    let mut value = 0_u32;
    for _ in 0..4 {
        value = value * 16 + chars.next()?.to_digit(16)?;
    }
    Some(value)
}

fn is_flat_json_value(token: &str) -> bool {
    is_json_string(token) || is_json_number(token) || matches!(token, "true" | "false" | "null")
}

fn is_json_number(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut index = 0;
    if bytes.first() == Some(&b'-') {
        index += 1;
    }
    if index >= bytes.len() {
        return false;
    }
    if bytes[index] == b'0' {
        index += 1;
    } else if bytes[index].is_ascii_digit() {
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    } else {
        return false;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == start {
            return false;
        }
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == start {
            return false;
        }
    }
    index == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::{
        Grammar, GrammarOutput, GrammarState, TokenAlphabet, TokenMatch, Transition,
        generate_with_grammar,
    };
    use crate::EngineError;
    use crate::sampler::Sampler;
    use proptest::prelude::*;
    use quarrel_core::SamplingParams;

    const KEY_NAME: &str = r#""name""#;
    const KEY_ARGS: &str = r#""arguments""#;

    fn greedy() -> SamplingParams {
        let mut params = SamplingParams::new();
        params.temperature = 0.0;
        params.max_tokens = 32;
        params
    }

    fn stochastic() -> SamplingParams {
        let mut params = greedy();
        params.temperature = 0.8;
        params
    }

    fn alphabet(extra: &[&str]) -> TokenAlphabet {
        let base = [
            "{",
            "}",
            KEY_NAME,
            KEY_ARGS,
            ":",
            ",",
            r#""calc""#,
            r#""other""#,
        ];
        let strings = base
            .into_iter()
            .chain(extra.iter().copied())
            .enumerate()
            .map(|(index, text)| (index as i32, text.to_owned()));
        TokenAlphabet::from_strings(strings)
    }

    fn id_of(map: &TokenAlphabet, text: &str) -> i32 {
        map.iter()
            .find_map(|(id, candidate)| (candidate == Some(text)).then_some(id))
            .unwrap_or_else(|| panic!("alphabet has no token for {text}"))
    }

    fn feed(state: &mut GrammarState<'_>, map: &TokenAlphabet, strings: &[&str]) {
        for text in strings {
            state
                .advance(map, id_of(map, text))
                .unwrap_or_else(|error| panic!("advance {text}: {error}"));
        }
    }

    #[test]
    fn valid_tool_call_strings_are_accepted() {
        let map = alphabet(&[
            r#""a""#,
            r#""flag""#,
            r#""n""#,
            r#""nothing""#,
            r#""hello""#,
            "-12.5e2",
            "true",
            "null",
        ]);
        let grammar = Grammar::json_tool_call();
        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
                r#""hello""#,
                ",",
                r#""flag""#,
                ":",
                "true",
                ",",
                r#""nothing""#,
                ":",
                "null",
                ",",
                r#""n""#,
                ":",
                "-12.5e2",
                "}",
                "}",
            ],
        );
        assert!(state.is_accepting());
        assert_eq!(state.remaining(), 0);
        assert_eq!(state.consumed_tokens().len(), 25);
        assert!(state.allowed_token_ids(&map, map.len()).is_empty());
        assert!(
            state
                .advance(&map, id_of(&map, "}"))
                .is_err_and(|error| matches!(error, EngineError::GrammarAlreadyAccepting { .. }))
        );
    }

    #[test]
    fn empty_arguments_are_accepted() {
        let map = alphabet(&[]);
        let grammar = Grammar::json_tool_call();
        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                "}",
                "}",
            ],
        );
        assert!(state.is_accepting());
    }

    #[test]
    fn pinned_name_accepts_only_the_requested_function_string() {
        let map = alphabet(&[]);
        let grammar = Grammar::json_tool_call_for(r#""calc""#);
        let mut state = grammar.initial_state().unwrap();
        feed(&mut state, &map, &["{", KEY_NAME, ":"]);
        assert_eq!(state.allowed_token_ids(&map, map.len()), vec![6]);
        assert!(matches!(
            state.advance(&map, id_of(&map, r#""other""#)),
            Err(EngineError::GrammarRejected { .. })
        ));
        feed(&mut state, &map, &[r#""calc""#]);
        assert_eq!(state.current_state(), Some("comma"));
    }

    #[test]
    fn pinned_argument_tokens_reject_structural_literals() {
        let map = TokenAlphabet::from_strings(
            [
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                r#""a""#,
                "1",
                "}",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, text)| (id as i32, text.to_owned())),
        );
        let grammar =
            Grammar::json_tool_call_with(r#""calc""#, &[r#""a""#.to_owned()], &["1".to_owned()]);
        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &["{", KEY_NAME, ":", r#""calc""#, ",", KEY_ARGS, ":", "{"],
        );
        let allowed = state.allowed_token_ids(&map, map.len());
        assert!(allowed.contains(&id_of(&map, r#""a""#)));
        assert!(!allowed.contains(&id_of(&map, KEY_ARGS)));
        assert!(matches!(
            state.advance(&map, id_of(&map, KEY_ARGS)),
            Err(EngineError::GrammarRejected { .. })
        ));
        feed(&mut state, &map, &[r#""a""#, ":", "1", "}", "}"]);
        assert!(state.is_accepting());
    }

    #[test]
    fn grammar_sampling_validates_original_logits_and_keeps_state_on_error() {
        let map = alphabet(&[]);
        let grammar = Grammar::json_tool_call_for(r#""calc""#);
        let state = grammar.initial_state().unwrap();
        let mut sampler = Sampler::new(7);
        assert!(matches!(
            sampler.sample_with_grammar(&[f32::NAN, 1.0], &greedy(), &[], &map, &state),
            Err(EngineError::InvalidLogits { .. })
        ));
        assert_eq!(state.current_state(), Some("root"));
    }

    #[test]
    fn invalid_tool_call_strings_are_rejected_with_typed_errors() {
        let map = alphabet(&[r#""a""#, r#""n""#, "1", "[", "]", "[]", "01", "calc"]);
        let grammar = Grammar::json_tool_call();

        let cases: &[&[&str]] = &[
            &[KEY_NAME],                        // missing opening brace
            &["{", KEY_ARGS],                   // wrong first member
            &["{", KEY_NAME, ":", "calc"],      // name is not a JSON string
            &["{", KEY_NAME, ":", r#""calc""#], // arguments missing
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "[",
                "]",
                "}",
                "}",
            ],
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
                "{",
                "}",
                "}",
                "}",
            ],
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
                "01",
                "}",
                "}",
            ],
        ];
        for tokens in cases {
            let mut state = grammar.initial_state().unwrap();
            let mut error = None;
            for text in *tokens {
                if let Err(reason) = state.advance(&map, id_of(&map, text)) {
                    error = Some(reason);
                    break;
                }
            }
            assert!(
                !state.is_accepting()
                    && error
                        .is_none_or(|reason| matches!(reason, EngineError::GrammarRejected { .. })),
                "expected rejection for {tokens:?}, got accepting={}",
                state.is_accepting()
            );
        }

        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
            ],
        );
        assert!(matches!(
            state.advance(&map, id_of(&map, "[")),
            Err(EngineError::GrammarRejected { .. })
        ));

        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
                "1",
                ",",
            ],
        );
        assert!(matches!(
            state.advance(&map, id_of(&map, r#""a""#)),
            Err(EngineError::GrammarDuplicateKey { .. })
        ));
        assert!(matches!(
            state.advance(&map, 999),
            Err(EngineError::GrammarTokenString { token_id: 999 })
        ));
    }

    #[test]
    fn escaped_duplicate_keys_are_rejected() {
        let map = TokenAlphabet::from_strings(
            [
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                r#""a""#,
                r#""\u0061""#,
                "1",
                "}",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, text)| (id as i32, text.to_owned())),
        );
        let grammar = Grammar::json_tool_call();
        let mut state = grammar.initial_state().unwrap();
        feed(
            &mut state,
            &map,
            &[
                "{",
                KEY_NAME,
                ":",
                r#""calc""#,
                ",",
                KEY_ARGS,
                ":",
                "{",
                r#""a""#,
                ":",
                "1",
                ",",
            ],
        );
        assert!(matches!(
            state.advance(&map, id_of(&map, r#""\u0061""#)),
            Err(EngineError::GrammarDuplicateKey { .. })
        ));
    }

    #[test]
    fn masking_is_deterministic_and_only_allows_live_tokens() {
        let map = alphabet(&[r#""a""#, "1"]);
        let grammar = Grammar::json_tool_call();
        let state = grammar.initial_state().unwrap();
        let ids = state.allowed_token_ids(&map, map.len());
        assert_eq!(ids, vec![id_of(&map, "{")]);

        let mut logits = vec![0.0_f32; map.len()];
        state.mask_logits(&map, &mut logits).unwrap();
        for (index, score) in logits.iter().enumerate() {
            if index == id_of(&map, "{") as usize {
                assert_eq!(*score, 0.0);
            } else {
                assert_eq!(*score, f32::NEG_INFINITY);
            }
        }
        assert_eq!(state.allowed_mask(&map, 2), vec![true, false]);
        let sparse = TokenAlphabet::from_strings([(999, "{".to_owned())]);
        assert_eq!(
            state.allowed_mask(&sparse, 2),
            vec![false, false],
            "ids beyond the logit slice are never allowed"
        );
        assert!(matches!(
            state.mask_logits(&sparse, &mut [0.0, 1.0]),
            Err(EngineError::GrammarNoAllowedTokens { .. })
        ));
    }

    #[test]
    fn greedy_and_sampled_paths_stay_inside_the_grammar() {
        let map = alphabet(&[r#""a""#, r#""text""#, r#""hello""#, "7", "false"]);
        let grammar = Grammar::json_tool_call();
        for (params, seed) in [(greedy(), 1), (stochastic(), 19)] {
            for repetition in 0..3 {
                let output: GrammarOutput = generate_with_grammar(
                    |_| Ok(vec![0.25; map.len()]),
                    &params,
                    seed + repetition,
                    32,
                    &map,
                    &grammar,
                )
                .unwrap();
                assert!(output.accepting, "output: {}", output.text);
                assert_eq!(output.token_strings.len(), output.tokens.len());
                assert_eq!(output.text, output.token_strings.concat());
                assert!(
                    output.text.starts_with(r#"{"name":""#)
                        && output.text.contains(r#","arguments":{"#),
                    "output: {}",
                    output.text
                );
                assert!(output.text.ends_with("}}"));
                assert!(output.tokens.iter().all(|token| token.logprob.is_finite()));
            }
        }
    }

    #[test]
    fn custom_grammar_can_be_validated() {
        let broken = Grammar::new(
            "start",
            [(
                "start".to_owned(),
                vec![Transition::literal("a", ["missing"])],
            )],
        );
        assert!(matches!(
            broken.validate(),
            Err(EngineError::GrammarInvalidState { .. })
        ));
        assert!(matches!(
            Grammar::new("start", std::iter::empty()).initial_state(),
            Err(EngineError::GrammarInvalidState { .. })
        ));

        let ambiguous = Grammar::new(
            "start",
            [
                (
                    "start".to_owned(),
                    vec![
                        Transition::literal(r#""s""#, ["done"]),
                        Transition::matching(TokenMatch::JsonString, ["done"]),
                    ],
                ),
                ("done".to_owned(), vec![]),
            ],
        );
        let map = TokenAlphabet::from_strings([(0, r#""s""#.to_owned())]);
        let mut state = ambiguous.initial_state().unwrap();
        assert!(matches!(
            state.advance(&map, 0),
            Err(EngineError::GrammarAmbiguous { .. })
        ));
    }

    proptest! {
        #[test]
        fn arbitrary_finite_logits_and_alphabets_never_panic(
            entries in prop::collection::vec((any::<i32>(), prop::option::of(any::<String>())), 0..12),
            mut logits in proptest::collection::vec(-30.0f32..30.0, 0..24),
            seed in any::<u64>(),
            steps in 0usize..12,
        ) {
            let alphabet = TokenAlphabet::from_entries(entries);
            let grammar = Grammar::json_tool_call();
            let mut state = grammar.initial_state().expect("built-in grammar is valid");
            let mut params = greedy();
            params.max_tokens = 32;
            let mut sampler = Sampler::new(seed);
            for _ in 0..steps {
                if state.is_accepting() {
                    break;
                }
                if state.mask_logits(&alphabet, &mut logits).is_err() {
                    break;
                }
                match sampler.sample_with_grammar(&logits, &params, &[0, -1, 99], &alphabet, &state) {
                    Ok((token, next)) => {
                        let _ = alphabet.string_for(token.id);
                        state = next;
                    }
                    Err(_) => break,
                }
            }
            let _ = generate_with_grammar(
                |_| Ok(logits.clone()),
                &params,
                seed,
                4,
                &alphabet,
                &grammar,
            );
        }
    }
}
