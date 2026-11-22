//! A small, dependency-free Jinja subset for turning chat turns into prompt text.
//!
//! Supported syntax:
//!
//! * `{{ variable }}` — inserts a value from the context (or the current loop scope).
//! * `{% for message in messages %}` / `{% endfor %}` — repeats a block for every message.
//! * `{% if variable %}...{% endif %}` and `{% if not variable %}...{% endif %}`.
//! * `{% set name = value %}` — binds a name to a quoted literal or another variable.
//!
//! Inside a `for` block the loop variable exposes `.role` and `.content` (for example
//! `{{ message.role }}`). Undefined variables render as the empty string, and an `if` on an
//! undefined or empty variable is false. Whitespace around tag contents is ignored; text outside
//! tags is preserved verbatim. Anything else is reported as [`CoreError::Template`].
//!
//! [`fallback`] is available when a model ships without a usable template.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::CoreError;

/// One turn in a chat conversation.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Speaker role, typically `system`, `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Text(String),
    Expression(String),
    Tag(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    Expression,
    Tag,
}

impl Delimiter {
    fn open(self) -> &'static str {
        match self {
            Self::Expression => "{{",
            Self::Tag => "{%",
        }
    }

    fn close(self) -> &'static str {
        match self {
            Self::Expression => "}}",
            Self::Tag => "%}",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Expression => "expression",
            Self::Tag => "tag",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Text(String),
    Expression(String),
    For {
        variable: String,
        body: Vec<Node>,
    },
    If {
        condition: String,
        negated: bool,
        body: Vec<Node>,
    },
    Set {
        name: String,
        value: String,
    },
}

fn template_error(message: impl Into<String>) -> CoreError {
    CoreError::Template(message.into())
}

fn slice(template: &str, start: usize, end: usize) -> Result<&str, CoreError> {
    template
        .get(start..end)
        .ok_or_else(|| template_error("template contains invalid byte offsets"))
}

fn tokenize(template: &str) -> Result<Vec<Token>, CoreError> {
    let mut tokens = Vec::new();
    let mut cursor = 0;

    while cursor < template.len() {
        let rest = slice(template, cursor, template.len())?;
        let expression_at = rest.find(Delimiter::Expression.open());
        let tag_at = rest.find(Delimiter::Tag.open());
        let Some((offset, delimiter)) = (match (expression_at, tag_at) {
            (Some(expression), Some(tag)) => {
                if expression <= tag {
                    Some((expression, Delimiter::Expression))
                } else {
                    Some((tag, Delimiter::Tag))
                }
            }
            (Some(expression), None) => Some((expression, Delimiter::Expression)),
            (None, Some(tag)) => Some((tag, Delimiter::Tag)),
            (None, None) => None,
        }) else {
            push_text(&mut tokens, rest);
            break;
        };

        let start = cursor + offset;
        push_text(&mut tokens, slice(template, cursor, start)?);

        let after_open = start + delimiter.open().len();
        let close = delimiter.close();
        let after_open_text = slice(template, after_open, template.len())?;
        let close_offset = after_open_text.find(close).ok_or_else(|| {
            template_error(format!(
                "unterminated {} delimiter (`{}`)",
                delimiter.label(),
                delimiter.open()
            ))
        })?;
        let inner_end = after_open + close_offset;
        let inner = slice(template, after_open, inner_end)?.trim().to_owned();
        let token = match delimiter {
            Delimiter::Expression => Token::Expression(inner),
            Delimiter::Tag => Token::Tag(inner),
        };
        tokens.push(token);
        cursor = inner_end + close.len();
    }

    Ok(tokens)
}

fn push_text(tokens: &mut Vec<Token>, text: &str) {
    if !text.is_empty() {
        tokens.push(Token::Text(text.to_owned()));
    }
}

fn take_part<'a>(
    parts: &mut impl Iterator<Item = &'a str>,
    what: &str,
) -> Result<&'a str, CoreError> {
    parts
        .next()
        .ok_or_else(|| template_error(format!("missing {what} in tag")))
}

fn ensure_end<'a>(parts: &mut impl Iterator<Item = &'a str>, tag: &str) -> Result<(), CoreError> {
    if parts.next().is_some() {
        return Err(template_error(format!("unexpected tokens in {tag}")));
    }
    Ok(())
}

fn parse_nodes(
    tokens: &[Token],
    cursor: &mut usize,
    stop: Option<&str>,
) -> Result<Vec<Node>, CoreError> {
    let mut nodes = Vec::new();

    while let Some(token) = tokens.get(*cursor) {
        *cursor += 1;
        match token {
            Token::Text(text) => {
                if !text.is_empty() {
                    nodes.push(Node::Text(text.clone()));
                }
            }
            Token::Expression(expression) => nodes.push(Node::Expression(expression.clone())),
            Token::Tag(tag) => {
                let mut parts = tag.split_whitespace();
                let Some(keyword) = parts.next() else {
                    return Err(template_error("empty tag"));
                };
                match keyword {
                    "for" => {
                        let variable = take_part(&mut parts, "loop variable")?.to_owned();
                        let preposition = take_part(&mut parts, "`in`")?;
                        let collection = take_part(&mut parts, "collection")?;
                        ensure_end(&mut parts, "`for` tag")?;
                        if preposition != "in" || collection != "messages" {
                            return Err(template_error(
                                "only `for <name> in messages` is supported",
                            ));
                        }
                        let body = parse_nodes(tokens, cursor, Some("endfor"))?;
                        nodes.push(Node::For { variable, body });
                    }
                    "if" => {
                        let first = take_part(&mut parts, "condition")?;
                        let (negated, condition) = if first == "not" {
                            (true, take_part(&mut parts, "condition after `not`")?)
                        } else {
                            (false, first)
                        };
                        ensure_end(&mut parts, "`if` tag")?;
                        let body = parse_nodes(tokens, cursor, Some("endif"))?;
                        nodes.push(Node::If {
                            condition: condition.to_owned(),
                            negated,
                            body,
                        });
                    }
                    "set" => {
                        let assignment = tag
                            .strip_prefix("set")
                            .ok_or_else(|| template_error("malformed `set` tag"))?;
                        let (name, value) = assignment
                            .split_once('=')
                            .ok_or_else(|| template_error("`set` requires `name = value`"))?;
                        let name = name.trim();
                        let value = value.trim();
                        if name.is_empty() || value.is_empty() {
                            return Err(template_error("`set` requires a name and a value"));
                        }
                        nodes.push(Node::Set {
                            name: name.to_owned(),
                            value: value.to_owned(),
                        });
                    }
                    "endfor" | "endif" => {
                        if stop == Some(keyword) {
                            return Ok(nodes);
                        }
                        return Err(template_error(format!("unexpected {{% {keyword} %}}")));
                    }
                    other => {
                        return Err(template_error(format!("unsupported tag: {other}")));
                    }
                }
            }
        }
    }

    if let Some(stop_tag) = stop {
        return Err(template_error(format!("missing {{% {stop_tag} %}}")));
    }
    Ok(nodes)
}

fn lookup(key: &str, environment: &BTreeMap<String, String>) -> String {
    match environment.get(key.trim()) {
        Some(value) => value.clone(),
        None => String::new(),
    }
}

fn resolve_value(value: &str, environment: &BTreeMap<String, String>) -> String {
    let value = value.trim();
    let first = value.chars().next();
    let last = value.chars().last();
    if let (Some(quote), Some(closing)) = (first, last)
        && quote == closing
        && (quote == '"' || quote == '\'')
        && let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|text| text.strip_suffix(quote))
    {
        return inner.to_owned();
    }
    lookup(value, environment)
}

fn render_nodes(
    nodes: &[Node],
    environment: &mut BTreeMap<String, String>,
    messages: &[ChatMessage],
    output: &mut String,
) -> Result<(), CoreError> {
    for node in nodes {
        match node {
            Node::Text(text) => output.push_str(text),
            Node::Expression(expression) => {
                output.push_str(&resolve_value(expression, environment))
            }
            Node::Set { name, value } => {
                let resolved = resolve_value(value, environment);
                environment.insert(name.clone(), resolved);
            }
            Node::If {
                condition,
                negated,
                body,
            } => {
                let truthy = !resolve_value(condition, environment).is_empty();
                if truthy != *negated {
                    render_nodes(body, environment, messages, output)?;
                }
            }
            Node::For { variable, body } => {
                for message in messages {
                    let mut scope = environment.clone();
                    scope.insert(variable.clone(), message.content.clone());
                    scope.insert(format!("{variable}.role"), message.role.clone());
                    scope.insert(format!("{variable}.content"), message.content.clone());
                    render_nodes(body, &mut scope, messages, output)?;
                }
            }
        }
    }
    Ok(())
}

/// Renders `template` using `context`, `messages`, and the supported Jinja subset.
///
/// See the [module documentation](self) for the accepted syntax and undefined-variable rules.
#[must_use = "returns an error for malformed templates"]
pub fn render(
    template: &str,
    context: &BTreeMap<String, String>,
    messages: &[ChatMessage],
) -> Result<String, CoreError> {
    let tokens = tokenize(template)?;
    let mut cursor = 0;
    let nodes = parse_nodes(&tokens, &mut cursor, None)?;
    let mut environment = context.clone();
    let mut output = String::new();
    render_nodes(&nodes, &mut environment, messages, &mut output)?;
    Ok(output)
}

/// Produces a simple `role: content` transcript, one message per line.
///
/// This is intended as a prompt when no chat template is available; it returns an empty string
/// for an empty conversation.
#[must_use]
pub fn fallback(messages: &[ChatMessage]) -> String {
    let mut lines = Vec::with_capacity(messages.len());
    for message in messages {
        lines.push(format!("{}: {}", message.role, message.content));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde::{Serialize, de::DeserializeOwned};

    use super::*;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".to_owned(),
                content: "Be brief.".to_owned(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: "Hello!".to_owned(),
            },
        ]
    }

    fn round_trip<T>(value: &T) -> T
    where
        T: Serialize + DeserializeOwned,
    {
        let json = serde_json::to_string(value).expect("serialize value");
        serde_json::from_str(&json).expect("deserialize value")
    }

    #[test]
    fn chat_message_is_serializable_and_comparable() {
        let message = ChatMessage {
            role: "user".to_owned(),
            content: "hi".to_owned(),
        };
        assert_eq!(round_trip(&message), message);
        assert_ne!(
            message,
            ChatMessage {
                role: "user".to_owned(),
                content: "bye".to_owned()
            }
        );
    }

    #[test]
    fn renders_context_variables_and_preserves_text() {
        let context = BTreeMap::from([("name".to_owned(), "Ada".to_owned())]);
        let rendered = render("Hello {{ name }}!\n", &context, &[]).expect("render");
        assert_eq!(rendered, "Hello Ada!\n");
    }

    #[test]
    fn renders_for_loop_with_message_attributes_and_whitespace() {
        let template = "  {%   for message in messages  %}\n{{ message.role }}: {{  message.content  }}\n{%  endfor  %}";
        let rendered = render(template, &BTreeMap::new(), &sample_messages()).expect("render");
        assert_eq!(rendered, "  \nsystem: Be brief.\n\nuser: Hello!\n");
    }

    #[test]
    fn renders_if_set_and_not() {
        let context = BTreeMap::from([("enabled".to_owned(), "yes".to_owned())]);
        let template = "{% if enabled %}{% set greeting = 'hello' %}{{ greeting }}{% endif %}";
        assert_eq!(render(template, &context, &[]).expect("render"), "hello");

        let empty_context = BTreeMap::new();
        assert_eq!(render(template, &empty_context, &[]).expect("render"), "");
        assert_eq!(
            render("{% if not enabled %}off{% endif %}", &empty_context, &[]).expect("render"),
            "off"
        );
    }

    #[test]
    fn renders_nested_blocks_and_copies_context_into_loop_scope() {
        let context = BTreeMap::from([("prefix".to_owned(), ">>".to_owned())]);
        let template = "{% for message in messages %}{% if message.role %}{{ prefix }}{{ message.role }}={{ message.content }};{% endif %}{% endfor %}";
        let rendered = render(template, &context, &sample_messages()).expect("render");
        assert_eq!(rendered, ">>system=Be brief.;>>user=Hello!;");
    }

    #[test]
    fn undefined_variables_render_empty() {
        assert_eq!(
            render("[{{ missing }}]", &BTreeMap::new(), &[]).expect("render"),
            "[]"
        );
    }

    #[test]
    fn malformed_templates_return_errors() {
        let templates = [
            "unterminated {{ name",
            "{%  %}",
            "unterminated {% for message in messages %}",
            "{% for message in messages %}{{ message.content }}",
            "{% endif %}",
            "{% for message in wrong %}{% endfor %}",
            "{% set broken %}",
            "{% unknown %}",
        ];
        for template in templates {
            assert!(matches!(
                render(template, &BTreeMap::new(), &[]),
                Err(CoreError::Template(_))
            ));
        }
    }

    #[test]
    fn fallback_returns_one_line_per_message() {
        assert_eq!(fallback(&[]), "");
        assert_eq!(
            fallback(&sample_messages()),
            "system: Be brief.\nuser: Hello!"
        );
    }

    #[test]
    fn rendered_template_snapshot() {
        let template =
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";
        let rendered = render(template, &BTreeMap::new(), &sample_messages()).expect("render");
        insta::assert_snapshot!(rendered);
    }

    proptest! {
        #[test]
        fn render_arbitrary_strings_never_panics(
            template in any::<String>(),
            key in any::<String>(),
            value in any::<String>(),
            role in any::<String>(),
            content in any::<String>(),
        ) {
            let context = BTreeMap::from([(key, value)]);
            let messages = vec![ChatMessage { role, content }];
            let result = render(&template, &context, &messages);
            if !template.contains("{{") && !template.contains("{%") {
                prop_assert_eq!(result.expect("plain text always renders"), template);
            }
        }
    }
}
