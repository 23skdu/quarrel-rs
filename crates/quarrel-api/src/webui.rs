//! Self-contained browser UI embedded in the single server binary.
//!
//! The HTML, CSS and JavaScript live in one asset with `include_str!`; there is no CDN dependency
//! and no separate file to deploy. The page talks to the same public HTTP endpoints as any other
//! client, so API-key and rate-limit behaviour is identical.

/// Complete HTML document served at `/`.
pub const WEBUI_HTML: &str = include_str!("../assets/webui.html");

#[cfg(test)]
mod tests {
    use super::WEBUI_HTML;

    #[test]
    fn webui_wires_all_controls_and_has_no_external_urls() {
        for needle in [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/docs",
            "/metrics",
            "chat-stream",
            "run-chat",
            "run-embeddings",
            "refresh-models",
        ] {
            assert!(
                WEBUI_HTML.contains(needle),
                "missing WebUI control: {needle}"
            );
        }
        for external in ["http://", "https://", "//"] {
            assert!(
                !WEBUI_HTML.contains(external),
                "WebUI must not depend on external assets: {external}"
            );
        }
    }
}
