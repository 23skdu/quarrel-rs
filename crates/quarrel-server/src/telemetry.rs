//! Optional OTLP trace export and W3C trace-context propagation for the HTTP plane.
//!
//! The `otel` feature is off by default. Without it [`init`] installs exactly the same
//! `tracing-subscriber` stack as before and returns a no-op [`TelemetryGuard`]; no
//! collector is contacted and no OpenTelemetry dependency is linked into the binary.
//!
//! With `otel` enabled [`init`] adds a [`tracing_opentelemetry`] layer backed by an
//! OTLP span exporter. The exporter is built with `SpanExporter::builder().build()`,
//! which reads the standard `OTEL_EXPORTER_OTLP_*` and `OTEL_EXPORTER_OTLP_TRACES_*`
//! variables (protocol, endpoint, headers, timeout, compression, TLS). When none are
//! set the standard OTLP default endpoint applies: no collector is needed to start,
//! but spans are only delivered once one is reachable (export failures are
//! non-fatal background errors).
//!
//! # Privacy
//!
//! Spans and log events carry only bounded route/method/status labels. Request bodies,
//! prompts, completions and token text are never recorded by this module, preserving
//! the existing tracing policy.
//!
//! # Integration
//!
//! Inbound W3C `traceparent` headers are extracted by [`propagate_trace`], which
//! parents an OTel server span to the remote context. Outbound callers can use
//! [`inject_context`] to propagate the current span to downstream services.

#[cfg(feature = "otel")]
use axum::http::{HeaderMap, HeaderName, HeaderValue};

/// W3C `traceparent` header name.
pub const TRACEPARENT_HEADER: &str = "traceparent";

/// Errors returned when the optional telemetry pipeline cannot be installed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TelemetryError {
    /// The OTLP exporter could not be configured from the environment.
    #[cfg(feature = "otel")]
    #[error("OTLP span exporter configuration failed: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),

    /// The process-wide tracing subscriber was already installed.
    #[error("tracing subscriber already installed: {0}")]
    Subscriber(#[from] tracing_subscriber::util::TryInitError),

    /// Pending spans could not be flushed during shutdown.
    #[cfg(feature = "otel")]
    #[error("OpenTelemetry provider shutdown failed: {0}")]
    Shutdown(#[from] opentelemetry_sdk::error::OTelSdkError),
}

/// Handle for the installed tracing pipeline.
///
/// Dropping it (or calling [`shutdown`](Self::shutdown)) flushes pending spans when
/// the `otel` feature is enabled. Without the feature this is a zero-sized no-op.
#[must_use = "keep the guard alive until shutdown to flush pending spans"]
#[derive(Debug)]
pub struct TelemetryGuard {
    #[cfg(feature = "otel")]
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl TelemetryGuard {
    /// Returns `true` when OTLP export is compiled in and active.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        cfg!(feature = "otel")
    }

    /// Flush pending spans and shut the provider down; a no-op without `otel`.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Shutdown`] if the provider cannot be shut down.
    pub fn shutdown(self) -> Result<(), TelemetryError> {
        #[cfg(feature = "otel")]
        if let Some(provider) = self.provider {
            provider.shutdown()?;
        }
        #[cfg(not(feature = "otel"))]
        let _ = self;
        Ok(())
    }
}

/// Install the tracing stack, adding an OTLP layer when `otel` is enabled.
///
/// `json` selects newline-delimited JSON output, exactly as
/// `QUARREL_LOG_JSON` did before this module existed.
///
/// # Errors
///
/// Returns a typed error if the exporter cannot be built from the environment or
/// if a global tracing subscriber is already installed.
pub fn init(json: bool) -> Result<TelemetryGuard, TelemetryError> {
    #[cfg(feature = "otel")]
    {
        init_otel(json)
    }
    #[cfg(not(feature = "otel"))]
    {
        init_plain(json);
        Ok(TelemetryGuard::disabled())
    }
}

impl TelemetryGuard {
    #[cfg(not(feature = "otel"))]
    fn disabled() -> Self {
        Self {}
    }
}

fn log_filter() -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
}

#[cfg(not(feature = "otel"))]
fn init_plain(json: bool) {
    use tracing_subscriber::prelude::*;

    if json {
        tracing_subscriber::registry()
            .with(log_filter())
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(log_filter())
            .with(tracing_subscriber::fmt::layer())
            .init();
    }
}

#[cfg(feature = "otel")]
fn init_otel(json: bool) -> Result<TelemetryGuard, TelemetryError> {
    use opentelemetry::global;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::prelude::*;

    // The builder resolves protocol/endpoint/headers from the standard OTEL_*
    // environment variables; no code-specific configuration is required.
    let exporter = opentelemetry_otlp::SpanExporter::builder().build()?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name("quarrel-server")
                .build(),
        )
        .build();
    let tracer = provider.tracer("quarrel-server");
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    global::set_text_map_propagator(TraceContextPropagator::new());

    let registry = tracing_subscriber::registry()
        .with(log_filter())
        .with(telemetry);
    if json {
        registry
            .with(tracing_subscriber::fmt::layer().json())
            .try_init()?;
    } else {
        registry.with(tracing_subscriber::fmt::layer()).try_init()?;
    }
    Ok(TelemetryGuard {
        provider: Some(provider),
    })
}

#[cfg(feature = "otel")]
struct HeaderExtractor<'a>(&'a HeaderMap);

#[cfg(feature = "otel")]
impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|key| key.as_str()).collect()
    }
}

#[cfg(feature = "otel")]
struct HeaderInjector<'a>(&'a mut HeaderMap);

#[cfg(feature = "otel")]
impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

#[cfg(feature = "otel")]
fn propagator() -> opentelemetry_sdk::propagation::TraceContextPropagator {
    opentelemetry_sdk::propagation::TraceContextPropagator::new()
}

/// Extract a W3C `traceparent` from HTTP headers, if it is present and valid.
///
/// The returned context is an in-memory OpenTelemetry [`Context`]; no network
/// access is involved. Returns `None` for absent, malformed or invalid-span headers.
///
/// [`Context`]: opentelemetry::Context
#[cfg(feature = "otel")]
#[must_use]
pub fn extract_context(headers: &HeaderMap) -> Option<opentelemetry::Context> {
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry::trace::TraceContextExt as _;
    let context = propagator().extract(&HeaderExtractor(headers));
    context.span().span_context().is_valid().then_some(context)
}

/// Inject `context` as a W3C `traceparent` into outgoing HTTP headers.
///
/// Existing `traceparent`/`tracestate` entries are replaced; other headers are
/// left untouched. Nothing is logged, so trace ids never mix with prompt text.
#[cfg(feature = "otel")]
pub fn inject_context(context: &opentelemetry::Context, headers: &mut HeaderMap) {
    use opentelemetry::propagation::TextMapPropagator;
    propagator().inject_context(context, &mut HeaderInjector(headers));
}

/// axum middleware that parents the request's server span to an inbound
/// `traceparent`. Used only when the `otel` feature is enabled.
#[cfg(feature = "otel")]
pub async fn propagate_trace(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use tracing::Instrument as _;
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;

    let method = request.method().to_string();
    // A bounded label, never the raw URI, query string or body.
    let endpoint = crate::api::endpoint_label(request.uri().path());
    let span = tracing::info_span!(
        "http.request",
        otel.kind = "server",
        "http.request.method" = %method,
        "url.path" = endpoint,
    );
    if let Some(context) = extract_context(request.headers()) {
        // Setting the parent before the span is entered lets the OTel layer
        // create the server span as a child of the remote caller.
        let _ = span.set_parent(context);
    }
    next.run(request).instrument(span).await
}

#[cfg(all(test, feature = "otel"))]
mod tests {
    use super::{
        TRACEPARENT_HEADER, TelemetryGuard, extract_context, inject_context, propagate_trace,
    };
    use axum::{
        Router,
        body::Body,
        http::{HeaderMap, Request, StatusCode},
        middleware::from_fn,
        routing::get,
    };
    use opentelemetry::{
        Context,
        trace::{
            SpanContext, SpanId, SpanKind, TraceContextExt as _, TraceFlags, TraceId, TraceState,
            TracerProvider as _,
        },
    };
    use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber as _;
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;
    use tracing_subscriber::prelude::*;

    #[derive(Clone, Debug, Default)]
    struct SharedExporter(Arc<Mutex<Vec<SpanData>>>);

    impl SharedExporter {
        fn spans(&self) -> Vec<SpanData> {
            self.0.lock().unwrap().clone()
        }
    }

    impl SpanExporter for SharedExporter {
        async fn export(&self, batch: Vec<SpanData>) -> opentelemetry_sdk::error::OTelSdkResult {
            self.0.lock().unwrap().extend(batch);
            Ok(())
        }

        fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
            Ok(())
        }
    }

    fn remote_context() -> SpanContext {
        SpanContext::new(
            TraceId::from_bytes([1; 16]),
            SpanId::from_bytes([2; 8]),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        )
    }

    #[test]
    fn guard_without_provider_reports_no_op_when_otel_is_disabled() {
        // The feature is enabled in this test binary, but a guard without a
        // provider must still shut down cleanly (the no-op path).
        let guard = TelemetryGuard { provider: None };
        assert!(guard.is_enabled());
        guard.shutdown().unwrap();
    }

    #[test]
    fn traceparent_round_trip_uses_w3c_format() {
        let remote = remote_context();
        let source = Context::new().with_remote_span_context(remote.clone());
        let mut headers = HeaderMap::new();
        inject_context(&source, &mut headers);

        assert_eq!(
            headers.get(TRACEPARENT_HEADER).unwrap().to_str().unwrap(),
            "00-01010101010101010101010101010101-0202020202020202-01"
        );
        let extracted = extract_context(&headers).expect("valid traceparent");
        let extracted_span = extracted.span();
        let actual = extracted_span.span_context();
        assert_eq!(actual.trace_id(), remote.trace_id());
        assert_eq!(actual.span_id(), remote.span_id());
        assert_eq!(actual.trace_flags(), remote.trace_flags());
    }

    #[test]
    fn malformed_or_absent_traceparent_is_ignored() {
        let mut headers = HeaderMap::new();
        assert!(extract_context(&headers).is_none());
        headers.insert(TRACEPARENT_HEADER, "not-a-traceparent".parse().unwrap());
        assert!(extract_context(&headers).is_none());
        headers.insert(
            TRACEPARENT_HEADER,
            "00-00000000000000000000000000000000-0000000000000000-01"
                .parse()
                .unwrap(),
        );
        assert!(extract_context(&headers).is_none());
    }

    #[tokio::test]
    async fn http_middleware_parents_the_server_span_to_the_remote_context() {
        let exporter = SharedExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("test-http");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        let app = Router::new()
            .route(
                "/v1/models",
                get(|| async {
                    tracing::info_span!("handler").in_scope(|| {
                        let current = tracing::Span::current().context();
                        assert_eq!(
                            current.span().span_context().trace_id(),
                            remote_context().trace_id()
                        );
                    });
                    StatusCode::OK
                }),
            )
            .layer(from_fn(propagate_trace));

        let remote = remote_context();
        let mut headers = HeaderMap::new();
        inject_context(
            &Context::new().with_remote_span_context(remote.clone()),
            &mut headers,
        );
        let mut request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        *request.headers_mut() = headers;

        let response = app
            .oneshot(request)
            .with_subscriber(subscriber)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        provider.shutdown().unwrap();
        let spans = exporter.spans();
        let server = spans
            .iter()
            .find(|span| span.name == "http.request")
            .expect("server span exported");
        assert_eq!(server.parent_span_id, remote.span_id());
        assert!(server.parent_span_is_remote);
        assert_eq!(server.span_kind, SpanKind::Server);
        assert!(
            spans
                .iter()
                .all(|span| !span.name.to_string().contains("prompt"))
        );
    }
}
