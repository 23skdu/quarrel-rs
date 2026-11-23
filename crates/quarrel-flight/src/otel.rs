//! W3C trace-context propagation for the Flight/gRPC plane.
//!
//! This module is compiled only with the `otel` feature. It never starts an
//! exporter or contacts a collector: OTLP export is owned by `quarrel-server`.
//! All work happens in memory through
//! [`opentelemetry_sdk::propagation::TraceContextPropagator`].
//!
//! # Server integration (exact wiring)
//!
//! Tonic's `trace_fn` is called once per inbound RPC with the request metadata
//! still attached, and keeps the returned [`tracing::Span`] entered for the whole
//! handler future. That makes it the correct hook for remote-parent extraction:
//!
//! ```ignore
//! tonic::transport::Server::builder()
//!     .trace_fn(quarrel_flight::otel::flight_trace_fn)
//!     .add_service(quarrel_flight::flight_server(service))
//!     .serve(address)
//!     .await?;
//! ```
//!
//! Without this hook the Flight service still works, but inbound `traceparent`
//! headers are ignored — the crate does **not** claim implicit propagation.
//!
//! # Client integration
//!
//! [`PropagatingInterceptor`] injects the current span as `traceparent` into
//! outgoing gRPC metadata:
//!
//! ```ignore
//! let channel = tonic::transport::Channel::from_shared(url)?
//!     .connect()
//!     .await?;
//! let mut client = arrow_flight::FlightClient::with_interceptor(
//!     channel,
//!     quarrel_flight::otel::PropagatingInterceptor,
//! );
//! ```
//!
//! Arrow payloads, prompts and generated token text are never read or logged by
//! this module; the server span uses static fields only (no method paths).

use opentelemetry::{
    Context,
    propagation::{Extractor, Injector, TextMapPropagator},
    trace::TraceContextExt as _,
};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tonic::{
    Request,
    metadata::{AsciiMetadataKey, KeyRef, MetadataMap, MetadataValue},
    service::Interceptor,
};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

/// W3C `traceparent` metadata key.
pub const TRACEPARENT_HEADER: &str = "traceparent";
/// W3C `tracestate` metadata key.
pub const TRACESTATE_HEADER: &str = "tracestate";

/// Error returned when a trace context cannot be written into gRPC metadata.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContextError {
    /// The propagator produced a value that is not valid ASCII metadata.
    #[error("invalid trace-context metadata value: {0}")]
    InvalidValue(String),
}

struct MetadataExtractor<'a>(&'a MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .filter_map(|key| match key {
                KeyRef::Ascii(key) => Some(key.as_str()),
                KeyRef::Binary(_) => None,
            })
            .collect()
    }
}

struct MetadataInjector<'a>(&'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        // `MetadataMap::insert` only accepts owned/'static keys, so parse into an
        // owned ASCII key first.
        if let (Ok(name), Ok(value)) = (
            key.parse::<AsciiMetadataKey>(),
            MetadataValue::try_from(value),
        ) {
            self.0.insert(name, value);
        }
    }
}

fn propagator() -> TraceContextPropagator {
    TraceContextPropagator::new()
}

/// Extract a W3C `traceparent` from inbound gRPC metadata, if present and valid.
///
/// Returns `None` for missing, malformed or all-zero identifiers. No network
/// access is involved.
#[must_use]
pub fn extract_context(metadata: &MetadataMap) -> Option<Context> {
    let context = propagator().extract(&MetadataExtractor(metadata));
    context.span().span_context().is_valid().then_some(context)
}

/// Inject `context` as W3C `traceparent`/`tracestate` into outgoing gRPC metadata.
///
/// # Errors
///
/// Returns [`ContextError::InvalidValue`] if a propagated value cannot be
/// represented as ASCII metadata (this cannot happen for a valid W3C context).
pub fn inject_context(context: &Context, metadata: &mut MetadataMap) -> Result<(), ContextError> {
    let mut injector = MetadataInjector(metadata);
    propagator().inject_context(context, &mut injector);
    Ok(())
}

/// Tonic `trace_fn` that parents each inbound Flight RPC to its remote caller.
///
/// Use it as `Server::builder().trace_fn(flight_trace_fn)`, as shown in the
/// [module documentation](self). The span has static fields only — message
/// payloads, prompts and token text are never read or recorded.
#[must_use]
pub fn flight_trace_fn(request: &Request<()>) -> tracing::Span {
    let span = tracing::info_span!("grpc.request", otel.kind = "server", "rpc.system" = "grpc",);
    if let Some(context) = extract_context(request.metadata()) {
        // Set the parent before the span is entered so the OTel layer exports a
        // child span instead of a new root.
        let _ = span.set_parent(context);
    }
    span
}

/// Client interceptor that injects the current span into outgoing Flight calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct PropagatingInterceptor;

impl Interceptor for PropagatingInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, tonic::Status> {
        inject_context(&tracing::Span::current().context(), request.metadata_mut())
            .map_err(|error| tonic::Status::internal(error.to_string()))?;
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MetadataExtractor, PropagatingInterceptor, TRACEPARENT_HEADER, extract_context,
        flight_trace_fn, inject_context,
    };
    use opentelemetry::{
        Context,
        propagation::Extractor as _,
        trace::{
            SpanContext, SpanId, SpanKind, TraceContextExt as _, TraceFlags, TraceId, TraceState,
            TracerProvider as _,
        },
    };
    use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
    use std::sync::{Arc, Mutex};
    use tonic::{
        Request,
        metadata::{MetadataMap, MetadataValue},
        service::Interceptor as _,
    };
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
            TraceId::from_bytes([3; 16]),
            SpanId::from_bytes([4; 8]),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        )
    }

    #[test]
    fn traceparent_round_trip_through_grpc_metadata() {
        let remote = remote_context();
        let source = Context::new().with_remote_span_context(remote.clone());
        let mut metadata = MetadataMap::new();
        inject_context(&source, &mut metadata).unwrap();

        assert_eq!(
            metadata.get(TRACEPARENT_HEADER).unwrap().to_str().unwrap(),
            "00-03030303030303030303030303030303-0404040404040404-01"
        );
        let extracted = extract_context(&metadata).expect("valid traceparent");
        assert_eq!(
            extracted.span().span_context().trace_id(),
            remote.trace_id()
        );
        assert_eq!(extracted.span().span_context().span_id(), remote.span_id());
    }

    #[test]
    fn invalid_or_missing_metadata_is_ignored() {
        let mut metadata = MetadataMap::new();
        assert!(extract_context(&metadata).is_none());
        metadata.insert(TRACEPARENT_HEADER, "garbage".parse().unwrap());
        assert!(extract_context(&metadata).is_none());
        let zero = "00-00000000000000000000000000000000-0000000000000000-01";
        metadata.insert(TRACEPARENT_HEADER, zero.parse().unwrap());
        assert!(extract_context(&metadata).is_none());
    }

    #[test]
    fn extractor_and_injector_ignore_binary_metadata_keys() {
        let mut metadata = MetadataMap::new();
        metadata.insert_bin("payload-bin", MetadataValue::from_bytes(b"prompt text"));
        let extractor = MetadataExtractor(&metadata);
        assert!(!extractor.keys().iter().any(|key| key.contains("bin")));
        inject_context(&Context::new(), &mut metadata).unwrap();
    }

    #[tokio::test]
    async fn trace_fn_and_client_interceptor_propagate_the_same_trace() {
        let exporter = SharedExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("test-flight");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        let remote = remote_context();
        let mut inbound = Request::new(());
        inject_context(
            &Context::new().with_remote_span_context(remote.clone()),
            inbound.metadata_mut(),
        )
        .unwrap();
        let (server_span, outbound, trace_id) = {
            let _default = tracing::subscriber::set_default(subscriber);
            let server_span = flight_trace_fn(&inbound);
            let _entered = server_span.enter();
            let mut interceptor = PropagatingInterceptor;
            let outbound = interceptor.call(Request::new(())).unwrap();
            let current = tracing::Span::current().context();
            let trace_id = current.span().span_context().trace_id();
            drop(_entered);
            (server_span, outbound, trace_id)
        };
        assert_eq!(trace_id, remote.trace_id());
        assert_eq!(
            extract_context(outbound.metadata())
                .expect("client injected a traceparent")
                .span()
                .span_context()
                .trace_id(),
            remote.trace_id()
        );

        drop(server_span);
        provider.shutdown().unwrap();
        let spans = exporter.spans();
        let server = spans
            .iter()
            .find(|span| span.name == "grpc.request")
            .expect("server span exported");
        assert_eq!(server.parent_span_id, remote.span_id());
        assert!(server.parent_span_is_remote);
        assert_eq!(server.span_kind, SpanKind::Server);
    }
}
