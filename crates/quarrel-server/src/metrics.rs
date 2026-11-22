//! Prometheus metric registry for the bootstrap server.
//!
//! Names are constants so that the roadmap, alert rules and dashboards can be drift-checked
//! against a single source of truth.

use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// HTTP request counter.
pub const REQUESTS_TOTAL: &str = "quarrel_requests_total";
/// HTTP request latency histogram.
pub const REQUEST_DURATION: &str = "quarrel_request_duration_seconds";
/// Build and Arrow schema identity gauge.
pub const BUILD_INFO: &str = "quarrel_build_info";
/// Process resident set size gauge (populated by Phase 7).
pub const PROCESS_RSS: &str = "quarrel_process_resident_bytes";
/// Typed engine error counter (populated by Phase 7).
pub const ENGINE_ERRORS: &str = "quarrel_engine_errors_total";
/// Prefill/decode token counter.
pub const INFERENCE_TOKENS: &str = "quarrel_inference_tokens_total";
/// End-to-end inference latency histogram.
pub const INFERENCE_DURATION: &str = "quarrel_inference_duration_seconds";
/// Time-to-first-token histogram.
pub const TIME_TO_FIRST_TOKEN: &str = "quarrel_time_to_first_token_seconds";
/// Latest decode throughput gauge.
pub const DECODE_TOKENS_PER_SECOND: &str = "quarrel_decode_tokens_per_second";
/// Model load latency histogram.
pub const MODEL_LOAD_DURATION: &str = "quarrel_model_load_duration_seconds";
/// Mapped model weight bytes gauge.
pub const MODEL_WEIGHTS_BYTES: &str = "quarrel_model_weights_bytes";
/// Layers resident on GPU gauge.
pub const GPU_LAYERS_ACTIVE: &str = "quarrel_gpu_layers_active";
/// Layers resident on CPU gauge.
pub const CPU_LAYERS_ACTIVE: &str = "quarrel_cpu_layers_active";
/// Host/device layer transfer counter.
pub const LAYER_OFFLOAD_TRANSFERS: &str = "quarrel_layer_offload_transfers_total";
/// Host/device layer transfer latency histogram.
pub const LAYER_OFFLOAD_DURATION: &str = "quarrel_layer_offload_duration_seconds";
/// Used KV cache blocks gauge.
pub const KV_CACHE_BLOCKS_USED: &str = "quarrel_kv_cache_blocks_used";
/// KV cache capacity gauge.
pub const KV_CACHE_BLOCKS_CAPACITY: &str = "quarrel_kv_cache_blocks_capacity";
/// Prefix/KV lookup counter.
pub const KV_CACHE_LOOKUPS: &str = "quarrel_kv_cache_lookups_total";
/// KV cache hit counter.
pub const KV_CACHE_HITS: &str = "quarrel_kv_cache_hits_total";
/// KV cache eviction counter.
pub const KV_CACHE_EVICTIONS: &str = "quarrel_kv_cache_evictions_total";
/// Speculative-decoding acceptance counter.
pub const SPECULATIVE_TOKENS: &str = "quarrel_speculative_tokens_total";
/// Grammar/vocab rejection counter.
pub const SAMPLING_REJECTIONS: &str = "quarrel_sampling_rejections_total";
/// Flight RPC batch counter.
pub const FLIGHT_BATCHES: &str = "quarrel_flight_batches_total";
/// Arrow payload byte counter.
pub const FLIGHT_BYTES: &str = "quarrel_flight_bytes_total";
/// Flight batch latency histogram.
pub const FLIGHT_BATCH_DURATION: &str = "quarrel_flight_batch_duration_seconds";
/// Active SSE stream gauge.
pub const SSE_STREAMS_ACTIVE: &str = "quarrel_sse_streams_active";
/// Probe outcome counter.
pub const HEALTH_CHECKS: &str = "quarrel_health_checks_total";

/// Every canonical metric name from the roadmap §7 contract.
///
/// Alerts and dashboard panels must reference names from this list. Engine metrics are declared
/// here now so drift is caught before the emitting code lands; see `docs/metrics.md` for the
/// emitted-vs-reserved status of each series.
pub const ALL_METRICS: &[&str] = &[
    BUILD_INFO,
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    INFERENCE_TOKENS,
    INFERENCE_DURATION,
    TIME_TO_FIRST_TOKEN,
    DECODE_TOKENS_PER_SECOND,
    MODEL_LOAD_DURATION,
    MODEL_WEIGHTS_BYTES,
    GPU_LAYERS_ACTIVE,
    CPU_LAYERS_ACTIVE,
    LAYER_OFFLOAD_TRANSFERS,
    LAYER_OFFLOAD_DURATION,
    KV_CACHE_BLOCKS_USED,
    KV_CACHE_BLOCKS_CAPACITY,
    KV_CACHE_LOOKUPS,
    KV_CACHE_HITS,
    KV_CACHE_EVICTIONS,
    SPECULATIVE_TOKENS,
    SAMPLING_REJECTIONS,
    FLIGHT_BATCHES,
    FLIGHT_BYTES,
    FLIGHT_BATCH_DURATION,
    SSE_STREAMS_ACTIVE,
    PROCESS_RSS,
    ENGINE_ERRORS,
    HEALTH_CHECKS,
];

/// Handle to the installed recorder, used both for emitting and rendering metrics.
#[derive(Debug, Clone)]
pub struct Metrics {
    handle: PrometheusHandle,
}

impl Metrics {
    /// Install the global recorder and return a handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the recorder cannot be built or installed (for example when another
    /// recorder has already been installed in the process).
    pub fn install() -> Result<Self, metrics_exporter_prometheus::BuildError> {
        let handle = PrometheusBuilder::new()
            .set_buckets(&[
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ])?
            .install_recorder()?;

        metrics::gauge!(BUILD_INFO,
            "version" => env!("CARGO_PKG_VERSION").to_owned(),
            "arrow_schema" => "pending".to_owned(),
        )
        .set(1.0);
        metrics::gauge!(PROCESS_RSS).set(rss_bytes());

        Ok(Self { handle })
    }

    /// Record one completed HTTP request.
    pub fn record_request(
        &self,
        endpoint: &'static str,
        method: &str,
        status: u16,
        elapsed: Duration,
    ) {
        metrics::counter!(REQUESTS_TOTAL,
            "endpoint" => endpoint,
            "method" => method.to_owned(),
            "status" => status.to_string(),
        )
        .increment(1);
        metrics::histogram!(REQUEST_DURATION,
            "endpoint" => endpoint,
            "status" => status.to_string(),
        )
        .record(elapsed.as_secs_f64());
        metrics::gauge!(PROCESS_RSS).set(rss_bytes());
    }

    /// Record a typed engine error.
    pub fn record_engine_error(&self, code: &'static str) {
        metrics::counter!(ENGINE_ERRORS, "code" => code).increment(1);
    }

    /// Render a Prometheus exposition snapshot.
    #[must_use]
    pub fn render(&self) -> String {
        self.handle.render()
    }

    /// Drive periodic histogram maintenance.
    pub fn run_upkeep(&self) {
        self.handle.run_upkeep();
    }
}

/// Read resident set size from `/proc/self/statm` (Linux) without allocating a process handle.
fn rss_bytes() -> f64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
            return 0.0;
        };
        let mut fields = statm.split_whitespace();
        let _total = fields.next();
        let Some(resident_pages) = fields.next() else {
            return 0.0;
        };
        let Ok(pages) = resident_pages.parse::<u64>() else {
            return 0.0;
        };
        let page_size = unsafe { libc_sysconf() };
        (pages * page_size) as f64
    }

    #[cfg(not(target_os = "linux"))]
    0.0
}

/// Page size from `sysconf`, isolated so the `unsafe` call is easy to audit.
///
/// # Safety
///
/// `sysconf(_SC_PAGESIZE)` has no preconditions and returns `-1` on failure, which is handled
/// below. No pointers are passed and no global state is mutated.
#[cfg(target_os = "linux")]
unsafe fn libc_sysconf() -> u64 {
    unsafe extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    const SC_PAGESIZE: i32 = 30;
    let pagesize = unsafe { sysconf(SC_PAGESIZE) };
    if pagesize > 0 { pagesize as u64 } else { 4096 }
}

#[cfg(test)]
pub(crate) fn test_metrics() -> Metrics {
    use std::sync::OnceLock;
    static METRICS: OnceLock<Metrics> = OnceLock::new();
    METRICS
        .get_or_init(|| Metrics::install().expect("one recorder per process"))
        .clone()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{ALL_METRICS, test_metrics};

    const ALERTS: &str = include_str!("../../../deploy/prometheus/alerts/quarrel.rules.yml");
    const DASHBOARD: &str =
        include_str!("../../../deploy/grafana/dashboards/quarrel-overview.json");

    #[test]
    fn render_contains_build_info() {
        let metrics = test_metrics();
        metrics.run_upkeep();
        let rendered = metrics.render();
        assert!(rendered.contains("quarrel_build_info"), "{rendered}");
    }

    #[test]
    fn all_metric_names_are_canonical_and_unique() {
        let mut seen = HashSet::new();
        for name in ALL_METRICS {
            assert!(name.starts_with("quarrel_"), "non-canonical name: {name}");
            assert!(
                name.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
                "non-canonical name: {name}"
            );
            assert!(seen.insert(*name), "duplicate metric name: {name}");
        }
        assert_eq!(ALL_METRICS.len(), 27);
    }

    #[test]
    fn alert_and_dashboard_contracts_reference_every_metric_name() {
        let contract = format!("{ALERTS}\n{DASHBOARD}");
        for name in ALL_METRICS {
            assert!(
                contract.contains(name),
                "metric missing from alerts and dashboard: {name}"
            );
        }
    }
}
