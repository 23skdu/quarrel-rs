# Metrics, alerts and dashboard

This document is the operator's companion to the [section 7 metric contract](roadmap.md).
Every panel in `deploy/grafana/dashboards/quarrel-overview.json` is listed below with its
query and the question it answers. All metric names come from section 7; histograms are
queried through their standard `_bucket` series.

## Emitted today vs reserved

Only the control-plane series are registered by `crates/quarrel-server/src/metrics.rs` today:

| Metric | Type | Labels | Status |
|--------|------|--------|--------|
| `quarrel_build_info` | gauge | `version`, `arrow_schema` | emitted |
| `quarrel_requests_total` | counter | `endpoint`, `method`, `status` | emitted |
| `quarrel_request_duration_seconds` | histogram | `endpoint`, `status` | emitted |
| `quarrel_process_resident_bytes` | gauge | — | emitted |
| `quarrel_engine_errors_total` | counter | `code` | emitted |

The `status` label on `quarrel_requests_total` currently carries the numeric HTTP status
(e.g. `200`, `503`), which is why the dashboard and alerts use `status=~"5.."`.

Every other section 7 name is **reserved for future engine work**: token, TTFT, decode
throughput, model load, layer offload, KV-cache, Flight, speculative sampling, SSE and
health-check metrics. All 27 names are declared as constants in
`crates/quarrel-server/src/metrics.rs`; a unit test asserts that every constant appears in the
alert rules or dashboard so the three contracts cannot drift silently. Panels and alerts that use
those names are deliberately wired up
ahead of the implementation; they render as “no data” and cannot fire until the engine
emits the series. Do not read an empty reserved panel as evidence that the behaviour is
healthy.

## Dashboard panels

Panel titles match the Grafana JSON exactly. “Status” says whether the metric behind the
panel is emitted today or reserved.

| Panel | Query (PromQL) | Status | Operational question |
|-------|----------------|--------|----------------------|
| Request rate | `sum(rate(quarrel_requests_total[5m]))` | emitted | How much HTTP traffic is the control plane serving? |
| Server error rate | `sum(rate(quarrel_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(quarrel_requests_total[5m])), 1)` | emitted | What share of requests are failing with 5xx? |
| Resident memory | `quarrel_process_resident_bytes` | emitted | How large is the current process RSS? |
| Build identity | `quarrel_build_info` | emitted | Which binary version and Arrow schema revision is running? |
| HTTP latency by endpoint | `histogram_quantile(0.50, sum by (le, endpoint) (rate(quarrel_request_duration_seconds_bucket[5m])))`<br>`histogram_quantile(0.95, sum by (le, endpoint) (rate(quarrel_request_duration_seconds_bucket[5m])))`<br>`histogram_quantile(0.99, sum by (le, endpoint) (rate(quarrel_request_duration_seconds_bucket[5m])))` | emitted | Are p50/p95/p99 latencies acceptable per endpoint? |
| Requests by endpoint and status | `sum by (endpoint, status) (rate(quarrel_requests_total[5m]))` | emitted | Which endpoint/status combination generates the traffic or errors? |
| Engine errors by code | `sum by (code) (rate(quarrel_engine_errors_total[5m]))` | emitted | Which typed engine error codes are occurring? |
| Time to first token (p95) | `histogram_quantile(0.95, sum by (le, model) (rate(quarrel_time_to_first_token_seconds_bucket[5m])))` | reserved | How long do users wait before the first token, per model? |
| Decode throughput (tokens/s) | `sum by (model) (quarrel_decode_tokens_per_second)`<br>`sum by (model) (rate(quarrel_inference_tokens_total{operation="decode"}[5m]))` | reserved | What decode rate is each model sustaining (latest gauge vs 5m average)? |
| KV-cache saturation | `sum by (model, pool) (quarrel_kv_cache_blocks_used) / clamp_min(sum by (model, pool) (quarrel_kv_cache_blocks_capacity), 1)` | reserved | Which KV pool is close to exhaustion and likely to evict? |
| Flight payload bytes | `sum by (method, direction) (rate(quarrel_flight_bytes_total[5m]))` | reserved | How much Arrow payload traffic crosses each Flight method? |
| Flight batches | `sum by (method, status) (rate(quarrel_flight_batches_total[5m]))` | reserved | What is the Flight batch rate and failure mix per method? |
| Layer offload transfers | `sum by (direction) (rate(quarrel_layer_offload_transfers_total[5m]))` | reserved | How often are layers moved host-to-device or device-to-host? |
| Active layers by device | `sum by (model) (quarrel_gpu_layers_active)`<br>`sum by (model) (quarrel_cpu_layers_active)` | reserved | Is the intended number of layers resident on GPU vs CPU? |
| Layer offload latency (p95) | `histogram_quantile(0.95, sum by (le, direction) (rate(quarrel_layer_offload_duration_seconds_bucket[5m])))` | reserved | Are offload transfers adding tail latency? |
| Model load duration (p95) | `histogram_quantile(0.95, sum by (le, model) (rate(quarrel_model_load_duration_seconds_bucket[5m])))` | reserved | How long do cold/warm model loads take at the tail? |
| Resident memory trend | `max by (instance) (quarrel_process_resident_bytes)` | emitted | Is RSS trending upward over a longer window than the stat panel shows? |
| Roadmap status | n/a (markdown panel) | informational | Which metrics are real today, and which are reserved? |

Section 7 names that are deliberately not given a panel yet
(`quarrel_model_weights_bytes`, `quarrel_kv_cache_lookups_total`,
`quarrel_kv_cache_hits_total`, `quarrel_kv_cache_evictions_total`,
`quarrel_speculative_tokens_total`, `quarrel_sampling_rejections_total`,
`quarrel_flight_batch_duration_seconds`, `quarrel_sse_streams_active`,
`quarrel_health_checks_total`, `quarrel_inference_duration_seconds`) are all reserved.

## Alert rules

`deploy/prometheus/alerts/quarrel.rules.yml` keeps the original control-plane rules and
adds the section 7 alert contract. Reserved-metric rules are inert until engine work lands.

| Alert | Condition | Status |
|-------|-----------|--------|
| `QuarrelHighErrorRate` | 5xx share of HTTP requests > 5% for 5m | emitted |
| `QuarrelHighChatLatency` | chat completion p95 latency > 2s for 10m | emitted |
| `QuarrelMemoryGrowth` | RSS derivative > 5 MB/s over 30m for 10m (acute growth) | emitted |
| `QuarrelUnavailable` | `up{job="quarrel"} == 0` for 2m | emitted |
| `QuarrelRecordersStale` | build-info timestamp older than 1h | emitted |
| `QuarrelHighTtftP95` | TTFT p95 > 0.5s for 10m | reserved |
| `QuarrelKvCacheSaturation` | KV blocks used / capacity > 90% for 10m | reserved |
| `QuarrelFlightErrorRate` | Flight batches with error/failure status > 5% for 5m | reserved |
| `QuarrelModelLoadFailures` | engine errors with a model/load/gguf code > 0 for 5m | emitted metric, best-effort proxy until a dedicated failure counter exists |
| `QuarrelSustainedRssGrowth` | RSS derivative > 100 kB/s over 1h for 30m (slow leak) | emitted |

The Helm chart's optional `PrometheusRule` (`monitoring.prometheusRule.enabled`) ships a
small baseline rule set; the full rules above come from `deploy/prometheus/alerts/`.
Keep both in sync when adding new section 7 metrics.

## Validation

```bash
python3 -m json.tool deploy/grafana/dashboards/quarrel-overview.json >/dev/null
yq deploy/prometheus/alerts/quarrel.rules.yml >/dev/null
make metrics-check
```

`make metrics-check` greps the emitted control-plane names across the Rust registry, the
dashboard and the alert rules. It intentionally does not assert that reserved metrics are
emitted.
