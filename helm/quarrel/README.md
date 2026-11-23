# quarrel Helm chart

Deploys the `quarrel-server` control plane with a model PVC, configurable health probes,
optional GPU scheduling and NetworkPolicy, and optional Prometheus Operator resources.

```bash
helm lint helm/quarrel --strict
helm template quarrel helm/quarrel > /dev/null
helm template quarrel helm/quarrel -f helm/quarrel/values-monitoring.yaml > /dev/null
helm install quarrel helm/quarrel --namespace quarrel --create-namespace
```

`values-monitoring.yaml` is a committed example that enables the ServiceMonitor,
PrometheusRule, Grafana dashboard ConfigMap and NetworkPolicy, shortens the probe intervals
and raises the model PVC size. It also adds `release: prometheus` labels for a typical
Operator selector and standard OTLP endpoint variables; traces are exported only when the
image is built with the server's `otel` feature. The Prometheus Operator CRDs and a Grafana
sidecar must exist in the cluster for those resources to be used.

The dashboard ConfigMap renders `files/quarrel-overview.json`, which must stay byte-identical
to `grafana/dashboards/quarrel-overview.json`. `make metrics-check` enforces that and checks
all metric names against the Rust registry and alert rules.

## Values

### Workload and image

| Value | Default | Purpose |
|-------|---------|---------|
| `replicaCount` | `1` | Deployment replicas (ignored when autoscaling is enabled) |
| `image.repository` | `quarrel-rs` | Image repository |
| `image.tag` | `local` | Image tag (falls back to chart appVersion when empty) |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy |
| `imagePullSecrets` | `[]` | Image pull secret references |
| `nameOverride` / `fullnameOverride` | `""` | Override generated resource names |
| `serviceAccount.create` | `true` | Create a ServiceAccount |
| `serviceAccount.name` | `""` | Use this name instead of the generated one |
| `serviceAccount.annotations` | `{}` | ServiceAccount annotations |
| `podAnnotations` / `podLabels` | `{}` | Extra pod metadata |
| `terminationGracePeriodSeconds` | `30` | Pod termination grace period |
| `nodeSelector` / `tolerations` / `affinity` | `{}` / `[]` / `{}` | Pod scheduling constraints |

### Service, config and storage

| Value | Default | Purpose |
|-------|---------|---------|
| `service.type` | `ClusterIP` | Service type |
| `service.port` | `8080` | Container and Service HTTP port (also used for `--listen`) |
| `config.logJson` | `true` | Render `QUARREL_LOG_JSON` in the ConfigMap |
| `config.rustLog` | `info,quarrel_server=debug` | Render `RUST_LOG` in the ConfigMap |
| `config.apiKey` | `""` | When set, render a Secret and require a Bearer token for `/v1/*` |
| `config.extraEnv` | `{}` | Extra environment variables in the ConfigMap (for example `OTEL_EXPORTER_OTLP_*`) |
| `persistence.enabled` | `true` | Create a PVC when `existingClaim` is empty |
| `persistence.existingClaim` | `""` | Reuse an existing, populated model PVC (no PVC is created) |
| `persistence.size` | `10Gi` | Requested PVC size when the chart creates it |
| `persistence.storageClass` | `""` | StorageClass name; empty uses the cluster default |
| `persistence.accessModes` | `[ReadWriteOnce]` | PVC access modes |
| `persistence.mountPath` | `/models` | Model mount path, passed to `--models-dir` |
| `persistence.readOnly` | `true` | Mount the model volume read-only |

### GPU resources

| Value | Default | Purpose |
|-------|---------|---------|
| `gpu.enabled` | `false` | Add the GPU device resource to container requests and limits |
| `gpu.resourceName` | `nvidia.com/gpu` | Device-plugin resource key (e.g. `amd.com/gpu`) |
| `gpu.count` | `1` | Device count placed in both requests and limits |
| `gpu.extraRequests` | `{}` | Extra GPU-related requests (e.g. vendor memory resources) |
| `gpu.extraLimits` | `{}` | Extra GPU-related limits |
| `gpu.runtimeClassName` | `""` | Optional RuntimeClass (e.g. `nvidia`); empty omits the field |

With the default `resources` block and `gpu.enabled=true`, the container requests
`cpu: 250m`, `memory: 256Mi`, `nvidia.com/gpu: 1` and limits `cpu: "2"`,
`memory: 1Gi`, `nvidia.com/gpu: 1`. Existing `resources` values are preserved.

### Probes and security

| Value | Default | Purpose |
|-------|---------|---------|
| `livenessProbe.enabled` | `true` | Include the liveness probe; `false` omits it |
| `livenessProbe.*` | `httpGet /healthz`, 5s delay, 15s period | Any Kubernetes liveness probe field |
| `readinessProbe.enabled` | `true` | Include the readiness probe; `false` omits it |
| `readinessProbe.*` | `httpGet /readyz`, 5s delay, 10s period | Any Kubernetes readiness probe field |
| `startupProbe.enabled` | `false` | Include a startup probe; `false` omits it |
| `startupProbe.*` | `httpGet /healthz`, 30 failures, 10s period | Any Kubernetes startup probe field |
| `podSecurityContext` | non-root `1000:1000`, `fsGroup: 1000`, `RuntimeDefault` seccomp | Pod security context |
| `securityContext` | no privilege escalation, read-only root FS, drop `ALL` | Container security context |

The `enabled` key is stripped before rendering; every other key is passed through as-is,
so probes can be reconfigured without template changes.

### Network and monitoring

| Value | Default | Purpose |
|-------|---------|---------|
| `networkPolicy.enabled` | `false` | Create a NetworkPolicy selecting the pod |
| `networkPolicy.labels` | `{}` | Extra NetworkPolicy labels |
| `networkPolicy.ingressPeers` | `[]` | NetworkPolicyPeer list for the HTTP port; empty allows all peers |
| `networkPolicy.egressRules` | `[]` | NetworkPolicyEgressRule list; empty allows all egress |
| `monitoring.serviceMonitor.enabled` | `false` | Create a ServiceMonitor for `/metrics` |
| `monitoring.serviceMonitor.interval` | `15s` | Scrape interval |
| `monitoring.serviceMonitor.scrapeTimeout` | `""` | Scrape timeout; empty uses the Prometheus default |
| `monitoring.serviceMonitor.labels` | `{}` | Extra ServiceMonitor labels (Operator selector) |
| `monitoring.serviceMonitor.relabelings` | `[]` | Standard target relabelings |
| `monitoring.serviceMonitor.metricRelabelings` | `[]` | Standard metric relabelings before storage |
| `monitoring.prometheusRule.enabled` | `false` | Create a baseline PrometheusRule |
| `monitoring.prometheusRule.labels` | `{}` | Extra PrometheusRule labels (Operator selector) |
| `monitoring.prometheusRule.additionalRules` | `[]` | Extra PrometheusRule rules appended to the baseline set |
| `monitoring.grafanaDashboard.enabled` | `false` | Publish the committed overview dashboard as a discoverable ConfigMap |
| `monitoring.grafanaDashboard.label` / `value` | `grafana_dashboard` / `1` | Sidecar discovery label |
| `monitoring.grafanaDashboard.annotations` | `{}` | Extra dashboard ConfigMap annotations |
| `monitoring.grafanaDashboard.folder` | `""` | Optional `k8s-sidecar-target-directory` value |
| `ingress.enabled` | `false` | Create an Ingress for the HTTP API |
| `ingress.className` | `""` | Ingress class name |
| `ingress.annotations` / `ingress.tls` | `{}` / `[]` | Ingress annotations and TLS |
| `ingress.hosts` | `quarrel.local` `/` (Prefix) | Hosts and paths |
| `autoscaling.enabled` | `false` | Create an HPA and drop the fixed replica count |
| `autoscaling.minReplicas` / `maxReplicas` | `1` / `5` | HPA bounds |
| `autoscaling.targetCPUUtilizationPercentage` | `80` | CPU target |
| `autoscaling.targetMemoryUtilizationPercentage` | `""` | Optional memory target |

## Golden render tests

Rendered YAML for two configurations is committed under `tests/golden/`. There is no test
framework dependency and no Make target; compare a fresh render with the committed file
using standard tools:

```bash
helm template quarrel helm/quarrel --namespace quarrel \
  | diff -u helm/quarrel/tests/golden/default.yaml -
helm template quarrel helm/quarrel --namespace quarrel \
  -f helm/quarrel/values-monitoring.yaml \
  | diff -u helm/quarrel/tests/golden/monitoring.yaml -
```

Regenerate a golden after an intentional template change with the same `helm template`
command redirected to the golden file. The release name (`quarrel`) and namespace must
match the commands above; other names are derived from them.

For local development without Kubernetes, use `make docker-up` instead.
