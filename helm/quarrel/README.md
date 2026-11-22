# quarrel Helm chart

Deploys the `quarrel-server` control plane with a model PVC, health probes and optional
Prometheus Operator resources.

```bash
helm lint helm/quarrel --strict
helm template quarrel helm/quarrel \
  --set monitoring.serviceMonitor.enabled=true \
  --set monitoring.prometheusRule.enabled=true
helm install quarrel helm/quarrel --namespace quarrel --create-namespace
```

Key values:

| Value | Default | Purpose |
|-------|---------|---------|
| `image.repository` / `image.tag` | `quarrel-rs` / `local` | Image to deploy |
| `config.apiKey` | `""` | Bearer token required for `/v1/*` |
| `persistence.enabled` | `true` | Create a PVC for GGUF models |
| `persistence.existingClaim` | `""` | Reuse an existing model PVC |
| `monitoring.serviceMonitor.enabled` | `false` | Create a ServiceMonitor |
| `monitoring.prometheusRule.enabled` | `false` | Create a PrometheusRule |
| `ingress.enabled` | `false` | Expose the HTTP API externally |

For local development without Kubernetes, use `make docker-up` instead.
