{{/* Chart name, overridable. */}}
{{- define "quarrel.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name. */}}
{{- define "quarrel.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Chart label. */}}
{{- define "quarrel.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Common labels. */}}
{{- define "quarrel.labels" -}}
helm.sh/chart: {{ include "quarrel.chart" . }}
{{ include "quarrel.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: quarrel-rs
{{- end -}}

{{/* Selector labels. */}}
{{- define "quarrel.selectorLabels" -}}
app.kubernetes.io/name: {{ include "quarrel.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* ServiceAccount name. */}}
{{- define "quarrel.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "quarrel.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Name of the model PVC. */}}
{{- define "quarrel.pvcName" -}}
{{- default (printf "%s-models" (include "quarrel.fullname" .)) .Values.persistence.existingClaim -}}
{{- end -}}

{{/* ConfigMap name. */}}
{{- define "quarrel.configMapName" -}}
{{- printf "%s-config" (include "quarrel.fullname" .) -}}
{{- end -}}

{{/* Secret name. */}}
{{- define "quarrel.secretName" -}}
{{- printf "%s-secret" (include "quarrel.fullname" .) -}}
{{- end -}}
