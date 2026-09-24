{{- define "nilam-ocr-npwp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nilam-ocr-npwp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else if contains (include "nilam-ocr-npwp.name" .) .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "nilam-ocr-npwp.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/* Label yang dimiliki semua pod release, apa pun service-nya. */}}
{{- define "nilam-ocr-npwp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nilam-ocr-npwp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nilam-ocr-npwp.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{ include "nilam-ocr-npwp.selectorLabels" . }}
app.kubernetes.io/part-of: nilam-ocr
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Argumen: dict "root" $ "name" <nama service>. Selector Deployment/Service/PDB satu komponen. */}}
{{- define "nilam-ocr-npwp.componentSelectorLabels" -}}
{{ include "nilam-ocr-npwp.selectorLabels" .root }}
app.kubernetes.io/component: {{ .name }}
{{- end }}

{{/* Argumen: dict "root" $ "name" <nama> "svc" <values service>. */}}
{{- define "nilam-ocr-npwp.componentLabels" -}}
{{ include "nilam-ocr-npwp.labels" .root }}
app.kubernetes.io/component: {{ .name }}
app.kubernetes.io/version: {{ include "nilam-ocr-npwp.imageTag" . | quote }}
{{- end }}

{{- define "nilam-ocr-npwp.componentName" -}}
{{- printf "%s-%s" (include "nilam-ocr-npwp.fullname" .root) .name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nilam-ocr-npwp.imageTag" -}}
{{- .svc.image.tag | default .root.Values.image.tag | default .root.Chart.AppVersion }}
{{- end }}

{{- define "nilam-ocr-npwp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nilam-ocr-npwp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Isi ConfigMap satu service. Argumen: dict "root" $ "svc" <values service>.
URL antar service menunjuk ke Service per komponen (<release>-<nama>).
*/}}
{{- define "nilam-ocr-npwp.serviceEnv" -}}
{{- $root := .root }}
{{- $svc := .svc }}
{{- $env := dict "PORT" ($svc.port | toString) "ENVIRONMENT" $root.Values.environment }}
{{- if $svc.pipeline }}
{{- $_ := set $env "ORCHESTRATION_URL" $root.Values.orchestration.url }}
{{- $_ := set $env "ORCHESTRATION_CALLBACK_PATH" $root.Values.orchestration.callbackPath }}
{{- $_ := set $env "ORCHESTRATION_TIMEOUT_SECONDS" ($root.Values.orchestration.timeoutSeconds | toString) }}
{{- $_ := set $env "ORCHESTRATION_CALLBACK_FORMAT" ($root.Values.orchestration.callbackFormat | default "stage") }}
{{- end }}
{{- range $svc.upstreams }}
{{- $upstream := index $root.Values.services . }}
{{- $host := include "nilam-ocr-npwp.componentName" (dict "root" $root "name" .) }}
{{- $_ := set $env (printf "%s_SERVICE_URL" (upper .)) (printf "http://%s:%v" $host $upstream.port) }}
{{- end }}
{{- range $key, $value := $root.Values.commonEnv }}
{{- $_ := set $env $key ($value | toString) }}
{{- end }}
{{- range $key, $value := $svc.env }}
{{- $_ := set $env $key ($value | toString) }}
{{- end }}
{{- range $key, $value := $env }}
{{- if ne $value "" }}
{{ $key }}: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}
