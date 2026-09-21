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

{{- define "nilam-ocr-npwp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nilam-ocr-npwp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nilam-ocr-npwp.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{ include "nilam-ocr-npwp.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/part-of: nilam-ocr
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nilam-ocr-npwp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nilam-ocr-npwp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "nilam-ocr-npwp.serviceEnv" -}}
{{- $root := .root }}
{{- $svc := .svc }}
{{- $env := dict "PORT" ($svc.port | toString) "ENVIRONMENT" $root.Values.environment }}
{{- if $svc.pipeline }}
{{- $_ := set $env "ORCHESTRATION_URL" (required "orchestration.url is required: ekstraksi, structuring and scoring refuse to start without it" $root.Values.orchestration.url) }}
{{- $_ := set $env "ORCHESTRATION_CALLBACK_PATH" $root.Values.orchestration.callbackPath }}
{{- $_ := set $env "ORCHESTRATION_TIMEOUT_SECONDS" ($root.Values.orchestration.timeoutSeconds | toString) }}
{{- end }}
{{- range $svc.upstreams }}
{{- $upstream := index $root.Values.services . }}
{{- $_ := set $env (printf "%s_SERVICE_URL" (upper .)) (printf "http://%s:%v" (include "nilam-ocr-npwp.fullname" $root) $upstream.port) }}
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
