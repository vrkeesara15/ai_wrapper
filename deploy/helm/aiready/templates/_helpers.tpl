{{/*
Shared helpers. Conventions:
  * Kubernetes object names are the component names (gateway, rag, model-chat ...), so the wiring overlay
    (files/overlays/k8s/services.yaml) can use short in-namespace DNS names. One release per namespace.
  * Every pod carries aiready.io/zone, which the NetworkPolicies select on.
  * Booleans are resolved with hasKey, never `default`, because `default` treats false as unset.
*/}}

{{- define "aiready.chart" -}}
{{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Common labels. Input: dict root name zone */}}
{{- define "aiready.labels" -}}
helm.sh/chart: {{ include "aiready.chart" .root }}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/part-of: aiready
app.kubernetes.io/managed-by: {{ .root.Release.Service }}
app.kubernetes.io/version: {{ .root.Chart.AppVersion | quote }}
aiready.io/component: {{ .name }}
aiready.io/zone: {{ .zone }}
aiready.io/stage: {{ .root.Values.global.stage }}
{{- end }}

{{/* Selector labels. Input: dict root name */}}
{{- define "aiready.selectorLabels" -}}
app.kubernetes.io/instance: {{ .root.Release.Name }}
aiready.io/component: {{ .name }}
{{- end }}

{{/* Image reference. Input: dict root image(registry repository tag digest) */}}
{{- define "aiready.image" -}}
{{- $img := .image -}}
{{- $reg := .root.Values.global.imageRegistry | default $img.registry -}}
{{- $tag := $img.tag | default .root.Chart.AppVersion | toString -}}
{{- $repo := $img.repository -}}
{{- if $reg }}{{ $repo = printf "%s/%s" $reg $repo }}{{ end -}}
{{- if $img.digest -}}
{{ printf "%s@%s" $repo $img.digest }}
{{- else -}}
{{ printf "%s:%s" $repo $tag }}
{{- end -}}
{{- end }}

{{- define "aiready.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/* Restricted Pod Security Standard - pod level. Input: root */}}
{{- define "aiready.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: {{ .Values.podSecurity.runAsUser }}
runAsGroup: {{ .Values.podSecurity.runAsGroup }}
fsGroup: {{ .Values.podSecurity.fsGroup }}
fsGroupChangePolicy: OnRootMismatch
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/* Restricted Pod Security Standard - container level. */}}
{{- define "aiready.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
runAsNonRoot: true
privileged: false
capabilities:
  drop: [ALL]
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Scheduling. Input: dict root pool tolerations selector(map) spread(bool)
pool "" -> global.nodeSelector. Tolerations: global + the ones passed in.
*/}}
{{- define "aiready.scheduling" -}}
{{- $root := .root -}}
{{- if .pool }}
nodeSelector:
  aiready.io/pool: {{ .pool }}
{{- else if $root.Values.global.nodeSelector }}
nodeSelector:
{{- toYaml $root.Values.global.nodeSelector | nindent 2 }}
{{- end }}
{{- $tol := concat ($root.Values.global.tolerations | default list) (.tolerations | default list) }}
{{- if $tol }}
tolerations:
{{- toYaml $tol | nindent 2 }}
{{- end }}
{{- if and $root.Values.global.topologySpread .spread }}
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
{{- toYaml .selector | nindent 8 }}
  - maxSkew: 1
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
{{- toYaml .selector | nindent 8 }}
{{- end }}
{{- end }}

{{/* Tolerations for a named node pool. Input: dict root pool */}}
{{- define "aiready.poolTolerations" -}}
{{- if eq .pool "batch" }}
{{- toYaml .root.Values.batch.tolerations }}
{{- else if hasPrefix "gpu" .pool }}
{{- toYaml .root.Values.modelServing.gpuTolerations }}
{{- else }}
[]
{{- end }}
{{- end }}

{{/* ConfigMap key for a packaged config file path (ConfigMap keys cannot contain "/"). */}}
{{- define "aiready.configKey" -}}
{{ trimPrefix "files/config/" . | replace "/" "__" }}
{{- end }}

{{/*
Volumes for platform pods. Input: dict root secrets(list)
*/}}
{{- define "aiready.platformVolumes" -}}
{{- $root := .root -}}
- name: config
  configMap:
{{- if $root.Values.config.existingConfigMap }}
    name: {{ $root.Values.config.existingConfigMap }}
{{- else }}
    name: aiready-config
    items:
{{- range $path, $_ := $root.Files.Glob "files/config/**" }}
      - key: {{ include "aiready.configKey" $path }}
        path: {{ trimPrefix "files/config/" $path }}
{{- end }}
{{- end }}
- name: overlay
  configMap:
    name: aiready-overlay
- name: tmp
  emptyDir:
    sizeLimit: 1Gi
- name: data
  emptyDir:
    sizeLimit: 5Gi
{{- include "aiready.secretVolume" . }}
{{- end }}

{{/* Secret volume (not used in vault mode: the agent injector writes /run/secrets itself). */}}
{{- define "aiready.secretVolume" -}}
{{- $root := .root -}}
{{- if and .secrets (ne $root.Values.secrets.mode "vault") }}
- name: secrets
  secret:
    secretName: {{ $root.Values.secrets.existingSecret }}
    defaultMode: 0440
    items:
{{- range .secrets }}
      - key: {{ . }}
        path: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{- define "aiready.platformVolumeMounts" -}}
- name: config
  mountPath: /etc/aiready/config
  readOnly: true
- name: overlay
  mountPath: /etc/aiready/overlay
  readOnly: true
- name: tmp
  mountPath: /tmp
- name: data
  mountPath: /var/lib/aiready
{{- if and .secrets (ne .root.Values.secrets.mode "vault") }}
- name: secrets
  mountPath: /run/secrets
  readOnly: true
{{- end }}
{{- end }}

{{/* Vault agent injector annotations. Input: dict root secrets(list) */}}
{{- define "aiready.vaultAnnotations" -}}
{{- $root := .root -}}
{{- if and .secrets (eq $root.Values.secrets.mode "vault") }}
{{- $v := $root.Values.secrets.vault }}
vault.hashicorp.com/agent-inject: "true"
vault.hashicorp.com/role: {{ $v.role | quote }}
vault.hashicorp.com/secret-volume-path: /run/secrets
vault.hashicorp.com/agent-run-as-user: {{ $root.Values.podSecurity.runAsUser | quote }}
vault.hashicorp.com/agent-run-as-group: {{ $root.Values.podSecurity.runAsGroup | quote }}
{{- if $v.preserveSecretCase }}
vault.hashicorp.com/preserve-secret-case: "true"
{{- end }}
{{- range .secrets }}
{{- $path := printf $v.pathTemplate . }}
vault.hashicorp.com/agent-inject-secret-{{ . }}: {{ $path | quote }}
vault.hashicorp.com/agent-inject-template-{{ . }}: {{ printf "{{- with secret %q -}}{{ .Data.data.%s }}{{- end -}}" $path $v.field | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/* Model endpoint URL for a model name ("" when neither set nor served by the chart). Input: dict root name suffix */}}
{{- define "aiready.modelUrl" -}}
{{- $explicit := get .root.Values.modelEndpoints .name -}}
{{- if $explicit -}}
{{ $explicit }}
{{- else if .root.Values.modelServing.enabled -}}
{{ printf "http://model-%s:8000%s" .name .suffix }}
{{- else if (get .root.Values.components "mock-models").enabled -}}
http://mock-models:8200/v1
{{- end -}}
{{- end }}

{{/* Environment shared by platform containers. Input: dict root name c(component) */}}
{{- define "aiready.env" -}}
{{- $root := .root -}}
{{- $c := .c -}}
- name: AIREADY_SERVICE
  value: {{ .name | quote }}
{{- if $c.port }}
- name: AIREADY_PORT
  value: {{ $c.port | quote }}
{{- end }}
- name: AIREADY_ENV
  value: {{ $root.Values.global.environment | quote }}
- name: AIREADY_CONFIG_DIR
  value: /etc/aiready/config
- name: AIREADY_CONFIG_OVERLAY
  value: /etc/aiready/overlay
- name: AIREADY_SECRET_DIRS
  value: /run/secrets
- name: AIREADY_AUDIT_PATH
  value: /var/lib/aiready/audit/audit.jsonl
- name: AIREADY_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ $root.Values.global.otlpEndpoint | quote }}
- name: OTEL_SERVICE_NAME
  value: {{ .name | quote }}
- name: OTEL_RESOURCE_ATTRIBUTES
  value: {{ printf "service.namespace=aiready,deployment.environment=%s" $root.Values.global.environment | quote }}
- name: AIREADY_TRACE_COLLECTOR_URL
  value: {{ $root.Values.global.traceCollectorUrl | quote }}
{{- range $pair := list (list "CHAT" "chat" "/v1") (list "SMALL" "small" "/v1") (list "GUARD" "guard" "/v1") (list "EMBED" "embed" "") (list "RERANK" "rerank" "") }}
{{- $url := include "aiready.modelUrl" (dict "root" $root "name" (index $pair 1) "suffix" (index $pair 2)) }}
{{- if $url }}
- name: AIREADY_MODEL_{{ index $pair 0 }}_URL
  value: {{ $url | quote }}
{{- if eq (index $pair 1) "embed" }}
- name: EMBEDDING_URL
  value: {{ $url | quote }}
{{- else if eq (index $pair 1) "rerank" }}
- name: RERANK_URL
  value: {{ $url | quote }}
{{- end }}
{{- end }}
{{- end }}
{{- with $root.Values.state.stateUrl }}
- name: AIREADY_STATE_URL
  value: {{ . | quote }}
{{- end }}
{{- with $root.Values.state.redisUrl }}
- name: AIREADY_REDIS_URL
  value: {{ . | quote }}
{{- end }}
{{- with $root.Values.state.kvType }}
- name: AIREADY_KV_TYPE
  value: {{ . | quote }}
{{- end }}
{{- with $root.Values.state.queueType }}
- name: AIREADY_QUEUE_TYPE
  value: {{ . | quote }}
{{- end }}
{{- if $root.Values.state.qdrantUrl }}
- name: QDRANT_URL
  value: {{ $root.Values.state.qdrantUrl | quote }}
{{- else if $root.Values.qdrant.enabled }}
- name: QDRANT_URL
  value: "http://qdrant:6333"
{{- end }}
{{- with $root.Values.legacyApp.url }}
- name: LEGACY_APP_URL
  value: {{ . | quote }}
{{- end }}
{{- with $root.Values.legacyApp.replicaUrl }}
- name: LEGACY_REPLICA_URL
  value: {{ . | quote }}
{{- end }}
{{- if and $c.egress $root.Values.egressProxy.enabled }}
- name: HTTPS_PROXY
  value: "http://egress-proxy:3128"
- name: HTTP_PROXY
  value: "http://egress-proxy:3128"
- name: NO_PROXY
  value: ".svc,.svc.cluster.local,.cluster.local,localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
{{- end }}
{{- range $k, $v := $root.Values.global.env }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- range $k, $v := $c.env }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end }}

{{/*
Resolve a component against .Values.defaults: a key the component sets wins as a whole (so
`hpa: {enabled: false}` or `secrets: []` work); unset keys come from defaults. Returns YAML (use fromYaml).
Input: dict root c
*/}}
{{- define "aiready.resolve" -}}
{{- $d := .root.Values.defaults -}}
{{- $out := deepCopy .c -}}
{{- range $k := list "replicas" "resources" "hpa" "pdb" "secrets" "terminationGracePeriodSeconds" }}
{{- if not (hasKey $out $k) }}{{ $_ := set $out $k (get $d $k) }}{{ end }}
{{- end }}
{{- toYaml $out -}}
{{- end }}

{{/* true when the component is scaled by an autoscaler (replicas is then left to it). Input: resolved c + root */}}
{{- define "aiready.autoscaled" -}}
{{- $kedaOn := and .root.Values.keda.enabled .c.keda .c.keda.enabled -}}
{{- if or $kedaOn (and .c.hpa .c.hpa.enabled) }}true{{ end -}}
{{- end }}
