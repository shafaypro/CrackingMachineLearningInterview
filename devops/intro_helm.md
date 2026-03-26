# Helm – Kubernetes Package Manager (2026 Edition)

**Helm** is the package manager for Kubernetes. It lets you define, install, and upgrade complex Kubernetes applications using reusable packages called **charts**.

---

## What is Helm?

Without Helm, deploying an application on Kubernetes means managing dozens of YAML files (Deployments, Services, ConfigMaps, Secrets, Ingress, HPAs...). Helm bundles all of these into a single **chart** with templating and versioning.

```
Helm Chart  →  helm install  →  Release (running app in K8s)
```

| Helm Concept | Kubernetes Analogy |
|--------------|-------------------|
| Chart | Package (like a Homebrew formula) |
| Release | Installed instance of a chart |
| Repository | Package registry (like npm/PyPI) |
| Values | Configuration for a release |

---

## Installation

```bash
# macOS
brew install helm

# Linux
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

---

## Core Workflow

```bash
# Add a chart repository
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update   # refresh repo index

# Search for charts
helm search repo postgres
helm search hub redis    # search Artifact Hub

# Install a chart (creates a release)
helm install my-postgres bitnami/postgresql \
  --namespace db \
  --create-namespace \
  --set auth.postgresPassword=secret

# Install with custom values file
helm install my-app ./my-chart -f values-production.yaml

# List releases
helm list
helm list -A   # all namespaces

# Upgrade a release
helm upgrade my-postgres bitnami/postgresql \
  --set auth.postgresPassword=newsecret

# Upgrade or install (idempotent)
helm upgrade --install my-app ./my-chart -f values.yaml

# Rollback to previous release
helm rollback my-postgres 1

# Uninstall
helm uninstall my-postgres

# Release history
helm history my-postgres
```

---

## Chart Structure

```
my-chart/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default values
├── charts/                 # Chart dependencies
├── templates/              # Kubernetes manifests (templated)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── _helpers.tpl        # Named template helpers
│   └── NOTES.txt           # Post-install instructions
└── .helmignore
```

### Chart.yaml

```yaml
apiVersion: v2
name: my-app
description: My application Helm chart
type: application
version: 1.2.0         # Chart version
appVersion: "2.5.1"   # App version (Docker image tag)

dependencies:
  - name: postgresql
    version: "~12.0"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
```

### values.yaml

```yaml
# Default configuration
replicaCount: 2

image:
  repository: myrepo/my-app
  tag: ""   # defaults to Chart.appVersion
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false
  className: "nginx"
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    postgresPassword: ""   # override in production

env: {}
  # KEY: value
```

---

## Templates

Templates are Kubernetes YAML + **Go templating**.

### Deployment Template

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
          {{- if .Values.env }}
          env:
            {{- range $key, $val := .Values.env }}
            - name: {{ $key }}
              value: {{ $val | quote }}
            {{- end }}
          {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

### _helpers.tpl (Named Templates)

```yaml
# templates/_helpers.tpl
{{/*
Expand the name of the chart.
*/}}
{{- define "my-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "my-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "my-app.labels" -}}
helm.sh/chart: {{ include "my-app.chart" . }}
{{ include "my-app.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
```

### Template Syntax Reference

```yaml
# Value access
{{ .Values.image.repository }}
{{ .Chart.Name }}
{{ .Release.Name }}
{{ .Release.Namespace }}

# Conditionals
{{- if .Values.ingress.enabled }}
...
{{- end }}

{{- if eq .Values.environment "production" }}
  replicas: 5
{{- else }}
  replicas: 1
{{- end }}

# Loops
{{- range .Values.ingress.hosts }}
- host: {{ .host }}
{{- end }}

{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}

# String functions
{{ "hello" | upper }}           → HELLO
{{ .Values.name | trunc 63 }}   → truncate
{{ .Values.name | trimSuffix "-" }}
{{ .Values.name | replace "-" "_" }}
{{ .Values.config | toYaml | nindent 4 }}
{{ .Values.password | b64enc }}
{{ .Values.password | quote }}
{{ printf "%s-%s" .Release.Name .Chart.Name }}

# Default values
{{ .Values.image.tag | default "latest" }}
{{ .Values.replicas | default 1 }}

# Whitespace control (- trims whitespace)
{{- /* trim leading whitespace */ -}}
```

---

## Values Override Strategies

```bash
# Override individual values
helm install my-app ./chart --set replicaCount=3
helm install my-app ./chart --set image.tag=2.0 --set ingress.enabled=true

# Override with values file
helm install my-app ./chart -f values-production.yaml

# Multiple values files (merged in order, last wins)
helm install my-app ./chart \
  -f values.yaml \
  -f values-production.yaml \
  -f secrets.yaml

# View effective values
helm get values my-app
helm get values my-app --all   # including defaults
```

### Environment-specific Values

```yaml
# values-production.yaml
replicaCount: 5

image:
  tag: "1.5.2"

ingress:
  enabled: true
  hosts:
    - host: api.myapp.com

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
```

---

## Debugging & Testing

```bash
# Render templates without installing (dry run)
helm template my-app ./chart -f values.yaml

# Dry run with server validation
helm install my-app ./chart --dry-run

# Debug rendered templates
helm template my-app ./chart --debug

# Validate chart
helm lint ./chart
helm lint ./chart -f values-production.yaml

# Inspect installed release
helm get all my-app
helm get manifest my-app
helm get notes my-app

# Test release
helm test my-app
```

---

## OCI Registries (2026 Standard)

Helm 3.8+ supports storing charts in **OCI registries** (like Docker Hub, GHCR, ECR).

```bash
# Push chart to OCI registry
helm package ./my-chart
helm push my-chart-1.0.0.tgz oci://ghcr.io/myorg/charts

# Pull and install from OCI
helm install my-app oci://ghcr.io/myorg/charts/my-chart --version 1.0.0

# Login to OCI registry
helm registry login ghcr.io -u $GITHUB_USER -p $GITHUB_TOKEN
```

---

## Popular Charts in 2026

```bash
# Ingress NGINX
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# cert-manager (TLS automation)
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# Prometheus + Grafana
helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# ArgoCD
helm upgrade --install argocd argo/argo-cd \
  --namespace argocd --create-namespace

# External Secrets Operator
helm upgrade --install external-secrets \
  external-secrets/external-secrets \
  --namespace external-secrets --create-namespace

# KEDA (event-driven autoscaling)
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace
```

---

## Cheat Sheet

```bash
# Repository
helm repo add name url
helm repo update
helm repo list
helm search repo term

# Install / Upgrade
helm install release chart [-f vals] [--set k=v]
helm upgrade release chart
helm upgrade --install release chart   # upsert
helm uninstall release [-n namespace]

# Inspect
helm list [-A]
helm status release
helm get values release [--all]
helm get manifest release
helm history release

# Rollback
helm rollback release [revision]

# Templates
helm template release chart [-f vals]
helm lint chart [-f vals]

# Package & Push (OCI)
helm package ./chart
helm push chart.tgz oci://registry/path
```
