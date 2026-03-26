# Kubernetes – Complete Guide (2026 Edition)

Kubernetes (K8s) is the de-facto standard for **container orchestration** — automating deployment, scaling, self-healing, and management of containerized applications at scale.

---

## Table of Contents
1. [What is Kubernetes?](#what-is-kubernetes)
2. [Architecture](#architecture)
3. [Core Objects](#core-objects)
4. [kubectl Basics](#kubectl-basics)
5. [Workloads](#workloads)
6. [Services & Networking](#services--networking)
7. [Storage](#storage)
8. [Configuration](#configuration)
9. [Namespaces & RBAC](#namespaces--rbac)
10. [Ingress](#ingress)
11. [Helm](#helm)
12. [Scaling & Auto-scaling](#scaling--auto-scaling)
13. [Observability](#observability)
14. [Kubernetes in 2026](#kubernetes-in-2026)
15. [Cheat Sheet](#cheat-sheet)

---

## What is Kubernetes?

Kubernetes solves the problem of running containers in production:
- **Where** should each container run?
- **How many** replicas should run?
- What happens when a container **crashes**?
- How do containers **find each other**?
- How do you **roll out** a new version without downtime?

Kubernetes answers all of these automatically.

```
You declare desired state  →  Kubernetes makes it real  →  Kubernetes keeps it that way
```

### When to Use Kubernetes (vs Docker Compose)

| Scenario | Use |
|----------|-----|
| Local dev, small team, single host | Docker Compose |
| Production, multi-host, scale, HA | Kubernetes |
| Managed cloud service | EKS (AWS), GKE (GCP), AKS (Azure) |

---

## Architecture

### Control Plane (Master)

```
┌─────────────────────────────────────────────────────────┐
│                     Control Plane                        │
│  ┌──────────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │  API Server  │  │   etcd   │  │  Controller Manager│ │
│  │  (kube-api)  │  │(key-val  │  │ (reconciliation    │ │
│  │              │  │  store)  │  │  loops)            │ │
│  └──────────────┘  └──────────┘  └────────────────────┘ │
│  ┌──────────────┐                                        │
│  │  Scheduler   │  (decides which node runs each pod)    │
│  └──────────────┘                                        │
└─────────────────────────────────────────────────────────┘

┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Worker Node  │  │   Worker Node  │  │   Worker Node  │
│  ┌──────────┐ │  │  ┌──────────┐ │  │  ┌──────────┐ │
│  │ kubelet  │ │  │  │ kubelet  │ │  │  │ kubelet  │ │
│  │ kube-    │ │  │  │ kube-    │ │  │  │ kube-    │ │
│  │ proxy    │ │  │  │ proxy    │ │  │  │ proxy    │ │
│  │ Pods...  │ │  │  │ Pods...  │ │  │  │ Pods...  │ │
│  └──────────┘ │  │  └──────────┘ │  │  └──────────┘ │
└───────────────┘  └───────────────┘  └───────────────┘
```

| Component | Role |
|-----------|------|
| **API Server** | Front door of K8s. All communication goes through it. |
| **etcd** | Distributed key-value store. Stores all cluster state. |
| **Controller Manager** | Runs control loops (ensure desired state == actual state) |
| **Scheduler** | Assigns pods to nodes based on resources, affinity rules |
| **kubelet** | Agent on each node. Ensures containers run as specified. |
| **kube-proxy** | Manages network rules on each node (Service routing) |
| **Container Runtime** | Runs containers (containerd, CRI-O) |

---

## Core Objects

Everything in Kubernetes is an **object** with a `kind`, `apiVersion`, `metadata`, and `spec`.

```yaml
apiVersion: apps/v1       # API group and version
kind: Deployment          # Object type
metadata:
  name: my-app            # Name of this object
  namespace: production   # Namespace (logical isolation)
  labels:                 # Key-value tags
    app: my-app
    version: "1.0"
spec:                     # Desired state
  ...
```

### The Core Objects

| Object | Purpose |
|--------|---------|
| **Pod** | Smallest deployable unit. One or more containers. |
| **Deployment** | Manages Pods: rolling updates, rollback, replicas |
| **Service** | Stable network endpoint for a group of Pods |
| **ConfigMap** | Store non-secret config data |
| **Secret** | Store sensitive data (passwords, tokens, keys) |
| **PersistentVolume** | Storage resource in the cluster |
| **Namespace** | Virtual cluster for isolation |
| **Ingress** | HTTP/HTTPS routing rules into the cluster |
| **ServiceAccount** | Identity for processes running in Pods |
| **HorizontalPodAutoscaler** | Auto-scale Pods based on CPU/memory |

---

## kubectl Basics

`kubectl` is the CLI for interacting with Kubernetes.

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes
kubectl get nodes -o wide

# Context management
kubectl config get-contexts
kubectl config use-context my-cluster
kubectl config current-context

# Get resources
kubectl get pods                        # in current namespace
kubectl get pods -n kube-system        # specific namespace
kubectl get pods -A                    # all namespaces
kubectl get pods -l app=my-app         # by label
kubectl get all                        # everything in namespace

# Describe resource (events, details)
kubectl describe pod my-pod
kubectl describe deployment my-app

# Logs
kubectl logs my-pod
kubectl logs -f my-pod                 # follow
kubectl logs my-pod -c container-name  # specific container
kubectl logs -l app=my-app --tail=100  # by label

# Execute command
kubectl exec -it my-pod -- bash
kubectl exec my-pod -- env

# Apply/delete manifests
kubectl apply -f deployment.yaml
kubectl apply -f ./manifests/          # all files in dir
kubectl delete -f deployment.yaml
kubectl delete pod my-pod

# Edit live resource
kubectl edit deployment my-app

# Port forward to local machine
kubectl port-forward pod/my-pod 8080:80
kubectl port-forward svc/my-service 8080:80

# Copy files
kubectl cp my-pod:/app/logs/app.log ./app.log
```

---

## Workloads

### Pod

The smallest unit. Almost never created directly — use Deployments instead.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: app
      image: nginx:1.25
      ports:
        - containerPort: 80
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
```

### Deployment

Manages a **ReplicaSet** of Pods. Handles rolling updates and rollbacks.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # extra pods during update
      maxUnavailable: 0   # zero downtime
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: myrepo/my-app:1.0
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
```

### Deployment Commands

```bash
# Rollout status
kubectl rollout status deployment/my-app

# Rollout history
kubectl rollout history deployment/my-app

# Rollback to previous version
kubectl rollout undo deployment/my-app

# Rollback to specific revision
kubectl rollout undo deployment/my-app --to-revision=2

# Update image
kubectl set image deployment/my-app app=myrepo/my-app:2.0

# Scale
kubectl scale deployment my-app --replicas=5
```

### StatefulSet

For **stateful applications** (databases, Kafka, Zookeeper) that need stable network identities and persistent storage.

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: "postgres"
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
          volumeMounts:
            - name: pgdata
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:               # Each pod gets its own PVC
    - metadata:
        name: pgdata
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

StatefulSet pods get stable names: `postgres-0`, `postgres-1`, `postgres-2`.

### DaemonSet

Runs **one pod per node**. Used for logging agents, monitoring, network plugins.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    metadata:
      labels:
        name: fluentd
    spec:
      containers:
        - name: fluentd
          image: fluentd:v1.16
```

### Job & CronJob

```yaml
# One-off Job
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  template:
    spec:
      containers:
        - name: migration
          image: myapp:latest
          command: ["python", "manage.py", "migrate"]
      restartPolicy: OnFailure

---
# Scheduled CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-job
spec:
  schedule: "0 2 * * *"   # 2am daily (cron syntax)
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: cleanup
              image: myapp:latest
              command: ["python", "cleanup.py"]
          restartPolicy: OnFailure
```

---

## Services & Networking

A **Service** provides a stable IP and DNS name for a set of Pods (which come and go).

### Service Types

```yaml
# ClusterIP (default) — internal only
apiVersion: v1
kind: Service
metadata:
  name: my-app-svc
spec:
  selector:
    app: my-app
  ports:
    - port: 80          # Service port
      targetPort: 8000  # Pod port
  type: ClusterIP

---
# NodePort — accessible on each node's IP
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080   # 30000-32767

---
# LoadBalancer — provisions cloud load balancer
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8000
```

### DNS in Kubernetes

Every Service gets a DNS name: `<service>.<namespace>.svc.cluster.local`

```bash
# From within the cluster:
curl http://my-app-svc                              # same namespace
curl http://my-app-svc.production                  # cross-namespace
curl http://my-app-svc.production.svc.cluster.local
```

---

## Storage

### PersistentVolume (PV) & PersistentVolumeClaim (PVC)

```yaml
# StorageClass (dynamic provisioning)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
reclaimPolicy: Retain

---
# PersistentVolumeClaim — request storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-data
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi

---
# Use PVC in a Pod
spec:
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: my-data
  containers:
    - name: app
      volumeMounts:
        - mountPath: /data
          name: data
```

---

## Configuration

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"
  config.yaml: |
    database:
      host: postgres
      port: 5432
```

```yaml
# Use ConfigMap as environment variables
envFrom:
  - configMapRef:
      name: app-config

# Use specific key
env:
  - name: LOG_LEVEL
    valueFrom:
      configMapKeyRef:
        name: app-config
        key: LOG_LEVEL

# Mount as file
volumes:
  - name: config
    configMap:
      name: app-config
containers:
  - volumeMounts:
      - name: config
        mountPath: /etc/config
```

### Secret

```bash
# Create secret from command line
kubectl create secret generic db-secret \
  --from-literal=password=mysecret \
  --from-literal=username=admin

# Create from file
kubectl create secret generic tls-secret \
  --from-file=tls.crt=./cert.crt \
  --from-file=tls.key=./cert.key
```

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  password: bXlzZWNyZXQ=   # base64 encoded
  username: YWRtaW4=
```

> **Note:** Kubernetes Secrets are base64-encoded, not encrypted by default. Use **Sealed Secrets**, **External Secrets Operator**, or **Vault** for production.

---

## Namespaces & RBAC

### Namespaces

```bash
# Create namespace
kubectl create namespace staging

# Set default namespace
kubectl config set-context --current --namespace=staging

# Resource quotas per namespace
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: staging-quota
  namespace: staging
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    pods: "20"
EOF
```

### RBAC

```yaml
# Role (namespace-scoped)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: staging
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: staging
subjects:
  - kind: ServiceAccount
    name: ci-runner
    namespace: staging
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

---

## Ingress

Ingress routes external HTTP(S) traffic to internal Services.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.myapp.com
      secretName: api-tls
  rules:
    - host: api.myapp.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-svc
                port:
                  number: 80
```

```bash
# Install NGINX Ingress Controller
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace
```

---

## Scaling & Auto-scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: AverageValue
          averageValue: 400Mi
```

### Vertical Pod Autoscaler (VPA)

Automatically adjusts CPU/memory requests based on actual usage.

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: my-app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  updatePolicy:
    updateMode: "Auto"
```

### Cluster Autoscaler

Scales the **number of nodes** in the cluster based on pending pods.

---

## Observability

### Health Probes

```yaml
containers:
  - name: app
    livenessProbe:       # Is the container alive? (restart if fails)
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 15
      failureThreshold: 3
    readinessProbe:      # Is the container ready to receive traffic?
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
    startupProbe:        # Is the app done starting up?
      httpGet:
        path: /healthz
        port: 8080
      failureThreshold: 30
      periodSeconds: 10
```

### Metrics Stack (Prometheus + Grafana)

```bash
# Install kube-prometheus-stack via Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

### Logging Stack (EFK / Loki)

```bash
# Loki + Grafana (lightweight)
helm repo add grafana https://grafana.github.io/helm-charts
helm upgrade --install loki grafana/loki-stack \
  --namespace monitoring --set grafana.enabled=true
```

---

## Kubernetes in 2026

### Key Trends

| Trend | Description |
|-------|-------------|
| **Gateway API** | Successor to Ingress. More expressive routing. Becoming standard. |
| **WebAssembly (Wasm)** | Wasm workloads running alongside containers via `runwasi` |
| **Karpenter** | Node auto-provisioner (replaces Cluster Autoscaler on AWS) |
| **Cilium** | eBPF-powered networking, replacing kube-proxy in many clusters |
| **ArgoCD / Flux** | GitOps — declarative continuous delivery for K8s |
| **Crossplane** | Kubernetes-native infrastructure provisioning (K8s for cloud resources) |
| **KEDA** | Event-driven autoscaling (scale on Kafka lag, queue depth, etc.) |
| **OpenTelemetry** | Standard for traces, metrics, logs across the cluster |
| **vCluster** | Virtual Kubernetes clusters inside a real cluster |

### Managed Kubernetes in 2026

| Cloud | Service | Notable Feature |
|-------|---------|-----------------|
| AWS | EKS | EKS Auto Mode (fully managed node groups) |
| GCP | GKE | Autopilot mode (serverless K8s) |
| Azure | AKS | Node auto-provisioner GA |
| All | Fargate/Cloud Run | Serverless containers (no node management) |

---

## Cheat Sheet

```bash
# === CLUSTER ===
kubectl cluster-info
kubectl get nodes -o wide
kubectl top nodes / kubectl top pods

# === PODS ===
kubectl get pods [-n ns] [-A] [-l label=val]
kubectl describe pod <name>
kubectl logs -f <pod> [-c container]
kubectl exec -it <pod> -- bash
kubectl port-forward pod/<pod> local:remote

# === DEPLOYMENTS ===
kubectl get deploy
kubectl rollout status deploy/<name>
kubectl rollout history deploy/<name>
kubectl rollout undo deploy/<name>
kubectl set image deploy/<name> container=image:tag
kubectl scale deploy/<name> --replicas=N

# === SERVICES ===
kubectl get svc
kubectl expose deploy <name> --port=80 --target-port=8000 --type=LoadBalancer

# === APPLY/DELETE ===
kubectl apply -f file.yaml
kubectl apply -f directory/
kubectl delete -f file.yaml
kubectl delete pod <name> --grace-period=0 --force

# === DEBUGGING ===
kubectl get events --sort-by=.lastTimestamp
kubectl describe node <node>
kubectl run debug --image=busybox --rm -it -- sh
kubectl debug pod/<pod> -it --image=busybox

# === SECRETS/CONFIGS ===
kubectl create secret generic name --from-literal=key=val
kubectl create configmap name --from-file=./config
kubectl get secret name -o jsonpath='{.data.key}' | base64 -d

# === NAMESPACES ===
kubectl get ns
kubectl create ns <name>
kubectl config set-context --current --namespace=<name>

# === CLEANUP ===
kubectl delete pods --field-selector=status.phase=Failed -A
kubectl delete evicted pods -A
```
