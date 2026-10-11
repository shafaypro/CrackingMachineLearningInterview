# Google Cloud (GCP) for ML Engineers

Vertex AI gets most of the attention in GCP interviews, but the questions that separate candidates usually sit around it: who is allowed to read the training data, how a private training job reaches Cloud Storage without a public IP, why the BigQuery bill spiked, and whether a model belongs on Cloud Run, GKE or a Vertex AI endpoint. This guide covers the core GCP services an ML or AI engineer touches outside Vertex AI itself and how they fit together.

For Vertex AI components (Pipelines, Custom Training, Model Registry, Endpoints, Feature Store) see [Google Vertex AI Interview Guide](./intro_vertex_ai.md). For a SageMaker vs Vertex AI vs Azure ML comparison see [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md).

> GCP product names change often, especially in data governance and generative AI. Where a name is known to have changed recently, this guide says so. Always check the current Google Cloud documentation and pricing pages before quoting limits or prices in a design.

---

## Table of Contents

1. [How the pieces fit together](#how-the-pieces-fit-together)
2. [Resource hierarchy and identity](#resource-hierarchy-and-identity)
3. [Networking basics](#networking-basics)
4. [Storage](#storage)
5. [Compute](#compute)
6. [Data and analytics](#data-and-analytics)
7. [Generative AI on GCP](#generative-ai-on-gcp)
8. [Observability](#observability)
9. [Cost control](#cost-control)
10. [Reference architectures](#reference-architectures)
11. [Code examples](#code-examples)
12. [Interview Q&A](#interview-qa)
13. [Common Pitfalls](#common-pitfalls)
14. [Related Topics](#related-topics)

---

## How the pieces fit together

| Layer | Main GCP services | What an ML engineer uses it for |
|---|---|---|
| Identity and security | IAM, service accounts, Workload Identity Federation, Cloud KMS, Secret Manager, VPC Service Controls, Cloud Audit Logs | Least-privilege access to data and models, keyless auth, exfiltration control |
| Networking | VPC, firewall rules, Private Google Access, Cloud NAT, Private Service Connect | Private training and serving, controlled egress |
| Storage | Cloud Storage, Persistent Disk / Hyperdisk, Filestore, Parallelstore, Cloud Storage FUSE | Datasets, checkpoints, model artifacts |
| Compute | Compute Engine (GPUs), Cloud TPU, GKE, Cloud Run, Cloud Run functions, Batch | Training, serving, glue code, batch jobs |
| Data and analytics | BigQuery, BigQuery ML, Dataflow, Dataproc, Pub/Sub, Cloud Composer, Workflows, Dataplex | Feature engineering, streaming, orchestration, governance |
| ML platform | Vertex AI (see [intro_vertex_ai.md](./intro_vertex_ai.md)) | Pipelines, training, registry, endpoints, monitoring |
| Generative AI | Gemini and partner models on Vertex AI, Model Garden, grounding, Vertex AI Search / RAG Engine, embeddings | LLM apps, RAG, semantic search |
| Operations | Cloud Monitoring, Cloud Logging, Cloud Trace, Billing budgets | Health, debugging, cost |
| Build and deploy | Artifact Registry, Cloud Build | Container images for training and serving |

A useful mental model: **BigQuery and Cloud Storage are the data plane, Vertex AI is the ML control plane, and IAM plus VPC Service Controls decide who and what can cross the boundary.**

---

## Resource hierarchy and identity

### Resource hierarchy

```text
Organization (example.com)
 |-- Folder: ml-platform
 |    |-- Project: ml-dev
 |    |-- Project: ml-staging
 |    `-- Project: ml-prod
 `-- Folder: data
      `-- Project: data-lake-prod
```

- **Organization**: the root node, tied to a Google Workspace or Cloud Identity domain. Organization policies (for example "disable service account key creation" or "restrict resource locations") are set here or on folders.
- **Folders**: group projects by team or environment. Policies and IAM bindings inherit downward.
- **Projects**: the unit of billing, API enablement, quotas and most IAM. Separate projects per environment is the standard way to isolate dev from prod.

IAM allow policies are **additive down the tree**: a role granted on a folder applies to every project beneath it, and a child cannot remove what a parent grants. IAM deny policies exist for explicit guardrails, but most day-to-day access is modeled with allow policies.

### IAM roles

| Role type | Examples | When to use |
|---|---|---|
| Basic | Owner, Editor, Viewer | Avoid in production. Far too broad (Editor can modify most resources in the project). |
| Predefined | `roles/bigquery.dataViewer`, `roles/storage.objectViewer`, `roles/aiplatform.user` | Default choice. Maintained by Google as new permissions are added. |
| Custom | Your own list of permissions | When predefined roles are too broad. You own the maintenance. |

IAM Conditions let you scope a binding further (for example to a bucket prefix by resource name, or to a time window).

### Service accounts and keys

A **service account** is the identity for workloads: a training job, a Cloud Run service, a Dataflow worker. Good practice:

- One service account per workload, with only the roles it needs (training reads `gs://features/*`, does not write to prod endpoints).
- **Attach** the service account to the resource (VM, Cloud Run service, Vertex AI custom job) so code gets short-lived tokens through Application Default Credentials.
- **Avoid service account keys.** A JSON key is a long-lived credential that leaks through laptops, CI logs and git history. Many organizations enforce the `iam.disableServiceAccountKeyCreation` organization policy (Google has been enabling secure-by-default policies like this on new organizations; check yours).
- Use **service account impersonation** for humans who need to act as a workload identity temporarily.

### Keyless identity outside and inside GCP

| Mechanism | Problem it solves | How it works |
|---|---|---|
| **Workload Identity Federation** | CI/CD (GitHub Actions, GitLab), AWS or Azure workloads, on-prem jobs need GCP access | An external OIDC/SAML token (or AWS/Azure credential) is exchanged through Google's Security Token Service for a short-lived GCP token. Either grant roles directly to the federated principal or let it impersonate a service account. |
| **Workload Identity Federation for GKE** (previously "GKE Workload Identity") | Pods need GCP access without node-level credentials | A Kubernetes service account maps to an IAM principal, so each pod gets only its own permissions instead of inheriting the node's service account. |

Always restrict a federation provider with an **attribute condition** (for example a specific GitHub repository and branch), otherwise any token from that issuer could be accepted.

### Encryption, secrets and perimeter controls

| Service | What it does | ML relevance |
|---|---|---|
| **Cloud KMS** | Manages encryption keys. Data is encrypted at rest by default with Google-managed keys; KMS adds customer-managed encryption keys (CMEK), with Cloud HSM and external key options. | Regulated datasets, model artifacts, and Vertex AI resources that must use CMEK. Revoking the key makes the data unreadable. |
| **Secret Manager** | Versioned storage for API keys, DB passwords, third-party tokens, with IAM per secret. | Third-party API keys for LLM providers, feature store credentials. Pin versions in prod, reference `latest` only where rotation is automatic. |
| **VPC Service Controls (VPC-SC)** | A service perimeter around projects that blocks Google API calls (Cloud Storage, BigQuery, Vertex AI, and others) crossing the perimeter, even with valid credentials. | Prevents copying training data to a personal bucket or a project outside the perimeter. Supports access levels, ingress/egress rules and a dry-run mode. |
| **Cloud Audit Logs** | Admin Activity logs (always on), Data Access logs (mostly off by default and must be enabled; BigQuery is an exception), System Event and Policy Denied logs. | Who read the PII table, who deployed a model, which calls VPC-SC blocked. |

**What interviewers listen for**

- Projects per environment, IAM granted to groups and service accounts, not to individual users.
- Predefined roles over basic roles; least privilege per workload.
- "No service account keys": attached service accounts inside GCP, Workload Identity Federation outside it.
- Understanding that IAM answers "who can call this API" while VPC-SC answers "from where, and to where can data move".
- Data Access audit logs must be explicitly enabled for most services, and they can be voluminous.

---

## Networking basics

| Concept | Key fact | Why ML engineers care |
|---|---|---|
| **VPC network** | Global resource; one VPC spans all regions. | Training in one region and serving in another can share one network. |
| **Subnets** | Regional, each with its own IP range. | GKE node and pod ranges must be planned up front; running out of IPs blocks scale-up. |
| **Firewall rules** | Stateful; apply by network tags or service accounts. Implied rules deny ingress and allow egress unless you add rules. Hierarchical and network firewall policies add org-wide control. | Restrict who can reach notebooks, inference servers and multi-node training ports. |
| **Private Google Access** | Lets VMs with only internal IPs reach Google APIs (Cloud Storage, BigQuery, Artifact Registry). Enabled per subnet. | Training VMs without external IPs can still read data and pull images. |
| **Cloud NAT** | Managed outbound NAT for resources without external IPs. Outbound only. | Pulling pip packages or Hugging Face weights from a private VM or GKE cluster. |
| **Private Service Connect (PSC)** | Private endpoints inside your VPC for Google APIs or for producer services in other VPCs. | Private Vertex AI endpoints, private access to managed services, consumer/producer separation. |
| **Shared VPC** | A host project owns the network; service projects attach to it. | Central network team, many ML projects. |

Serverless products (Cloud Run, Cloud Run functions) reach a VPC through Direct VPC egress or Serverless VPC Access connectors.

**What interviewers listen for**

- "No public IPs on training or serving nodes" combined with Private Google Access for Google APIs and Cloud NAT for the public internet.
- Knowing that Private Google Access is about Google APIs, while Cloud NAT is about everything else on the internet.
- PSC as the way to consume managed services privately, and VPC-SC as a separate, API-level control (network privacy is not exfiltration protection).

---

## Storage

### Cloud Storage

Cloud Storage (GCS) is the default home for raw data, training shards, checkpoints and model artifacts.

| Storage class | Minimum storage duration | Typical ML use |
|---|---|---|
| Standard | None | Active training data, checkpoints, serving artifacts |
| Nearline | 30 days | Data accessed roughly monthly |
| Coldline | 90 days | Old experiment outputs, quarterly audits |
| Archive | 365 days | Regulatory retention of raw data and model lineage |

Colder classes have lower storage cost but retrieval charges and minimum durations (early deletion is billed as if stored for the minimum). **Autoclass** can move objects between classes automatically based on access. Check current pricing for exact numbers.

Key behaviors:

- **Strong global consistency**: read-after-write, read-after-metadata-update, read-after-delete and object listing are strongly consistent. A training job that lists a prefix right after a writer finishes will see the new objects.
- **Location types**: regional, dual-region, multi-region. Keep training data in the same region as the compute that reads it to avoid egress cost and latency.
- **Lifecycle rules**: delete or change class by age, version count or other conditions (for example delete intermediate checkpoints after 14 days, keep the final model).
- **Object versioning**: keeps noncurrent versions on overwrite or delete. Pair it with lifecycle rules, otherwise old versions accumulate cost silently. Soft delete adds a retention window for deleted objects (enabled by default on newer buckets; check the current default).
- **Uniform bucket-level access**: use IAM only (no per-object ACLs). Recommended.
- **Signed URLs**: time-limited access for a client without GCP credentials (see [code example](#cloud-storage-v4-signed-url)).

### File and block storage for training data

| Option | What it is | Good for | Watch out for |
|---|---|---|---|
| **Persistent Disk / Hyperdisk** | Network block storage attached to VMs. Hyperdisk is the newer family with independently provisioned performance; a Hyperdisk ML variant targets read-heavy model and data loading across many VMs. | Boot disks, local datasets on a single node, fast scratch | Mostly single-writer; sharing across many nodes is limited |
| **Local SSD** | Physically attached, very fast, ephemeral | Data caching and scratch during training | Data is lost when the VM stops |
| **Filestore** | Managed NFS | Shared home directories, small to medium shared datasets, legacy code expecting POSIX | Throughput scales with tier and capacity; cost |
| **Parallelstore / Managed Lustre** | Managed parallel file systems for HPC and AI | Very high throughput, many small files, frequent checkpointing at large scale | Newer products; check regional availability, tiers and naming |
| **Cloud Storage FUSE** | Mounts a GCS bucket as a file system (Vertex AI training exposes buckets under `/gcs/`; GKE has a CSI driver) | Reusing file-based data loaders on GCS without copying | Not fully POSIX; enable its caching options for repeated epochs; many tiny files hurt throughput |

A common pattern is to **shard data into large files** (TFRecord, WebDataset tar shards, Parquet) in GCS and stream them, which usually beats both FUSE on millions of tiny files and copying everything to disk.

**What interviewers listen for**

- Co-locating data and compute in the same region.
- Lifecycle rules for checkpoints, versioning paired with cleanup.
- Matching storage to the access pattern: object storage for durable data, parallel or cached file systems when the loader is I/O bound.
- Awareness that GCS is strongly consistent (an older S3-style eventual consistency assumption is a red flag).

---

## Compute

| Service | What it is | Use it for | Avoid when |
|---|---|---|---|
| **Compute Engine + GPUs** | VMs with attached NVIDIA GPUs (accelerator-optimized machine families) | Custom training clusters, full control over drivers and OS | You want managed scheduling and do not want to run VMs |
| **Cloud TPU** | Google-designed accelerators accessed as TPU VMs or slices | Large dense training and inference with JAX or PyTorch/XLA | Your model relies on custom CUDA kernels or GPU-only libraries |
| **GKE Standard** | Managed Kubernetes, you manage node pools | Multi-tenant platforms, custom serving stacks (vLLM, Triton, KServe), GPU and TPU node pools | Small team with no Kubernetes expertise |
| **GKE Autopilot** | Google manages nodes, you pay per pod resource requests | Kubernetes without node operations; supports GPU and TPU workloads through compute classes | You need node-level customization or privileged daemons |
| **Cloud Run** | Serverless containers, request-based autoscaling including scale to zero; also Cloud Run jobs for run-to-completion tasks | Stateless APIs, lightweight model servers, LLM gateways | Long-lived stateful workloads, complex multi-container topologies |
| **Cloud Run functions** (formerly Cloud Functions) | Event-driven functions on Cloud Run infrastructure | Glue: GCS upload triggers, Pub/Sub handlers | Heavy compute or large models |
| **Batch** | Managed batch job scheduler on Compute Engine VMs (supports GPUs and Spot VMs) | Embarrassingly parallel preprocessing, offline inference, simulations | You already run a cluster scheduler such as GKE or Slurm |

### GPUs

GPU-backed VMs come from accelerator-optimized machine families, and the available NVIDIA models, counts per VM and regions change frequently. Check current docs and quotas before committing to a design; GPU quota is per region and often the real blocker. For multi-node training, care about high-bandwidth networking between nodes, placement, and whether your framework uses NCCL efficiently. Obtaining large GPU or TPU capacity can require reservations or Google's scheduling options for capacity (for example Dynamic Workload Scheduler; check current offerings).

### TPUs

TPUs are Google-designed accelerators built around large matrix multiply units and compiled through XLA.

- **Good at**: large, dense, regular tensor workloads (transformers, large embedding and recommendation models) at scale, especially with JAX; PyTorch is supported through PyTorch/XLA.
- **TPU VMs**: you SSH into the host VM attached to the TPU chips and run your code there directly.
- **Slices**: groups of TPU chips connected by a high-speed interconnect, used as one training unit. Multiple slices can be combined for larger jobs.
- **Tradeoffs**: static shapes and XLA compilation matter (dynamic shapes trigger recompiles), custom CUDA kernels do not port, and debugging tooling differs from the GPU world. Generation-specific specs differ; look them up rather than quoting from memory.

### Pricing levers (no numbers here, check current pricing)

| Lever | What it is | Fit |
|---|---|---|
| **Spot VMs** | Spare capacity at a large discount; can be preempted at any time with a short notice | Checkpointed training, batch inference, preprocessing. Not for latency-critical serving without fallback. |
| **Committed use discounts (CUDs)** | Discount for committing to a level of usage or spend for one or three years | Steady baseline: always-on serving, recurring training |
| **Sustained use discounts** | Automatic discount for some machine types that run much of the month | Long-running VMs on eligible families |
| **Reservations** | Guaranteed capacity in a zone | Scarce accelerators for a planned training run |

Spot VMs replaced legacy preemptible VMs and, unlike preemptible VMs, do not have a fixed maximum runtime.

### Cloud Run with GPUs

Cloud Run supports attaching NVIDIA GPUs to services (NVIDIA L4 was the first supported type). This makes it a reasonable option for small to medium open models and embedding servers that benefit from scale to zero. Check current GPU types, regions, quotas and cold-start behavior; loading multi-gigabyte weights on each cold start is the main practical constraint.

**What interviewers listen for**

- Choosing the least operationally expensive option that meets the requirements (Cloud Run before GKE before raw VMs, unless there is a reason).
- Spot VMs paired with checkpointing; CUDs only for proven steady-state usage.
- A clear TPU vs GPU rationale tied to framework, model shape and kernel dependencies, not brand preference.
- Mentioning GPU quotas and regional capacity as a real constraint.

---

## Data and analytics

### BigQuery

BigQuery is a serverless data warehouse that **separates storage from compute**: data sits in columnar managed storage, queries run on a shared pool of compute units called **slots**.

| Topic | Key points |
|---|---|
| Pricing models (general terms) | **On-demand**: billed by bytes processed per query. **Capacity-based (editions)**: you pay for slot capacity (reservations, optionally autoscaling) and query bytes are not billed individually. Storage is billed separately (logical or physical bytes). Check current pricing and edition features. |
| Partitioning | Splits a table by a time-unit column, ingestion time, or an integer range. Queries that filter on the partition column scan only matching partitions (**partition pruning**). `require_partition_filter` forces callers to include that filter. |
| Clustering | Sorts data within partitions by up to four columns. Filters and aggregations on clustered columns read fewer blocks. Works well on high-cardinality columns such as `user_id`. |
| Columnar scans | Cost and speed depend on columns read. `SELECT *` reads every column; `LIMIT` does not reduce bytes scanned on a non-clustered table. |
| Cost guardrails | `maximum_bytes_billed` per query, dry runs to estimate bytes, custom quotas per project or user, `INFORMATION_SCHEMA.JOBS` to find expensive queries. |
| ML integration | BigQuery ML, remote models backed by Vertex AI, vector search over embeddings, and direct reads from Vertex AI training through the BigQuery Storage Read API. |

```sql
-- Partitioned and clustered table for event features
CREATE TABLE analytics.events (
  event_ts   TIMESTAMP,
  user_id    STRING,
  event_name STRING,
  amount     NUMERIC
)
PARTITION BY DATE(event_ts)
CLUSTER BY user_id, event_name
OPTIONS (require_partition_filter = TRUE);
```

### BigQuery ML

BigQuery ML trains and runs models with SQL: linear and logistic regression, boosted trees, k-means, matrix factorization, time series forecasting, imported models (for example TensorFlow or ONNX), and remote models that call Vertex AI or Gemini. Function names for the generative AI integration have changed over time, so check current docs.

```sql
CREATE OR REPLACE MODEL analytics.churn_model
OPTIONS (model_type = 'LOGISTIC_REG', input_label_cols = ['churned']) AS
SELECT tenure_days, sessions_30d, support_tickets_30d, churned
FROM analytics.churn_training
WHERE snapshot_date < '2026-01-01';

SELECT user_id, predicted_churned, predicted_churned_probs
FROM ML.PREDICT(
  MODEL analytics.churn_model,
  (SELECT user_id, tenure_days, sessions_30d, support_tickets_30d
   FROM analytics.churn_scoring)
);
```

Use it for fast baselines and analyst-owned models where the data is already in BigQuery. Move to Vertex AI custom training when you need custom architectures, GPUs, or a richer serving story.

### Pipelines, streaming and orchestration

| Service | What it is | Use it for |
|---|---|---|
| **Dataflow** | Managed runner for Apache Beam (batch and streaming, Python and Java), with autoscaling | Feature pipelines, streaming aggregations with windows and late data, one codebase for batch and streaming |
| **Dataproc** | Managed Spark and Hadoop clusters, plus a serverless Spark option (product naming has changed; check docs) | Lift-and-shift Spark jobs, PySpark feature engineering, teams already invested in Spark |
| **Pub/Sub** | Global, managed messaging with push and pull subscriptions | Event ingestion, decoupling producers and consumers, fan-out. At-least-once by default; ordering keys, dead-letter topics, an exactly-once delivery option for pull subscriptions, and subscriptions that write straight to BigQuery or Cloud Storage. |
| **Cloud Composer** | Managed Apache Airflow | Cross-system DAGs with many operators, existing Airflow skills, data plus ML orchestration |
| **Workflows** | Serverless orchestration of HTTP and Google API calls, defined in YAML or JSON | Lightweight sequences: call an API, start a Vertex AI pipeline, wait, notify. No cluster to run. |
| **Dataplex** | Data governance: catalog, data quality, lineage, discovery across lakes and warehouses (Data Catalog functionality has moved into Dataplex) | Finding datasets, documenting owners, data quality checks, lineage for audits |

Orchestrator choice in one line: **Vertex AI Pipelines** for ML DAGs with artifact lineage, **Cloud Composer** for broad cross-system data DAGs, **Workflows** for cheap, simple API sequencing.

**What interviewers listen for**

- Partition plus cluster design and knowing that `LIMIT` does not save money.
- Explaining on-demand vs capacity pricing in terms of predictability and utilization, not exact prices.
- Dataflow for streaming with event-time windows and late data; Dataproc when Spark code already exists.
- Pub/Sub delivery semantics and idempotent consumers.
- Picking an orchestrator based on what is being orchestrated.

---

## Generative AI on GCP

> This area changes fastest. Product names (Vertex AI Search, Agent Builder, AI Applications, RAG Engine, Gemini Enterprise) have shifted repeatedly. Describe capabilities in interviews and treat names as secondary.

| Capability | What it is | Notes |
|---|---|---|
| **Gemini models on Vertex AI** | Google's first-party multimodal models via the Vertex AI API | Use the Google Gen AI SDK (`google-genai`) with `vertexai=True`; the older `vertexai.generative_models` module has been deprecated. Pay-as-you-go or provisioned throughput. |
| **Model Garden** | Catalog of first-party, partner and open models | Partner models (for example Anthropic Claude, Mistral, and others) are offered as managed APIs; open models (for example Llama or Gemma families) can be deployed to your own Vertex AI endpoints or GKE. |
| **Grounding** | Connect model responses to a source of truth | Grounding with Google Search for public facts; grounding on your own data through Vertex AI Search or a retrieval layer you build. |
| **Vertex AI Search / RAG Engine** | Managed retrieval: ingestion, chunking, embedding, indexing, ranking | Fastest path to RAG over documents. Less control over chunking and ranking than a custom pipeline. |
| **Embeddings** | Text and multimodal embedding models | Store vectors in Vertex AI Vector Search, BigQuery (vector search), AlloyDB or Cloud SQL with pgvector, or a third-party vector database. |
| **Evaluation and safety** | Gen AI evaluation service, safety filters, Model Armor (prompt and response screening) | Check availability and naming; see [LLM evaluation](../mlops/intro_llm_evaluation.md). |

Architecture notes:

- Keep the model ID in configuration, not code; models are retired on published schedules.
- Data residency: choose regional endpoints where required, and know whether a global endpoint is acceptable for your data.
- Quotas and throughput: pay-as-you-go endpoints have quotas; provisioned throughput buys reserved capacity for predictable, high-volume traffic.
- VPC-SC and CMEK support for generative AI features vary; verify per feature for regulated workloads.

See [code examples](#vertex-ai-gemini-with-google-search-grounding-and-embeddings) for a model-agnostic Gen AI SDK call, and [RAG](../ai_genai/intro_rag.md) and [Embeddings](../ai_genai/intro_embeddings.md) for the retrieval side.

**What interviewers listen for**

- Managed RAG vs custom RAG as a control vs speed tradeoff.
- Grounding and citations as the answer to hallucination concerns, plus evaluation.
- Quotas, provisioned throughput, region and data residency, and model version pinning.
- Choosing a vector store based on where the data and queries already live.

---

## Observability

| Service | What it covers | ML example |
|---|---|---|
| **Cloud Monitoring** | Metrics, dashboards, alerting policies, uptime checks, SLOs; Managed Service for Prometheus for Prometheus-style metrics | Endpoint p95 latency, GPU utilization, Pub/Sub backlog age, Dataflow system lag |
| **Cloud Logging** | Central logs, Log Router, sinks to BigQuery, Cloud Storage or Pub/Sub, retention buckets, exclusion filters | Structured prediction logs exported to BigQuery for offline analysis and drift checks |
| **Cloud Trace** | Distributed tracing (OpenTelemetry compatible) | Where time goes in a RAG request: retrieval vs reranking vs LLM call |
| **Vertex AI Model Monitoring** | Skew and drift detection for Vertex AI models | See [intro_vertex_ai.md](./intro_vertex_ai.md#monitoring-and-operations) and [Model Monitoring](../mlops/intro_model_monitoring.md) |

These are grouped under the Google Cloud Observability umbrella. Logging ingestion and retention are billed, so high-volume debug logs or full request payloads can become a cost problem; use exclusion filters and sampling.

**What interviewers listen for**

- System metrics (latency, errors, saturation) and model metrics (drift, prediction distribution) and business KPIs, all alerting.
- Structured logs routed to BigQuery for analysis, with PII handling.
- Alerting on symptoms (SLO burn) rather than every metric.

---

## Cost control

| Lever | How | Notes |
|---|---|---|
| **Labels** | Label every resource with `team`, `env`, `cost-center`, `model` | Labels flow into the billing export, enabling per-model cost reporting |
| **Budgets and alerts** | Cloud Billing budgets with threshold alerts, optionally publishing to Pub/Sub | Budgets alert; they do not cap spend by themselves. Automated actions (for example disabling a dev project's billing) must be built on the Pub/Sub notification. |
| **Billing export to BigQuery** | Detailed cost data queryable with SQL | Foundation for dashboards and anomaly detection |
| **BigQuery controls** | `maximum_bytes_billed`, dry runs, partition pruning, clustering, avoid `SELECT *`, custom quotas, consider capacity pricing for heavy steady usage | Most BigQuery surprises come from full scans by scheduled queries or dashboards |
| **Spot VMs** | Training, preprocessing, batch inference with checkpointing | Check current pricing |
| **Autoscaling to zero** | Cloud Run, GKE autoscaling (node pools to zero where possible), Batch | Vertex AI online endpoints keep at least one replica for most model types; check current scale-to-zero support |
| **Idle cleanup** | Undeploy unused Vertex AI endpoint models, stop idle Workbench instances, delete orphaned disks and old snapshots, lifecycle rules on buckets | Recommender surfaces idle VMs and disks |
| **Commitments** | CUDs for steady baseline usage | Only after usage is stable |

No prices or discount percentages are given here on purpose: they change, and vary by region and resource. Check current pricing pages.

**What interviewers listen for**

- Cost visibility first (labels, billing export), then guardrails (budgets, quotas, `maximum_bytes_billed`), then optimization.
- Architectural cost decisions: batch prediction instead of an always-on endpoint, scale to zero for spiky traffic, Spot for fault-tolerant jobs.
- Knowing that a budget alert does not stop spending.

---

## Reference architectures

### (a) Batch training pipeline

```text
  Sources                 Prep                      Train and register                    Score
+--------------+     +----------------+     +---------------------------------+     +------------------+
| Cloud Storage|---->|                |     | Vertex AI Pipelines             |     | Vertex AI        |
| (raw files)  |     |   Dataflow     |---->|  validate -> train (GPU/TPU,    |---->| Batch Prediction |
+--------------+     |   (Beam batch) |     |  Spot) -> evaluate -> gate      |     |  input: BigQuery |
+--------------+     |                |     +----------------+----------------+     |  output: BigQuery|
| BigQuery     |---->|                |                      |                      +---------+--------+
| (curated)    |     +-------+--------+                      v                                |
+--------------+             |                     +------------------+                       v
                             v                     | Model Registry   |             downstream apps,
                    GCS shards / BigQuery          | (versioned,      |             dashboards
                    training tables                |  aliases)        |
                                                   +------------------+
  Orchestration: Cloud Composer or a Vertex AI pipeline schedule
  Identity: one service account per stage; VPC-SC perimeter around data and ML projects
```

Walkthrough:

1. Raw files land in Cloud Storage and curated tables live in BigQuery.
2. A Dataflow batch job cleans, joins and computes features, writing sharded files to GCS or a BigQuery training table. Simple feature logic can stay in BigQuery SQL instead.
3. A Vertex AI pipeline validates data, trains (custom container, GPUs or TPUs, Spot where checkpointed), evaluates against the current production model, and only registers the model if it passes the gate.
4. The Model Registry stores the version with lineage back to the pipeline run.
5. Batch prediction reads from BigQuery and writes scores back to BigQuery, so no always-on endpoint is needed.

### (b) Real-time inference

```text
  Client
    |
    v
+---------------------------+       +------------------------------------+
| API layer                 |       | Online features                    |
| Cloud Run service         |------>| Vertex AI Feature Store online     |
| (auth, validation,        |<------| serving, Bigtable or Memorystore   |
|  feature lookup)          |       +------------------------------------+
+-------------+-------------+
              |
              v
+---------------------------+       +------------------------------------+
| Model server              |       | Cloud Logging -> BigQuery          |
| Vertex AI endpoint        |------>| (request/prediction logs)          |
| (traffic split, min/max   |       | Cloud Monitoring alerts            |
|  replicas) or the model   |       | Vertex AI Model Monitoring         |
|  inside Cloud Run / GKE   |       +------------------------------------+
+---------------------------+
```

Walkthrough:

1. A Cloud Run service is the entry point: authentication, input validation and feature lookup. It scales with requests and can scale to zero in non-prod.
2. Features computed offline are synced to a low-latency online store; request-time features are computed in the API layer using the same logic as training (shared library) to avoid skew.
3. The model runs on a Vertex AI endpoint (traffic splits for canaries, autoscaling between min and max replicas), or directly inside Cloud Run or GKE for small models or custom stacks.
4. Requests and predictions are logged in structured form to BigQuery for monitoring, drift checks and future training labels.
5. Private networking: Cloud Run reaches private resources through Direct VPC egress; the endpoint can be exposed privately through Private Service Connect.

### (c) Streaming features

```text
+-------------+     +-------------+     +-------------------------------+
| Producers   |---->|  Pub/Sub    |---->| Dataflow (streaming Beam)     |
| apps, IoT,  |     |  topic      |     |  parse -> dedupe by event id  |
| CDC         |     +------+------+     |  -> event-time windows        |
+-------------+            |            |  -> aggregate (e.g. 5m count) |
                    dead-letter topic   +-------+---------------+-------+
                                                |               |
                                                v               v
                                     +-----------------+ +--------------------+
                                     | BigQuery        | | Online store       |
                                     | (offline store, | | (Bigtable, Feature |
                                     |  history for    | |  Store online      |
                                     |  training)      | |  serving, Redis)   |
                                     +-----------------+ +---------+----------+
                                                                   |
                                                                   v
                                                          real-time inference (b)
```

Walkthrough:

1. Producers publish events to Pub/Sub; failed messages go to a dead-letter topic.
2. A streaming Dataflow job deduplicates by event ID (Pub/Sub is at-least-once by default), assigns event-time windows with watermarks and allowed lateness, and computes aggregates.
3. The same aggregates go to two sinks: BigQuery as the historical offline store for training with point-in-time joins, and a low-latency online store for serving.
4. Writing both from one pipeline is what keeps training and serving features consistent. See [Feature Stores](../mlops/intro_feature_stores.md).

---

## Code examples

All examples use official Google Cloud Python client libraries and Application Default Credentials (ADC). On GCP, ADC picks up the attached service account; locally, `gcloud auth application-default login`. No key files.

### Cloud Storage V4 signed URL

```python
import datetime

import google.auth
from google.auth.transport import requests
from google.cloud import storage

BUCKET = "my-ml-artifacts"
OBJECT = "reports/eval_2026_10.html"

# On Cloud Run, GKE or Compute Engine the default credentials have no private key,
# so signing is delegated to the IAM signBlob API using an access token.
# The service account needs roles/iam.serviceAccountTokenCreator on itself.
credentials, project = google.auth.default()
credentials.refresh(requests.Request())

client = storage.Client(credentials=credentials, project=project)
blob = client.bucket(BUCKET).blob(OBJECT)

url = blob.generate_signed_url(
    version="v4",
    expiration=datetime.timedelta(minutes=15),  # V4 maximum is 7 days
    method="GET",
    service_account_email=credentials.service_account_email,
    access_token=credentials.token,
)
print(url)
```

### BigQuery query with a cost guardrail

```python
import datetime

from google.cloud import bigquery

client = bigquery.Client()

sql = """
SELECT user_id, COUNT(*) AS purchases_7d
FROM `my-project.analytics.events`
WHERE DATE(event_ts) BETWEEN @start AND @end   -- partition pruning
  AND event_name = 'purchase'                  -- clustered column
GROUP BY user_id
"""
params = [
    bigquery.ScalarQueryParameter("start", "DATE", datetime.date(2026, 10, 1)),
    bigquery.ScalarQueryParameter("end", "DATE", datetime.date(2026, 10, 7)),
]

# 1) Dry run: estimate bytes without running or billing the query
dry_cfg = bigquery.QueryJobConfig(
    dry_run=True, use_query_cache=False, query_parameters=params
)
dry_job = client.query(sql, job_config=dry_cfg)
print(f"Estimated bytes processed: {dry_job.total_bytes_processed:,}")

# 2) Real run: the job fails instead of billing more than the cap
cfg = bigquery.QueryJobConfig(
    maximum_bytes_billed=10 * 1024**3,  # 10 GiB cap
    query_parameters=params,
    labels={"team": "ml", "pipeline": "features"},  # shows up in billing data
)
rows = client.query(sql, job_config=cfg).result()
for row in rows:
    print(row.user_id, row.purchases_7d)
```

### Pub/Sub publish

```python
import json

from google.cloud import pubsub_v1

PROJECT_ID = "my-project"
TOPIC_ID = "click-events"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

event = {"event_id": "e-123", "user_id": "u-42", "action": "click"}

# data must be bytes; attributes are string key/value pairs.
future = publisher.publish(
    topic_path,
    data=json.dumps(event).encode("utf-8"),
    event_id=event["event_id"],  # lets consumers deduplicate
)
print("Published message id:", future.result())
```

### Secret Manager access

```python
from google.cloud import secretmanager

PROJECT_ID = "my-project"
SECRET_ID = "third-party-api-key"
VERSION = "3"  # pin in production; "latest" is fine with automated rotation

client = secretmanager.SecretManagerServiceClient()
name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/{VERSION}"

response = client.access_secret_version(request={"name": name})
api_key = response.payload.data.decode("utf-8")
# Do not log the value; pass it straight to the client that needs it.
```

### Vertex AI: Gemini with Google Search grounding, and embeddings

Model IDs are deliberately placeholders; read them from configuration and check the current model list in Model Garden.

```python
import os

from google import genai
from google.genai import types

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GENERATION_MODEL = os.environ["GENERATION_MODEL_ID"]  # from config, not hardcoded
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL_ID"]

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Grounded generation: the model can use Google Search results and return sources
response = client.models.generate_content(
    model=GENERATION_MODEL,
    contents="Summarize recent changes to our cloud provider's GPU offerings.",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.2,
    ),
)
print(response.text)

# Embeddings for retrieval
emb = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=["How do I rotate a secret?", "Spot VM preemption handling"],
)
vectors = [e.values for e in emb.embeddings]
print(len(vectors), len(vectors[0]))
```

### Keyless GitHub Actions auth (Workload Identity Federation)

```yaml
# .github/workflows/deploy.yml (excerpt)
permissions:
  contents: read
  id-token: write   # required to mint the GitHub OIDC token

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2   # check for the current major version
        with:
          workload_identity_provider: projects/123456789/locations/global/workloadIdentityPools/github/providers/my-repo
          service_account: deployer@my-project.iam.gserviceaccount.com
      # Later steps (gcloud, client libraries) use the short-lived credentials
```

---

## Interview Q&A

#### How is the GCP resource hierarchy structured, and why does it matter for an ML platform?

GCP resources sit in an organization, optional folders, and projects. Projects are the unit of billing, API enablement, quotas and most IAM bindings, and IAM allow policies inherit downward, so a role granted on a folder applies to every project in it. For ML this usually means separate dev, staging and prod projects, often with data in its own project that ML projects read from. The benefit is blast-radius control: a broken experiment or an overly broad role in dev cannot touch prod data or endpoints. The cost is cross-project wiring: service accounts need explicit grants on the data project, and networking may need Shared VPC. Organization policies at the org or folder level (restrict locations, disable service account keys, require CMEK) give guardrails that individual projects cannot override.

#### What is the difference between basic, predefined and custom IAM roles, and which should a training pipeline use?

Basic roles (Owner, Editor, Viewer) predate fine-grained IAM and are very broad; Editor alone can change most resources in a project, so they should not be used for workloads. Predefined roles are service-specific bundles maintained by Google, such as `roles/bigquery.dataViewer` or `roles/aiplatform.user`. Custom roles let you list exact permissions when no predefined role is narrow enough, but you must maintain them as services add permissions. A training pipeline should run as a dedicated service account with predefined roles scoped to the specific resources it needs: read on the feature dataset, write to the artifact bucket, and permission to create Vertex AI jobs. Use IAM Conditions or resource-level grants (bucket, dataset) rather than project-wide roles where possible. Review grants with IAM Recommender, which flags unused permissions.

#### Why should you avoid service account keys, and what do you use instead?

A service account key is a long-lived credential that works from anywhere until someone deletes it, so a leak through a laptop, CI log or git commit is a durable breach. Inside GCP, attach the service account to the resource (VM, Cloud Run, GKE through Workload Identity Federation for GKE, Vertex AI jobs) and let Application Default Credentials fetch short-lived tokens automatically. Outside GCP, use Workload Identity Federation to exchange an external identity token (GitHub OIDC, AWS, Azure, an on-prem IdP) for short-lived GCP credentials. For humans, use their own identity plus service account impersonation when they need a workload's permissions. Enforce this with the organization policy that disables key creation. The tradeoff is setup effort and some tools that still assume key files, but those can usually be fed ADC or federation configuration files instead.

#### How would you let GitHub Actions deploy to GCP without storing any keys?

Create a Workload Identity Pool and an OIDC provider for GitHub's token issuer, with attribute mappings from the token claims (repository, ref, workflow) and an attribute condition that only accepts your repository, ideally only the main branch or a protected environment. Grant the federated principal permission to impersonate a narrowly scoped deployer service account, or grant roles directly to the principal set where the target services support it. In the workflow, request `id-token: write` permission and use the `google-github-actions/auth` action with the provider resource name and service account. The job receives short-lived credentials that expire on their own, and nothing secret is stored in GitHub. The main risk is a loose attribute condition: without it, tokens from other repositories could be accepted. Audit logs show the federated identity, so deployments remain attributable.

#### How do VPC Service Controls help prevent data exfiltration, and what do they not cover?

VPC-SC draws a service perimeter around a set of projects and blocks Google API calls that would move data across that boundary, for example copying a BigQuery table or GCS object to a project outside the perimeter, even if the caller has valid IAM permissions. That covers the classic insider or stolen-credential scenario that IAM alone cannot stop. Access levels (for example corporate device or IP ranges) and ingress/egress rules allow specific, audited exceptions such as a vendor project that needs read access. Roll it out in dry-run mode first, because it breaks legitimate flows (CI, BI tools, cross-project pipelines) that you did not know existed. It does not stop someone from exfiltrating data through an application you expose, through screenshots, or through outbound internet from a VM (that is handled by firewall rules and no NAT), and not every product or feature is supported, so check the supported services list. Pair it with Data Access audit logs and CMEK for regulated ML data.

#### Your team's BigQuery bill doubled this month. How do you investigate and fix it?

Start with data, not guesses: query `INFORMATION_SCHEMA.JOBS` (by project or organization) for the most expensive queries by bytes billed or slot time, grouped by user, service account and labels, and look at the billing export in BigQuery to confirm whether it is query compute or storage. Common culprits are a scheduled query or dashboard doing full scans, a new pipeline doing `SELECT *` on a wide table, a missing partition filter, or a join that explodes. Fixes include adding partition filters and `require_partition_filter`, clustering on common filter columns, selecting only needed columns, materializing intermediate results, and caching dashboard queries. Add guardrails: `maximum_bytes_billed` on pipeline queries, custom daily quotas per user or project, and budget alerts. If usage is heavy and steady, evaluate capacity-based pricing, which trades per-query billing for predictable slot cost but can slow queries if under-provisioned. Finally, add job labels so the next spike can be attributed in minutes.

#### How do partitioning and clustering differ in BigQuery, and when do you use each?

Partitioning splits a table into segments by a date or timestamp column, ingestion time, or an integer range, and the query planner skips whole partitions that a filter excludes, so the bytes estimate before the query already reflects the savings. Clustering sorts data within each partition by up to four columns, so filters on those columns read fewer storage blocks, but the exact savings are only known after the query runs. Partition on the column almost every query filters by, usually event date, and cluster on high-cardinality columns used in filters or joins, such as `user_id`. Too many tiny partitions hurt performance and there are partition count limits, so do not partition on a high-cardinality key. For ML feature tables, date partitioning also makes point-in-time training snapshots and retention easy. Both are cheap to set up at table creation and painful to retrofit on large tables.

#### When would you choose TPUs over GPUs for training?

TPUs are a strong fit for large, dense, regular workloads such as transformer training or large embedding models, especially when the team uses JAX or is comfortable with XLA, and when the job is big enough to use a slice of many chips. They can offer good price-performance at scale and are well integrated with GCP. GPUs are the safer default when the code depends on custom CUDA kernels, GPU-specific libraries (some attention and quantization kernels, certain inference servers), dynamic shapes, or a broad PyTorch ecosystem without XLA work. Portability matters too: GPU code moves between clouds and on-prem more easily. In practice, I would benchmark a representative step on both with the real model and input pipeline, compare cost per training step and engineering effort, and check capacity availability in the target region. The worst outcome is choosing TPUs and then spending weeks fighting recompilation from dynamic shapes.

#### How do you choose between Cloud Run, GKE and Vertex AI endpoints for serving a model?

Vertex AI endpoints are the default when the model lives in the Vertex AI Model Registry and you want managed traffic splitting, autoscaling, model monitoring and private endpoints with little ops work; the tradeoff is less control over the serving stack and a minimum replica count for most model types. Cloud Run fits stateless HTTP model servers that are small to medium, spiky traffic that benefits from scale to zero, and API layers around models; GPU support makes it viable for smaller open models, with cold starts and model load time as the main constraint. GKE fits when you need a custom serving stack (vLLM, Triton, KServe), multi-model packing on shared GPUs, fine-grained autoscaling, or a platform team that already runs Kubernetes; it brings the most control and the most operational burden. I would decide using traffic pattern, model size and hardware, latency SLO, team skills and the monitoring and rollout features needed. A common hybrid is Cloud Run for the API and feature lookup with the model on a Vertex AI endpoint.

#### How would you feed a large image dataset from Cloud Storage to multi-node GPU training without starving the GPUs?

First check whether the job is actually I/O bound by watching GPU utilization and data loader wait time. Store data in the same region as the training cluster and pack millions of small images into large shards (WebDataset tar files or TFRecords) so reads are large and sequential. Stream shards with parallel reads and prefetching in the data loader, and use Cloud Storage FUSE with its file caching enabled if the code expects file paths, or a parallel file system such as Parallelstore or Managed Lustre (check availability) when you need very high throughput or POSIX semantics. Local SSD can cache the dataset per node for multi-epoch training. Filestore is an option for moderate scale but can become a bottleneck or expensive at very high throughput. Write checkpoints asynchronously so they do not stall training steps.

#### When would you use Dataflow, Dataproc or plain BigQuery SQL for feature engineering?

If the data already lives in BigQuery and the transformations are relational (joins, aggregations, window functions), BigQuery SQL is usually the simplest and cheapest to maintain, possibly managed with dbt. Dataflow is the choice for streaming features with event-time windows, late data and exactly-once processing, or when you want a single Apache Beam codebase for both batch backfills and streaming. Dataproc is the choice when the team already has Spark jobs or needs Spark libraries, and the serverless Spark option avoids managing clusters. The tradeoffs are skills and operations: Beam has a learning curve, Spark clusters need tuning, and SQL becomes hard to test when logic gets complex. Whatever you pick, the same feature logic must serve training and serving to avoid skew.

#### What delivery guarantees does Pub/Sub give, and how do you build a correct consumer?

Pub/Sub delivers at least once by default, so a message can arrive more than once, for example after an acknowledgment deadline expires. Ordering is only guaranteed per ordering key when ordering is enabled, and there is an exactly-once delivery option for pull subscriptions that has its own constraints. A correct consumer is idempotent: include a unique event ID, deduplicate (Dataflow can deduplicate by ID, or use upserts keyed by event ID in the sink), and make side effects safe to repeat. Configure a dead-letter topic with a max delivery attempts setting so poison messages do not block processing, and alert on backlog age. Acknowledge only after the work is durably done. Retaining acknowledged messages and using seek lets you replay after a bad deploy.

#### Cloud Composer, Workflows or Vertex AI Pipelines: how do you pick an orchestrator?

Vertex AI Pipelines orchestrates ML steps with tracked artifacts, lineage and caching, so it is the natural choice for the train, evaluate and register DAG itself. Cloud Composer is managed Airflow, good for broad data orchestration across many systems with existing operators, schedules and backfills, but it runs an always-on environment with a baseline cost and Airflow operational habits. Workflows is serverless and cheap for simple sequences of API calls, such as "wait for a BigQuery load, start a Vertex AI pipeline, notify Slack," but it is not a data processing engine and has limited reuse compared with Airflow. A common combination is Composer or Workflows triggering a Vertex AI pipeline when upstream data is ready. Choosing Composer only to run one nightly pipeline is often overkill.

#### How would you build a Gemini-based assistant grounded on internal documents, and what are the tradeoffs of managed vs custom retrieval?

The managed path uses Vertex AI Search or the Vertex AI RAG Engine (names change, so check current docs) to ingest documents from Cloud Storage or other connectors, handle chunking, embeddings and ranking, and then ground Gemini responses with citations. That is fast to deliver and includes access control features, but you get less control over chunking, hybrid retrieval and reranking, and debugging retrieval quality is harder. The custom path chunks documents yourself, embeds them with a Vertex AI embedding model, stores vectors in Vector Search, BigQuery or AlloyDB with pgvector, and builds retrieval and reranking in your own service. That gives full control and portability at the cost of more engineering and evaluation work. In both cases, enforce document-level permissions at retrieval time, evaluate with a labeled question set, keep model IDs in config, and log prompts and retrieved sources (with PII handling) for debugging.

#### How do you run training on Spot VMs without losing progress?

Spot VMs can be preempted at any time with a short notice, so the job must be restartable. Save checkpoints (model, optimizer state, data loader position, RNG state) to Cloud Storage at a frequency that bounds lost work, and on start always resume from the latest checkpoint. Use a scheduler that restarts the job automatically: Vertex AI custom training with Spot and restart settings, a GKE Job with Spot node pools, or Batch with retries. Handle the preemption signal to flush a final checkpoint if possible, but do not rely on it. For multi-node jobs, preempting one node kills the step for everyone, so elastic training or smaller node counts can help. The savings are significant (check current pricing) but are offset by lost compute if checkpoints are too infrequent, so measure the preemption rate in your region.

#### How do you let private training and serving workloads reach Google APIs and the internet without public IPs?

Create VMs, GKE nodes and Vertex AI workloads without external IP addresses, and enable Private Google Access on their subnets so they can reach Cloud Storage, BigQuery and Artifact Registry over Google's network. For specific internet destinations such as package mirrors or model hubs, add Cloud NAT, which allows outbound connections only, and restrict egress with firewall rules or a proxy. Prefer mirroring dependencies into Artifact Registry so production workloads do not need internet access at all. Use Private Service Connect for private endpoints to managed services, including private Vertex AI endpoints, and Direct VPC egress for Cloud Run to reach private resources. Remember that this is network privacy; it does not stop API-level data copying, which is what VPC Service Controls address.

---

## Common Pitfalls

| Problem | Why it hurts | Fix |
|---|---|---|
| Using basic roles (Editor) for workloads | Any compromise or bug can modify most of the project | Dedicated service accounts with predefined or custom roles, scoped to resources |
| Service account JSON keys in CI or notebooks | Long-lived credentials leak and stay valid | Attached service accounts, Workload Identity Federation, org policy disabling key creation |
| Federation provider without an attribute condition | Tokens from other repositories or tenants may be accepted | Restrict by repository, branch or environment claims |
| Assuming IAM prevents exfiltration | Valid credentials can still copy data out | VPC Service Controls perimeters, Data Access audit logs |
| `SELECT *` and no partition filter in BigQuery | Full table scans on every run, bill spikes | Select columns, partition and cluster, `require_partition_filter`, `maximum_bytes_billed` |
| Relying on `LIMIT` to cut BigQuery cost | Bytes scanned are usually unchanged | Filter on partition and cluster columns, use table preview for exploration |
| Training data in a different region from compute | Egress cost and slower input pipeline | Co-locate buckets, datasets and training regions |
| Millions of tiny files read through FUSE | GPUs idle waiting on I/O | Shard into large files, enable caching, or use a parallel file system |
| Object versioning without lifecycle rules | Noncurrent versions accumulate cost silently | Lifecycle rules for noncurrent versions and old checkpoints |
| Spot VMs without checkpointing | Preemption throws away hours of training | Frequent checkpoints to GCS, automatic restart and resume |
| Endpoints and Workbench instances left running | Steady spend for zero value | Undeploy idle models, idle shutdown, budget alerts, labels to find owners |
| Treating budgets as spending caps | Alerts fire, spend continues | Pub/Sub budget notifications with automated actions for non-prod |
| Hardcoding generative model IDs | Breaks when models are retired | Model IDs in configuration, scheduled upgrade and evaluation |
| Not enabling Data Access audit logs for sensitive data | Cannot answer "who read this table" during an incident | Enable for sensitive services and route to a retained log bucket |

---

## Related Topics

| Topic | Why It's Related |
|---|---|
| [Google Vertex AI Interview Guide](./intro_vertex_ai.md) | The ML platform this guide surrounds: pipelines, training, registry, endpoints |
| [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md) | Vertex AI vs SageMaker vs Azure ML |
| [AWS for ML Engineers](./aws_for_ml_engineers.md) | The same building blocks on AWS |
| [Azure for ML Engineers](./azure_for_ml_engineers.md) | The same building blocks on Azure |
| [Cloud Service Mapping](./cloud_service_mapping.md) | GCP, AWS and Azure equivalents side by side |
| [CI/CD for ML](../mlops/intro_cicd_for_ml.md) | Pipelines that use keyless auth and promotion gates |
| [Model Serving](../mlops/intro_model_serving.md) | Serving patterns behind Cloud Run, GKE and Vertex AI endpoints |
| [Model Monitoring](../mlops/intro_model_monitoring.md) | Drift and performance monitoring beyond infrastructure metrics |
| [Feature Stores](../mlops/intro_feature_stores.md) | Offline and online stores in the streaming architecture |
| [Kubernetes](../devops/intro_kubernetes.md) | Foundation for GKE |
| [Terraform](../devops/intro_terraform.md) | Managing projects, IAM and networks as code |
| [GitHub Actions](../devops/intro_github_actions.md) | CI/CD with Workload Identity Federation |
| [Observability](../devops/intro_observability.md) | Metrics, logs, traces and SLOs in general |
| [SQL Optimization](../data_engineering/intro_sql_optimization.md) | Query patterns that reduce BigQuery scans |
| [Apache Airflow](../data_engineering/intro_apache_airflow.md) | The engine behind Cloud Composer |
| [Data Processing Pipelines](../data_engineering/intro_data_processing_pipelines.md) | Batch and streaming design behind Dataflow and Dataproc |
| [RAG](../ai_genai/intro_rag.md) | Retrieval design for grounded generative AI |
| [Embeddings](../ai_genai/intro_embeddings.md) | Embedding models and vector search |
