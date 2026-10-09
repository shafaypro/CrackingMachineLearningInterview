# AWS for ML Engineers

Most ML interviews that mention AWS are not really about SageMaker. They are about everything around it: who is allowed to read the training data, how a GPU job reaches S3 without touching the internet, why the bill doubled last week, and how a model trained in one account ends up serving traffic in another. This guide covers the core AWS services an ML or AI engineer works with outside SageMaker itself and shows how they fit together.

SageMaker training, hosting, pipelines and the model registry are covered in depth in [AWS SageMaker Interview Guide](./intro_sagemaker.md), and cross-cloud platform comparisons live in [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md). This guide links to them rather than repeating them.

AWS renames and extends services often. Where a limit, name or feature is likely to change, this guide says so; treat the AWS documentation as the source of truth and check current pricing before quoting numbers in an interview.

---

## Table of Contents

1. [How the pieces fit together](#how-the-pieces-fit-together)
2. [Identity and security](#identity-and-security)
3. [Networking basics](#networking-basics)
4. [Storage](#storage)
5. [Compute](#compute)
6. [Data and analytics](#data-and-analytics)
7. [Generative AI with Amazon Bedrock](#generative-ai-with-amazon-bedrock)
8. [Observability](#observability)
9. [Cost control](#cost-control)
10. [Reference architectures](#reference-architectures)
11. [Code examples](#code-examples)
12. [Interview Q&A](#interview-qa)
13. [Common Pitfalls](#common-pitfalls)
14. [Related Topics](#related-topics)

---

## How the pieces fit together

| Layer | Core services | ML question it answers |
|---|---|---|
| Identity and security | IAM, STS, KMS, Secrets Manager, CloudTrail, Organizations | Who or what can touch the data and models, and can we prove it? |
| Networking | VPC, subnets, NAT gateway, security groups, VPC endpoints | Can the training job reach S3 and ECR without the public internet? |
| Storage | S3, EBS, EFS, FSx for Lustre | Where do datasets, checkpoints and artifacts live, and how fast can GPUs read them? |
| Compute | EC2 (GPU and accelerator families), Batch, ECS, Fargate, EKS, Lambda, SageMaker | What runs the training, batch scoring and serving code? |
| Data and analytics | Glue, Athena, EMR, Redshift, Lake Formation, Kinesis, MSK | How is raw data turned into governed, queryable training sets and features? |
| Orchestration and events | Step Functions, MWAA, EventBridge, SageMaker Pipelines | What triggers retraining, and what happens when a step fails? |
| Generative AI | Amazon Bedrock (models, Knowledge Bases, Agents, Guardrails) | How do we call foundation models and build RAG without hosting GPUs? |
| Observability | CloudWatch, X-Ray / OpenTelemetry, Model Monitor | Is the system healthy, and is the model still correct? |
| Cost | Tags, Budgets, Cost Explorer, Savings Plans, Spot | What does each model cost to train and serve, and who pays? |

A useful mental model for interviews: **IAM decides who, the VPC decides where from, KMS decides whether the bytes are readable, and CloudTrail records what happened.** Every ML architecture answer on AWS should touch all four.

---

## Identity and security

### IAM building blocks

| Concept | What it is | ML example |
|---|---|---|
| IAM user | Long-lived identity with optional access keys | Avoid for workloads; humans should use IAM Identity Center (SSO) instead |
| IAM role | Identity with no long-lived credentials, assumed to get temporary credentials | SageMaker execution role, ECS task role, Lambda execution role |
| Trust policy | Resource policy on a role saying who may assume it | Allow `sagemaker.amazonaws.com` to assume the training role |
| Identity-based policy | Permissions attached to a user, group or role | Training role may `s3:GetObject` on `datasets/fraud/*` |
| Resource-based policy | Permissions attached to the resource itself | S3 bucket policy, KMS key policy, ECR repository policy |
| Permissions boundary | Upper limit on what a role's policies can grant | Let data scientists create roles that can never exceed a safe ceiling |
| Service control policy (SCP) | Organization-level guardrail; grants nothing, only limits | Deny any action outside approved regions in all ML accounts |
| Session policy / tags | Extra restrictions or attributes passed at assume time | Attribute-based access control (ABAC) by `team` or `project` tag |

**Policy evaluation in one line:** everything is denied by default, an explicit `Deny` anywhere wins, and an action is allowed only if some applicable policy allows it and no boundary, SCP or session policy blocks it. Cross-account access needs both sides to agree: the caller's identity policy allows it, and the target resource policy (or role trust policy) allows the caller.

### Roles for workloads

| Where code runs | How it gets credentials |
|---|---|
| EC2 instance | Instance profile (role attached to the instance), served through the instance metadata service (use IMDSv2) |
| ECS task (EC2 or Fargate) | **Task role** for application calls; the separate **task execution role** is only for pulling images and writing logs |
| EKS pod | **IRSA** (IAM Roles for Service Accounts, via the cluster's OIDC provider) or **EKS Pod Identity** (newer, uses an agent add-on and association API; no per-cluster OIDC trust policy edits) |
| Lambda | Execution role |
| SageMaker job or endpoint | Execution role passed at creation; the caller needs `iam:PassRole` for that role |
| CI/CD (for example GitHub Actions) | OIDC federation to `sts:AssumeRoleWithWebIdentity`, so no stored access keys |
| Another account | `sts:AssumeRole` into a role whose trust policy names your account or role, optionally with an external ID for third parties |

STS issues temporary credentials (access key, secret key, session token) with a configurable duration. Role chaining (assuming a role from an assumed-role session) is limited to a shorter maximum session; check current docs for the exact limits.

### Encryption, secrets and audit

| Service | Purpose | Notes for ML |
|---|---|---|
| KMS | Managed encryption keys and envelope encryption | Use customer managed keys when you need key policies, cross-account grants or audit per key. Enable S3 Bucket Keys to cut KMS request volume on busy buckets |
| S3 encryption | SSE-S3 (default for new objects), SSE-KMS, DSSE-KMS, SSE-C | Reading an SSE-KMS object needs both S3 permission and `kms:Decrypt` on the key, a common cause of `AccessDenied` |
| Secrets Manager | Stores and rotates secrets (database passwords, API keys) | Inject at runtime by ARN; cache in-process; rotation is typically done with a Lambda function |
| SSM Parameter Store | Configuration values and simple secrets (`SecureString`) | Cheaper and simpler when you do not need managed rotation |
| VPC endpoints / PrivateLink | Private connectivity to AWS services | Lets a training job in a private subnet reach S3, ECR, STS and SageMaker APIs without a NAT gateway |
| S3 bucket policies | Resource-side rules on a bucket | Enforce TLS (`aws:SecureTransport`), require a specific VPC endpoint (`aws:SourceVpce`), restrict to your organization (`aws:PrincipalOrgID`) |
| CloudTrail | API audit log | Management events are recorded by default; S3 object-level **data events** must be enabled explicitly (and cost extra). Use an organization trail delivered to a locked-down log archive account |
| GuardDuty, Macie, Config | Threat detection, sensitive-data discovery in S3, configuration compliance | Macie helps find PII in training buckets before it ends up in a model |

New S3 buckets have Block Public Access enabled and ACLs disabled by default. Keep it that way and grant access with policies.

**What interviewers listen for:**

- Roles and temporary credentials everywhere; no access keys in code, notebooks, containers or environment files.
- Least privilege scoped to a bucket **and prefix**, not `s3:*` on `*`.
- Awareness that KMS key policies, bucket policies and SCPs can each block access independently.
- `iam:PassRole` as the control that stops someone from launching a job with a more powerful role than their own.
- A data perimeter story: VPC endpoints plus bucket and endpoint policies plus SCPs, not just "we encrypt everything".

---

## Networking basics

| Concept | What it does | ML relevance |
|---|---|---|
| VPC | Isolated virtual network in one region | Training jobs, endpoints and EKS nodes usually run inside one |
| Public subnet | Route table sends `0.0.0.0/0` to an internet gateway | Load balancers, NAT gateways, bastions (prefer SSM Session Manager over bastions) |
| Private subnet | No direct route from the internet | Training, inference, databases, vector stores |
| NAT gateway | Lets private subnets make outbound connections | Needed for pip installs or external APIs; zonal, so deploy one per AZ for resilience. Charges per hour and per GB processed |
| Internet gateway | Connects the VPC to the internet | Only public subnets route to it |
| Security group | Stateful, allow-only firewall attached to ENIs | Reference other security groups (for example "endpoint SG allows 443 from app SG") |
| Network ACL | Stateless, ordered allow and deny rules at the subnet level | Coarse subnet guardrails; return traffic must be allowed explicitly |
| Gateway VPC endpoint | Route table entry for S3 or DynamoDB | No hourly or data processing charge at time of writing; same-region only |
| Interface VPC endpoint (PrivateLink) | ENI with a private IP in your subnet for a service | Works for most AWS APIs (SageMaker API and runtime, ECR, STS, CloudWatch Logs, Bedrock, Secrets Manager) and for your own services exposed via an NLB. Hourly and per-GB charges apply |
| VPC peering / Transit Gateway | Connect VPCs | Shared services VPC, cross-account networking |

**Security groups vs NACLs:** security groups are stateful (return traffic is automatically allowed), support only allow rules, and attach to network interfaces. NACLs are stateless, support allow and deny, evaluate rules in number order, and apply to whole subnets. Most teams do almost everything with security groups and leave NACLs near default.

**Gateway vs interface endpoints:** S3 supports both. Use the gateway endpoint for in-VPC traffic to S3 (no extra charge, and it removes S3 traffic from the NAT gateway). Use an S3 interface endpoint when traffic comes from on-premises over Direct Connect or VPN, or from another region's network path that cannot use a route table entry.

A fully private SageMaker or EKS setup typically needs: an S3 gateway endpoint, interface endpoints for ECR (`ecr.api` and `ecr.dkr`), STS, CloudWatch Logs, and the SageMaker API and runtime (or Bedrock runtime), plus any other service the code calls. Missing one shows up as a job that hangs or times out rather than a clean permission error.

**What interviewers listen for:**

- Training and inference in private subnets, with egress either blocked or controlled.
- Using gateway endpoints for S3 to cut NAT data processing cost and keep traffic private.
- Knowing that a "hanging" job in a VPC is often a missing endpoint or security group rule.
- Clear explanation of stateful vs stateless filtering.

---

## Storage

### S3 essentials

S3 is the system of record for datasets, features, checkpoints, model artifacts and logs.

| Topic | What to know |
|---|---|
| Consistency | Strong read-after-write consistency for PUTs, overwrites, deletes and LIST operations in all regions (since late 2020). No more "eventual consistency" workarounds |
| Prefixes and throughput | Request rate scales per prefix (AWS documents at least 3,500 writes and 5,500 reads per second per partitioned prefix). Spread hot data across prefixes, and use fewer, larger files (for example 100 MB to 1 GB Parquet or WebDataset shards) rather than millions of tiny ones |
| Multipart upload | Upload large objects in parallel parts, retry parts independently. Required for objects above the single-PUT limit (5 GB at time of writing) and recommended well before that. Add a lifecycle rule to abort incomplete multipart uploads, or orphaned parts keep costing storage |
| Versioning | Keeps prior versions on overwrite or delete (delete adds a delete marker). Protects datasets and model artifacts from accidental loss; required for replication. Pair with lifecycle rules for noncurrent versions |
| Object Lock | Write-once-read-many retention for compliance or audit evidence |
| Lifecycle rules | Transition objects to cheaper classes or expire them by age, prefix or tag |
| Replication | Same-region or cross-region replication for DR, data residency or cross-account copies |
| Event notifications | Trigger Lambda, SQS, SNS or EventBridge when new data lands |

### Storage classes

| Class | Use for | Watch out for |
|---|---|---|
| S3 Standard | Active datasets, current artifacts | Highest storage price per GB |
| S3 Intelligent-Tiering | Data with unknown or changing access patterns | Small per-object monitoring fee, so poor fit for millions of tiny objects |
| S3 Standard-IA / One Zone-IA | Infrequently read data that must be available immediately | Retrieval fees and minimum storage duration; One Zone-IA lives in a single AZ |
| S3 Glacier Instant Retrieval | Archives read rarely but needed in milliseconds | Higher retrieval cost, minimum duration |
| S3 Glacier Flexible Retrieval / Deep Archive | Old raw data, compliance archives, retired model versions | Retrieval takes minutes to hours; not for training reads |
| S3 Express One Zone | Very low latency, high request rate data close to compute (directory buckets) | Single AZ, different bucket type and API behavior; check feature support |

### Block and file storage for training

| Option | Type | Strengths | Limits | Typical ML use |
|---|---|---|---|---|
| Instance store (local NVMe) | Ephemeral block | Fastest local I/O | Lost when the instance stops or is reclaimed | Data cache, scratch space |
| EBS | Network block volume, one AZ | Persistent, snapshots, predictable IOPS on provisioned types | Attached to one instance in most cases; AZ-bound | Boot volumes, single-node training scratch, databases |
| EFS | Managed NFS, multi-AZ | Shared by many instances and pods, elastic size | Lower per-client throughput than a parallel file system; small-file metadata overhead | Shared home directories, notebooks, small shared datasets, config |
| FSx for Lustre | Parallel file system | Very high aggregate throughput for many GPU nodes; can link to an S3 bucket and lazy-load objects | Provisioned capacity cost; scratch file systems are not durable | Multi-node distributed training on large datasets |
| S3 directly | Object storage | Unlimited scale, cheapest durable option | Per-request latency; needs parallel reads | Streaming via SageMaker FastFile or Pipe mode, Mountpoint for Amazon S3, or the S3 connector for PyTorch |

Rule of thumb: start by streaming sharded data from S3. Move to FSx for Lustre when profiling shows GPUs waiting on I/O across many nodes or many epochs re-read the same data. Use EFS for shared, low-throughput files, not as a high-throughput training data source.

**What interviewers listen for:**

- Knowing S3 is strongly consistent now, and that prefix layout and file size drive throughput.
- Lifecycle rules for raw data, intermediate outputs, old checkpoints and incomplete multipart uploads.
- Versioning (or immutable, versioned paths) for datasets and artifacts, so a model can be traced to the exact data it saw.
- Matching storage to GPU throughput needs instead of defaulting to EFS or EBS.

---

## Compute

### GPU and accelerator families

Describe these generically in interviews; instance names and specs change every year.

| Family | Hardware | Typical use |
|---|---|---|
| P family | High-end NVIDIA data center GPUs, high-bandwidth networking (EFA) on larger sizes | Large-scale and distributed training, large-model inference |
| G family | NVIDIA GPUs aimed at inference, graphics and smaller training | Cost-efficient inference, fine-tuning smaller models, experimentation |
| Trn (AWS Trainium) | AWS-designed training accelerators | Training with the AWS Neuron SDK, often at lower cost per unit of work for supported models |
| Inf (AWS Inferentia) | AWS-designed inference accelerators | High-throughput, lower-cost inference for models supported by Neuron |
| CPU families (general purpose, compute optimized, memory optimized, Graviton/Arm) | CPUs | Classical ML, feature engineering, small-model inference, preprocessing |

Trainium and Inferentia require compiling or adapting models with the Neuron SDK, so check model and operator support before committing. For multi-node GPU training, look for Elastic Fabric Adapter (EFA) support and cluster placement groups.

### Purchase options

| Option | What it is | Best for | Tradeoff |
|---|---|---|---|
| On-Demand | Pay by the second or hour, no commitment | Spiky, unpredictable, or short workloads; production when capacity matters | Highest unit price |
| Spot | Spare capacity at a large discount; can be reclaimed with a two-minute warning | Fault-tolerant training with checkpoints, batch inference, hyperparameter sweeps | Interruptions; GPU Spot capacity can be scarce |
| Savings Plans | Commit to an hourly spend for 1 or 3 years. Compute Savings Plans cover EC2, Fargate and Lambda; SageMaker Savings Plans cover SageMaker instance usage | Steady baseline usage (always-on endpoints, recurring training) | Commitment risk if usage drops or shifts |
| Reserved Instances | Older commitment model tied to instance attributes | Existing commitments, some database services | Less flexible than Savings Plans |
| On-Demand Capacity Reservations | Reserve capacity in an AZ, billed whether used or not | Guaranteeing GPUs for a launch or critical job | Pays for idle capacity; no discount on its own |
| EC2 Capacity Blocks for ML | Reserve GPU instances for a defined future time window | Planned large training runs when on-demand GPU capacity is hard to get | Fixed window; plan ahead |

Do not quote discount percentages; they vary by instance type, region and time. Check current pricing.

### Spot interruption handling

1. **Checkpoint regularly** to S3 (or FSx linked to S3): model weights, optimizer state, LR scheduler, data loader position, RNG seeds and step count.
2. **Resume automatically** from the latest checkpoint on start. Make the training script idempotent.
3. **Watch for the interruption notice:** a two-minute warning is available from the instance metadata service and as an EventBridge event. EC2 can also send an earlier "rebalance recommendation" when the risk of interruption rises.
4. **Diversify capacity:** allow several instance types and AZs, and use a capacity-aware allocation strategy (for example `price-capacity-optimized`) in Auto Scaling groups, EC2 Fleet, Batch or Karpenter.
5. **Bound the loss:** checkpoint frequency should reflect how much recomputation you can afford, balanced against the time spent writing checkpoints.

SageMaker Managed Spot Training does much of this for you: set `use_spot_instances`, `max_wait`, and a `checkpoint_s3_uri`, and write checkpoints to the local checkpoint directory (`/opt/ml/checkpoints` by default) so SageMaker syncs them. See [AWS SageMaker Interview Guide](./intro_sagemaker.md#cost-optimization).

### Choosing a compute service

| Service | What it is | Fits ML when | Poor fit when |
|---|---|---|---|
| SageMaker AI | Managed training, tuning, hosting and pipelines | You want managed jobs and endpoints with little platform work | You need full control of the serving stack or already run everything on Kubernetes |
| AWS Batch | Managed job queues on EC2, Spot, Fargate or EKS | Many independent jobs: batch scoring, array jobs, sweeps, simulation; multi-node parallel jobs | Low-latency serving |
| ECS (EC2 launch type) | AWS-native container orchestration | Simple container services, including GPU tasks on GPU instances | Teams that need the Kubernetes ecosystem |
| ECS on Fargate | Serverless containers, no instances to manage | CPU inference services, feature APIs, preprocessing workers | GPU workloads (Fargate has not offered GPUs; check current docs) |
| EKS | Managed Kubernetes control plane | Platform teams running Ray, Kubeflow, vLLM or Triton, GPU node pools with Karpenter, multi-tenant clusters | Small teams without Kubernetes skills |
| Lambda | Event-driven functions, maximum 15-minute timeout, no GPUs | Glue code, S3-triggered preprocessing, light CPU inference, calling Bedrock or SageMaker endpoints | Long jobs, large models, GPU inference, steady high-throughput traffic where cold starts and per-invocation cost hurt |
| EC2 directly | Raw instances | Custom research setups, special drivers, full control | Anything you would rather not patch and babysit |

Lambda memory, package size and ephemeral storage limits have grown over time; check current docs rather than quoting them.

**What interviewers listen for:**

- Matching workload shape to compute: batch vs online, GPU vs CPU, steady vs bursty.
- Spot only with checkpointing and automatic resume, never for a job that cannot restart.
- Savings Plans for steady baseline, On-Demand or Spot for the variable part.
- Honest tradeoff between EKS flexibility and its operational cost compared with SageMaker.

---

## Data and analytics

| Service | What it does | ML use |
|---|---|---|
| AWS Glue Data Catalog | Central metadata store (databases, tables, schemas, partitions) compatible with the Hive metastore | Shared table definitions for Athena, EMR, Redshift Spectrum and Glue jobs |
| Glue crawlers | Infer schemas and partitions from data in S3 and register them | Quick cataloging of landing zones (prefer explicit schemas for production tables) |
| Glue jobs | Serverless Spark (and Python shell) ETL with job bookmarks for incremental processing | Cleaning, joining and featurizing raw data into curated Parquet or Iceberg tables |
| Glue Data Quality | Rule-based data quality checks | Block training on bad data |
| Athena | Serverless SQL over S3 using the Glue Data Catalog, billed by data scanned | Ad hoc exploration, label audits, building training sets with CTAS, querying prediction logs |
| EMR | Managed Spark, Hive, Presto/Trino and more, on EC2, on EKS or Serverless | Large or long-running Spark feature pipelines, custom cluster tuning |
| Redshift | Columnar data warehouse (provisioned or Serverless); Spectrum queries S3 | BI-grade feature tables, aggregations, SQL-first teams; Redshift ML for in-warehouse models |
| Lake Formation | Fine-grained permissions on Data Catalog tables (database, table, column, row and cell level, tag-based) | Give the fraud team's training role access to only non-PII columns; cross-account data sharing |
| Kinesis Data Streams | Durable, ordered, replayable stream with shards and multiple consumers | Real-time features, clickstream processing, custom consumers |
| Amazon Data Firehose (formerly Kinesis Data Firehose) | Fully managed buffered delivery into S3, Redshift, OpenSearch and other destinations, with optional transformation and format conversion | Landing inference logs and events into S3 as Parquet with no consumer code |
| MSK | Managed Apache Kafka (provisioned or Serverless) | Kafka-native teams, Kafka Connect ecosystem, high fan-out event backbones |
| Managed Service for Apache Flink | Managed Flink for stateful stream processing | Streaming feature computation (windows, aggregations) |
| Step Functions | Serverless state machines with retries, branching and service integrations (including SageMaker, Glue, Batch, Bedrock, Lambda) | Orchestrating ML workflows across AWS services; human approval steps |
| MWAA | Amazon Managed Workflows for Apache Airflow | Teams with existing Airflow DAGs, complex cross-system data dependencies |
| EventBridge | Event bus, rules, Scheduler and Pipes | Trigger retraining on a schedule or when new data lands; react to model registry approval or Spot interruption events |

### Streaming: Data Streams vs Firehose vs MSK

| Question | Kinesis Data Streams | Data Firehose | MSK |
|---|---|---|---|
| Do you write consumers? | Yes (Lambda, KCL apps, Flink) | No, delivery is managed | Yes (Kafka clients, Connect, Flink) |
| Replay | Yes, within the retention window | No | Yes, within retention |
| Latency | Sub-second to seconds | Buffered (seconds to minutes) | Sub-second to seconds |
| Ordering | Per shard (partition key) | Not a design goal | Per partition |
| Ops effort | Low (on-demand mode removes shard sizing) | Lowest | Highest of the three, even when managed |
| Pick it when | You need custom real-time processing on AWS | You need "get these events into S3 or a warehouse" | You already use Kafka or need its ecosystem |

### Orchestration: which tool?

| Tool | Strength | Choose when |
|---|---|---|
| SageMaker Pipelines | ML-native steps, lineage, model registry integration | The workflow is mostly SageMaker jobs |
| Step Functions | Serverless, deep AWS integrations, Standard (long-running, exactly-once) and Express (high-volume, short) workflows | Glue to SageMaker to Lambda to Bedrock workflows, event-driven pipelines, no servers to run |
| MWAA (Airflow) | Huge operator ecosystem, Python DAGs, backfills | Data platform already runs on Airflow or spans many non-AWS systems |
| EventBridge | Scheduling and event routing, not multi-step state | Triggering any of the above |

**What interviewers listen for:**

- A lake layout (raw, curated, features) in S3 with a catalog, open table formats (Iceberg) where updates or time travel matter, and Lake Formation for column and row access.
- Choosing Firehose for simple delivery and Data Streams or MSK only when replay or custom consumers are needed.
- Separating orchestration (Step Functions, Airflow, Pipelines) from triggers (EventBridge).
- Partitioning and columnar formats to keep Athena scans and costs down.

---

## Generative AI with Amazon Bedrock

Amazon Bedrock is a managed, serverless API for foundation models from several providers (including Amazon's own models). You call models without managing GPUs or model servers. AWS states that prompts and outputs are not used to train the base models and are not shared with model providers; confirm the current data-handling terms for your use case.

| Capability | What it is | Interview angle |
|---|---|---|
| Model access and APIs | `bedrock-runtime` with `InvokeModel` (provider-specific body) and the unified **Converse** API (same message format across models, tool use, streaming) | Converse makes swapping models easier; model availability differs by region, and some models may need access enabled first |
| Inference profiles | Cross-region inference profiles route requests across regions for throughput; application inference profiles let you tag usage for cost allocation | Higher availability and per-team cost tracking |
| Knowledge Bases | Managed RAG: ingest from data sources such as S3, chunk, embed, store in a vector store (for example OpenSearch Serverless or Aurora PostgreSQL with pgvector; check docs for the current list), then `Retrieve` or `RetrieveAndGenerate` with citations and metadata filters | Fast path to RAG; tradeoff is less control over chunking, retrieval and reranking than a custom pipeline |
| Agents | Managed agents that plan, call tools (action groups backed by Lambda or API schemas) and query Knowledge Bases. AWS also offers Bedrock AgentCore for running agents built with other frameworks (newer; check its current feature set) | Tool permissions, prompt injection risk, observability of tool calls |
| Guardrails | Configurable policies: content filters, denied topics, word filters, sensitive information (PII) masking or blocking, contextual grounding checks and more. Can be attached to model calls or used standalone via the `ApplyGuardrail` API | Defense in depth for inputs and outputs; not a replacement for authorization |
| Customization | Fine-tuning and other customization for supported models, and Custom Model Import for some open-weight architectures | Usually try prompting and RAG first |
| Batch inference | Asynchronous jobs over files in S3 | Cheaper for offline workloads such as bulk summarization or labeling (check current pricing) |
| Model invocation logging | Log prompts and responses to CloudWatch Logs or S3 | Auditing and evaluation; treat logs as sensitive data |

### On-demand vs Provisioned Throughput

| | On-demand | Provisioned Throughput |
|---|---|---|
| Billing | Per token (input and output), or per image or other unit | Per model unit per hour, with optional commitment terms |
| Capacity | Shared, subject to account quotas (requests and tokens per minute) | Dedicated throughput for a specific model |
| Best for | Most workloads, variable traffic, prototyping | Very high steady volume with strict throughput needs; historically required for serving many customized models (some custom models now support on-demand; check current docs) |
| Risk | Throttling at quota limits | Paying for idle capacity |

Before buying Provisioned Throughput, try quota increases, cross-region inference profiles, prompt caching (for supported models), smaller models for easy requests, and batch inference for offline work.

**What interviewers listen for:**

- When Bedrock beats self-hosting (time to market, no GPU ops, managed security) and when it does not (custom architectures, very high steady volume where dedicated hosting is cheaper, strict latency control, unsupported models).
- RAG access control: filter retrieval by the user's permissions (metadata filters or per-tenant indexes), not only by prompt instructions.
- Guardrails as one layer alongside IAM, input validation, output validation and logging.
- Token and cost tracking per team or feature, and quota planning.

For RAG design depth, see [RAG](../ai_genai/intro_rag.md) and [Vector Databases](../ai_genai/intro_vector_databases.md).

---

## Observability

| Tool | What it gives you | ML use |
|---|---|---|
| CloudWatch Metrics | Namespaced time series with dimensions; custom metrics (including via the embedded metric format in logs) | Endpoint invocations, latency and error metrics; custom metrics such as prediction score distribution or tokens per request |
| CloudWatch Logs | Log groups and streams, Logs Insights queries, metric filters, subscriptions | Training job logs, inference request logs, Bedrock invocation logs. Set retention: the default is to never expire |
| CloudWatch Alarms | Threshold or anomaly-detection alarms, composite alarms, actions (SNS, Auto Scaling, Lambda via EventBridge) | Page on p99 latency, 5xx rate, endpoint CPU or GPU utilization, failed pipeline runs |
| CloudWatch agent / Container Insights | Host and container metrics, including NVIDIA GPU metrics through the agent | Detect idle or saturated GPUs |
| X-Ray and OpenTelemetry | Distributed tracing and service maps; AWS Distro for OpenTelemetry (ADOT) is the recommended instrumentation path (check current docs on X-Ray SDK status) | Find whether latency comes from feature lookup, model call or post-processing |
| Amazon Managed Service for Prometheus and Managed Grafana | Prometheus-compatible metrics and dashboards | Common on EKS with GPU exporters |
| SageMaker Model Monitor | Data quality, model quality, bias and feature attribution drift for endpoints and batch jobs | Covered in [AWS SageMaker Interview Guide](./intro_sagemaker.md#monitoring-and-governance) and [Model Monitoring](../mlops/intro_model_monitoring.md) |

Infrastructure health (latency, errors, saturation) and model health (drift, quality, business KPIs) are different signals. A healthy endpoint can serve a badly drifted model with perfect latency.

**What interviewers listen for:**

- The four golden signals plus model-specific signals (input drift, prediction distribution, delayed ground-truth accuracy).
- Alarms tied to actions and owners, not just dashboards.
- Log retention and PII handling for request and prompt logs.
- Tracing across feature store, model and downstream calls.

---

## Cost control

| Lever | How | Why it matters for ML |
|---|---|---|
| Tagging | Tag every resource with `team`, `project`, `env`, `model`; activate the tags as cost allocation tags in the Billing console; enforce with tag policies or SCPs | Without tags you cannot say which model costs what |
| Budgets and anomaly detection | AWS Budgets with alerts (and optional actions), Cost Anomaly Detection, Cost Explorer, Data Exports (Cost and Usage Report) queried in Athena | Catch a forgotten GPU cluster in hours, not at month end |
| Spot | Training, sweeps, batch inference with checkpoints | Often the single largest saving for training |
| Commitments | Savings Plans (Compute, EC2 Instance, SageMaker) for steady baseline | Always-on endpoints and recurring training |
| Right-sizing | Watch GPU utilization and memory; use smaller GPUs, CPUs, Inferentia, quantization or batching for inference; Compute Optimizer for EC2 | Many inference endpoints run on GPUs they barely use |
| Idle cleanup | Delete idle endpoints, stop notebooks and Studio spaces (idle shutdown), scale dev endpoints to zero or use serverless or batch options, delete unused OpenSearch Serverless collections and Provisioned Throughput | Idle resources are the most common surprise bill |
| S3 lifecycle | Expire scratch data and old checkpoints, transition cold data, abort incomplete multipart uploads, limit noncurrent versions | Storage grows silently with every experiment |
| Data transfer | S3 gateway endpoint instead of NAT for S3 traffic; keep data and compute in the same region and AZ where possible; watch cross-AZ traffic in distributed training and service meshes | NAT gateway data processing charges on large dataset reads are a classic surprise |
| Log retention | Set CloudWatch Logs retention; avoid logging full payloads at high volume | Verbose inference logs can cost more than the model |
| GenAI usage | Track tokens per feature with application inference profiles and tags; cache; route easy requests to smaller models; use batch inference for offline jobs | Token spend scales with traffic and prompt length |

Never quote prices or discount percentages from memory in an interview; say you would check current pricing and model it with the AWS Pricing Calculator.

**What interviewers listen for:**

- Cost is attributed (tags, accounts per team or environment) before it is optimized.
- Alerts exist before the spike, not after.
- Concrete levers for training (Spot, right GPU, data pipeline efficiency) and serving (autoscaling, batching, smaller hardware, scale to zero).
- Awareness of hidden costs: NAT data processing, cross-AZ traffic, logs, idle vector stores, incomplete multipart uploads.

---

## Reference architectures

### Batch training pipeline

```text
  Sources (app DBs, event streams via Firehose, partner drops)
        |
        v
  +-------------------+    crawler or explicit DDL    +---------------------+
  | S3 raw zone       | ----------------------------> | Glue Data Catalog   |
  | (versioned, KMS)  |                               | + Lake Formation    |
  +-------------------+                               +---------------------+
        |
        |  Glue job or EMR Spark: validate, dedupe, join, featurize
        v
  +-------------------+
  | S3 curated zone   |  Parquet / Iceberg, partitioned by date
  +-------------------+
        |
        |  SageMaker training job (private subnets, VPC endpoints,
        |  Spot + checkpoints to S3, data streamed from S3 or FSx for Lustre)
        v
  +-------------------+     evaluate vs baseline     +----------------------+
  | model artifact S3 | ---------------------------> | Model Registry       |
  +-------------------+                              | (PendingApproval)    |
                                                     +----------------------+
                                                                |  approved
                                                                v
                                                     +----------------------+
                                                     | Batch Transform job  |
                                                     +----------------------+
                                                                |
                                                                v
                                     S3 predictions -> Athena / Redshift -> downstream apps

  Orchestration: SageMaker Pipelines or Step Functions, started by an EventBridge
  schedule or a "new data landed" event. Alarms on failure via CloudWatch + SNS.
```

Walkthrough: raw data lands in a versioned, encrypted S3 zone and is registered in the Glue Data Catalog, with Lake Formation controlling which roles see which columns. A Glue or EMR job produces curated, partitioned tables. A SageMaker training job runs in private subnets using its own execution role scoped to the curated prefix and the artifact prefix, on Spot with checkpoints. An evaluation step compares against the current production model; only passing models are registered, and approval (manual or automated) triggers a Batch Transform job that writes predictions back to S3 for analytics and downstream systems. Every run is tagged for cost and traceable from prediction back to dataset version.

### Real-time inference

```text
  Client
    |
    v
  API Gateway (auth, throttling, WAF)      or      ALB (internal, inside the VPC)
    |
    v
  Inference service: Lambda, ECS/Fargate, or pods on EKS
    |   - validate request, fetch features ----------------> Online features:
    |                                                         SageMaker Feature Store (online),
    |                                                         DynamoDB or ElastiCache
    v
  Model:  SageMaker real-time endpoint (target-tracking autoscaling on
          invocations per instance or latency)
     or:  model server pods on EKS (HPA on custom metrics + Karpenter GPU nodes)
    |
    v
  Response to client
    |
    +--> async: request/response sample -> Firehose -> S3 -> Model Monitor / Athena
    +--> CloudWatch metrics and alarms (p99 latency, 5xx, GPU utilization), X-Ray / OTel traces
```

Walkthrough: a thin service in front of the model handles auth, validation and feature lookup so the model container stays simple. Features come from a low-latency online store populated by the same pipeline that builds training features, which limits training-serving skew. The model runs on a SageMaker endpoint (least ops) or on EKS (most control, shared GPU pools). Autoscaling is based on load metrics with a minimum capacity that covers cold-start time. Requests and predictions are sampled to S3 asynchronously for drift monitoring and later labeling. Deployment safety (canary or blue/green, shadow testing) is covered in [AWS SageMaker Interview Guide](./intro_sagemaker.md#deployment-patterns) and [Model Serving](../mlops/intro_model_serving.md).

### RAG application on Bedrock

```text
  Documents (PDF, HTML, wiki exports)
        |
        v
  +---------------------------+
  | S3 docs bucket            |  versioned, KMS, metadata files with ACL / tenant tags
  +---------------------------+
        |
        |  Knowledge Base sync: parse -> chunk -> embed (Bedrock embedding model)
        v
  +---------------------------+
  | Bedrock Knowledge Base    | ---> vector store: OpenSearch Serverless
  +---------------------------+      (or another supported store)
        ^
        | 1. Retrieve (query + metadata filter for the user's tenant / groups)
        |
  User -> App (ECS/Lambda behind API Gateway, authenticated user identity)
        |
        | 2. Converse (or RetrieveAndGenerate) with retrieved chunks
        |    + Guardrail (PII masking, denied topics, grounding check)
        v
  +---------------------------+
  | Bedrock foundation model  | ---> answer + citations ---> user
  +---------------------------+

  Private access via VPC interface endpoints for Bedrock; invocation logs to S3/CloudWatch;
  offline evaluation set re-run on every prompt, model or chunking change.
```

Walkthrough: documents land in S3 with metadata describing who may see them. The Knowledge Base ingests, chunks and embeds them into a vector store. At query time the app authenticates the user, retrieves with a metadata filter derived from the user's identity (so the model never sees documents the user cannot access), then calls a Bedrock model through the Converse API with a guardrail attached. Answers return with citations. Invocation logging and an offline evaluation set catch regressions when prompts, models or chunking change. Move to a custom pipeline (your own chunking, hybrid search, reranking) when the managed Knowledge Base cannot meet quality targets; see [Enterprise RAG Search](../system_design/enterprise_rag_search.md).

---

## Code examples

These are minimal boto3 sketches. Bucket names, ARNs and IDs are placeholders. Credentials come from the environment (role, SSO profile), never from code.

### Least-privilege IAM policy for one S3 prefix

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListOnlyTheFraudPrefix",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::ml-datasets-prod",
      "Condition": { "StringLike": { "s3:prefix": ["fraud/v3/*"] } }
    },
    {
      "Sid": "ReadObjectsUnderThePrefix",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::ml-datasets-prod/fraud/v3/*"
    },
    {
      "Sid": "DecryptWithTheDatasetKey",
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:us-east-1:111122223333:key/EXAMPLE-KEY-ID"
    }
  ]
}
```

### Bucket policy: only reachable through your VPC endpoint

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAccessOutsideTheTrainingVpcEndpoint",
      "Effect": "Deny",
      "Principal": "*",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::ml-datasets-prod/*",
      "Condition": { "StringNotEquals": { "aws:SourceVpce": "vpce-0example1234567890" } }
    },
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::ml-datasets-prod", "arn:aws:s3:::ml-datasets-prod/*"],
      "Condition": { "Bool": { "aws:SecureTransport": "false" } }
    }
  ]
}
```

Test deny-based bucket policies carefully: a broad deny can also lock out administrators and the console.

### Presigned S3 URLs

```python
import boto3

s3 = boto3.client("s3")

# Download link valid for 15 minutes. The URL carries the signer's permissions,
# and it stops working when the signer's temporary credentials expire, even if
# ExpiresIn is longer.
download_url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "ml-reports", "Key": "eval/2026-10/fraud-v3.html"},
    ExpiresIn=900,
)

# Upload link: lets a labeling tool push one object without AWS credentials.
upload_url = s3.generate_presigned_url(
    "put_object",
    Params={"Bucket": "ml-raw-uploads", "Key": "incoming/batch-0042.parquet"},
    ExpiresIn=900,
)

print(download_url)
print(upload_url)
```

### Assuming a role with STS

```python
from typing import Optional

import boto3


def session_for_role(role_arn: str, session_name: str, external_id: Optional[str] = None) -> boto3.Session:
    """Return a boto3 Session that uses temporary credentials for role_arn."""
    params = {"RoleArn": role_arn, "RoleSessionName": session_name, "DurationSeconds": 3600}
    if external_id:
        params["ExternalId"] = external_id  # used when a third party assumes your role
    creds = boto3.client("sts").assume_role(**params)["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


prod = session_for_role("arn:aws:iam::444455556666:role/ModelDeployer", "deploy-fraud-v3")
print(prod.client("sts").get_caller_identity()["Arn"])
```

### Reading a secret from Secrets Manager

```python
import json
import time

import boto3

_client = boto3.client("secretsmanager")
_cache = {}
_TTL_SECONDS = 300  # refresh periodically so rotated secrets are picked up


def get_secret(secret_id: str) -> dict:
    cached = _cache.get(secret_id)
    if cached and time.time() - cached[0] < _TTL_SECONDS:
        return cached[1]
    value = json.loads(_client.get_secret_value(SecretId=secret_id)["SecretString"])
    _cache[secret_id] = (time.time(), value)
    return value


db = get_secret("prod/feature-db/readonly")
print(sorted(db.keys()))  # never log the secret values themselves
```

AWS also provides caching libraries and a Lambda extension for Parameters and Secrets that handle this pattern.

### Bedrock Converse API (model-agnostic)

```python
import os

import boto3

# Use a model ID or inference profile ID available in your account and region.
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
GUARDRAIL_ID = os.environ.get("BEDROCK_GUARDRAIL_ID")

client = boto3.client("bedrock-runtime")

request = {
    "modelId": MODEL_ID,
    "system": [{"text": "You answer questions about internal ML runbooks. Be concise."}],
    "messages": [
        {"role": "user", "content": [{"text": "How do we roll back a production endpoint?"}]}
    ],
    "inferenceConfig": {"maxTokens": 512, "temperature": 0.2},
}
if GUARDRAIL_ID:
    request["guardrailConfig"] = {
        "guardrailIdentifier": GUARDRAIL_ID,
        "guardrailVersion": os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
    }

response = client.converse(**request)
print(response["output"]["message"]["content"][0]["text"])
print(response["usage"])       # input and output token counts, useful for cost tracking
print(response["stopReason"])  # for example "end_turn", "max_tokens", "guardrail_intervened"
```

### Retrieving from a Knowledge Base, then generating

```python
import os

import boto3

agent_runtime = boto3.client("bedrock-agent-runtime")
runtime = boto3.client("bedrock-runtime")

question = "What is our policy for retraining the fraud model?"
results = agent_runtime.retrieve(
    knowledgeBaseId=os.environ["KNOWLEDGE_BASE_ID"],
    retrievalQuery={"text": question},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)["retrievalResults"]

context = "\n\n".join(r["content"]["text"] for r in results)
prompt = f"Answer using only this context. Say if it is not covered.\n\n{context}\n\nQuestion: {question}"

answer = runtime.converse(
    modelId=os.environ["BEDROCK_MODEL_ID"],
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    inferenceConfig={"maxTokens": 400},
)
print(answer["output"]["message"]["content"][0]["text"])
```

In production, add a metadata `filter` to `vectorSearchConfiguration` based on the caller's identity so retrieval respects document permissions.

### Detecting a Spot interruption notice (IMDSv2)

```python
import urllib.error
import urllib.request

IMDS = "http://169.254.169.254/latest"


def spot_interruption_pending(timeout: float = 1.0) -> bool:
    """True if EC2 has scheduled this Spot instance for interruption."""
    token_request = urllib.request.Request(
        f"{IMDS}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    token = urllib.request.urlopen(token_request, timeout=timeout).read().decode()
    action_request = urllib.request.Request(
        f"{IMDS}/meta-data/spot/instance-action",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        urllib.request.urlopen(action_request, timeout=timeout)
        return True  # 200 means an interruption action is scheduled
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False  # nothing scheduled
        raise


# Inside the training loop (save_checkpoint is your own function):
# if step % 50 == 0 and spot_interruption_pending():
#     save_checkpoint(step)  # write to S3 or FSx, then exit cleanly
```

From inside a container, IMDS may need a higher hop limit on the instance to be reachable; EventBridge Spot interruption events are an alternative signal.

---

## Interview Q&A

#### How would you give a training job read access to one S3 prefix and nothing else?

Create a dedicated execution role for the job and attach an identity policy that allows `s3:GetObject` only on `arn:aws:s3:::bucket/prefix/*` and `s3:ListBucket` on the bucket with an `s3:prefix` condition limited to that prefix. If the data is encrypted with SSE-KMS, the role also needs `kms:Decrypt` on that specific key, and the key policy must allow it. Write access should go to a separate output prefix, ideally in another bucket, so the job cannot overwrite its own training data. For defense in depth, add a bucket policy that denies access unless the request comes through your VPC endpoint or from principals in your organization. Restrict who can launch jobs with this role using `iam:PassRole`. The tradeoff is more roles to manage, which teams handle with infrastructure as code or attribute-based access control using tags.

#### What is the difference between an IAM user and an IAM role, and why should ML workloads use roles?

An IAM user is a long-lived identity that can have permanent access keys, while a role has no permanent credentials and is assumed to obtain temporary credentials from STS. Workloads such as training jobs, ECS tasks, Lambda functions and EKS pods should always use roles, because the credentials rotate automatically and are never stored in code, images or notebooks. Leaked long-lived keys are one of the most common causes of cloud incidents, and ML repos and notebooks are frequent leak sources. Humans should sign in through IAM Identity Center or federation and also receive role-based temporary credentials. CI/CD systems should use OIDC federation to assume a role rather than storing keys as secrets. The remaining use for IAM users is narrow, such as legacy tools that cannot use federation.

#### How do pods on EKS get AWS permissions without static keys?

There are two mechanisms: IAM Roles for Service Accounts (IRSA) and EKS Pod Identity. IRSA uses the cluster's OIDC provider; the role's trust policy trusts that provider and a specific namespace and service account, and the SDK exchanges the projected service account token for role credentials. EKS Pod Identity is newer: you install the Pod Identity agent add-on and create an association between a service account and a role, and the role trusts the EKS Pod Identity service principal instead of a per-cluster OIDC provider. Pod Identity is simpler to reuse across many clusters because you do not edit trust policies per cluster, while IRSA is widely supported and works in more environments. Either way, each workload should get its own service account and role rather than relying on the node's instance role, which every pod on the node could otherwise use. Blocking pod access to the node's instance metadata is part of that hardening.

#### When would you use a gateway VPC endpoint versus an interface endpoint?

Gateway endpoints exist only for S3 and DynamoDB; they are route table entries, have no extra charge at time of writing, and keep traffic from private subnets off the NAT gateway. Interface endpoints (PrivateLink) create network interfaces with private IPs in your subnets, support most AWS services and your own services, can be protected with security groups, and have hourly and per-GB charges. For a private ML environment you typically use a gateway endpoint for S3 plus interface endpoints for ECR, STS, CloudWatch Logs, SageMaker API and runtime, Bedrock runtime and Secrets Manager. An S3 interface endpoint makes sense when on-premises clients reach S3 over Direct Connect or VPN, since they cannot use a VPC route table. Both endpoint types support endpoint policies, which are useful for restricting access to your own buckets. The common mistake is forgetting the S3 gateway endpoint, so terabytes of training data flow through the NAT gateway and are billed as data processing.

#### Your AWS bill jumped overnight. How do you find the cause and stop it from happening again?

Start in Cost Explorer grouped by service, then by usage type, region, linked account and cost allocation tag, to see whether the jump is compute, storage, data transfer or a managed service. Cost Anomaly Detection, if enabled, may already point to the service and account. Typical ML culprits are GPU instances or endpoints left running, a training job stuck in a retry loop, an autoscaling group that scaled up and never down, NAT gateway data processing from dataset reads, cross-AZ transfer in distributed jobs, a large CloudWatch Logs ingest from verbose inference logging, or a burst of foundation model tokens. Use the Cost and Usage Report in Athena to drill down to resource IDs, then CloudTrail to see who or what created them. Stop the bleeding first (scale down, delete, or apply a deny via SCP), then fix the root cause. Prevention means mandatory tags, Budgets with alerts per team and account, anomaly detection, idle-resource cleanup automation, and default lifecycle and log retention policies.

#### How would you prevent data exfiltration from an ML training environment?

Build a data perimeter with several independent layers. Run training and notebooks in private subnets with no internet egress (or egress only through an inspected proxy), and give them VPC endpoints for the AWS services they need. Attach endpoint policies that allow only your organization's buckets, and bucket policies that deny access unless requests come from your VPC endpoints and your organization's principals (`aws:SourceVpce`, `aws:PrincipalOrgID`, `aws:ResourceOrgID`). Use SCPs (and resource control policies where appropriate) so no role in the ML accounts can disable these controls or write to buckets outside the organization. Encrypt with customer managed KMS keys whose key policies limit use to approved roles, enable SageMaker network isolation for jobs that do not need network access, and turn on CloudTrail data events, GuardDuty and Macie for detection. The tradeoff is friction: data scientists lose easy pip installs and public dataset downloads, so provide a private package mirror and an approved ingestion path.

#### How do you run large training jobs on Spot without losing days of progress?

Make the job resumable: checkpoint model weights, optimizer state, scheduler, data loader position and RNG state to S3 at an interval chosen from how much recomputation you can tolerate versus checkpoint write time. On startup the script should find the latest valid checkpoint and resume, and checkpoints should be written atomically (write then rename, or write a manifest last) so a half-written file is never loaded. Listen for the two-minute interruption notice from instance metadata or EventBridge and trigger a final checkpoint. Diversify instance types and AZs and use a capacity-aware allocation strategy, because GPU Spot capacity for a single instance type can disappear for hours. For multi-node distributed training, an interruption of one node usually stops the whole job, so elastic training frameworks or falling back to On-Demand or Capacity Blocks for the critical final phase can be worthwhile. SageMaker Managed Spot Training handles restarts and checkpoint syncing; you still have to write resume logic.

#### How would you design a multi-account AWS setup for an ML organization?

Use AWS Organizations with AWS Control Tower to create a landing zone: a management account used only for billing and organization policy, a log archive account for CloudTrail and Config, and a security tooling account. Put ML workloads in separate accounts per environment (dev, staging, prod) and often per team or product, grouped into organizational units that receive different SCPs and Control Tower controls. A shared services account can host the model registry, ECR images and CI/CD tooling, and a data lake account can own curated data shared via Lake Formation. Humans access accounts through IAM Identity Center with permission sets, and pipelines assume deployment roles across accounts. Accounts give strong blast-radius isolation, clean cost attribution and separate service quotas, at the cost of more cross-account plumbing for KMS keys, bucket policies, ECR repository policies and networking. Account vending through Control Tower Account Factory and infrastructure as code keeps that overhead manageable.

#### How do you deploy a model trained in a dev account into a production account?

Do not copy credentials or let production read from dev freely; promote an immutable, approved artifact. One common pattern: the training account writes the model artifact to S3 encrypted with a customer managed KMS key and registers it in a model registry, then grants the production deployment role read access through the bucket policy and KMS key policy. The container image lives in ECR with a repository policy that lets the production account pull it, or it is replicated into the production account's registry. SageMaker model package groups support resource policies (and sharing via AWS RAM; check current docs), so production can reference the approved model package directly. A pipeline in a deployment or production account then assumes a role, creates the model and endpoint in production, runs smoke tests and shifts traffic gradually. An alternative is to copy the artifact and image into the production account at approval time, which decouples prod from dev availability at the cost of duplicate storage.

#### How do you choose between EBS, EFS, FSx for Lustre and streaming from S3 for training data?

The deciding factors are dataset size, number of nodes reading concurrently, how many epochs re-read the data, and whether profiling shows GPUs waiting on I/O. For a single-node job with a dataset that fits on local disk, copying from S3 to local NVMe or EBS once is simple and fast. For multi-node jobs over large datasets, streaming sharded files from S3 (SageMaker FastFile or Pipe mode, Mountpoint for Amazon S3, or the S3 connector for PyTorch) avoids provisioning anything and scales well if files are large and spread across prefixes. FSx for Lustre adds a high-throughput parallel file system, linked to S3, that helps when many GPU nodes repeatedly read the same data or need POSIX semantics, but it costs provisioned capacity and must be cleaned up. EFS is good for shared code, configs and notebooks but generally not for feeding a large GPU cluster. Many teams also fix "slow storage" by repacking millions of small files into large shards, which helps every option.

#### When would you serve a model with Lambda, ECS on Fargate, EKS, or a SageMaker endpoint?

Lambda suits light CPU models, pre and post processing, or calls to Bedrock and SageMaker endpoints, especially with spiky low traffic; its limits are the 15-minute timeout, no GPUs, package size constraints and cold starts. ECS on Fargate works well for CPU inference services where you want containers without managing instances. EKS fits when you need GPUs with custom serving stacks (vLLM, Triton, Ray Serve), want to bin-pack many models onto shared GPU nodes, or already have a platform team on Kubernetes; it has the highest operational burden. SageMaker endpoints give managed GPU or CPU hosting with autoscaling, variants for canary testing, async and serverless options, and integration with Model Monitor, with less control over the runtime. The choice is mostly about traffic shape, hardware needs and team skills, not model accuracy. Many organizations use more than one: SageMaker for classic models and EKS for LLM serving, for example.

#### Kinesis Data Streams, Data Firehose or MSK: which would you use for a real-time feature pipeline?

If you only need to land events in S3 or a warehouse for later training, Data Firehose is the simplest: no consumers, managed buffering, optional format conversion to Parquet. If you need to compute features in near real time with replay ability and multiple independent consumers, Kinesis Data Streams with Lambda or Managed Service for Apache Flink is the AWS-native choice, and on-demand capacity mode removes shard planning. MSK makes sense when the company already standardizes on Kafka, needs Kafka Connect or the Kafka client ecosystem, or wants portability across environments, accepting more tuning and operational work. Ordering is guaranteed per shard or partition, so choose partition keys (such as user ID) that keep related events together without creating hot partitions. A common pattern combines them: stream processing writes fresh features to an online store, and Firehose archives raw events to S3 for training and backfills.

#### When would you use Bedrock instead of hosting an open-weights model yourself on SageMaker or EKS?

Bedrock is the default when you want strong models quickly, have variable traffic, and do not want to manage GPU capacity, model servers, scaling or patching; you pay per token and get Guardrails, Knowledge Bases and logging built in. Self-hosting fits when you need a model or architecture Bedrock does not offer, deep control over inference (custom decoding, specific quantization, LoRA adapters swapped per tenant), strict latency control, or when steady high volume makes dedicated GPUs cheaper per token than on-demand pricing. Self-hosting brings GPU capacity planning, autoscaling with slow cold starts, observability and security patching onto your team. Data residency and privacy can be satisfied by either, so check the specific requirements rather than assuming self-hosting is required. A practical approach is to prototype on Bedrock, measure quality and cost per request, and only move a workload to self-hosting when the numbers or a hard requirement justify it.

#### What is the difference between on-demand and Provisioned Throughput in Bedrock, and how do you decide?

On-demand bills per token (or other unit) on shared capacity and is subject to account quotas on requests and tokens per minute, so heavy traffic can be throttled. Provisioned Throughput reserves dedicated capacity for a specific model, billed per model unit per hour with optional commitment terms, so you pay whether or not you use it. Historically it was required to serve many customized models; check current docs, since some custom models now support on-demand inference. Decide by measuring steady-state token throughput and peak requirements: if traffic is variable or modest, stay on-demand and request quota increases or use cross-region inference profiles. If traffic is large, steady and latency-critical, or a custom model requires it, compare the cost of provisioned units against projected on-demand spend. Batch inference and prompt caching are often cheaper ways to reduce cost before committing.

#### How would you secure a multi-tenant RAG application built on Bedrock?

Authentication happens in the application, and the user's identity must drive retrieval: tag each document chunk with tenant and group metadata at ingestion and apply a metadata filter on every retrieve call, or use separate indexes or Knowledge Bases per tenant when isolation requirements are strict. Never rely on the prompt to tell the model to ignore other tenants' data, since prompt injection inside documents or user input can override instructions. Attach a guardrail for PII masking, denied topics and grounding checks, and validate outputs before they reach downstream tools. Keep the app, Knowledge Base and vector store in private networking with VPC endpoints, encrypt the source bucket and vector store with customer managed keys, and give the app role only the specific `bedrock:` actions and resources it needs. Turn on invocation logging, but treat logs as sensitive data with restricted access and retention. Finally, test with adversarial documents and cross-tenant queries as part of the evaluation suite.

#### How does S3's consistency model and prefix layout affect ML pipelines?

Since late 2020, S3 provides strong read-after-write consistency for writes, overwrites, deletes and listings, so a pipeline step can list and read objects that a previous step just wrote without sleep-and-retry workarounds. That does not make concurrent writers to the same key safe; the last writer wins, so pipelines should write to unique, versioned paths (for example including a run ID) and publish a manifest or commit marker when a dataset is complete. Open table formats like Apache Iceberg add atomic commits and snapshots on top of S3 for tables that are updated. Request throughput scales per prefix, so spreading heavy reads across prefixes and using fewer, larger files helps data loaders at scale. Millions of tiny files also inflate request costs and listing time, so compaction into shards is usually worth it.

#### How should an inference container get a database password or third-party API key?

Store the secret in Secrets Manager (or SSM Parameter Store for simple cases), encrypted with a KMS key, and grant the container's task role, pod role or SageMaker execution role permission to read only that secret ARN. The application fetches the value at startup or on demand and caches it with a short TTL so rotation is picked up; ECS can also inject secrets as environment variables at container start, at the cost of not seeing rotations until restart. Never bake secrets into images, model artifacts, notebooks or plain environment configuration checked into Git. Enable automatic rotation where supported, and prefer IAM authentication over passwords when the target supports it, such as IAM database authentication for RDS. CloudTrail records `GetSecretValue` calls, which helps investigate misuse.

---

## Common Pitfalls

| Problem | Why it hurts | Fix |
|---|---|---|
| Access keys in notebooks, code or Docker images | Keys leak through Git, logs and shared images and stay valid until revoked | Use roles everywhere, OIDC for CI, IAM Identity Center for humans; scan repos for secrets |
| One shared "ML admin" role with `s3:*` and `sagemaker:*` on `*` | Any compromised job can read or delete everything | One role per workload, scoped to bucket prefixes and specific actions; control `iam:PassRole` |
| Training data read through a NAT gateway | Large per-GB processing charges and unnecessary internet path | Add an S3 gateway endpoint and route tables for private subnets |
| Missing interface endpoints in a private VPC | Jobs hang or time out with unclear errors | Inventory every AWS API the job calls and add endpoints (ECR, STS, Logs, SageMaker, Bedrock, Secrets Manager) |
| SSE-KMS bucket with no `kms:Decrypt` grant | `AccessDenied` that looks like an S3 permission problem | Grant `kms:Decrypt` in the role policy and key policy; enable S3 Bucket Keys |
| Millions of tiny training files in one prefix | Slow data loading, idle GPUs, high request costs | Shard into large files, spread across prefixes, or cache on FSx for Lustre |
| Spot training without checkpoints | Interruptions throw away hours of GPU time | Checkpoint to S3, auto-resume, diversify instance types and AZs |
| Idle endpoints, notebooks, GPU nodes and vector store collections | Steady cost for zero value | Idle shutdown, scale-to-zero or serverless for dev, scheduled cleanup, Budgets alerts |
| No lifecycle rules on scratch and checkpoint prefixes | Storage and incomplete multipart uploads grow forever | Expiration and transition rules, abort incomplete multipart uploads, limit noncurrent versions |
| CloudWatch Logs with default retention | Logs never expire and costs accumulate; PII retained indefinitely | Set retention per log group; avoid logging full payloads |
| Untagged resources | Cannot attribute cost per model or team | Enforce tags with tag policies or SCPs; activate cost allocation tags |
| RAG access control done in the prompt | Prompt injection or bugs leak other users' documents | Filter retrieval by identity using metadata or separate indexes; guardrails as an extra layer |
| Quoting prices or limits from memory | Numbers change and undermine credibility | Say "check current pricing and quotas" and explain how you would model cost |

---

## Related Topics

| Topic | Why it's related |
|---|---|
| [AWS SageMaker Interview Guide](./intro_sagemaker.md) | Training, hosting, pipelines, registry and Model Monitor in depth |
| [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md) | SageMaker vs Vertex AI vs Azure ML |
| [GCP for ML Engineers](./gcp_for_ml_engineers.md) | The equivalent core services on Google Cloud |
| [Azure for ML Engineers](./azure_for_ml_engineers.md) | The equivalent core services on Azure |
| [Cloud Service Mapping](./cloud_service_mapping.md) | Side-by-side mapping of AWS, GCP and Azure services |
| [Google Vertex AI Interview Guide](./intro_vertex_ai.md) | Managed ML platform on GCP |
| [Azure Machine Learning Interview Guide](./intro_azure_ml.md) | Managed ML platform on Azure |
| [Model Serving](../mlops/intro_model_serving.md) | Serving patterns behind the real-time architecture |
| [Model Monitoring](../mlops/intro_model_monitoring.md) | Drift and quality monitoring beyond infrastructure metrics |
| [CI/CD for ML](../mlops/intro_cicd_for_ml.md) | Promotion pipelines across accounts |
| [Feature Stores](../mlops/intro_feature_stores.md) | Online and offline features for training and inference |
| [Terraform](../devops/intro_terraform.md) | Managing IAM, VPCs and endpoints as code |
| [Kubernetes](../devops/intro_kubernetes.md) | Background for EKS-based training and serving |
| [Docker](../devops/intro_docker.md) | Containers for SageMaker, ECS, EKS and Lambda |
| [Observability](../devops/intro_observability.md) | Metrics, logs and traces fundamentals |
| [Apache Airflow](../data_engineering/intro_apache_airflow.md) | Background for MWAA |
| [Apache Spark](../data_engineering/intro_apache_spark.md) | Background for Glue and EMR jobs |
| [Apache Kafka](../data_engineering/intro_apache_kafka.md) | Background for MSK |
| [Apache Iceberg](../data_engineering/intro_apache_iceberg.md) | Table format for S3 data lakes |
| [RAG](../ai_genai/intro_rag.md) | Retrieval-augmented generation design |
| [Vector Databases](../ai_genai/intro_vector_databases.md) | Vector stores behind Knowledge Bases |
| [LLM Security](../ai_genai/intro_llm_security.md) | Prompt injection and data leakage risks for Bedrock apps |
| [Enterprise RAG Search](../system_design/enterprise_rag_search.md) | System design walkthrough for a RAG platform |
