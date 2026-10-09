# AWS vs GCP vs Azure: Service Mapping for ML Engineers

Most ML engineers know one cloud well and the other two loosely. Interviewers know this, so cross-cloud questions usually test whether you understand the underlying concept well enough to translate it, not whether you memorized three product catalogs.

How to use this guide in an interview:

1. **Name the concept first.** "We need a workload identity for the training job so it can read the bucket without a static key."
2. **Then name the service on the cloud you know.** "On AWS that is the SageMaker AI execution role."
3. **Then translate, and say where the analogy breaks.** "On GCP it would be a service account attached to the Vertex AI custom job; the difference is that GCP grants the role on the bucket or project rather than through a policy document attached to the role."

This guide focuses on the infrastructure around ML: identity, networking, storage, compute, data, GenAI and cost. For the ML platforms themselves (Amazon SageMaker AI, Vertex AI, Azure Machine Learning) feature by feature, see the [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md); the ML platform table below is intentionally brief.

> Product names in this space change often (several services in these tables were renamed in 2024 and 2025). Names are believed current as of 2026, and the ones most likely to drift are marked "check current name". In an interview, getting the concept right matters more than getting the newest brand name.

---

## Table of Contents

1. [Service mapping tables](#service-mapping-tables)
   - [Identity and access](#identity-and-access)
   - [Networking](#networking)
   - [Storage](#storage)
   - [Compute](#compute)
   - [Data and streaming](#data-and-streaming)
   - [ML platform](#ml-platform)
   - [Generative AI](#generative-ai)
   - [Observability and cost](#observability-and-cost)
2. [Key behavioural differences](#key-behavioural-differences)
3. [Cloud-agnostic ML architecture](#cloud-agnostic-ml-architecture)
4. [Migration and multi-cloud scenarios](#migration-and-multi-cloud-scenarios)
5. [Terraform example: one versioned bucket on three clouds](#terraform-example-one-versioned-bucket-on-three-clouds)
6. [Interview Q&A](#interview-qa)
7. [Common Pitfalls](#common-pitfalls)
8. [Related Topics](#related-topics)

---

## Service mapping tables

Read each row as "same job, different implementation". Where the Notes column says there is no clean equivalent, say that in an interview instead of forcing a match: it signals real experience.

### Identity and access

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Resource hierarchy | AWS Organizations: management account, organizational units (OUs), member accounts | Organization, folders, projects | Microsoft Entra tenant, management groups, subscriptions, resource groups | AWS accounts are hard isolation boundaries with their own IAM. GCP projects inherit IAM from folders and the org. Azure has two levels below management groups: subscriptions (billing, quota, policy scope) and resource groups (lifecycle grouping). See [Key behavioural differences](#key-behavioural-differences). |
| Human identity and SSO | AWS IAM Identity Center (federates to an external IdP) | Cloud Identity or Google Workspace; Workforce Identity Federation for external IdPs | Microsoft Entra ID (formerly Azure AD) | Entra ID is both the directory and the IdP for Azure, so Azure identity is the most tightly integrated with corporate accounts. |
| Permission bundle | IAM policies (JSON documents of actions, resources, conditions) | IAM roles (predefined or custom collections of permissions) | Azure RBAC role definitions (built-in or custom) | On AWS the policy is the unit you attach. On GCP and Azure you bind a role to a principal at a scope. AWS "role" means an assumable identity, not a permission bundle, which confuses people moving between clouds. |
| Workload identity for compute | IAM role via instance profile, task role, Lambda or SageMaker AI execution role; EKS Pod Identity or IRSA for pods | Service account attached to the VM, job or Cloud Run service; Workload Identity Federation for GKE for pods | Managed identity (system-assigned or user-assigned); Microsoft Entra Workload ID for AKS pods | All three issue short-lived credentials from a metadata endpoint or token exchange. On GCP a service account is both an identity and a resource you can grant access to (for example, who may impersonate it). |
| Keyless CI federation | IAM OIDC identity provider plus `sts:AssumeRoleWithWebIdentity` | Workload Identity Federation (workload identity pools and providers) | Workload identity federation via federated credentials on an app registration or user-assigned managed identity | All three trust an external OIDC issuer (for example GitHub Actions) and exchange its token for short-lived cloud credentials. Scope the trust to repository, branch or environment claims. |
| Secrets | AWS Secrets Manager (also SSM Parameter Store for config) | Secret Manager | Azure Key Vault (secrets) | Rotation differs: Secrets Manager has managed rotation with Lambda; Secret Manager rotation schedules send notifications that you act on; Key Vault emits near-expiry events. Check current rotation features before claiming "automatic". |
| Key management | AWS KMS (CloudHSM for dedicated HSMs) | Cloud KMS (Cloud HSM, Cloud External Key Manager) | Azure Key Vault keys, Azure Key Vault Managed HSM | GCP calls customer-managed keys "CMEK". Not every managed ML resource supports customer-managed keys on every cloud; verify per service. |
| Audit logs | AWS CloudTrail | Cloud Audit Logs | Azure Activity Log (control plane), resource logs via diagnostic settings, Entra sign-in and audit logs | Defaults differ: CloudTrail records management events by default but data events (such as S3 object reads) must be enabled. GCP Admin Activity logs are always on, while Data Access logs are mostly off by default (BigQuery is an exception). Azure data-plane logs need diagnostic settings. |
| Org-wide guardrails | Service control policies (SCPs) and resource control policies (RCPs) in AWS Organizations | Organization Policy Service constraints; IAM deny policies | Azure Policy (assigned at management group, subscription or resource group) | SCPs and RCPs cap the maximum permissions and never grant anything. GCP organization policies restrict resource configuration (allowed locations, no service account keys, no external IPs), not who can act. Azure Policy evaluates resource properties with effects such as deny, audit, modify and deployIfNotExists. |
| Data exfiltration perimeter | "Data perimeter" pattern: SCPs, RCPs and VPC endpoint policies combined | VPC Service Controls (service perimeters around projects) | No single equivalent: private endpoints, disabling public network access, Azure Policy; Network Security Perimeter for supported services (check availability) | VPC Service Controls is the most direct product for "data in this perimeter cannot leave". On AWS and Azure you assemble the same guarantee from several controls. |

### Networking

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Virtual network | VPC (regional); subnets are zonal (one Availability Zone each) | VPC network (global); subnets are regional | Virtual network (VNet, regional); subnets span the region's zones | GCP is the outlier: one VPC can hold subnets in many regions that talk over internal IPs without peering. |
| Network sharing across teams | VPC sharing via AWS Resource Access Manager; Transit Gateway | Shared VPC (host project, service projects); Network Connectivity Center | Hub-and-spoke with VNet peering; Azure Virtual WAN | Shared VPC is a very common GCP enterprise pattern: networking lives in a host project and ML projects attach to it. |
| Peering | VPC peering (non-transitive) | VPC Network Peering (non-transitive) | VNet peering (non-transitive; gateway transit is an option) | Non-transitivity bites in all three. Hubs (Transit Gateway, NCC, Virtual WAN or a hub VNet) solve it. |
| Private access to managed services | Interface VPC endpoints (AWS PrivateLink); gateway endpoints for S3 and DynamoDB | Private Service Connect (PSC) endpoints; Private Google Access for subnets without external IPs | Private Endpoint (Azure Private Link); service endpoints (older, coarser) | All three give a private IP inside your network for a managed service. DNS is the usual failure point: the service hostname must resolve to the private IP. |
| Publishing your own private service | PrivateLink endpoint service (behind a Network Load Balancer) | PSC service attachment (published service) | Azure Private Link service (behind a Standard Load Balancer) | Useful for exposing a model-serving API to another account, project or tenant without peering networks. |
| Outbound NAT | NAT gateway (per Availability Zone) | Cloud NAT (regional, configured on a Cloud Router) | Azure NAT Gateway | Training jobs in private subnets need NAT or private endpoints to pull packages and images. |
| Firewalling | Security groups (stateful, per network interface); network ACLs (stateless, per subnet) | VPC firewall rules and firewall policies (target by network tag or service account) | Network security groups (on subnet or NIC); Azure Firewall | GCP rules can target by service account, which maps neatly to workload identity. |
| L7 load balancer | Application Load Balancer | Application Load Balancer (global external, regional external, internal) | Azure Application Gateway (regional); Azure Front Door (global) | GCP's global external load balancer uses a single anycast IP in front of backends in many regions. AWS and Azure typically add a global layer (Global Accelerator, CloudFront, Front Door) for the same effect. |
| L4 load balancer | Network Load Balancer | Network Load Balancer (proxy or passthrough) | Azure Load Balancer | Used for gRPC or TCP model servers and for private service publishing. |
| CDN | Amazon CloudFront | Cloud CDN (Media CDN for large media) | Azure Front Door | Older Azure CDN offerings are being retired in favour of Front Door; check current status. |
| DNS | Amazon Route 53 (public and private hosted zones) | Cloud DNS (public and private zones) | Azure DNS and Azure Private DNS zones | Private endpoint DNS on Azure (the `privatelink.*` zones) is a classic source of "works from my laptop, fails from the cluster" bugs. |
| Hybrid connectivity | AWS Direct Connect; Site-to-Site VPN | Cloud Interconnect; Cloud VPN | Azure ExpressRoute; VPN Gateway | Relevant when training data stays on premises or another cloud. |

### Storage

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Object storage | Amazon S3 | Cloud Storage (GCS) | Azure Blob Storage inside a storage account; ADLS Gen2 when hierarchical namespace is enabled | Azure adds a level: buckets map to containers inside a storage account, and many settings (redundancy, network rules, versioning) live on the account. |
| Storage classes | S3 Standard, Standard-IA, One Zone-IA, Intelligent-Tiering, Glacier Instant Retrieval, Glacier Flexible Retrieval, Glacier Deep Archive; S3 Express One Zone for low latency | Standard, Nearline, Coldline, Archive; Autoclass for automatic tiering | Hot, Cool, Cold, Archive access tiers | Retrieval semantics differ: GCS colder classes stay online with retrieval fees and minimum storage durations; Azure Archive is offline and needs rehydration; S3 Glacier Flexible Retrieval and Deep Archive need a restore. Never put training data you read every epoch in an archive tier. |
| Lifecycle rules | S3 Lifecycle | Object Lifecycle Management | Blob lifecycle management policies | Use these to expire old checkpoints and intermediate pipeline artifacts. |
| Block storage | Amazon EBS | Persistent Disk and Hyperdisk | Azure Managed Disks | Each also has ephemeral local NVMe (instance store, Local SSD, temp or NVMe disks) that is fast but lost when the VM stops. |
| Shared file (NFS/SMB) | Amazon EFS; FSx for NetApp ONTAP, OpenZFS, Windows File Server | Filestore; NetApp Volumes | Azure Files (SMB, NFS); Azure NetApp Files | General shared storage, usually not the fastest option for large-scale training input. |
| High-throughput file for training | Amazon FSx for Lustre (can link to an S3 bucket) | Google Cloud Managed Lustre (newer; check region availability); Cloud Storage FUSE with caching | Azure Managed Lustre (can integrate with Blob Storage) | Lustre keeps GPUs fed when many workers read small files. Object storage mounts (Mountpoint for Amazon S3, Cloud Storage FUSE, BlobFuse2) are simpler and often enough for sharded, sequential reads. |
| Data lake catalog | AWS Glue Data Catalog; AWS Lake Formation for fine-grained access; Amazon S3 Tables for managed Iceberg tables | Dataplex Universal Catalog; BigLake for open table formats (check current names) | Microsoft Purview for governance; OneLake catalog in Microsoft Fabric; Unity Catalog on Azure Databricks | No clean one-to-one match. On Azure the catalog depends on whether the data platform is Fabric, Databricks or both. |

### Compute

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Virtual machines | Amazon EC2 | Compute Engine | Azure Virtual Machines | GCP also offers custom machine types for some families. |
| VM autoscaling groups | EC2 Auto Scaling groups | Managed instance groups (MIGs) | Virtual Machine Scale Sets | Same concept: a template plus desired count plus health checks. |
| NVIDIA GPU VMs | P and G instance families | Accelerator-optimized A-series and G-series machine types; GPUs attachable to N1 | NC, ND and NV series | Family names and generations change often; check the current catalog and, more importantly, your regional GPU quota. Azure ND series also includes AMD GPU options. |
| First-party ML accelerators | AWS Trainium (training) and AWS Inferentia (inference), via the AWS Neuron SDK | Cloud TPU, via JAX, PyTorch/XLA or TensorFlow | No first-party accelerator offered as a general VM family to my knowledge; use GPU VMs | Porting to Trainium or TPU is a software project (compiler, kernels, sharding), not just an instance swap. |
| Spot capacity | EC2 Spot Instances (two-minute interruption notice); SageMaker AI Managed Spot Training | Spot VMs (short preemption notice, around 30 seconds; no maximum runtime unlike legacy preemptible VMs) | Azure Spot Virtual Machines (eviction notice via Scheduled Events, around 30 seconds); "low priority" tier in Azure ML compute clusters | Checkpoint often and make the job resumable. Spot availability for large GPU types varies a lot by region and time. |
| Reserved or scheduled GPU capacity | On-Demand Capacity Reservations; EC2 Capacity Blocks for ML | Reservations; Dynamic Workload Scheduler (check current modes) | On-demand capacity reservations | Getting GPUs at all is often the real constraint. Ask about quota and reservation strategy, not just price. |
| Managed Kubernetes | Amazon EKS (EKS Auto Mode for managed nodes) | GKE (Standard and Autopilot) | AKS (AKS Automatic for a more managed mode; check status) | Kubernetes is the most portable compute layer, but node pools, GPU drivers, autoscalers and identity integration still differ per cloud. |
| Serverless containers | Amazon ECS on AWS Fargate; AWS App Runner for simple web services (check current status) | Cloud Run (services and jobs; GPU support in selected regions) | Azure Container Apps (serverless GPU support in selected regions) | Cloud Run and Container Apps scale to zero on request traffic. Fargate runs tasks without servers but has no GPU support, so there is no exact AWS equivalent of serverless GPU containers. |
| Functions | AWS Lambda | Cloud Run functions (formerly Cloud Functions) | Azure Functions | Fine for glue (triggering pipelines on events), poor for model inference beyond small CPU models. |
| Batch jobs | AWS Batch | Batch (Google Cloud Batch) | Azure Batch | Useful for embarrassingly parallel preprocessing or scoring outside the ML platform. |
| Container registry | Amazon ECR | Artifact Registry (replaced the deprecated Container Registry) | Azure Container Registry | Cross-cloud image pulls cost egress and add latency; mirror images per cloud. |
| Large training clusters | Amazon SageMaker HyperPod; AWS ParallelCluster | GKE with GPU or TPU node pools; Cluster Toolkit | Azure CycleCloud; ND series with InfiniBand; Azure ML compute clusters | Interconnect differs (EFA on AWS, Google's GPU networking stack, InfiniBand on Azure ND), which matters for multi-node training throughput. |

### Data and streaming

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Data warehouse | Amazon Redshift (provisioned or Redshift Serverless) | BigQuery | Microsoft Fabric Warehouse; Azure Synapse Analytics dedicated and serverless SQL pools; Databricks SQL on Azure Databricks | BigQuery is serverless by design. Microsoft's direction is Fabric, but many enterprises still run Synapse. See [BigQuery vs Redshift vs Synapse and Fabric](#bigquery-vs-redshift-vs-synapse-and-fabric). |
| Managed Spark | Amazon EMR (on EC2, on EKS, Serverless); AWS Glue Spark jobs | Dataproc; Google Cloud Serverless for Apache Spark (formerly Dataproc Serverless) | Azure Databricks; Fabric Spark; Synapse Spark pools | Databricks is available on all three clouds and is often the portable choice for Spark-heavy teams. |
| ETL and data integration | AWS Glue | Dataflow (Apache Beam); Cloud Data Fusion | Azure Data Factory; Data Factory in Fabric | Dataflow runs Beam, so pipelines are portable in principle to other Beam runners (Flink, Spark). |
| Managed Airflow | Amazon Managed Workflows for Apache Airflow (MWAA) | Cloud Composer | Apache Airflow jobs in Microsoft Fabric (check current name); the older Data Factory managed Airflow option has been announced for retirement | Airflow DAGs are portable; operators and connections are not. Many Azure teams self-host Airflow on AKS. |
| Streaming log or event bus | Amazon Kinesis Data Streams; Amazon MSK (managed Kafka) | Pub/Sub; Google Cloud Managed Service for Apache Kafka | Azure Event Hubs (with a Kafka-compatible endpoint) | Kinesis and Event Hubs are partitioned logs where consumers track offsets. Pub/Sub is subscription-based with per-message acknowledgement, plus ordering keys and seek for replay. Managed Kafka gives the most portable semantics. |
| Queues and pub/sub messaging | Amazon SQS, Amazon SNS, Amazon EventBridge | Pub/Sub, Cloud Tasks, Eventarc | Azure Service Bus, Storage Queues, Event Grid | Pub/Sub covers both streaming ingest and messaging on GCP; the other clouds split these across products. |
| Stream processing | Amazon Managed Service for Apache Flink | Dataflow (Beam streaming) | Azure Stream Analytics; Spark Structured Streaming on Azure Databricks; Fabric Real-Time Intelligence | No first-party managed Flink on Azure that I would rely on today; teams use Databricks or run Flink on AKS. |
| NoSQL (key-value, document, wide-column) | Amazon DynamoDB | Firestore; Bigtable (wide-column) | Azure Cosmos DB | Bigtable is a common online feature store backend on GCP. Cosmos DB exposes several APIs (NoSQL, MongoDB, Cassandra and others). |
| In-memory cache | Amazon ElastiCache (Valkey, Redis OSS, Memcached); Amazon MemoryDB | Memorystore | Azure Managed Redis; Azure Cache for Redis (older offering, retirement announced; check timeline) | Used for low-latency feature lookups and prediction caching. |
| Postgres with pgvector | Amazon RDS for PostgreSQL; Amazon Aurora PostgreSQL | Cloud SQL for PostgreSQL; AlloyDB | Azure Database for PostgreSQL flexible server | pgvector on managed Postgres is the most portable vector option across clouds. |
| Vector-capable search | Amazon OpenSearch Service (k-NN); Amazon S3 Vectors (newer); Aurora with pgvector | Vertex AI Vector Search; BigQuery vector search; AlloyDB vector search | Azure AI Search; vector search in Azure Cosmos DB | Product boundaries differ: Azure AI Search is a full search engine (keyword, vector, hybrid, semantic ranking), while Vertex AI Vector Search is primarily an approximate nearest-neighbour index. |

### ML platform

This table is deliberately short. The [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md) covers the platforms feature by feature, and the provider guides ([SageMaker](./intro_sagemaker.md), [Vertex AI](./intro_vertex_ai.md), [Azure ML](./intro_azure_ml.md)) cover each one in depth.

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Platform | Amazon SageMaker AI (inside the broader Amazon SageMaker umbrella with SageMaker Unified Studio) | Vertex AI | Azure Machine Learning (GenAI work increasingly in Microsoft Foundry) | Azure splits classic ML (Azure ML workspace) from GenAI (Foundry projects), although they share some underlying resources. |
| Notebooks | SageMaker Studio (JupyterLab and Code Editor spaces) | Vertex AI Workbench instances; Colab Enterprise | Azure ML compute instances | All are VMs underneath; idle shutdown is the main cost control. |
| Training | Training jobs; HyperPod for large clusters | Custom training jobs | Command jobs on compute clusters or serverless compute | Container contract differs (paths and environment variables); see the [migration walkthrough](#migrating-a-sagemaker-ai-pipeline-to-vertex-ai). |
| Pipelines | SageMaker Pipelines | Vertex AI Pipelines (Kubeflow Pipelines SDK, TFX) | Azure ML pipelines (components) | Pipeline definitions are not portable between clouds. Vertex AI Pipelines uses the open KFP format, which also runs on self-managed Kubeflow. |
| Experiment tracking | Managed MLflow in SageMaker AI | Vertex AI Experiments | MLflow-compatible tracking built into the workspace | MLflow is the portable layer. Azure ML speaks the MLflow protocol natively; Vertex AI needs a self-hosted MLflow if you want it. |
| Model registry | SageMaker Model Registry (model package groups, approval status) | Vertex AI Model Registry (versions, aliases) | Azure ML model registry; registries for cross-workspace sharing | Approval semantics differ: SageMaker has an explicit approval status; on Vertex AI you typically use aliases and labels plus your own gate. |
| Feature store | SageMaker Feature Store | Vertex AI Feature Store (BigQuery as offline source) | Azure ML managed feature store | See [Feature Stores](../mlops/intro_feature_stores.md) for the general pattern. |
| Online endpoints | Real-time, serverless and asynchronous inference endpoints | Vertex AI endpoints | Managed online endpoints; Kubernetes online endpoints | Traffic splitting across model versions exists on all three. |
| Batch prediction | Batch Transform | Batch prediction | Batch endpoints | Batch scoring is often cheaper done directly in the warehouse or Spark when the model is small. |
| Model monitoring | SageMaker Model Monitor; SageMaker Clarify | Vertex AI Model Monitoring | Azure ML model monitoring | All cover drift and data quality; business KPI monitoring is still yours. |
| AutoML | Autopilot capabilities in SageMaker Canvas | Vertex AI AutoML | Azure Automated ML | Good baselines; outputs are the least portable artefact on this list. |

### Generative AI

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Managed foundation model API | Amazon Bedrock (Amazon Nova plus third-party models) | Gemini API on Vertex AI; Vertex AI Model Garden for partner and open models | Microsoft Foundry (formerly Azure AI Foundry), including Azure OpenAI models and other catalog models | Catalogs overlap but are not identical, and availability differs by region. The "home" model families differ: OpenAI models on Azure, Gemini on Google Cloud, Nova on AWS. Check the current catalog per region before committing. |
| Self-hosting open-weight models | SageMaker AI endpoints and JumpStart; Bedrock Custom Model Import | Model Garden deployment to Vertex AI endpoints; GKE | Foundry managed compute deployments; Azure ML online endpoints | Same tradeoff everywhere: you manage capacity and scaling, but control the model and serving stack (for example vLLM). |
| Fine-tuning | Bedrock model customization; SageMaker AI training jobs | Vertex AI tuning (supervised fine-tuning for Gemini and open models) | Fine-tuning in Foundry (including Azure OpenAI models) | Which base models can be tuned, and with which methods, differs and changes frequently. |
| Reserved throughput | Bedrock Provisioned Throughput | Vertex AI Provisioned Throughput | Provisioned deployments (provisioned throughput units) in Foundry | All three sell reserved capacity for predictable latency on top of pay-per-token. |
| Managed RAG | Amazon Bedrock Knowledge Bases | Vertex AI RAG Engine; Vertex AI Search (check current branding) | Azure AI Search with integrated vectorization; Foundry knowledge features (check current names) | Managed RAG speeds up a first version; chunking, retrieval evaluation and access control still need engineering. See [RAG](../ai_genai/intro_rag.md). |
| Agents | Amazon Bedrock Agents; Amazon Bedrock AgentCore | Vertex AI Agent Engine; Agent Development Kit (ADK) | Foundry Agent Service | Agent products are the fastest-changing names in this guide. Interviewers care more about tool permissions, memory and evaluation than the brand. |
| Guardrails and content safety | Amazon Bedrock Guardrails | Gemini safety settings; Model Armor | Azure AI Content Safety (content filters, Prompt Shields) | Coverage differs (PII redaction, prompt injection detection, grounding checks). Layer your own checks for domain-specific policy. |
| GenAI evaluation | Bedrock evaluations | Gen AI evaluation service in Vertex AI | Foundry evaluations | Useful for regression suites; keep your golden dataset portable. |

### Observability and cost

| Concept | AWS | GCP | Azure | Notes on differences |
|---|---|---|---|---|
| Metrics and logs | Amazon CloudWatch (metrics, Logs, alarms) | Cloud Monitoring and Cloud Logging | Azure Monitor (metrics, Log Analytics workspaces, alerts) | Query languages differ (CloudWatch Logs Insights, Logging query language or Log Analytics in BigQuery, KQL). Logs volume is a real cost on all three. |
| Distributed tracing and APM | AWS X-Ray; CloudWatch Application Signals | Cloud Trace | Application Insights | All three accept OpenTelemetry data in some form, which is the portable instrumentation choice. |
| Managed Prometheus and Grafana | Amazon Managed Service for Prometheus; Amazon Managed Grafana | Google Cloud Managed Service for Prometheus | Azure Monitor managed service for Prometheus; Azure Managed Grafana | Good for GPU and Kubernetes metrics (for example DCGM exporter) with the same dashboards on every cloud. |
| Budgets and alerts | AWS Budgets; AWS Cost Anomaly Detection | Cloud Billing budgets and alerts | Cost Management budgets and anomaly alerts | Budgets alert by default; they do not stop spending. Capping requires automation (for example a budget notification that triggers a function to stop resources). |
| Detailed billing data | AWS Cost and Usage Report via Data Exports | Cloud Billing export to BigQuery | Cost Management exports to a storage account | All three now offer some support for the FinOps FOCUS format, which helps normalise multi-cloud cost data. |
| Cost allocation metadata | Tags (cost allocation tags must be activated in Billing) | Labels (flow into billing export); separate resource manager tags used for policy and IAM conditions | Tags on resources, resource groups and subscriptions | GCP's labels vs tags split surprises people. Azure tags are not inherited by default; use Azure Policy to enforce or inherit them. |
| Quotas | Service Quotas (per account, per Region) | Cloud Quotas (per project, per region, plus some global GPU quotas) | Quotas per subscription, per region, often per VM family | GPU quota is a common launch blocker; request it early. |
| Rightsizing recommendations | AWS Compute Optimizer; AWS Trusted Advisor | Recommender (Active Assist) | Azure Advisor | Treat recommendations as hints; ML workloads are bursty and confuse utilisation-based advice. |

---

## Key behavioural differences

Mapping tables hide the differences that actually cause incidents and wrong interview answers. These are the ones worth explaining out loud.

### Global VPC on GCP vs regional networks on AWS and Azure

- **GCP:** a VPC network is a global resource. Subnets are regional, but VMs in `us-central1` and `europe-west4` on the same VPC reach each other over internal IPs with no peering. Firewall rules and routes are defined once for the whole network.
- **AWS:** a VPC lives in one Region, and each subnet sits in one Availability Zone. Multi-region means multiple VPCs connected with peering or Transit Gateway, and route tables to match.
- **Azure:** a VNet lives in one region, and subnets span the region's availability zones. Multi-region means multiple VNets with peering or Virtual WAN.

Why it matters for ML: on GCP a training job in one region reading from a store in another region "just works" on the network, which makes it easy to pay for cross-region traffic without noticing. On AWS and Azure the network topology forces you to make the cross-region decision explicitly.

### Projects vs accounts vs subscriptions and resource groups

| | AWS account | GCP project | Azure subscription | Azure resource group |
|---|---|---|---|---|
| Isolation strength | Strong: separate IAM, separate quotas | Medium: IAM inherited from folders and org | Strong for billing, quota and policy | Weak: a lifecycle and RBAC grouping |
| Typical granularity | Per team per environment | Per workload per environment (cheap to create) | Per environment or business unit | Per application or ML workspace |
| Billing | Consolidated through the management account | Linked to a billing account (separate from the hierarchy) | Linked to a billing account or profile | Rolls up to subscription |

A common mapping: one AWS account per environment maps to one GCP project per environment and to one Azure subscription per environment, with resource groups per application inside it. Azure ML adds a twist: a workspace has dependent resources (storage account, key vault, container registry, Application Insights) that usually sit in the same resource group, so deleting the group deletes all of them.

### IAM models

- **AWS:** policies are JSON documents of `Effect`, `Action`, `Resource` and `Condition`. They attach to principals (identity-based) or to resources (resource-based, such as S3 bucket policies and KMS key policies). Evaluation starts from an implicit deny; an explicit `Deny` anywhere (identity policy, resource policy, permission boundary, SCP, RCP) wins. Cross-account access usually needs both sides: the caller's identity policy and the target's resource or trust policy.
- **GCP:** allow policies bind principals to roles on a resource (organization, folder, project or individual resource) and are inherited downward. You cannot remove an inherited allow lower in the tree; you either grant it lower in the first place or add an IAM deny policy, which is evaluated before allow policies. IAM Conditions add attribute-based restrictions.
- **Azure:** a role assignment binds a principal to a role definition at a scope (management group, subscription, resource group or resource) and is inherited downward. Azure Policy is a separate system that governs what resources may look like, not who can act. Two traps: control-plane roles (such as Owner or Contributor) do not automatically grant data-plane access (for example reading blobs needs a data role like Storage Blob Data Reader), and Microsoft Entra directory roles are a separate permission system from Azure RBAC roles. Deny assignments exist but are created by platform features (such as deployment stacks or managed applications), not directly by you.

Interview shorthand: AWS is "policy documents with explicit deny", GCP is "role bindings inherited down a resource tree, plus deny policies", Azure is "role assignments at scopes, plus Azure Policy for configuration".

### BigQuery vs Redshift vs Synapse and Fabric

- **BigQuery** is serverless: there is no cluster to size. Storage and compute are separate, and you pay either for data scanned (on-demand) or for reserved compute capacity (slots via editions). For ML this means ad hoc feature exploration needs no infrastructure, but unpartitioned `SELECT *` queries over large tables become a cost problem fast. BigQuery ML and remote models let you train or call models from SQL.
- **Redshift** started as a provisioned cluster you size, now with managed storage and a Serverless option that scales compute automatically. Tuning concepts (distribution keys, sort keys, workload management) still matter more than in BigQuery.
- **Azure** is in transition: Synapse dedicated SQL pools are provisioned, Synapse serverless SQL pools query the lake on demand, and Microsoft Fabric bundles warehouse, lakehouse, Spark and pipelines on a shared capacity model with OneLake storage. Many Azure ML teams instead use Azure Databricks as the warehouse and feature engineering layer.

When an interviewer asks "what is the Azure BigQuery?", the honest answer is "it depends on whether the company is on Fabric, Synapse or Databricks", followed by the concept: a columnar warehouse with separated storage and compute.

### Egress and data movement charges

All three clouds charge for data leaving the cloud to the internet, and for traffic between regions. Traffic between zones in the same region is also billed on some clouds and not others, and the rules change, so check the current pricing pages rather than quoting from memory. Ingress is generally free. Consequences for ML:

- Training where the data lives is usually cheaper than moving the data to the GPUs.
- Cross-cloud architectures pay egress on every training read and every replication run, so design for one bulk copy plus incremental syncs, not repeated reads across clouds.
- Container image pulls, dataset downloads from public hubs and log shipping to a third-party observability vendor are easy to forget.
- All three providers have announced waived egress for customers moving data out to leave the cloud (on request, with conditions). That helps one-time migrations, not ongoing multi-cloud traffic.

---

## Cloud-agnostic ML architecture

"Cloud-agnostic" should not mean "use nothing managed". It means keeping the parts that are expensive to rewrite portable, and accepting managed services where the switching cost is small or the value is large.

### What to keep portable

| Layer | Portable choice | Why it is worth it |
|---|---|---|
| Packaging | Containers (OCI images) for training and serving | The same image runs on SageMaker AI, Vertex AI, Azure ML, Kubernetes or a laptop. |
| Orchestration and runtime | Kubernetes (EKS, GKE, AKS) with Kubeflow, Argo or Airflow | Workflow definitions move between clusters; the cluster itself is commodity. |
| Infrastructure as code | Terraform or OpenTofu with one module per cloud behind a common interface | Rebuilding environments is repeatable. Resources are not portable, but the workflow and review process are. See [Terraform](../devops/intro_terraform.md). |
| Experiment tracking and registry | MLflow | Runs, metrics, model versions and model formats survive a platform move. See [MLflow](../mlops/intro_mlflow.md). |
| Data format | Parquet with open table formats (Apache Iceberg, Delta Lake) | Engines on every cloud read them; the warehouse becomes replaceable. See [Iceberg](../data_engineering/intro_apache_iceberg.md) and [Delta Lake](../data_engineering/intro_delta_lake.md). |
| Feature logic | Feature definitions in code (SQL, dbt models, Python transforms) separated from the store | The store can change; the definitions and their tests should not. |
| Model serving interface | Open model servers (for example vLLM, Triton, KServe) behind a stable HTTP or gRPC contract | Clients do not care which cloud serves the model. |
| Telemetry | OpenTelemetry SDKs and the Collector | Swap the backend (CloudWatch, Cloud Monitoring, Azure Monitor or a vendor) without touching application code. See [Observability](../devops/intro_observability.md). |
| CI/CD | GitHub Actions or similar with OIDC federation to each cloud | One pipeline, no long-lived cloud keys. See [GitHub Actions](../devops/intro_github_actions.md). |

### Where managed services usually win

- **Warehouses** (BigQuery, Redshift, Fabric or Synapse): running your own is rarely worth it; keep data in open formats so the engine is swappable.
- **Foundation model APIs** (Bedrock, Gemini on Vertex AI, Foundry): frontier closed models are only available as managed APIs. Hide them behind an internal gateway or client library so the provider is a configuration choice.
- **Identity, KMS, secrets and audit:** always use the native service. Abstracting these usually weakens security.
- **Spot and accelerator scheduling:** managed training services handle capacity, retries and checkpoint plumbing that is tedious to rebuild.
- **Small teams:** a single managed ML platform beats a portable platform nobody has time to operate.

### Portable training and serving stack

```text
                         +----------------------------------------+
   git push ------------>|  CI (OIDC to each cloud, no keys)      |
                         |  build images, terraform/tofu plan     |
                         +-------------------+--------------------+
                                             |
   ==================== PORTABLE CORE (same code everywhere) ====================
                                             v
   +-------------------+   +-------------------------+   +-----------------------+
   | Data              |   | Training                |   | Serving               |
   | Iceberg / Delta   |-->| container image         |-->| container image       |
   | tables on object  |   | (PyTorch, JAX, sklearn) |   | (vLLM, Triton, FastAPI|
   | storage; feature  |   | run by Kubeflow / Argo  |   |  or KServe)           |
   | logic in SQL/dbt  |   | or a managed job        |   | stable HTTP/gRPC API  |
   +-------------------+   +-----------+-------------+   +-----------+-----------+
                                       |                             |
                                       v                             v
                           +-----------------------+     +-----------------------+
                           | MLflow tracking and   |     | OpenTelemetry SDK     |
                           | model registry        |     | and Collector         |
                           +-----------------------+     +-----------------------+
   ==============================================================================
                                             |
                 thin cloud-specific adapters (Terraform modules, config)
                                             |
        +--------------------------+---------+---------+--------------------------+
        v                          v                   v                          v
   AWS adapter               GCP adapter          Azure adapter          Managed extras
   S3, EKS, ECR,             GCS, GKE,            Blob/ADLS, AKS, ACR,   (opt-in, behind
   Pod Identity or IRSA,     Artifact Registry,   Entra Workload ID,     interfaces)
   KMS, Secrets Manager,     WIF for GKE, KMS,    Key Vault,             Bedrock, Gemini,
   CloudWatch                Secret Manager,      Azure Monitor          Foundry, BigQuery,
                             Cloud Monitoring                            managed endpoints
```

The adapters should be thin and boring: storage URIs, identity bindings, registry endpoints and exporters. If business logic leaks into an adapter, portability is gone.

---

## Migration and multi-cloud scenarios

### Migrating a SageMaker AI pipeline to Vertex AI

Assume a SageMaker Pipeline that runs a processing job, a training job, an evaluation step, registers a model with an approval status, and deploys to a real-time endpoint.

1. **Inventory the contract, not the code.** List inputs (S3 prefixes, Feature Store groups), outputs (model artifacts, metrics), the IAM execution role's permissions, triggers (EventBridge rules, schedules) and consumers of the endpoint.
2. **Move data first.** Bulk-copy training data from S3 to GCS with Storage Transfer Service, then decide whether the source of truth moves or stays (if it stays on AWS, you now have a cross-cloud data path; see the next scenario). If features live in SageMaker Feature Store, rebuild them as BigQuery tables registered with Vertex AI Feature Store.
3. **Adapt the training container.** SageMaker containers read channels from `/opt/ml/input/data/<channel>` (or `SM_CHANNEL_*` variables) and write the model to `/opt/ml/model` (`SM_MODEL_DIR`). Vertex AI custom training passes an output location in `AIP_MODEL_DIR` (a `gs://` path) and exposes Cloud Storage buckets through Cloud Storage FUSE under `/gcs/`. Make the script take paths as arguments so both work.
4. **Adapt the serving container.** SageMaker expects `GET /ping` and `POST /invocations` on port 8080. Vertex AI custom containers read the port and routes from `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE` and `AIP_PREDICT_ROUTE`, and requests use an `{"instances": [...]}` envelope with `{"predictions": [...]}` responses. A small routing shim keeps one image working on both.
5. **Rewrite the pipeline definition.** SageMaker Pipelines steps become KFP v2 components and a pipeline compiled for Vertex AI Pipelines. This is a rewrite, not a translation; keep step logic in the container so the DSL layer is thin.
6. **Replace governance.** SageMaker Model Registry approval status has no direct equivalent: use Vertex AI Model Registry versions with aliases (for example `champion`) and labels, with the approval gate in CI or a pipeline step. Replace EventBridge triggers with Cloud Scheduler, Pub/Sub or Eventarc.
7. **Translate identity.** The SageMaker AI execution role becomes a dedicated service account on the custom job and endpoint, with roles granted on the specific bucket, dataset and Artifact Registry repository. Replace any CI access keys with Workload Identity Federation.
8. **Run in parallel, then cut over.** Shadow traffic or dual-run batch scoring, compare predictions and latency, then shift traffic and keep the SageMaker endpoint available for rollback until confidence is high.

### Training on one cloud, serving on another

Typical reason: TPUs or reserved GPUs on GCP for training, while the product and its users run on AWS.

- **Data gravity:** keep training data next to the training compute. Replicate only what training needs (curated feature tables, not raw logs), on a schedule, in an open format.
- **Egress:** the big transfer should be the training dataset copy, and that should be incremental. Model artifacts are small by comparison; ship them once per release.
- **Artifacts:** export the model in a framework-neutral or serving-ready format (for example SavedModel, ONNX or safetensors weights) and register it in a shared MLflow registry or copy it to the serving cloud's object storage. Build the serving image in CI and push it to the serving cloud's registry.
- **Identity federation between clouds:** do not copy access keys across clouds. A GCP workload can obtain AWS credentials with `AssumeRoleWithWebIdentity` using a Google-issued OIDC token for its service account. In the other direction, GCP Workload Identity Federation accepts AWS identities directly, and Microsoft Entra workload identity federation accepts external OIDC issuers. For workloads without a cloud identity, AWS IAM Roles Anywhere uses X.509 certificates.
- **Serving parity:** feature computation at serving time on AWS must match what training used on GCP. Keep feature logic in shared code and run a skew check that compares online features against the offline training snapshot.
- **Networking:** for private paths use VPN or a cross-cloud interconnect product instead of the public internet if data is sensitive; check current offerings, as cross-cloud interconnect products are relatively new.

### Choosing a cloud for a new ML team

Structure the answer around constraints, in this order:

1. **Where the data and the company already are.** Existing identity, security review, data warehouse and procurement agreements usually decide it. Fighting the company's main cloud costs more than any feature gap.
2. **Accelerator access.** If the roadmap needs large-scale training, ask which cloud can actually provide GPU or TPU capacity and quota in your regions, and whether the team can use Trainium or TPU software stacks.
3. **Model access for GenAI.** If a specific model family is required (OpenAI models, Gemini, a particular partner model), check which cloud offers it in the required regions with the needed data-residency terms.
4. **Team skills.** A team fluent in one cloud ships faster there.
5. **Compliance and residency.** Regions, certifications and sovereign cloud offerings.
6. **Total cost of ownership,** not list prices: engineering time, egress, idle resources and negotiated commitments.

Then keep the [portable core](#what-to-keep-portable) so the decision is not irreversible.

---

## Terraform example: one versioned bucket on three clouds

The same intent (a private object store with versioning) expressed for each provider. Provider configuration is kept minimal: in practice you would split this into per-cloud modules, use remote state and supply credentials through OIDC federation in CI. Bucket and account names must be globally unique (the Azure storage account name must be 3 to 24 lowercase letters and digits).

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.20"
    }
  }
}

# Minimal provider configuration. Credentials come from the environment
# (for example OIDC federation in CI), never from this file.
provider "aws" {
  region = "us-east-1"
}

provider "google" {
  project = "my-ml-project"
  region  = "us-central1"
}

provider "azurerm" {
  features {}
  # subscription_id is read from ARM_SUBSCRIPTION_ID if not set here.
}

# ---------- AWS: S3 bucket with versioning ----------
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "example-ml-artifacts-aws"
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------- GCP: Cloud Storage bucket with versioning ----------
resource "google_storage_bucket" "ml_artifacts" {
  name                        = "example-ml-artifacts-gcp"
  location                    = "US-CENTRAL1"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }
}

# ---------- Azure: storage account + container with blob versioning ----------
resource "azurerm_resource_group" "ml" {
  name     = "rg-ml-artifacts"
  location = "eastus"
}

resource "azurerm_storage_account" "ml_artifacts" {
  name                            = "examplemlartifacts"
  resource_group_name             = azurerm_resource_group.ml.name
  location                        = azurerm_resource_group.ml.location
  account_tier                    = "Standard"
  account_replication_type        = "ZRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_storage_container" "ml_artifacts" {
  name                  = "ml-artifacts"
  storage_account_id    = azurerm_storage_account.ml_artifacts.id
  container_access_type = "private"
}
```

Points worth saying in an interview:

- **Versioning lives in different places.** AWS uses a separate `aws_s3_bucket_versioning` resource (since AWS provider v4 split bucket settings out), GCP uses a block on the bucket, and Azure sets it on the storage account, so it applies to every container in that account.
- **Public access defaults.** New S3 buckets block public access by default. On GCP, `public_access_prevention` and uniform bucket-level access make IAM the only access path. On Azure, `allow_nested_items_to_be_public = false` stops containers from being made public.
- **Azure's extra layer.** The storage account is where redundancy, network rules, TLS and versioning live; the container is closer to an S3 or GCS bucket. Older azurerm examples pass `storage_account_name` to the container; recent 4.x releases use `storage_account_id`.
- **Versioning costs storage.** Pair it with lifecycle rules that expire noncurrent versions, or checkpoints will accumulate indefinitely.

---

## Interview Q&A

#### What is the GCP equivalent of an AWS IAM role for a workload?

The closest equivalent is a service account attached to the workload: the VM, the Vertex AI custom job, the Cloud Run service, or a Kubernetes service account mapped through Workload Identity Federation for GKE. Like an AWS role used by EC2 or SageMaker AI, it gives the workload short-lived credentials from the metadata server, so there are no static keys. The difference is where permissions live. On AWS you attach policy documents to the role. On GCP you grant roles to the service account on the resources it needs (a bucket, a dataset, a project), and those grants are inherited down the hierarchy. A GCP service account is also a resource, so you control who can impersonate it, which is the rough counterpart of an AWS role trust policy. On Azure the equivalent is a managed identity plus Azure RBAC role assignments at the right scope. A common mistake is downloading service account keys on GCP "because it is easier"; organization policy can and usually should block key creation.

#### How do PrivateLink, Private Service Connect and Private Endpoint compare?

All three put a private IP address for a managed service (or a partner's service) inside your own network, so traffic never traverses the public internet and you can disable public access to the service. AWS PrivateLink creates interface endpoints per service per VPC, and S3 and DynamoDB also have cheaper gateway endpoints that work through route tables. GCP Private Service Connect creates an endpoint for Google APIs or for a published service; Private Google Access is a separate, simpler option that lets VMs without external IPs reach Google APIs. Azure Private Endpoint attaches a NIC for a specific resource (for example one storage account or one Azure ML workspace), which is more granular than per-service. In every case DNS is what usually breaks: the public hostname must resolve to the private IP from inside the network, and Azure in particular relies on `privatelink.*` private DNS zones. For ML, remember that a private workspace or endpoint also needs private paths to its dependencies (storage, registry, key vault), or jobs fail in confusing ways.

#### How do AWS accounts, GCP projects and Azure subscriptions map to each other?

For environment separation the usual mapping is one AWS account, one GCP project, or one Azure subscription per team per environment. They are not equally strong boundaries, though. An AWS account is a hard boundary: separate IAM, separate quotas, and cross-account access must be explicitly allowed on both sides. A GCP project is cheaper to create and is the unit for APIs, quotas and billing linkage, but IAM granted on a parent folder or the organization flows into it. An Azure subscription is the billing, quota and policy boundary, while resource groups inside it are lightweight lifecycle groupings with their own role assignments. So "one ML workspace per resource group in a per-environment subscription" on Azure is roughly analogous to "one project per environment" on GCP. The tradeoff is blast radius against overhead: more boundaries mean better isolation but more networking, IAM and quota requests to manage.

#### How would you let a GitHub Actions workflow deploy to all three clouds without long-lived keys?

Use OIDC federation on each cloud. GitHub issues a signed OIDC token per workflow run that includes claims such as repository, branch and environment. On AWS you register GitHub as an IAM OIDC identity provider and create a role whose trust policy restricts the `sub` claim; the workflow calls `AssumeRoleWithWebIdentity`. On GCP you create a workload identity pool and provider with an attribute condition on the repository, then either grant roles directly to the federated principal or let it impersonate a service account. On Azure you add a federated credential to an app registration or user-assigned managed identity matching the repository and environment. The benefits are no secrets to rotate or leak and a clear audit trail per run. The main risk is overly broad trust conditions, such as trusting every repository in an organization or every branch, which would let a pull request from a fork or a feature branch deploy to production.

#### When is multi-cloud worth it for ML, and when is it a trap?

It is worth it when there is a concrete reason that a single cloud cannot satisfy: accelerator capacity or a specific chip (for example TPUs) available only elsewhere, a required foundation model only offered on another cloud, customers or regulators requiring deployment in their cloud, or an acquisition that brought a second cloud with it. It is a trap when the motivation is vague "avoid lock-in" or "negotiate leverage" for a team that has not yet mastered one cloud. The costs are real: duplicate identity and network setups, egress on every cross-cloud data path, two sets of on-call knowledge, lowest-common-denominator tooling and slower delivery. A middle ground that usually works is a portable core (containers, Terraform, MLflow, open table formats) on one primary cloud, with a second cloud used for one well-bounded job such as training or a specific model API. I would also separate "multi-cloud capable" (could move in months) from "active multi-cloud" (runs on two clouds today), because the second is much more expensive.

#### How would you design a feature pipeline that could move between clouds?

Separate feature definitions from feature infrastructure. Write the transformations in SQL (for example dbt models) or Python in a framework that runs on several engines (Spark, Beam or DuckDB for small data), and keep them in version control with tests. Store offline features as Iceberg or Delta tables on object storage, so BigQuery, Redshift, Spark on Databricks or Fabric can all read them. For online serving, put a thin interface in front of the key-value store (DynamoDB, Bigtable, Cosmos DB or Redis) so the serving code calls `get_features(entity_id)` and not a cloud SDK; an open-source feature store such as Feast can provide that layer. Orchestrate with Airflow or another portable scheduler, and keep cloud-specific connection details in configuration. The tradeoff is that you give up some managed conveniences, such as native BigQuery-backed feature serving in Vertex AI Feature Store, so do this only if moving clouds is a realistic requirement.

#### Why might the same training job cost very differently across clouds?

Several factors differ even when the GPU model is identical. Instance shapes bundle different amounts of CPU, memory, local disk and network bandwidth with the same GPU, and some shapes only come in large multi-GPU sizes. Spot availability and interruption rates differ by cloud, region and time, and frequent interruptions without good checkpointing waste paid compute. Storage throughput matters: if the data loader cannot keep GPUs busy, you pay for idle accelerators, and the fast-storage options (Lustre-based services, local NVMe caching) differ. Data location adds egress if the data lives on another cloud or region. Managed ML services price their instances separately from raw VMs, and discount instruments (savings plans, committed use discounts, reservations) have different scopes and terms. Finally, switching hardware (for example to TPUs or Trainium) can change cost a lot in either direction, but only after engineering work to port and tune the model. I would compare cost per completed training run, including failed and restarted attempts, not hourly instance price.

#### What is the Azure equivalent of BigQuery, and why is the answer not a single service?

The concept is a managed columnar warehouse with separated storage and compute. On Azure that can be a Microsoft Fabric Warehouse or Lakehouse, Azure Synapse Analytics (dedicated SQL pools are provisioned, serverless SQL pools query the lake on demand), or Databricks SQL on Azure Databricks. Microsoft is steering new analytics work toward Fabric, but many enterprises still run Synapse or standardize on Databricks. BigQuery differs from most of these in being serverless by default: there is no cluster to size, and cost scales with data scanned or reserved slots. That changes ML habits, because exploration is easy but careless queries are expensive, so partitioning, clustering and query cost controls matter. In an interview I would ask which data platform the company uses before mapping, and then talk about where features are computed and how the ML platform reads them.

#### How does Pub/Sub differ from Kinesis Data Streams and Event Hubs for ML event pipelines?

Kinesis Data Streams and Event Hubs are partitioned logs: producers write to shards or partitions, ordering is per partition, and consumers track their own position and can replay within the retention window. Pub/Sub is a topic-and-subscription service: each subscription gets every message, delivery is tracked per message with acknowledgements, ordering is opt-in through ordering keys, and replay uses seek to a timestamp or snapshot when retention is configured. For ML this changes how you build things like feature backfills and exactly-once processing. Log semantics make "reprocess the last day from offset X" natural, while Pub/Sub is simpler to scale and operate with no shard management. If portability matters, managed Kafka exists on all three clouds (Amazon MSK, Managed Service for Apache Kafka on GCP, and the Kafka endpoint on Event Hubs), which keeps the client code and semantics consistent.

#### If you know Amazon Bedrock, how would you explain Vertex AI and Microsoft Foundry to a team?

All three are managed APIs for foundation models with similar building blocks: a model catalog, pay-per-token and provisioned throughput, fine-tuning, managed RAG, agent services, guardrails and evaluation. On Google Cloud, Gemini is the first-party model family accessed through Vertex AI, and Model Garden adds partner and open models that can be called as APIs or deployed to your own endpoints. On Azure, Microsoft Foundry (formerly Azure AI Foundry) is the umbrella for model access, including Azure OpenAI models, plus agents, evaluation and content safety. The practical differences are which models are available in which regions, data residency terms, quota models and how identity and networking integrate (IAM roles, service accounts, Entra ID and private endpoints). Agent and RAG product names change frequently, so I would map them by function. For portability, put an internal client or gateway in front of all providers and keep prompts, evaluation sets and guardrail policies in your own repository.

#### How do you prevent data exfiltration from an ML environment on each cloud?

Combine identity, network and organization controls, because any one layer alone has gaps. On GCP, VPC Service Controls is the most direct tool: it draws a perimeter around projects so that even a valid credential cannot copy data from BigQuery or Cloud Storage to a project outside the perimeter. On AWS you build a data perimeter from SCPs and RCPs that restrict access to trusted identities and resources, VPC endpoint policies that limit which buckets can be reached from inside the network, and bucket policies that require access through your endpoints. On Azure you disable public network access on storage and workspaces, use private endpoints, restrict outbound traffic (Azure ML managed virtual networks support an outbound mode that allows only approved destinations), and enforce it with Azure Policy. On all three, notebooks with open internet egress are the weak point, so restrict outbound traffic and package mirrors as well. The tradeoff is developer friction: package installs and model downloads need approved mirrors or allow-lists.

#### How would you serve a model trained on TPUs to users whose application runs on AWS?

First, make sure the model artifact is not tied to the TPU: JAX or PyTorch/XLA training can export weights in a standard format, and the serving stack on AWS will run them on GPUs or CPUs (or on Inferentia after compiling with the Neuron SDK). I would build the serving image in CI and push it to Amazon ECR, then deploy to a SageMaker AI endpoint or to EKS with an open model server. The artifact moves once per release from GCS to S3, using a pipeline step whose identity is federated (no long-lived keys). Online features must be computed on AWS with the same logic used for training on GCP, so I would keep feature code shared and run a skew check. Monitoring data flows back to GCP for retraining in batches, not per request, to limit egress. The tradeoff is operating two clouds for one model; it is worth it if TPU access substantially lowers training cost or time.

#### What breaks first when you lift a SageMaker AI training script onto Vertex AI or Azure ML?

Usually the container contract, not the model code. SageMaker AI injects channel paths and the model output directory through `/opt/ml/...` paths and `SM_*` environment variables, and uploads `/opt/ml/model` to S3 at the end. Vertex AI custom training passes `AIP_MODEL_DIR` and mounts buckets through Cloud Storage FUSE under `/gcs/`, while Azure ML passes inputs and outputs as command arguments bound to datastore URIs. Next come credentials and SDK calls (boto3 calls to S3 or Secrets Manager inside the script), distributed training environment setup, and checkpoint locations for spot recovery. The fix is to make the script take every path and secret as arguments or environment variables, read and write through `fsspec`-style URIs or mounted paths, and keep cloud SDK calls out of training code. Then a thin launcher per cloud maps the platform's contract onto the script's arguments.

#### How do spot capacity semantics differ, and how do you make training robust to them?

All three clouds sell interruptible capacity at a discount with a short warning before reclaim: EC2 Spot gives a two-minute notice, while GCP Spot VMs and Azure Spot VMs give a much shorter notice (around 30 seconds). GCP Spot VMs have no maximum runtime, unlike the older preemptible VMs, and Azure lets you choose eviction behavior and a maximum price. Managed ML services wrap this differently: SageMaker AI Managed Spot Training syncs a checkpoint directory to S3 and resumes, Vertex AI custom training can run on Spot VMs, and Azure ML compute clusters offer a low-priority tier. Regardless of cloud, the job must checkpoint to durable storage frequently enough that losing the time since the last checkpoint is acceptable, and resume idempotently, including optimizer state and data loader position. For multi-node training, one interrupted node stops the whole job, so elastic training or smaller jobs reduce wasted compute. I would use spot for fault-tolerant training and hyperparameter search, and on-demand or reserved capacity for latency-sensitive serving.

#### How do you keep cost attribution consistent across three clouds?

Define one tagging schema (for example `team`, `env`, `project`, `cost_center`, `model`) and enforce it with each cloud's policy engine: SCPs or tag policies on AWS, organization policies and required labels in Terraform on GCP, and Azure Policy on Azure. Remember the naming differences: AWS tags must be activated as cost allocation tags, GCP uses labels for billing and separate tags for policy, and Azure tags are not inherited by default. Export detailed billing data from each cloud (Cost and Usage Report data exports, BigQuery billing export, Cost Management exports) into one place, ideally normalised to the FOCUS format, and build dashboards on that. ML-specific gaps include shared GPU clusters, where you need Kubernetes-level allocation by namespace or label, and managed services that create untagged child resources. Enforce tags in CI through Terraform modules, because tags added by hand drift quickly.

---

## Common Pitfalls

| Problem | Why it hurts | Fix |
|---|---|---|
| Forcing a one-to-one service match in interviews | Signals memorization and leads to wrong designs (for example "Pub/Sub is just Kinesis") | Name the concept, map the service, then state where the analogy breaks |
| Assuming a GCP VPC is regional like AWS and Azure | Cross-region traffic happens silently over the same network and shows up on the bill | Pin training and data to one region on purpose; review cross-region flows |
| Treating Azure Owner or Contributor as data access | Jobs fail reading blobs despite "full" permissions, or people grant overly broad roles to compensate | Assign data-plane roles (for example Storage Blob Data Reader) to the job's managed identity |
| Downloading service account keys or access keys for CI | Long-lived secrets leak through logs, laptops and forks | Use OIDC workload identity federation on every cloud; block key creation with org policy |
| Private endpoints without private DNS | Services resolve to public IPs, so traffic fails or bypasses the private path | Configure private DNS zones and test resolution from inside the cluster and job subnets |
| Reading training data across clouds every epoch | Egress costs and slow, unpredictable data loading | Copy once, sync incrementally and train next to the data |
| Pipeline logic written in a cloud-specific DSL | Migration becomes a rewrite of business logic, not just orchestration | Keep step logic in containers; keep the DSL layer thin |
| Assuming budgets cap spend | Budgets only alert by default, so runaway GPU jobs keep running | Add automated actions on budget notifications and set quotas as hard limits |
| Archive tier for active datasets | Restore delays and retrieval fees stall training | Use lifecycle rules on old checkpoints only; keep active data in standard or infrequent tiers |
| Quoting a new product name with full confidence | Names in GenAI and data platforms change often, and a wrong one undermines credibility | Describe the capability first and hedge the brand ("currently called...") |
| Abstracting identity, KMS and secrets "for portability" | Weakens security and duplicates mature native services | Use native identity and key services; keep portability in data, code and packaging |

---

## Related Topics

| Topic | Why it is related |
|---|---|
| [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md) | Feature-by-feature comparison of SageMaker AI, Vertex AI and Azure ML |
| [AWS SageMaker Interview Guide](./intro_sagemaker.md) | Depth on the AWS ML platform |
| [Google Vertex AI Interview Guide](./intro_vertex_ai.md) | Depth on the GCP ML platform |
| [Azure Machine Learning Interview Guide](./intro_azure_ml.md) | Depth on the Azure ML platform |
| [AWS for ML Engineers](./aws_for_ml_engineers.md) | AWS infrastructure services around ML |
| [GCP for ML Engineers](./gcp_for_ml_engineers.md) | GCP infrastructure services around ML |
| [Azure for ML Engineers](./azure_for_ml_engineers.md) | Azure infrastructure services around ML |
| [Terraform](../devops/intro_terraform.md) | Infrastructure as code across clouds |
| [Kubernetes](../devops/intro_kubernetes.md) | The most portable compute layer |
| [Docker](../devops/intro_docker.md) | Container images as the unit of portability |
| [GitHub Actions](../devops/intro_github_actions.md) | CI with OIDC federation to each cloud |
| [Observability](../devops/intro_observability.md) | OpenTelemetry and monitoring patterns |
| [Apache Iceberg](../data_engineering/intro_apache_iceberg.md) | Open table format for portable data |
| [Delta Lake](../data_engineering/intro_delta_lake.md) | Open table format used heavily with Databricks |
| [Apache Kafka](../data_engineering/intro_apache_kafka.md) | Portable streaming semantics across clouds |
| [Apache Spark](../data_engineering/intro_apache_spark.md) | Engine available on every cloud |
| [Apache Airflow](../data_engineering/intro_apache_airflow.md) | Portable orchestration |
| [MLflow](../mlops/intro_mlflow.md) | Portable tracking and model registry |
| [Feature Stores](../mlops/intro_feature_stores.md) | Online and offline feature patterns |
| [RAG](../ai_genai/intro_rag.md) | The pattern behind managed RAG services |
