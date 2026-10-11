# Microsoft Azure for ML Engineers

Azure Machine Learning is only one piece of an ML system on Azure. Around it sit identity, networking, storage, compute, data platforms, generative AI services, monitoring and billing, and most production incidents and interview follow-ups happen in those surrounding layers rather than inside the ML workspace. This guide covers the core Azure services an ML or AI engineer touches outside Azure ML itself and how they fit together.

For the Azure ML workspace, jobs, registries and endpoints in depth, see the [Azure Machine Learning Interview Guide](./intro_azure_ml.md). For a cross-cloud comparison of SageMaker, Vertex AI and Azure ML, see [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md).

> Azure renames services often. Where a service has a recent former name, it is noted in parentheses. Check the current Microsoft documentation before quoting a name, SKU or feature status in a design review.

---

## Table of Contents

1. [How the pieces fit together](#how-the-pieces-fit-together)
2. [Resource hierarchy and identity](#resource-hierarchy-and-identity)
3. [Networking basics](#networking-basics)
4. [Storage](#storage)
5. [Compute](#compute)
6. [Data and analytics](#data-and-analytics)
7. [Generative AI](#generative-ai)
8. [Observability](#observability)
9. [Cost control](#cost-control)
10. [Reference architectures](#reference-architectures)
11. [Code examples](#code-examples)
12. [Interview Q&A](#interview-qa)
13. [Common Pitfalls](#common-pitfalls)
14. [Related Topics](#related-topics)

---

## How the pieces fit together

```text
                    Microsoft Entra ID (identities, groups, managed identities)
                                         |
          Management groups -> Subscriptions -> Resource groups -> Resources
                                         |
   +-----------------+-------------------+-------------------+------------------+
   |                 |                   |                   |                  |
 Storage          Data / analytics     ML platform         GenAI              Serving
 (Blob, ADLS)     (Data Factory,       (Azure ML           (Azure OpenAI in   (managed online
                  Fabric, Databricks,   workspace,          Microsoft Foundry, endpoints, AKS,
                  Event Hubs)           registries)         AI Search)         Container Apps)
   |                 |                   |                   |                  |
   +-----------------+---------+---------+-------------------+------------------+
                               |
            VNets, private endpoints, Private DNS, NAT gateway
                               |
            Azure Monitor / Log Analytics / Application Insights
                               |
            Cost Management (tags, budgets, alerts)
```

A useful mental model for interviews: **identity decides who can call what, networking decides from where, storage and data services hold the bytes, compute does the work, and monitoring plus cost management tell you whether it is healthy and affordable.**

---

## Resource hierarchy and identity

### Hierarchy

| Level | What it is | ML platform use |
|---|---|---|
| Microsoft Entra tenant | The directory of users, groups, apps and identities | One per organization, usually |
| Management group | A container of subscriptions for policy and RBAC inheritance | "Platform", "Landing zones/Corp", "Sandbox" groups |
| Subscription | Billing, quota and scale boundary | Separate dev, test and prod; separate shared ML platform services |
| Resource group | Lifecycle container for related resources | One per workload per environment (e.g. `rg-churn-prod`) |
| Resource | An individual service instance | Storage account, Key Vault, Azure ML workspace, AKS cluster |

Policies and role assignments applied at a higher level inherit downward. GPU quota is granted per subscription, per region and per VM family, which is one practical reason ML teams care about subscription layout.

### Identity and access

| Concept | What it does | Notes for ML engineers |
|---|---|---|
| Microsoft Entra ID (formerly Azure AD) | Identity provider for users, groups, apps and Azure resources | All modern Azure data-plane auth (Storage, Key Vault, Azure OpenAI, AI Search) can use Entra tokens instead of keys |
| Azure RBAC | Role assignment = principal + role definition + scope | Scope is a management group, subscription, resource group or resource. Assign at the narrowest scope that works |
| Control plane vs data plane roles | `Owner`, `Contributor`, `Reader` manage resources; data roles grant access to the data inside | Reading blobs with Entra auth needs a data role such as `Storage Blob Data Reader`; calling Azure OpenAI needs a role such as `Cognitive Services OpenAI User` |
| Service principal | The identity of an application in a tenant | Created from an app registration; can authenticate with a secret, a certificate, or a federated credential |
| Managed identity | A service principal whose credentials Azure creates and rotates | No secret in your code or config. **System-assigned** lives and dies with one resource; **user-assigned** is its own resource and can be shared |
| Workload identity federation | Trust an external OIDC token instead of a secret | GitHub Actions (OIDC token from the workflow) and AKS pods (Microsoft Entra Workload ID, which replaced the deprecated pod-managed identity) exchange their token for an Entra token |
| Key Vault | Stores secrets, keys and certificates | Use the Azure RBAC permission model (roles such as `Key Vault Secrets User`) rather than legacy access policies for new vaults |
| Azure Policy | Evaluates resources against rules; effects include audit, deny, modify and deploy-if-not-exists | Deny public network access, require tags, restrict regions or allowed VM sizes |
| Activity log | Subscription-level log of control-plane operations (who created, changed or deleted what) | Retained for 90 days by default; export it with diagnostic settings to Log Analytics for longer retention and querying |

**Rule of thumb:** humans get access through Entra groups, workloads get access through managed identities, CI/CD gets access through workload identity federation, and the few remaining third-party secrets live in Key Vault and are read with a managed identity.

**What interviewers listen for**

- You separate control-plane permissions (create a resource) from data-plane permissions (read the data in it).
- You default to managed identities and federated credentials and can explain why long-lived client secrets are a liability.
- You know when to choose system-assigned vs user-assigned identity (lifecycle vs reuse and pre-provisioned role assignments).
- You use Azure Policy for guardrails and the activity log for audit, instead of relying on people following a wiki.

---

## Networking basics

| Concept | What it does | ML relevance |
|---|---|---|
| Virtual network (VNet) | Private address space in one region | Hosts AKS nodes, VMs, private endpoints, integrated app services |
| Subnet | A range inside a VNet; some services need a dedicated or delegated subnet | Plan address space early: AKS and private endpoints consume IPs quickly |
| Network security group (NSG) | Stateful allow/deny rules on a subnet or NIC | Restrict which subnets can reach inference services |
| Service endpoint | Lets a subnet reach a PaaS service over the Azure backbone; the service still has a public endpoint, and its firewall allows your subnet | Simpler, but the resource keeps a public IP and the scope is the whole service type, not one instance |
| Private endpoint (Private Link) | A NIC with a private IP in your VNet mapped to one specific resource instance | Lets you disable public network access entirely on Storage, Key Vault, Azure OpenAI, AI Search, ACR, Azure ML |
| Private DNS zone | Resolves the service hostname to the private endpoint IP inside your network | e.g. `privatelink.blob.core.windows.net`, `privatelink.vaultcore.azure.net`, `privatelink.openai.azure.com`; missing or unlinked zones are the most common private endpoint bug |
| NAT gateway | Provides predictable, scalable outbound internet access for a subnet | Needed for pulling packages or calling external APIs from private subnets; gives a stable egress IP for allow-lists |
| Hub-and-spoke or Virtual WAN | Shared connectivity (firewall, VPN, ExpressRoute) in a hub, workloads in spokes | Typical enterprise landing zone layout |

Azure ML also offers a **workspace managed virtual network** that Microsoft manages for compute and endpoints, with outbound modes such as "allow internet outbound" and "allow only approved outbound". See the [Azure ML guide](./intro_azure_ml.md) for the workspace side; this guide focuses on the network around it.

Azure has been retiring implicit "default outbound access" for new virtual networks, so plan explicit egress (NAT gateway, Azure Firewall or load balancer outbound rules) rather than relying on it. Check the current status for your region and deployment date.

**What interviewers listen for**

- A crisp distinction: a private endpoint gives one resource a private IP; a service endpoint keeps the public endpoint but restricts it to your subnet.
- DNS is part of the design: private endpoints without the right Private DNS zone linked to the VNet will resolve to the public IP and fail when public access is disabled.
- You think about egress as well as ingress (package mirrors, model downloads, external APIs, data exfiltration risk).

---

## Storage

### Blob Storage access tiers

| Tier | Best for | Tradeoff |
|---|---|---|
| Hot | Active training data, feature snapshots read often | Highest storage cost, lowest access cost |
| Cool | Data read occasionally, recent checkpoints | Lower storage cost, higher access cost, minimum retention period (30 days) with early deletion charges |
| Cold | Rarely read data that must stay online | Lower again, longer minimum retention (90 days) |
| Archive | Compliance copies, raw data you may never read again | Offline: must be rehydrated (taking hours) before reading; longest minimum retention (180 days) |

**Lifecycle management** policies move or delete blobs automatically based on age (since creation, last modification or, if access tracking is enabled, last access). A typical ML policy: keep the current training snapshot hot, move checkpoints older than a few weeks to cool, archive raw extracts after their retention review, delete temporary scoring outputs.

### Other storage options

| Service | What it is | When ML engineers use it |
|---|---|---|
| ADLS Gen2 | A storage account with **hierarchical namespace** enabled | Data lakes for Spark, Databricks, Fabric shortcuts and Azure ML datastores. Real directories give atomic renames (important for Spark and Delta commits) and POSIX-like ACLs per directory. Accessed via `abfss://` |
| Azure Files | Managed SMB and NFS file shares | Shared home directories, legacy tools that expect a file system |
| Azure Managed Lustre | Managed Lustre parallel file system that can import from and export to Blob | High-throughput reads for large distributed training jobs where Blob streaming becomes the bottleneck. Check region availability and current Blob integration features before committing |
| Premium block blob | SSD-backed blob storage | Lower-latency small-object reads, e.g. many small image files |

Good hygiene for ML storage accounts: disable shared key access where possible (forcing Entra auth), disable public network access and use private endpoints, enable soft delete and versioning for datasets that matter, and keep training data and model artifacts in separate containers with separate role assignments.

**What interviewers listen for**

- You know archive is offline and that cool, cold and archive have minimum retention periods, so tiering short-lived data can cost more, not less.
- You can explain why hierarchical namespace matters for analytics engines (directory operations, ACLs) rather than just saying "it is a data lake".
- You match storage to the access pattern: Blob for durable data, a parallel file system or local NVMe caching for throughput-bound training.

---

## Compute

| Service | What it is | Use it for | Watch out for |
|---|---|---|---|
| GPU virtual machines (N-series) | VM families with NVIDIA or AMD GPUs. Broadly: **NC** for compute (training and inference), **ND** for large-scale deep learning (multi-GPU, high-bandwidth interconnect such as InfiniBand), **NV** for visualization and virtual desktops | Custom training, research boxes, self-managed inference | Quota is per family per region and often must be requested; availability varies by region; pick sizes from current docs rather than memory |
| Spot VMs | Unused capacity at a discount that Azure can evict with short notice (delivered through Scheduled Events) | Fault-tolerant training with checkpointing, batch scoring, hyperparameter sweeps | Eviction policy (deallocate or delete), no SLA, must checkpoint frequently |
| Reservations and Azure savings plan for compute | One- or three-year commitments in exchange for lower rates (reservations are tied to a VM family and region; savings plans to an hourly spend amount) | Steady, predictable baseload such as always-on inference | Commit only to what you will use; forecast from real utilization |
| Azure Kubernetes Service (AKS) | Managed Kubernetes | Multi-model serving, custom inference stacks (e.g. vLLM, Triton), shared platforms | You own upgrades, node images, scaling configs and security hardening |
| Azure Container Apps | Serverless containers on a managed Kubernetes-based platform, scaling with KEDA rules, including to zero | APIs, workers, lightweight model services without cluster management | Less control than AKS. Serverless GPU support exists in selected regions; check GPU types, regions and quotas before designing around it |
| Azure Functions | Event-driven serverless functions | Glue: react to a blob landing, trigger a pipeline, small preprocessing | Not a fit for heavy model serving or long GPU work |
| Azure Batch | Managed pools of VMs for large-scale parallel and HPC jobs | Embarrassingly parallel scoring, simulations, preprocessing at scale | You manage task scheduling semantics and pool images; Azure ML jobs are often simpler for ML-specific work |
| Azure Container Registry (ACR) | Private container registry | Training and inference images | Pull with a managed identity (`AcrPull`); private endpoints require the Premium tier |

### AKS for ML

- **Node pools:** a system node pool runs cluster add-ons; user node pools run workloads. Separate CPU and GPU user pools so CPU workloads never land on expensive GPU nodes, using taints and tolerations plus node selectors.
- **GPU node pools:** GPU nodes need drivers plus the NVIDIA device plugin (or NVIDIA GPU Operator) so pods can request `nvidia.com/gpu`. User pools can scale down to zero nodes with the cluster autoscaler; the system pool cannot.
- **Autoscaling:** the cluster autoscaler adds and removes nodes; the Horizontal Pod Autoscaler scales pods on metrics; the **KEDA add-on** scales on event sources such as queue length, Event Hubs lag or Prometheus queries, which suits batch and async inference workers.
- **Identity:** use Microsoft Entra Workload ID so pods get Entra tokens via a federated credential on a user-assigned managed identity.
- **Azure ML integration:** an AKS cluster (or Azure Arc-enabled cluster) can be attached to an Azure ML workspace as Kubernetes compute, giving Azure ML-managed deployments on infrastructure you control.

**What interviewers listen for**

- You can describe GPU families at the level of "compute vs large-scale training vs visualization" and say you would check current sizes and quota rather than reciting specs.
- Spot answers include checkpointing and eviction handling, not just "it is cheaper".
- You can argue managed endpoints vs AKS in terms of team skills, control, multi-model density and operational burden.

---

## Data and analytics

| Service | What it is | ML use |
|---|---|---|
| Azure Data Factory | Managed orchestration and data movement: pipelines, copy activities, mapping data flows, integration runtimes (Azure, self-hosted for on-premises, Azure-SSIS) | Land data from SaaS, databases and on-premises into ADLS; schedule preprocessing; trigger Azure ML pipelines |
| Microsoft Fabric | SaaS analytics platform built around **OneLake** (one logical lake per tenant, Delta Parquet as the table format), with lakehouses, warehouses, Data Factory pipelines, notebooks, real-time analytics and Power BI on shared capacity | Analytics-heavy organizations, Power BI users, teams wanting one governed SaaS data platform. Shortcuts reference data in ADLS Gen2 or other clouds without copying |
| Azure Synapse Analytics | Earlier integrated analytics service (dedicated and serverless SQL pools, Spark pools, pipelines) | Existing estates. Microsoft positions Fabric as the strategic direction for new analytics investment; Synapse remains supported, but check current guidance before starting new work on it |
| Azure Databricks | First-party Azure service running the Databricks platform: Spark, Delta Lake, Unity Catalog, MLflow, model serving | Large-scale feature engineering, Spark-native ML teams, multi-cloud consistency |
| Event Hubs | Managed event ingestion with partitions and consumer groups; exposes an **Apache Kafka-compatible endpoint** (Standard tier and above) | Streaming features, clickstream, telemetry; existing Kafka clients can connect by changing configuration. Capture writes raw events to Blob or ADLS |
| Azure Stream Analytics | Managed stream processing with a SQL-like query language | Windowed aggregates for real-time features, anomaly flags, routing events |
| Azure Cosmos DB | Globally distributed NoSQL database with several APIs, throughput in request units (provisioned, autoscale or serverless) | Low-latency online feature or session lookups, chat history, app state. The NoSQL API supports vector indexing and search |
| Azure AI Search (formerly Azure Cognitive Search) | Search service with full-text, vector and **hybrid** search (results fused with Reciprocal Rank Fusion), an optional semantic ranker, indexers and skillsets, and integrated vectorization | The retrieval layer for most enterprise RAG on Azure |

### Fabric vs Databricks vs Synapse at a glance

| Question | Leans Fabric | Leans Databricks |
|---|---|---|
| Who are the main users? | Analysts, BI developers, mixed-skill teams | Data engineers and ML engineers comfortable with Spark and code |
| Billing model | Capacity (F SKU) shared by all workloads; can be paused | Usage-based compute (DBUs) plus the underlying cloud resources, or serverless |
| Multi-cloud | Azure-centric SaaS | Same platform on Azure, AWS and GCP |
| ML depth | Notebooks, MLflow experiments and models, good for lighter ML | Mature ML and serving features, Unity Catalog governance for features and models |

Both use Delta Lake tables, and Fabric can read data in ADLS through shortcuts, so many organizations use both: Databricks for heavy engineering and ML, Fabric for BI and self-service analytics.

**What interviewers listen for**

- You can name where data lands (ADLS or OneLake), what moves it (Data Factory, Event Hubs), what transforms it (Spark in Databricks or Fabric, Stream Analytics) and what serves it (Cosmos DB, AI Search).
- You hedge appropriately on Fabric vs Synapse: Fabric is the strategic direction, but existing Synapse estates are real and migrations take planning.
- You know AI Search hybrid retrieval (keyword plus vector plus optional semantic reranking) usually beats pure vector search on enterprise documents.

---

## Generative AI

### Azure OpenAI in Microsoft Foundry

Azure OpenAI provides OpenAI models hosted in Azure. It is documented as part of **Foundry Models** and managed through **Azure AI Foundry (now Microsoft Foundry)**, which also hosts models from other providers, evaluation, tracing and agents. Newer Foundry projects sit on a Foundry resource; older hub-based projects build on Azure ML hubs. Check which model your organization uses, because networking and RBAC setup differ.

Core concepts:

| Concept | What it means |
|---|---|
| Resource | The Azure resource with an endpoint such as `https://<resource>.openai.azure.com/`, its own networking, keys and RBAC |
| Deployment | A named instance of a specific model and version inside the resource. Your code calls the **deployment name**, not the model name, so you can swap versions behind a stable name |
| Quota | Granted per subscription, per region, per model, measured in tokens per minute (TPM) for standard deployments; requests per minute limits derive from it |
| Rate limiting | Exceeding a deployment's limits returns HTTP 429 with retry hints in the response headers |
| Content filtering | Default filters classify prompts and completions for categories such as hate, sexual, violence and self-harm at configurable severity levels, plus features such as prompt shields for jailbreak and indirect prompt injection attacks. Filter configurations are attached to deployments |

### Deployment types (in general terms)

| Type | Billing and capacity | Data processing location | Typical use |
|---|---|---|---|
| Standard (regional) | Pay per token, shared capacity in one region | That region | Workloads with strict regional processing requirements |
| Global Standard | Pay per token, traffic routed across Azure's global capacity | Any region where the model is deployed (data at rest stays in your chosen geography) | Default for many apps: higher available quota, better resilience |
| Data Zone Standard | Pay per token, routed within a data zone (for example US or EU) | Within that data zone | A middle ground for data residency |
| Provisioned (regional, data zone or global) | Reserved throughput units (PTUs), billed for the reservation whether used or not | Depends on the variant | Predictable latency for steady, high-volume traffic |
| Batch (global or data zone) | Asynchronous jobs at a lower price than standard, longer turnaround | Depends on the variant | Offline enrichment, bulk classification, evaluations |

Exact names, discounts and model availability change; check the current deployment types page before quoting specifics.

### Other GenAI building blocks

- **Foundry Agent Service:** a managed runtime for agents that handles conversation state, tool calling and grounding (for example with Azure AI Search, files or OpenAPI tools) with enterprise networking and identity. Use it when you want a managed agent backend instead of hosting your own orchestration loop.
- **Prompt flow:** an orchestration, evaluation and deployment tool for LLM flows in Azure ML and Foundry. Treat it as a pointer only: Microsoft's newer investment is in Foundry agents, evaluations and its agent frameworks, so check its current support status before building on it.
- **Azure AI Content Safety:** the standalone moderation service behind much of the content filtering, also callable directly for user-generated content.
- **Azure API Management as an AI gateway:** policies for per-consumer token limits, load balancing across several Azure OpenAI backends with circuit breaking, token usage metrics and semantic caching. Useful when many teams share model capacity.

**What interviewers listen for**

- You call deployments by a configurable name and keep code model-agnostic.
- You use Entra ID auth with a managed identity and disable key-based (local) auth where possible.
- You explain the latency, cost and residency tradeoffs between global, data zone, regional and provisioned deployments.
- You treat content filters as a layer, not a complete safety story: you still need input validation, output checks and evaluation (see [LLM Security](../ai_genai/intro_llm_security.md)).

---

## Observability

| Service | What it does | ML use |
|---|---|---|
| Azure Monitor | The umbrella platform for metrics, logs, alerts and action groups | Alert on endpoint latency, error rates, GPU node health, quota usage |
| Log Analytics workspace | Log store queried with **KQL** (Kusto Query Language) | Central place for resource logs (via diagnostic settings), activity logs and app telemetry |
| Application Insights | Application performance monitoring (requests, dependencies, exceptions, traces), workspace-based and OpenTelemetry-friendly | Instrument inference APIs and LLM apps; Foundry tracing can send traces here |
| Container insights, Managed Prometheus and Managed Grafana | Kubernetes and Prometheus-style metrics | AKS inference clusters, GPU utilization exporters |
| Azure ML model monitoring | Data drift, prediction drift and data quality checks on deployed models | See [Azure ML guide](./intro_azure_ml.md) and [Model Monitoring](../mlops/intro_model_monitoring.md) |

Resource logs are not collected by default for most services: you create **diagnostic settings** to send them to Log Analytics, Storage or Event Hubs. A small KQL example for an inference API instrumented with Application Insights:

```kusto
requests
| where timestamp > ago(1h)
| summarize p95_ms = percentile(duration, 95), failures = countif(success == false), total = count()
    by bin(timestamp, 5m), name
| order by timestamp asc
```

**What interviewers listen for**

- You monitor three layers: infrastructure (nodes, GPUs, quota), service (latency, errors, 429s, token usage) and model quality (drift, evaluation scores, user feedback).
- You know diagnostic settings must be configured explicitly and that log ingestion and retention cost money.
- You can write or at least read a basic KQL query.

---

## Cost control

| Lever | What to do |
|---|---|
| Tags | Tag every resource with owner, cost center, environment and project; enforce required tags with Azure Policy so cost reports can be grouped and chargebacks work |
| Cost Management budgets and alerts | Create budgets at subscription or resource group scope with alerts on actual and forecasted spend, routed to action groups (email, Teams, automation); enable cost anomaly alerts |
| Spot VMs | Use for interruptible training and batch scoring with checkpointing |
| Autoscale to zero | Azure ML compute clusters with minimum nodes of zero, AKS user node pools that scale to zero, Container Apps scaling to zero |
| Deallocate, do not just stop | A VM shut down from inside the OS is still allocated and still billed for compute; "Stopped (deallocated)" releases the hardware and stops compute charges (disks and some IPs are still billed) |
| Idle shutdown for compute instances | Configure idle shutdown and schedules on Azure ML compute instances so notebooks do not run all weekend |
| Commitments | Reservations or savings plans only for measured, steady baseload |
| Provisioned AI capacity | PTUs are billed whether used or not; size from measured traffic and keep bursty traffic on pay-per-token deployments |
| Network costs | Data transfer between regions and to the internet is charged; private endpoints and NAT gateways have hourly and per-GB processing charges. Keep training compute in the same region as its data |
| Log costs | Set retention per table, avoid sending verbose debug logs to Log Analytics in production, use cheaper log tiers for high-volume, rarely queried data |
| Restrict expensive SKUs | Use Azure Policy "allowed VM sizes" and per-subscription quota to stop accidental large GPU deployments |

**What interviewers listen for**

- You attach cost controls to ownership (tags, budgets per team) instead of a monthly surprise for the platform team.
- You mention the stop vs deallocate trap and idle compute instances, which are classic real-world waste.
- You can reason about network and logging costs, not only VM costs.

---

## Reference architectures

### (a) Batch training and batch scoring

```text
 Source systems (DBs, SaaS, files)
          |
          v
 Azure Data Factory (copy, schedule)  ---->  ADLS Gen2 (raw -> curated, Delta tables)
                                                   |
                                                   v
                                     Azure Databricks / Fabric Spark
                                     (feature engineering, data checks)
                                                   |
                                                   v
                              Azure ML pipeline (train -> evaluate -> register)
                              on compute cluster (Spot where tolerable, scale to 0)
                                                   |
                                                   v
                              Azure ML registry (versioned model, approval gate)
                                                   |
                                                   v
                              Azure ML batch endpoint (scheduled scoring)
                                                   |
                                                   v
                              ADLS Gen2 / Cosmos DB / warehouse (predictions)
```

**Walkthrough:** Data Factory lands raw data in ADLS Gen2. Spark in Databricks or Fabric builds curated feature tables in Delta format with data quality checks. An Azure ML pipeline, triggered on a schedule or by CI/CD, trains on a compute cluster that scales to zero, evaluates against the current production model and registers the candidate in an Azure ML registry shared across workspaces. After approval, a batch endpoint scores the latest data and writes predictions back to storage or a serving database. Every step runs as a managed identity with data roles on only the containers it needs, and storage, the workspace and the registry are reachable only through private endpoints.

### (b) Real-time inference

```text
 Clients (internal apps, partners)
          |
          v
 Azure API Management (auth with Entra, rate limits, versioning)   <- public or internal
          |  (private networking from here on)
          v
 +-------------------------------+        +-------------------------------+
 | Option 1: Azure ML managed    |   or   | Option 2: AKS                 |
 | online endpoint               |        | (GPU and CPU user node pools, |
 | (blue/green traffic split,    |        |  KEDA/HPA, Workload ID,       |
 |  autoscale, managed VNet)     |        |  internal load balancer)      |
 +-------------------------------+        +-------------------------------+
          |                                         |
          v                                         v
 Feature lookups: Cosmos DB / Redis      Model artifacts: registry, ACR images
          |
          v
 Application Insights + Azure Monitor alerts + model monitoring
```

**Walkthrough:** API Management is the single front door: it validates Entra tokens, applies per-client quotas and lets you version the API independently of the model. Behind it, either a managed online endpoint (less to operate, built-in traffic splitting for safe rollouts) or AKS (more control, better for many models per GPU or custom serving stacks) runs the model with no public endpoint. Online features come from a low-latency store reached via private endpoints, containers come from ACR with `AcrPull` via managed identity, and telemetry flows to Application Insights and Log Analytics with alerts on latency, errors and drift.

### (c) Enterprise RAG

```text
 Documents: Blob Storage / ADLS / SharePoint
          |
          v
 Azure AI Search indexer + skillset
 (crack documents, chunk, embed with an embedding deployment)
          |
          v
 Azure AI Search index
 (text fields + vector fields + permission metadata, hybrid search, semantic ranker)
          ^
          | query (hybrid retrieve, filtered by user's groups)
          |
 App (Container Apps / App Service / AKS)  <---- user signs in with Microsoft Entra ID
          |
          | grounded prompt (managed identity, Entra token auth)
          v
 Azure OpenAI deployment in Microsoft Foundry (private endpoint, content filtering)
          |
          v
 Answer with citations -> Application Insights traces, evaluation and feedback store
```

**Walkthrough:** An AI Search indexer pulls documents from Blob or ADLS (SharePoint ingestion options exist but have changed over time, so check current support), chunks them and calls an embedding deployment to populate vector fields. The app authenticates the user with Entra ID, runs a hybrid query (keyword plus vector, optionally semantic reranking) filtered by the user's group memberships so people only retrieve documents they are allowed to see, and sends the grounded prompt to a chat deployment using its managed identity. Search, Azure OpenAI and storage all sit behind private endpoints with key auth disabled; traces, token usage, retrieval quality and user feedback feed evaluation. For RAG design details see [RAG Engineering](../ai_genai/intro_rag_engineering.md).

---

## Code examples

All examples use `DefaultAzureCredential`, which tries several credential sources in order (environment variables, workload identity, managed identity, Azure CLI login and others). The same code works on a laptop after `az login` and in Azure with a managed identity. In production, consider a specific credential such as `ManagedIdentityCredential` to avoid slow or surprising fallbacks through the chain.

### Upload a file to Blob Storage with Entra auth

```python
# pip install azure-identity azure-storage-blob
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT_URL = "https://<storage-account>.blob.core.windows.net"

# The identity needs a data-plane role such as "Storage Blob Data Contributor"
# on the container or account. "Contributor" alone is not enough.
credential = DefaultAzureCredential()
service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
container = service.get_container_client("training-data")

with open("train.parquet", "rb") as data:
    container.upload_blob(
        name="churn/2026-10-01/train.parquet",
        data=data,
        overwrite=True,
    )
```

### Read a secret from Key Vault

```python
# pip install azure-identity azure-keyvault-secrets
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

VAULT_URL = "https://<vault-name>.vault.azure.net"

# The identity needs a role such as "Key Vault Secrets User" (RBAC permission model).
client = SecretClient(vault_url=VAULT_URL, credential=DefaultAzureCredential())
partner_api_key = client.get_secret("partner-api-key").value

# Use the value in memory; never log it or write it to disk.
```

Only use Key Vault for secrets you cannot eliminate (for example a third-party API key). For Azure services, use managed identity directly, as in the other examples.

### Call an Azure OpenAI deployment with Entra token auth

```python
# pip install openai azure-identity
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # https://<resource>.openai.azure.com/
    azure_ad_token_provider=token_provider,  # no API key; needs "Cognitive Services OpenAI User"
    # PLACEHOLDER: check the current supported api_version in the Azure OpenAI docs.
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "<check-current-api-version>"),
    max_retries=5,  # the SDK retries 429 and 5xx with backoff
)

DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # your deployment name, not a model name

response = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
        {"role": "system", "content": "You answer questions about internal ML runbooks."},
        {"role": "user", "content": "How do I roll back a managed online endpoint?"},
    ],
)
print(response.choices[0].message.content)
```

Azure OpenAI also offers a newer versioned `/openai/v1/` API path that works with the standard `OpenAI` client and does not need an `api_version` parameter; check the current docs for which path your organization standardizes on.

### Fall back across deployments on rate limits

```python
from openai import RateLimitError


def chat_with_fallback(targets, messages):
    """targets: list of (client, deployment_name) pairs, e.g. in different regions."""
    last_error = None
    for client, deployment_name in targets:
        try:
            return client.chat.completions.create(model=deployment_name, messages=messages)
        except RateLimitError as err:  # HTTP 429 after the client's own retries
            last_error = err
    raise last_error
```

In larger systems this logic usually lives in a gateway (API Management backend pools with circuit breakers) rather than in every application.

### GitHub Actions login with workload identity federation

```yaml
# The app registration or user-assigned managed identity has a federated credential
# trusting this repository (for example subject "repo:my-org/my-repo:environment:prod").
permissions:
  id-token: write   # allow the workflow to request an OIDC token
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: prod
    steps:
      - uses: actions/checkout@v4
      - uses: azure/login@v2
        with:
          client-id: ${{ vars.AZURE_CLIENT_ID }}
          tenant-id: ${{ vars.AZURE_TENANT_ID }}
          subscription-id: ${{ vars.AZURE_SUBSCRIPTION_ID }}
      # Later steps run as that identity; no client secret is stored in GitHub.
```

---

## Interview Q&A

#### What is the difference between a management group, a subscription and a resource group?

A resource group is a lifecycle container: resources that are deployed, updated and deleted together, such as everything for one model service in one environment. A subscription is a billing, quota and scale boundary, and it is where GPU quota and many service limits are granted. Management groups sit above subscriptions and exist so you can apply Azure Policy and role assignments once and have them inherit to many subscriptions. For ML, the practical consequences are that production and development should not share a subscription (so quota and blast radius are separate) and that guardrails such as "no public endpoints" belong at the management group level. The tradeoff is overhead: too many subscriptions create networking and identity sprawl, so most organizations standardize on a small set per workload and environment.

#### How would you remove secrets from an ML service's code using managed identities?

First, inventory every secret: storage keys, Azure OpenAI keys, database passwords and third-party tokens. For every Azure service that supports Entra auth (Storage, Key Vault, Azure OpenAI, AI Search, Cosmos DB, Azure SQL), assign the service's managed identity a narrow data-plane role and switch the code to `DefaultAzureCredential` or `ManagedIdentityCredential`. Then disable key-based or local auth on those resources so nobody can fall back to keys. For the secrets that genuinely cannot go away, such as a partner API key, store them in Key Vault and let the managed identity read them with `Key Vault Secrets User`. Finally, rotate the old keys, because anything that was in code or CI variables must be considered leaked. The main tradeoff is migration effort and the need for local developer access, which `az login` plus personal role assignments in dev handles cleanly.

#### When would you choose a system-assigned managed identity over a user-assigned one?

A system-assigned identity is created with a resource and deleted with it, so it is ideal when one resource needs its own identity and you want cleanup to be automatic. A user-assigned identity is a standalone resource that can be attached to several resources and created before them. That makes it better when several replicas or services need the same permissions, when you want role assignments to exist before the compute is deployed (avoiding a race in infrastructure-as-code), or for AKS workload identity and federated credentials. The downside of user-assigned identities is that they outlive their consumers and can accumulate stale permissions if nobody owns their lifecycle. A reasonable default is user-assigned for platform components and workloads managed by IaC, system-assigned for simple one-off resources.

#### Why can a user with Contributor on a storage account get a 403 when reading blobs?

`Contributor` is a control-plane role: it lets you manage the storage account resource, but reading blob data with Entra authentication requires a data-plane role such as `Storage Blob Data Reader` or `Storage Blob Data Contributor`. Contributor can often still list the account keys and read data that way, which is exactly why many organizations disable shared key access. Once shared keys are disabled, only data-plane role assignments work, which is the intended secure state. The same split exists elsewhere: managing an Azure OpenAI resource is different from calling its deployments, which needs a role such as `Cognitive Services OpenAI User`. Role assignments can also take a few minutes to propagate, so a brand-new assignment may briefly still return 403.

#### How does a GitHub Actions workflow deploy to Azure without storing a client secret?

It uses workload identity federation. You create a federated credential on an app registration or a user-assigned managed identity that trusts GitHub's OIDC issuer and a specific subject, such as a repository plus an environment or branch. The workflow requests an OIDC token (`id-token: write`), and the `azure/login` action exchanges it for a short-lived Entra token. Nothing long-lived is stored in GitHub, and the trust is scoped to that repository and environment. The tradeoff is that subjects must be precise: an overly broad subject (any branch, any pull request) could let untrusted code deploy, so production credentials should be tied to a protected environment with required reviewers. The same pattern powers AKS workload identity, where the trusted issuer is the cluster's OIDC issuer and the subject is a Kubernetes service account.

#### What is the difference between a private endpoint and a service endpoint?

A service endpoint lets traffic from a subnet reach a PaaS service over the Azure backbone and lets the service firewall allow that subnet, but the service keeps its public endpoint and the setting applies to the service type rather than one resource instance. A private endpoint creates a network interface with a private IP in your VNet that maps to one specific resource, so you can turn off public network access entirely and reach it from on-premises over VPN or ExpressRoute. Private endpoints are the standard for regulated ML workloads and for data exfiltration concerns because access is bound to a specific resource. They cost more and require Private DNS zones to be linked correctly, which is where most implementation bugs happen. Service endpoints are simpler and free but are rarely enough when the requirement is "no public endpoint".

#### How would you lock down an Azure OpenAI resource with private endpoints?

Create a private endpoint for the resource in a dedicated subnet and link the matching Private DNS zone (for example `privatelink.openai.azure.com`) to every VNet that must resolve it, including the hub if DNS is centralized. Then disable public network access on the resource and disable local (key) auth so only Entra-authenticated callers on the private network can reach it. Grant the calling app's managed identity `Cognitive Services OpenAI User` on that resource only. If other services call it on your behalf, such as AI Search using integrated vectorization or Foundry agents, check how they connect (trusted service access, shared private links or managed networks), because they will fail once public access is off. Verify from inside the VNet that the hostname resolves to a private IP, and add Azure Policy to deny public network access on new AI resources. The tradeoff is operational complexity: developer access needs VPN, a jump box or a dev environment that still permits it.

#### When would you serve a model on AKS instead of an Azure ML managed online endpoint?

Managed online endpoints are the default when you want Azure to run the infrastructure: you get autoscaling, blue/green traffic splitting, authentication and monitoring without owning a cluster. AKS makes sense when you need control the managed service does not give you: custom serving stacks such as vLLM or Triton with specific tuning, packing many models onto shared GPUs, sidecars, service mesh, or a platform team that already runs Kubernetes well. AKS also helps when inference is one part of a larger microservice system that already lives on the cluster. The cost is operational: upgrades, node images, GPU drivers, autoscaling, security policies and on-call all become yours. A middle path is attaching AKS to Azure ML as Kubernetes compute, keeping Azure ML's deployment workflow on infrastructure you control. I would choose based on team skills and model count, not on raw performance, since both can run the same container.

#### How would you choose between Microsoft Fabric and Azure Databricks for an ML team's data platform?

I would start from the users and existing estate. If most consumers are analysts and Power BI developers, and the organization wants one SaaS platform with capacity-based billing and minimal infrastructure, Fabric is a strong fit, and its data science features handle lighter ML. If the team is engineering-heavy, runs large Spark workloads, needs mature feature engineering, model serving and Unity Catalog governance, or must stay consistent across clouds, Databricks is usually the better center of gravity. They are not mutually exclusive: both use Delta tables, and Fabric shortcuts can read data that Databricks writes to ADLS, so a common pattern is Databricks for engineering and ML with Fabric for BI. I would also weigh cost models (shared capacity vs usage-based compute), existing skills and governance tooling. If the organization is on Synapse today, I would note that Fabric is Microsoft's strategic direction and plan migration deliberately rather than rushing it.

#### How do you handle 429 rate-limit errors from Azure OpenAI in production?

A 429 means the deployment exceeded its tokens-per-minute or requests-per-minute limits (or provisioned capacity is saturated), so the first fix is client-side: retry with exponential backoff and jitter, honoring the retry hints in the response headers, which the `openai` SDK does with `max_retries`. Next, reduce demand: set sensible `max_tokens`, trim prompts, cache repeated answers, and move non-interactive work to batch deployments. For capacity, request more quota, use Global Standard or Data Zone deployments that draw from larger pools, and spread traffic across multiple deployments or regions with failover, ideally in a gateway such as API Management with backend pools and circuit breakers. For steady high-volume traffic with strict latency needs, provisioned throughput gives predictable capacity, with pay-per-token deployments absorbing spillover. Finally, monitor 429 rates and token usage per consumer so one noisy client cannot starve others, using per-consumer token limits at the gateway. The tradeoff is complexity: multi-region fallback can change data residency and latency, so check that against compliance requirements.

#### How would you choose between global, data zone, regional and provisioned Azure OpenAI deployments?

Start with data residency: global deployments may process prompts in any region where the model is deployed, data zone deployments keep processing within a zone such as the US or EU, and regional deployments keep it in one region. If residency allows it, global standard is usually the best default because it offers larger quota and better availability. Provisioned deployments reserve throughput and give more predictable latency, but you pay for the reservation whether you use it or not, so they suit steady, high-volume production traffic measured over time. Batch deployments suit offline jobs where turnaround in hours is fine at a lower price. Many production systems combine them: provisioned for baseload, standard for bursts, batch for offline enrichment. I would confirm current names, model availability per region and pricing in the docs before committing.

#### How would you design a multi-subscription landing zone for ML?

I would follow the Azure landing zone pattern: a platform management group with subscriptions for identity, connectivity (hub VNet, firewall, DNS) and management (Log Analytics, monitoring), and a landing zones management group holding workload subscriptions. ML workloads get separate subscriptions per environment, for example `ml-dev`, `ml-test` and `ml-prod`, so GPU quota, budgets and blast radius are isolated, plus possibly a shared subscription for cross-environment assets like an Azure ML registry and container registry. Spokes peer to the hub, private endpoints use centrally managed Private DNS zones, and egress goes through the hub firewall or NAT gateways. Azure Policy at the management group level enforces tags, allowed regions, denial of public network access and allowed VM sizes, and diagnostic settings are deployed automatically. Access is through Entra groups with environment-specific roles, and CI/CD uses federated identities scoped per environment. The tradeoff is upfront effort and some friction for experimentation, which a loosely governed sandbox subscription with tight budgets can relieve.

#### How would you build a RAG system on Azure that respects document permissions?

Index documents into Azure AI Search with vector and text fields, and store permission metadata on each chunk, typically the Entra group IDs allowed to read the source document. At query time the app authenticates the user with Entra ID, resolves their group memberships and applies them as a security filter on the search query so that only permitted chunks are retrieved; check whether newer built-in permission-aware indexing features fit your sources. Permissions must be kept in sync as they change in the source system, which is the hardest operational part, so the indexer schedule or an event-driven update path needs monitoring. The model should never see unfiltered results, and the app should cite sources so users can verify answers. The tradeoff is freshness vs cost: frequent reindexing of permissions costs compute, while stale ACLs can leak information, so I would prioritize correctness for sensitive collections.

#### How do you keep GPU spend under control on Azure?

Make idle cost visible first: tags per team and project, budgets with forecast alerts, and a dashboard of GPU utilization by owner. Then remove idle capacity: Azure ML clusters with a minimum of zero nodes, AKS GPU node pools that scale to zero, compute instance idle shutdown, and deallocating (not just stopping inside the OS) any standalone VMs. Use Spot VMs for training and batch work with regular checkpointing, and keep on-demand capacity for serving and deadlines. Right-size: smaller GPUs for inference, quantized or distilled models where quality allows, and batch endpoints instead of always-on endpoints for offline scoring. Commitments such as reservations or savings plans only make sense once utilization data shows a stable baseload. Finally, enforce guardrails with Azure Policy on allowed VM sizes and subscription quota so one experiment cannot launch a large cluster unnoticed.

#### How would you monitor an LLM application on Azure end to end?

I would instrument the app with OpenTelemetry and send traces to Application Insights, capturing each request's retrieval step, model call, token usage, latency and errors, including 429s. Resource logs and metrics from Azure OpenAI, AI Search and the hosting platform go to a Log Analytics workspace through diagnostic settings, with KQL-based alerts on latency percentiles, error rates and throttling. On top of operational metrics, I would track quality: offline evaluation suites on every prompt or model change, online sampling with groundedness or relevance evaluators, and user feedback tied to trace IDs. Logging full prompts and responses has privacy implications, so I would redact or sample and set retention deliberately. Cost is part of monitoring too: tokens per request and per tenant, trended over time. The tradeoff is log volume vs insight, so I would keep detailed traces sampled and aggregate metrics complete.

#### How would you ingest streaming events from existing Kafka producers into Azure?

Event Hubs exposes a Kafka-compatible endpoint on Standard tier and above, so existing producers can often switch by changing the bootstrap server and authentication configuration rather than rewriting code. Event Hubs Capture can persist raw events to Blob or ADLS for replay and training data, while Stream Analytics, Spark Structured Streaming in Databricks or Fabric real-time features compute windowed aggregates for online features. Partitions determine parallelism, so the partition count and keys need planning up front. The tradeoff is that Event Hubs is not a full Kafka distribution: some Kafka features and ecosystem tools may differ, so teams that rely heavily on Kafka-specific features may prefer a managed Kafka offering. For authentication, I would use Entra ID with managed identities rather than connection strings.

---

## Common Pitfalls

| Problem | Why it hurts | Fix |
|---|---|---|
| Keys and connection strings in code, notebooks or CI variables | Leaks are hard to detect and rotate; anyone with the key has full data access | Managed identities with data-plane roles, workload identity federation for CI, Key Vault for unavoidable secrets, disable local auth |
| Granting `Owner` or `Contributor` at subscription scope "to make it work" | Massive blast radius, and it still may not grant data access | Narrow scope and data-plane roles; Entra groups; just-in-time elevation for admins |
| Private endpoint created but Private DNS zone not linked to the VNet | Names resolve to the public IP, calls fail once public access is disabled | Centrally managed Private DNS zones linked to all VNets that need them; test resolution from inside the network |
| Disabling public access without checking dependent services | Indexers, agents or pipelines that reach the resource break silently | Map every caller first; use trusted service access, shared private links or managed networks as appropriate |
| Stopping VMs from inside the OS | VM remains allocated and compute keeps billing | Stop and deallocate from the portal, CLI or automation; auto-shutdown schedules |
| Compute instances and GPU clusters left running | Large idle spend | Idle shutdown, minimum nodes of zero, budgets with alerts |
| Tiering short-lived data to cool, cold or archive | Early deletion charges and archive rehydration delays | Tier by real access patterns; lifecycle policies based on age and last access |
| Training compute in a different region from the data | Egress charges and slower throughput | Co-locate data and compute; replicate data deliberately if needed |
| One Azure OpenAI deployment shared by every team with no limits | One client causes 429s for everyone | Per-consumer token limits at a gateway, multiple deployments, quota planning |
| Hard-coding model names or API versions everywhere | Painful upgrades and inconsistent behavior | Deployment names and API version in configuration; test upgrades in a non-prod deployment |
| No diagnostic settings | No resource logs when an incident happens | Deploy diagnostic settings with Azure Policy or IaC to a central Log Analytics workspace |
| Single subscription for dev and prod ML | Shared quota, shared blast radius, unclear costs | Separate subscriptions per environment under a management group with shared policy |

---

## Related Topics

| Topic | Why It's Related |
|---|---|
| [Azure Machine Learning Interview Guide](./intro_azure_ml.md) | Workspace, jobs, registries and endpoints in depth |
| [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md) | Azure ML vs SageMaker vs Vertex AI |
| [AWS for ML Engineers](./aws_for_ml_engineers.md) | The equivalent surrounding services on AWS |
| [GCP for ML Engineers](./gcp_for_ml_engineers.md) | The equivalent surrounding services on Google Cloud |
| [Cloud Service Mapping](./cloud_service_mapping.md) | Azure, AWS and GCP services side by side |
| [Model Serving](../mlops/intro_model_serving.md) | Serving patterns behind managed endpoints and AKS |
| [Model Monitoring](../mlops/intro_model_monitoring.md) | Drift and performance monitoring concepts |
| [CI/CD for Machine Learning](../mlops/intro_cicd_for_ml.md) | Pipelines that use workload identity federation |
| [MLflow](../mlops/intro_mlflow.md) | Tracking and model registry used by Azure ML and Databricks |
| [Kubernetes](../devops/intro_kubernetes.md) | Foundations for AKS |
| [Terraform](../devops/intro_terraform.md) | Infrastructure as code for landing zones and ML resources |
| [GitHub Actions for CI/CD](../devops/intro_github_actions.md) | OIDC-based deployments to Azure |
| [Observability](../devops/intro_observability.md) | Metrics, logs and traces concepts behind Azure Monitor |
| [Apache Kafka](../data_engineering/intro_apache_kafka.md) | Kafka concepts used with the Event Hubs Kafka endpoint |
| [Delta Lake](../data_engineering/intro_delta_lake.md) | Table format used by Databricks and Fabric |
| [Data Engineering for AI](../data_engineering/intro_data_engineering_for_ai.md) | Data pipelines feeding training and RAG |
| [RAG Engineering](../ai_genai/intro_rag_engineering.md) | Retrieval design behind the enterprise RAG architecture |
| [Vector Databases](../ai_genai/intro_vector_databases.md) | Vector search concepts used in AI Search and Cosmos DB |
| [LLMOps](../ai_genai/intro_llmops.md) | Operating Azure OpenAI-based applications |
| [LLM Security](../ai_genai/intro_llm_security.md) | Prompt injection and defenses beyond content filtering |
