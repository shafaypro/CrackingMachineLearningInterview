# Cloud ML Platforms

This track covers the managed machine learning platforms most commonly discussed in ML engineer, MLOps, and applied AI interviews:

- AWS SageMaker
- Google Vertex AI
- Azure Machine Learning

Use this section when you need to explain how a team trains, deploys, monitors, and governs models without building the full platform stack from scratch.

---

## Table of Contents

1. [Why cloud ML platforms matter](#why-cloud-ml-platforms-matter)
2. [Shared building blocks](#shared-building-blocks)
3. [Platform selection framework](#platform-selection-framework)
4. [Common interview themes](#common-interview-themes)
5. [Learning path](#learning-path)
6. [Track links](#track-links)

---

## Why cloud ML platforms matter

Most real-world ML teams do not start from bare Kubernetes clusters and custom deployment tooling. They use a managed platform because it reduces:

- time to first production deployment
- infrastructure ownership burden
- operational risk around training jobs and model hosting
- friction around IAM, networking, registries, and auditability

In interviews, cloud ML questions often test whether you understand:

- the full model lifecycle, not just notebook experimentation
- the tradeoff between managed convenience and platform lock-in
- how platform features map to business requirements like latency, compliance, and cost

---

## Shared building blocks

Across SageMaker, Vertex AI, and Azure ML, you will see the same core concepts:

| Capability | What it does | Why interviewers care |
|---|---|---|
| Notebook / workbench | Interactive development environment | How experimentation starts |
| Training jobs | Managed compute for model training | Reproducibility and scale |
| Pipelines | Orchestrates multi-step ML workflows | CI/CD for ML |
| Model registry | Stores and versions approved models | Governance and rollback |
| Online endpoints | Real-time prediction serving | Low-latency production use |
| Batch inference | Large-scale asynchronous scoring | Offline scoring patterns |
| Feature store | Reusable online and offline features | Training-serving consistency |
| Monitoring | Drift, skew, performance, data quality | Reliability after launch |
| Security controls | IAM, network isolation, secrets, audit | Enterprise readiness |

The interview skill is not memorizing product names. It is being able to say:

1. where data enters
2. how training is triggered
3. where artifacts are stored
4. how models are promoted
5. how inference runs
6. how failures and regressions are detected

---

## Platform selection framework

When asked "Which platform would you choose?", structure the answer around constraints:

### 1. Existing cloud footprint

- Choose SageMaker if the company is already heavily invested in AWS data and infra services.
- Choose Vertex AI if BigQuery, GCS, and GCP analytics are already the core stack.
- Choose Azure ML if the org is Microsoft-centric with Azure security, AD, and Azure OpenAI adoption.

### 2. Team maturity

- Smaller teams benefit more from managed training, deployment, and registry workflows.
- Advanced platform teams may prefer more custom control, but still use managed primitives selectively.

### 3. Workload type

- Tabular AutoML-heavy teams may prefer Vertex AI or Azure Automated ML.
- Broad enterprise ML with many service integrations often fits SageMaker well.
- GenAI teams using Azure OpenAI may prefer Azure ML plus Azure-native governance.

### 4. Governance and compliance

- Regulated environments care about network isolation, audit trails, approval gates, and identity integration.
- The "best" platform is often the one that best aligns with the organization's security model.

### 5. Cost model

- Managed services reduce engineering cost but can increase infrastructure cost.
- The right comparison is total cost of ownership, not raw GPU hourly price alone.

---

## Common interview themes

You should be ready to answer:

- How would you design a training-to-deployment pipeline on a managed cloud platform?
- How do you prevent training-serving skew?
- When would you use real-time endpoints vs batch prediction?
- How do you implement canary rollout for a new model version?
- How would you monitor data drift and model degradation?
- How do you control cost for GPU training workloads?
- How do you keep secrets, PII, and regulated data isolated?

Strong answers usually include:

- explicit storage layers
- model versioning
- approval workflow
- observability
- rollback strategy
- cost controls

---

## Learning path

Recommended order:

1. Read the comparison guide first.
2. Read the provider-specific guide for the cloud you use most.
3. Compare how each platform handles training, pipelines, and endpoints.
4. Practice answering system design questions using one concrete platform.

---

## Track links

- [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md)
- [AWS SageMaker Interview Guide](./intro_sagemaker.md)
- [Google Vertex AI Interview Guide](./intro_vertex_ai.md)
- [Azure Machine Learning Interview Guide](./intro_azure_ml.md)
