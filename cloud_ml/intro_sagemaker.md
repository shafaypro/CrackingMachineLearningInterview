# AWS SageMaker Interview Guide

AWS SageMaker is one of the broadest managed ML platforms used in production. In interviews, it often appears in questions about end-to-end ML lifecycle design, real-time serving, training pipelines, and enterprise governance.

---

## Table of Contents

1. [What SageMaker is good at](#what-sagemaker-is-good-at)
2. [Core services you should know](#core-services-you-should-know)
3. [Typical production architecture](#typical-production-architecture)
4. [Training workflow example](#training-workflow-example)
5. [Deployment patterns](#deployment-patterns)
6. [Monitoring and governance](#monitoring-and-governance)
7. [Cost optimization](#cost-optimization)
8. [Interview Q and A](#interview-q-and-a)
9. [Common mistakes](#common-mistakes)
10. [Related topics](#related-topics)

---

## What SageMaker is good at

SageMaker is strong when you need:

- a managed path from experimentation to deployment
- broad integration with AWS services like S3, IAM, CloudWatch, ECR, Lambda, and EventBridge
- support for custom containers and custom training scripts
- registry, monitoring, and pipeline support inside one ecosystem

It is especially common in enterprise environments where the rest of the platform already runs on AWS.

---

## Core services you should know

| Service | Purpose | Interview shorthand |
|---|---|---|
| SageMaker Studio | Browser-based ML workspace | Notebook and experiment environment |
| Training Jobs | Managed model training | Scheduled or pipeline-driven training |
| Processing Jobs | Data prep and feature processing | Pre-training ETL |
| Pipelines | Multi-step ML workflow orchestration | CI/CD for ML |
| Model Registry | Versioning and approval state | Promotion and rollback control |
| Endpoints | Real-time model serving | Online inference |
| Batch Transform | Offline batch scoring | Large async inference jobs |
| Model Monitor | Drift, bias, and data quality monitoring | Post-deploy guardrails |
| Feature Store | Shared offline and online features | Training-serving consistency |
| JumpStart | Prebuilt models and templates | Faster bootstrap for foundation models |

---

## Typical production architecture

```text
Raw data -> S3 data lake -> Processing job -> Feature store / curated S3
        -> Training job -> Model artifacts in S3
        -> Model registry approval
        -> Deployment to endpoint or batch transform
        -> CloudWatch metrics + Model Monitor + alerts
```

A strong answer should mention:

- S3 for durable artifact and dataset storage
- IAM roles for least-privilege execution
- ECR when custom containers are needed
- CloudWatch for metrics and alerting
- approval gates before production deployment

---

## Training workflow example

```python
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 8,
    },
    use_spot_instances=True,
    max_run=3600,
    max_wait=5400,
)

step_train = TrainingStep(
    name="TrainClassifier",
    estimator=estimator,
    inputs={"train": "s3://ml-bucket/train/"},
)

pipeline = Pipeline(
    name="customer-churn-pipeline",
    steps=[step_train],
)
pipeline.upsert(role_arn=role)
```

What matters in interviews is not the exact SDK call. It is understanding:

- where the training data lives
- how the job is versioned and parameterized
- how artifacts are saved
- how retraining is triggered

---

## Deployment patterns

### Real-time endpoint

Use when:

- latency matters
- requests are interactive
- traffic is steady enough to justify warm instances

Examples:

- fraud scoring
- recommendation ranking
- chat moderation

### Asynchronous inference

Use when:

- payloads are large
- latency requirements are relaxed
- inference can take seconds or minutes

Examples:

- document processing
- batch image classification
- long-running generative tasks

### Batch Transform

Use when:

- you need to score many records at once
- output is written to storage instead of returned immediately
- jobs fit a scheduled or offline workflow

Examples:

- nightly customer churn scoring
- weekly risk refresh
- campaign audience scoring

### Multi-model endpoint

Use when:

- you have many smaller models
- traffic per model is low
- you want to consolidate hosting cost

Tradeoff:

- lower cost, but possible cold start and noisy-neighbor effects

---

## Monitoring and governance

In a mature SageMaker deployment, monitoring is not optional.

You should track:

- endpoint latency and error rate
- request and response schema drift
- feature distribution drift
- prediction distribution shifts
- business metrics like conversion, fraud capture, or escalation rate

Governance patterns:

- register every promoted model version
- require approval before production deployment
- store lineage from dataset to model artifact
- tag jobs and endpoints by owner, environment, and cost center
- isolate production in private subnets when needed

---

## Cost optimization

Common SageMaker cost controls:

- use spot training for fault-tolerant workloads
- shut down idle notebook resources
- right-size instance families for CPU vs GPU jobs
- prefer batch inference if always-on endpoints are unnecessary
- autoscale endpoint capacity based on real traffic patterns
- archive old artifacts and unused model versions

Interview tip:

Do not say "use the cheapest instance." Cost optimization means matching workload shape to compute shape without violating latency or reliability requirements.

---

## Interview Q and A

### Q1. When would you choose SageMaker over a self-managed Kubernetes platform?

Use SageMaker when speed, managed operations, and AWS integration matter more than full platform control. It reduces platform engineering overhead for training orchestration, hosting, registry, and monitoring. I would prefer self-managed infrastructure only if the team has strong platform maturity and specific requirements that SageMaker cannot satisfy cleanly.

### Q2. How would you build a retraining pipeline in SageMaker?

I would land data in S3, run a processing job for validation and feature generation, trigger a training job, evaluate against a baseline, register the model, require approval if metrics pass, then deploy gradually to a staging or production endpoint. The pipeline should be parameterized, versioned, observable, and reproducible.

### Q3. How do you avoid training-serving skew on AWS?

Use a shared feature definition and a consistent feature computation path. In practice that means storing canonical features in a feature store or shared transformation code, versioning schemas, validating inference payloads, and monitoring offline-vs-online feature drift after deployment.

### Q4. How would you deploy a new model with low risk?

Deploy to a separate endpoint or variant, route a small percentage of traffic, compare latency and business metrics, and keep rollback immediate. I would not replace the old model in place without staged validation.

### Q5. What is the difference between Batch Transform and an endpoint?

Batch Transform is for offline scoring over large datasets with outputs written to storage. Endpoints are for low-latency online inference where predictions are returned immediately to an application.

---

## Common mistakes

- treating notebooks as production pipelines
- skipping model approval and lineage tracking
- using real-time endpoints for workloads that should be batch jobs
- ignoring cost impact of idle endpoints and GPU overprovisioning
- assuming cloud-managed means observability is automatic

---

## Related topics

- [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md)
- [Google Vertex AI Interview Guide](./intro_vertex_ai.md)
- [Azure Machine Learning Interview Guide](./intro_azure_ml.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [MLflow](../mlops/intro_mlflow.md)
