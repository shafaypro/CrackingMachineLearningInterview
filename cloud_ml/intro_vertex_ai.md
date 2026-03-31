# Google Vertex AI Interview Guide

Vertex AI is Google's unified machine learning platform. It is especially strong in organizations that already rely on BigQuery, GCS, and GCP's analytics stack, and it appears frequently in interviews about pipelines, managed training, GenAI tooling, and platform selection.

---

## Table of Contents

1. [Where Vertex AI fits best](#where-vertex-ai-fits-best)
2. [Key platform components](#key-platform-components)
3. [Reference architecture](#reference-architecture)
4. [Pipeline example](#pipeline-example)
5. [Serving and rollout patterns](#serving-and-rollout-patterns)
6. [Monitoring and operations](#monitoring-and-operations)
7. [Cost and scaling](#cost-and-scaling)
8. [Interview Q and A](#interview-q-and-a)
9. [Common mistakes](#common-mistakes)
10. [Related topics](#related-topics)

---

## Where Vertex AI fits best

Vertex AI is a strong choice when:

- data already lives in BigQuery and GCS
- teams want managed pipelines with good GCP integration
- AutoML or Google-managed foundation model workflows matter
- the organization values a tight connection to the broader GCP data platform

It is often the cleanest choice for companies that already use:

- BigQuery for analytics
- Dataflow for pipelines
- GKE for surrounding platform services
- Pub/Sub for event-driven systems

---

## Key platform components

| Component | Purpose | Interview framing |
|---|---|---|
| Vertex AI Workbench | Managed notebook environment | Exploration and development |
| Custom Training | Managed training with custom code | Scalable training jobs |
| Vertex AI Pipelines | Kubeflow-based workflow orchestration | Reproducible ML DAGs |
| Model Registry | Central model version tracking | Governance and promotion |
| Endpoints | Online serving with traffic splits | Real-time inference |
| Batch Prediction | Offline scoring | Scheduled inference jobs |
| Feature Store | Managed feature management | Reuse and consistency |
| Vertex AI Experiments | Run tracking | Operational visibility |
| Model Garden / Studio | GenAI models and prompt tooling | Modern applied AI workflows |

---

## Reference architecture

```text
Event or batch data -> BigQuery / GCS
                   -> data validation and feature prep
                   -> Vertex AI Pipeline
                   -> custom training
                   -> model evaluation
                   -> model registry
                   -> endpoint deployment with traffic split
                   -> monitoring, logging, alerts
```

Good interview answers call out:

- data plane: BigQuery, GCS, feature store
- control plane: pipelines, registry, deployment policies
- observability: logs, metrics, drift alerts
- rollback: previous model version and traffic split reversal

---

## Pipeline example

```python
from kfp import dsl
from kfp.v2 import compiler

@dsl.component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_model(data_uri: str, model_dir: str):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(data_uri)
    X = df.drop("label", axis=1)
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    model.fit(X, y)
    joblib.dump(model, f"{model_dir}/model.joblib")

@dsl.pipeline(name="fraud-vertex-pipeline")
def fraud_pipeline(
    data_uri: str = "gs://ml-platform/fraud/train.csv",
    model_dir: str = "gs://ml-platform/models/fraud/latest",
):
    train_model(data_uri=data_uri, model_dir=model_dir)

compiler.Compiler().compile(
    pipeline_func=fraud_pipeline,
    package_path="fraud_pipeline.json",
)
```

What to say in an interview:

- pipeline definitions should be versioned in source control
- each run should capture parameters, metrics, and artifact paths
- promotion should depend on evaluation and approval gates, not just training success

---

## Serving and rollout patterns

### Online endpoint

Use when:

- low latency is needed
- you need live application integration
- traffic is interactive or user-facing

### Traffic splitting

Vertex AI endpoints support traffic splits between model versions. This is useful for:

- canary rollouts
- A/B testing
- safe migration to a new model

### Batch prediction

Use when:

- inference does not need immediate responses
- data is already in GCS or BigQuery
- throughput matters more than request latency

### GenAI serving

In modern interviews, Vertex AI may also come up in the context of:

- managed LLM access
- prompt evaluation workflows
- retrieval and grounding patterns around Gemini or Model Garden assets

---

## Monitoring and operations

A complete answer should mention:

- endpoint latency and error monitoring
- request payload validation
- prediction distribution monitoring
- data drift checks
- business KPI degradation checks

Operational patterns:

- separate dev, staging, and prod projects or environments
- use service accounts with narrow permissions
- keep training artifacts and deployment metadata traceable
- alert on both system metrics and business regressions

---

## Cost and scaling

Ways to control spend on Vertex AI:

- use autoscaling for endpoints
- prefer batch prediction where latency is not strict
- use preemptible resources when jobs tolerate interruption
- remove idle workbench instances
- keep feature computation close to data to avoid expensive movement

A strong answer connects cost to architecture decisions. For example, moving nightly scoring from always-on endpoints to batch prediction can dramatically cut serving spend.

---

## Interview Q and A

### Q1. When would you choose Vertex AI over SageMaker?

I would choose Vertex AI when the surrounding data platform is already centered on GCP, especially BigQuery and GCS. That reduces integration overhead and gives a cleaner operational model. If the company is already AWS-native, SageMaker may still be the better organizational fit even if the feature lists look similar.

### Q2. How would you deploy a model safely on Vertex AI?

Register the model, deploy it to an endpoint as a new version, route a small percentage of traffic to it, compare latency and business outcomes, then gradually increase traffic if it performs well. Keep the previous version live for fast rollback.

### Q3. How do Vertex AI Pipelines help in production?

They make training workflows reproducible, parameterized, and observable. Instead of ad hoc notebook runs, each pipeline execution becomes a traceable artifact with explicit stages for data prep, training, evaluation, and deployment.

### Q4. How would you integrate BigQuery with Vertex AI?

I would use BigQuery as the analytical source of truth for curated training data, export or materialize the required dataset into the training workflow, and keep schemas versioned. For production features, I would avoid last-minute transformations that are not shared between offline and online paths.

### Q5. What are the tradeoffs of managed endpoints?

They reduce operational burden and speed up deployment, but you give up some low-level serving control and may pay more than a highly optimized custom stack. For most teams, that tradeoff is worth it because reliability and delivery speed matter more than maximum control.

---

## Common mistakes

- building everything in notebooks instead of pipelines
- failing to version data sources and pipeline parameters
- skipping staged rollout and traffic split testing
- ignoring BigQuery cost and data movement implications
- treating managed monitoring as a replacement for business KPI tracking

---

## Related topics

- [Cloud ML Platforms Comparison](./intro_cloud_ml_platforms.md)
- [AWS SageMaker Interview Guide](./intro_sagemaker.md)
- [Azure Machine Learning Interview Guide](./intro_azure_ml.md)
- [Feature Stores](../mlops/intro_feature_stores.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
