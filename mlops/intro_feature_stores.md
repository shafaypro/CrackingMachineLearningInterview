# Feature Stores

A feature store is a centralized repository for storing, sharing, and serving ML features. It bridges the gap between data engineering and model training/serving, solving the consistency problem between offline training and online inference.

---

## Table of Contents
1. [What is a Feature Store?](#what-is-a-feature-store)
2. [Offline vs Online Store](#offline-vs-online-store)
3. [Point-in-Time Correctness](#point-in-time-correctness)
4. [Feast (Open Source)](#feast-open-source)
5. [Other Feature Store Solutions](#other-feature-store-solutions)
6. [When to Use a Feature Store](#when-to-use-a-feature-store)
7. [Interview Q&A](#interview-qa)
8. [Common Pitfalls](#common-pitfalls)
9. [Related Topics](#related-topics)

---

## What is a Feature Store?

Without a feature store, teams face two major problems:

1. **Training-serving skew**: Features computed differently in the pipeline that generates training data vs. the one serving predictions
2. **Feature duplication**: Team A and Team B both compute "user's average purchase in last 30 days" independently, with different logic

```
WITHOUT Feature Store:
  Data Science Team  →  Custom ETL → Training features (Python/pandas)
  Engineering Team   →  Custom SQL → Serving features (Java/SQL)
  → Different logic = training-serving skew = silent model degradation

WITH Feature Store:
  One feature definition → Registry → Offline store (training)
                                    → Online store (serving, <10ms)
  → Same logic guaranteed = consistent features everywhere
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Feature Registry** | Metadata store: defines features, their computation logic, owners, schemas |
| **Offline Store** | Historical feature data for model training (data warehouse, S3, BigQuery) |
| **Online Store** | Low-latency feature serving for inference (Redis, DynamoDB, Bigtable) |
| **Materialization Job** | Computes and populates offline/online store from raw data |
| **Feature Server** | API to retrieve features at inference time |

---

## Offline vs Online Store

| | Offline Store | Online Store |
|-|--------------|-------------|
| **Purpose** | Training data generation | Real-time inference serving |
| **Latency** | Hours to days | < 10ms |
| **Scale** | Petabytes (historical data) | Millions of real-time lookups/sec |
| **Storage** | Parquet, BigQuery, Redshift, S3 | Redis, DynamoDB, Bigtable, Cassandra |
| **Access pattern** | Batch reads (point-in-time joins) | Single-key lookups by entity ID |
| **Freshness** | Minutes to hours lag acceptable | Must be near real-time |

```
                Raw Data (data warehouse / streaming)
                            │
               ┌────────────┼────────────┐
               │                         │
        Offline Store              Online Store
        (Parquet / S3)             (Redis / Dynamo)
               │                         │
               │                         │
        Training jobs            Prediction service
        (batch, hours)           (real-time, <10ms)
```

---

## Point-in-Time Correctness

Point-in-time correctness ensures that when creating training data, only features that were available **before the label timestamp** are used. This prevents future leakage.

```python
# Example: predicting customer churn
# Event: customer churned on 2024-06-15
# We should only use features available before 2024-06-15

# WRONG: join on user_id only — gets current feature values (includes future data)
training_df = events.merge(features, on='user_id')

# CORRECT: point-in-time join — gets feature values as of event timestamp
# (each row gets the feature value that was current at the event_timestamp)
training_df = feature_store.get_historical_features(
    entity_df=events[['user_id', 'event_timestamp']],
    feature_refs=['user_features:total_purchases_30d', 'user_features:avg_order_value']
)
```

---

## Feast (Open Source)

Feast (Feature Store) is the most popular open-source feature store. It supports multiple offline/online backends.

### Installation and Setup

```bash
pip install feast
feast init my_feature_repo
cd my_feature_repo
```

### Define Feature Views

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define entity (the key)
user = Entity(name="user_id", description="User identifier")

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=90),  # How long to keep features in online store
    schema=[
        Field(name="total_purchases_30d", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="customer_lifetime_value", dtype=Float32),
    ],
    online=True,
    source=user_stats_source,
)
```

### Apply and Materialize

```bash
# Register feature definitions
feast apply

# Populate offline store from source data
# Populate online store (last N days)
feast materialize 2024-01-01T00:00:00 2024-12-31T00:00:00
```

### Retrieving Features

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path=".")

# --- Training: get historical features (point-in-time) ---
entity_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "event_timestamp": [
        datetime(2024, 6, 1),
        datetime(2024, 7, 15),
        datetime(2024, 8, 20),
    ]
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_stats:total_purchases_30d",
        "user_stats:avg_order_value",
        "user_stats:days_since_last_purchase",
    ],
).to_df()

print(training_df)

# --- Inference: get online features (real-time) ---
feature_vector = store.get_online_features(
    features=["user_stats:total_purchases_30d", "user_stats:avg_order_value"],
    entity_rows=[{"user_id": 1001}]
).to_dict()

print(feature_vector)
# {'user_id': [1001], 'total_purchases_30d': [15], 'avg_order_value': [45.3]}
```

### Feature Store in a Prediction Service

```python
from feast import FeatureStore
import joblib

store = FeatureStore(repo_path="/app/feature_repo")
model = joblib.load("model.pkl")

def predict(user_id: int) -> float:
    # Fetch features from online store (<10ms)
    features = store.get_online_features(
        features=["user_stats:total_purchases_30d", "user_stats:avg_order_value"],
        entity_rows=[{"user_id": user_id}]
    ).to_dict()

    feature_vector = [[
        features["total_purchases_30d"][0],
        features["avg_order_value"][0],
    ]]

    return model.predict(feature_vector)[0]
```

---

## Other Feature Store Solutions

| Solution | Type | Best For |
|----------|------|---------|
| **Feast** | Open source | Full control, cloud-agnostic |
| **Tecton** | Managed SaaS | Enterprise, production-grade |
| **Hopsworks** | Open source / Managed | Full MLOps platform |
| **Databricks Feature Store** | Managed | Databricks ecosystem |
| **AWS SageMaker Feature Store** | Managed | AWS ecosystem |
| **Google Vertex AI Feature Store** | Managed | GCP ecosystem |
| **Azure ML Feature Store** | Managed | Azure ecosystem |

---

## When to Use a Feature Store

### Use a Feature Store When:
- Multiple teams share features (reuse > 2 models)
- Training-serving skew is causing production incidents
- You have real-time inference requirements (< 50ms)
- You need point-in-time correct historical features
- Feature computation is expensive and needs caching

### Don't Need One When:
- Single model, single team
- Batch predictions only (no real-time serving)
- Simple features that are easy to compute inline
- Early-stage project (add it when pain is clear)

---

## Interview Q&A

**Q1: What problem does a feature store solve?**
Two main problems: (1) **Training-serving skew** — without a feature store, features computed in Python/pandas for training and in Java/SQL for serving often diverge, causing silent model degradation in production. (2) **Feature duplication** — multiple teams compute the same feature independently with different logic, wasting engineering effort and causing inconsistency.

**Q2: What is point-in-time correctness and why does it matter?**
When creating training data, we must use only feature values that were available *before* the label event. A naive join on entity ID would give the current (latest) feature value, which may include data from after the event — future leakage. Point-in-time joins fetch the feature value that was current at the exact event timestamp, preventing this leakage.

**Q3: What is the difference between online and offline feature stores?**
The offline store holds historical feature values for model training — it's optimized for batch reads (Parquet, BigQuery) and supports point-in-time queries. The online store holds the latest feature values for real-time inference — it's optimized for low-latency single-key lookups (Redis, DynamoDB, < 10ms). Materialization jobs sync data from offline to online.

**Q4: How does feature freshness work in a feature store?**
TTL (Time To Live) controls how long features stay in the online store. Materialization frequency determines how current the online store is. For real-time features (e.g., "user's last click"), you'd use streaming pipelines (Kafka → Redis) that update the online store continuously. For slower-changing features (e.g., "30-day purchase count"), batch materialization every few hours is sufficient.

**Q5: How would you implement a feature store for a fraud detection system?**
Fraud detection needs real-time features: transaction velocity (# transactions in last 1h), aggregated amounts, device fingerprint. Architecture: (1) Kafka for streaming transaction events; (2) Flink/Spark Streaming computes features in near real-time; (3) writes to Redis for online serving; (4) writes to S3/BigQuery for offline training. A feature store like Tecton or Feast manages the registry and both stores.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| No point-in-time joins | Future data leaks into training | Use feature store's historical retrieval |
| Materializing too infrequently | Stale features in online store | Match materialization frequency to feature TTL |
| No feature versioning | Breaking changes silently affect models | Version feature views; test before updating |
| Computing features in prediction service | Slow, inconsistent with training | Pre-compute and store; retrieve at inference |
| Over-engineering early | Adds complexity before value is clear | Add a feature store when you have >2 models sharing features |
| Not monitoring feature drift | Silent degradation | Add feature distribution monitoring |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [MLflow](./intro_mlflow.md) | Track which feature set version was used per experiment |
| [Model Serving](./intro_model_serving.md) | Feature store integrates with the serving layer |
| [Data Quality](./intro_data_quality.md) | Validate feature quality before materialization |
| [Feature Engineering](../classical_ml/intro_feature_engineering.md) | Feature engineering defines what goes in the store |
| [Study Pattern](../docs/study-pattern.md) | Feature Stores is an Advanced (🔴) MLOps topic |
