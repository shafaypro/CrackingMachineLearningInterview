# Feature Store Guide

A comprehensive guide to feature stores — centralized repositories for storing, versioning, and serving ML features.

---

## Table of Contents

1. [What is a Feature Store](#what-is-a-feature-store)
2. [Why Use a Feature Store](#why-use-a-feature-store)
3. [Online vs Offline Store](#online-vs-offline-store)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Tool Comparison](#tool-comparison)
6. [Feast Example](#feast-example)
7. [Interview Q&A](#interview-qa)
8. [References](#references)

---

## What is a Feature Store

A feature store is a data system that manages the creation, storage, and serving of features used by ML models. It acts as a bridge between raw data (in data warehouses or data lakes) and ML models (in training and production serving).

```
Raw Data Sources              Feature Store                   ML Models
─────────────────    →    ────────────────────    →    ──────────────────
Data Warehouse            Feature Definitions            Training Pipeline
Kafka Streams             Offline Store (S3)             Online Serving
Databases                 Online Store (Redis)           Batch Predictions
Event Logs                Feature Registry
```

---

## Why Use a Feature Store

### Problem 1: Training-Serving Skew

Without a feature store, data scientists compute features in notebooks (Python/SQL) and engineers re-implement them in production code (Java, C++). Differences in implementation lead to **training-serving skew** — the model sees different data at serving time than it was trained on.

**Feature store solution:** Define features once, use them for both training and serving.

### Problem 2: Feature Duplication

Different teams compute the same features independently — wasted compute, inconsistent results, no shared governance.

**Feature store solution:** Centralized feature registry with reusable, discoverable features.

### Problem 3: Data Leakage

When creating training datasets, using future data that wouldn't be available at prediction time leads to overly optimistic models that fail in production.

**Feature store solution:** Point-in-time correct joins — retrieve feature values as they existed at the time of each training example.

### Problem 4: Feature Freshness

Models need fresh features at serving time. Manually managing caching and refresh logic is error-prone.

**Feature store solution:** Automated feature pipelines with configurable freshness requirements.

---

## Online vs Offline Store

| Aspect | Online Store | Offline Store |
|--------|-------------|---------------|
| **Purpose** | Real-time feature retrieval for inference | Historical feature retrieval for training |
| **Latency** | Low latency (< 10ms) | High throughput, high latency OK |
| **Storage** | Redis, DynamoDB, Bigtable | S3, GCS, BigQuery, Parquet files |
| **Data size** | Current/recent feature values | Full historical feature values |
| **Access pattern** | Key-value lookup by entity ID | Bulk reads, time-range queries |
| **Use case** | Real-time model serving | Model training, backtesting |

### Data Flow

```
                    Batch Feature Pipeline
Raw Data → Compute Features → Write to Offline Store (S3)
                                        ↓
                              Materialize to Online Store (Redis)
                                        ↓
                           Production API → Read from Online Store
                                              (low latency, < 10ms)

                    Training
Training Pipeline → Read from Offline Store (point-in-time correct)
                  → Train model
```

---

## Feature Engineering Pipeline

```python
# Example: Feature pipeline architecture

# 1. Raw data ingestion (from data warehouse)
transactions_df = spark.read.parquet("s3://data-lake/transactions/")

# 2. Feature computation
def compute_user_features(df):
    """Compute user-level aggregated features."""
    return df.groupBy("user_id").agg(
        F.count("transaction_id").alias("transaction_count_30d"),
        F.sum("amount").alias("total_spend_30d"),
        F.avg("amount").alias("avg_transaction_amount_30d"),
        F.countDistinct("merchant_id").alias("unique_merchants_30d"),
        F.max("timestamp").alias("last_transaction_time")
    )

# 3. Write to offline store
user_features = compute_user_features(transactions_df)
user_features.write.parquet("s3://feature-store/user_features/")

# 4. Register features in feature registry
feature_definitions = {
    "user_features": {
        "entity": "user_id",
        "features": [
            "transaction_count_30d",
            "total_spend_30d",
            "avg_transaction_amount_30d"
        ],
        "freshness": "24h",
        "owner": "ml-team@company.com"
    }
}
```

---

## Tool Comparison

| Tool | Type | Online Store | Offline Store | Strengths | Weaknesses |
|------|------|-------------|---------------|-----------|------------|
| **Feast** | Open source | Redis, DynamoDB, Bigtable | BigQuery, Redshift, S3 | Flexible, community | Manual setup, limited governance |
| **Tecton** | Managed SaaS | Redis | S3, Snowflake | Enterprise-grade, real-time | Expensive, vendor lock-in |
| **Hopsworks** | Open source / managed | Redis, RonDB | Hudi on S3 | Full platform, feature monitoring | Complex deployment |
| **Vertex AI Feature Store** | GCP managed | Bigtable | BigQuery | Tight GCP integration | GCP-only |
| **AWS SageMaker Feature Store** | AWS managed | DynamoDB | S3 | AWS integration | AWS-only |
| **Databricks Feature Store** | Databricks managed | Delta tables | Delta Lake | Unity Catalog integration | Databricks-only |

---

## Feast Example

### Installation

```bash
pip install feast
```

### Define Features

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define entities (primary keys)
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created"
)

# Define feature view
user_stats_view = FeatureView(
    name="user_stats",
    entities=["user_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="transaction_count_30d", dtype=ValueType.INT64),
        Feature(name="total_spend_30d", dtype=ValueType.FLOAT),
        Feature(name="avg_transaction_amount_30d", dtype=ValueType.FLOAT),
        Feature(name="unique_merchants_30d", dtype=ValueType.INT64),
    ],
    source=user_stats_source,
)
```

### Initialize and Apply

```bash
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### Retrieve Features for Training

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="feature_repo")

# Training data with entity IDs and timestamps
training_df = pd.DataFrame({
    "user_id": [101, 102, 103, 104],
    "event_timestamp": [
        datetime(2024, 1, 15),
        datetime(2024, 1, 16),
        datetime(2024, 1, 17),
        datetime(2024, 1, 18),
    ],
    "label": [0, 1, 0, 1]  # fraud label
})

# Point-in-time correct feature retrieval
training_features = store.get_historical_features(
    entity_df=training_df,
    features=[
        "user_stats:transaction_count_30d",
        "user_stats:total_spend_30d",
        "user_stats:avg_transaction_amount_30d",
    ]
).to_df()

print(training_features.head())
```

### Online Feature Retrieval (Serving)

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Retrieve current features for a specific user
features = store.get_online_features(
    features=[
        "user_stats:transaction_count_30d",
        "user_stats:total_spend_30d",
        "user_stats:avg_transaction_amount_30d",
    ],
    entity_rows=[{"user_id": 101}]
).to_dict()

print(features)
# {'user_id': [101], 'transaction_count_30d': [42], 'total_spend_30d': [1250.50], ...}
```

---

## Interview Q&A

**Q1: What is a feature store and why do ML teams use it?** 🟢 Beginner

A feature store is a centralized system for storing and serving ML features. Teams use it to: (1) eliminate training-serving skew by defining features once for both training and production, (2) enable feature reuse across teams and models, (3) provide point-in-time correct feature retrieval to prevent data leakage, and (4) manage feature freshness and monitoring.

---

**Q2: What is the difference between the online store and offline store in a feature store?** 🟢 Beginner

The **online store** is a low-latency key-value store (Redis, DynamoDB) that serves the most recent feature values for real-time model inference. The **offline store** is a scalable storage system (S3, BigQuery) that holds the full historical feature history for model training. Data flows from offline to online via a materialization process.

---

**Q3: What is training-serving skew and how does a feature store prevent it?** 🟡 Intermediate

Training-serving skew occurs when features are computed differently during training vs production serving. Example: data scientists compute a 30-day rolling average in Python during training, but engineers implement it differently in Java for production. The model sees different feature values and performs worse.

A feature store prevents this by providing a single feature definition that is used for both training (reading from the offline store) and serving (reading from the online store). The same transformation logic runs in both contexts.

---

**Q4: What is point-in-time correctness and why is it important?** 🔴 Advanced

Point-in-time correctness means that when creating a training dataset, each training example only uses feature values that were available at the time of that event — not future data.

Without it, data leakage occurs. Example: if you train a fraud model and join the user's features as of today (which include transactions that happened after the fraud event), the model learns from future information and appears much more accurate than it will be in production.

Feast and other feature stores provide point-in-time correct joins by looking up the feature values that existed at the event's timestamp.

---

**Q5: How would you decide between building a custom feature pipeline vs using a feature store?** 🔴 Advanced

**Use a custom pipeline when:**
- Small team, few models, simple feature requirements
- Strong existing data infrastructure (dbt, Airflow) that already handles features
- Budget constraints — feature stores add operational overhead

**Use a feature store when:**
- Multiple models share features across teams
- Real-time serving requires low-latency feature lookup
- Training-serving skew is causing production issues
- Need point-in-time correct feature retrieval for training data
- Feature governance and discovery are important at scale

---

## References

- [Feast Documentation](https://docs.feast.dev/)
- [Tecton Feature Store Guide](https://www.tecton.ai/blog/what-is-a-feature-store/)
- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [Feature Store Comparison — featurestore.org](https://www.featurestore.org/)
- [Chip Huyen — Real-time Machine Learning Inference (Feature Stores)](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)
