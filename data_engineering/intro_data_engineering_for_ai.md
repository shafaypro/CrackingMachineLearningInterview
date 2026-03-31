# Data Engineering for AI

This guide covers the data layer that powers training, retrieval, online features, and production AI systems.

---

## Overview

AI systems depend on reliable data pipelines more than model demos usually reveal. Data engineering for AI includes batch and streaming ingestion, dataset and feature preparation, quality monitoring, storage formats that support training and retrieval, and versioning.

It matters because model quality, latency, and reliability are all downstream of data quality.

---

## Core Concepts

### Data pipelines

A data pipeline moves data from raw sources into training sets, feature stores, or retrieval corpora. In practice, pipelines must be observable, repeatable, and resilient to partial failures.

### Streaming data

Streaming systems matter when data freshness affects product quality, such as fraud scoring, recommender events, or live operational dashboards.

### Data versioning

If you cannot reproduce the data a model was trained on, you cannot explain regressions with confidence. Versioning applies to datasets, schemas, and transformation logic.

### Data quality monitoring

Teams should detect missing fields, schema changes, distribution drift, duplicated records, and late or out-of-order events.

---

## Key Skills

### Building reliable pipelines

In practice, this means idempotent jobs, retry-safe processing, observability and alerts, and lineage between source data and downstream outputs.

### Handling real-time data

A strong engineer understands event ordering, consumer lag, throughput bottlenecks, and backpressure.

### Data quality monitoring

This includes defining expectations, validating schemas, and alerting on freshness or distribution shifts before models silently degrade.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| Airflow | Workflow orchestration for scheduled pipelines | Batch ETL and recurring ML prep jobs |
| Spark | Distributed data processing engine | Large-scale batch transformations |
| Kafka | Event streaming platform | Real-time ingestion and event pipelines |
| dbt | SQL transformation framework | Warehouse-first modeling and testing |
| Delta Lake / Iceberg | Table formats with versioning and reliability features | Lakehouse storage for ML and analytics |

---

## Projects

### Data pipeline for ML

- Goal: Build a pipeline that creates a training dataset from raw source tables.
- Key components: scheduled orchestration, validation checks, partitioned outputs, metadata logging.
- Suggested tech stack: Airflow, Spark, dbt, object storage.
- Difficulty: Intermediate.

### Streaming ingestion system

- Goal: Consume events in real time and materialize clean downstream records.
- Key components: Kafka topics, consumer group, dead-letter queue, schema enforcement.
- Suggested tech stack: Kafka, Python or Spark Structured Streaming, Redis or Postgres.
- Difficulty: Advanced.

### Feature store system

- Goal: Serve consistent features for both training and online inference.
- Key components: offline store, online store, feature definitions, freshness monitoring.
- Suggested tech stack: Feast or custom store, Spark, Redis, warehouse.
- Difficulty: Advanced.

---

## Example Code

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("training-dataset").getOrCreate()

df = spark.read.parquet("s3://ml-data/raw/events/")

clean_df = df.dropna(subset=["user_id", "event_time"]).dropDuplicates()
clean_df.write.mode("overwrite").parquet("s3://ml-data/curated/training/")
```

---

## Suggested Project Structure

```text
feature-pipeline/
├── dags/
├── spark_jobs/
├── models/
├── tests/
├── configs/
└── README.md
```

---

## Related Topics

- [Apache Airflow](./intro_apache_airflow.md)
- [Apache Spark](./intro_apache_spark.md)
- [Apache Kafka](./intro_apache_kafka.md)
- [dbt](./intro_dbt.md)
- [Feature Stores](../mlops/intro_feature_stores.md)
