# Data Engineering for AI

This guide covers the data layer that powers training, retrieval, online features, and production AI systems.

---

## Overview

AI systems depend on reliable data pipelines more than model demos usually reveal. Data engineering for AI includes batch and streaming ingestion, dataset and feature preparation, quality monitoring, storage formats that support training and retrieval, and versioning.

It matters because model quality, latency, and reliability are all downstream of data quality.

For senior data engineering interviews, that foundation is not enough by itself. You also need to explain how the platform is architected, how domains are modeled, how keys and history are managed, and how the data layer stays governable as scale and organizational complexity grow.

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

## Interview Q&A

#### What makes a data pipeline "reliable" beyond just running on schedule?

Reliability means downstream consumers can trust the output without checking it. That requires: **idempotency** (re-running a task produces the same result, so retries and backfills are safe), **atomicity** (partial writes are never visible — write to a staging location and swap), **schema contracts** with validation at the boundary, **data quality checks** that fail the pipeline rather than publishing bad data, **freshness SLAs** that are monitored and alerted on, and **lineage** so you can answer "what does this table depend on and who breaks if it's wrong".

The distinction worth drawing in an interview: a pipeline that succeeds while producing wrong data is worse than one that fails, because failure is visible and silent corruption is not. That's why quality checks belong in the pipeline as blocking gates, not in a dashboard someone might look at.

#### How do you handle late-arriving and out-of-order data?

Separate **event time** from **processing time** and key everything on event time. Then define a **watermark** — how late you're willing to wait — and a policy for what arrives after it: drop, route to a side output for reconciliation, or trigger a restatement of the affected window.

For ML specifically, the consequence is that features must be computed **as-of** the event timestamp, not as-of now. A feature pipeline that joins current values onto historical labels leaks the future and is the most common source of "great offline, useless in production" models. Point-in-time correct joins are the fix, and they're what feature stores exist to provide.

#### Batch or streaming — how do you decide?

From the decision latency the business actually needs, not from what's fashionable. If a prediction made on yesterday's data is just as valuable, batch is dramatically cheaper to build, test, backfill, and operate. Streaming earns its complexity when freshness changes the outcome — fraud, real-time personalization, alerting, dynamic pricing.

The honest answer includes the hybrid: most production systems batch-compute the expensive historical features and stream only the handful that genuinely need to be fresh, serving both through one interface. And it acknowledges the cost: streaming systems need exactly-once or idempotent semantics, watermarking, state management, and a replay story — none of which a nightly job needs.

#### How do you version data for reproducible ML?

Three layers, and interviewers want all three. **Immutable raw data** — append-only, partitioned by ingest date, never mutated. **Versioned table formats** (Delta Lake, Iceberg, Hudi) for the curated layer, which give you time travel, so "train on the data as of March 1" is a query rather than an archaeology project. **Snapshot references in the model artifact** — every training run records the exact table versions, feature definitions, and code commit used.

Tools like DVC or LakeFS handle file-level versioning where table formats don't apply. The test of whether it works: can you reproduce a model trained six months ago, exactly, from its recorded metadata alone?

#### What data quality checks would you put on an ML feature pipeline?

Layered by cost and specificity:
- **Schema**: expected columns, types, and nullability — catches upstream changes immediately.
- **Volume**: row count within an expected range versus the same weekday historically — catches partial loads, which are the sneakiest failure.
- **Freshness**: max timestamp within the SLA.
- **Distribution**: null rate, cardinality, and summary statistics per column compared against a rolling baseline (PSI or a simple threshold) — catches semantic changes that pass schema checks.
- **Referential**: joins produce the expected row count; no unexpected fan-out from a duplicated key.
- **Business rules**: domain invariants — amounts non-negative, timestamps ordered, statuses in a known set.

Blocking versus alerting matters: schema and volume failures should stop the pipeline; distribution shifts should page someone, because they may be legitimate.

#### How do you prevent training/serving skew?

The structural answer is a single feature definition used by both paths, which is what a feature store provides — the same transformation code computes offline training features and online serving features, so they cannot diverge.

Without one: put transformations inside the model artifact (a pipeline object) rather than in notebook preprocessing; log the actual features used at serving time and periodically compare their distribution to the training set; and add an integration test that scores the same entity through both paths and asserts the feature vectors match. That last test is cheap and catches the majority of skew bugs before release.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Non-idempotent pipeline tasks | Retries and backfills duplicate or corrupt data | Deterministic partitioning; overwrite-by-partition, not append |
| Publishing partial writes | Consumers read half-loaded data | Write to staging, then atomic swap or table-format commit |
| Joining current feature values onto historical labels | Label leakage; offline scores are unreachable in production | Point-in-time correct joins on event time |
| Quality checks as dashboards, not gates | Bad data reaches consumers and nobody notices | Blocking assertions in the pipeline |
| Row-count checks against a fixed threshold | Weekly seasonality causes constant false alarms | Compare against the same weekday historically |
| Mutating raw data in place | Reproducibility is gone permanently | Append-only raw layer; transform into a curated layer |
| Streaming chosen by default | Large complexity cost for latency nobody needed | Start batch; stream only the features that need freshness |
| Separate transformation code for training and serving | Guaranteed skew over time | One shared definition, or a feature store |
| No lineage | Impact of a schema change is unknowable | Lineage tracking (dbt, OpenLineage, catalog integration) |
| Ignoring late data entirely | Silent undercounting in every window | Explicit watermark and a late-arrival policy |

---

## Related Topics

- [Data Modeling](./data-modeling.md)
- [Data Architecture](./data-architecture.md)
- [Apache Airflow](./intro_apache_airflow.md)
- [Apache Spark](./intro_apache_spark.md)
- [Apache Kafka](./intro_apache_kafka.md)
- [dbt](./intro_dbt.md)
- [Feature Stores](../mlops/intro_feature_stores.md)
