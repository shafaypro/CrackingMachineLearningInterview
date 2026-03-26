
## 📋 Table of Contents

1. [What is OpenClaw?](#1-what-is-openclaw)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Concepts](#3-core-concepts)
4. [Beginner: Getting Started](#4-beginner-getting-started)
5. [Intermediate: Building Real Pipelines](#5-intermediate-building-real-pipelines)
6. [Advanced: Production Patterns](#6-advanced-production-patterns)
7. [Common Patterns & Best Practices](#7-common-patterns--best-practices)
8. [Learning Path Summary](#8-learning-path-summary)
9. [CLI Command Reference](#9-cli-command-reference)
10. [Glossary](#10-glossary)

---

## 1. What is OpenClaw?

**OpenClaw** is an open-source data orchestration and lineage tracking framework designed to manage complex data pipelines in modern data platforms. It provides a declarative, graph-based approach to defining how data moves, transforms, and gets validated — from ingestion through to serving.

> 💡 **Why the name?** A claw grips and holds data at every stage of its journey. "Open" signals it's fully open-source, community-driven, and vendor-neutral.

Think of OpenClaw as the **connective tissue** of your data platform: it knows where your data comes from, where it goes, how it was transformed, and whether it can be trusted.

### Core Philosophy

| Principle | Description |
|-----------|-------------|
| **Declarative pipelines** | Define *what* you want, not *how* to do it |
| **Lineage-first** | Every transformation is tracked by default |
| **Observability over magic** | Surface what's happening, not just handle it silently |
| **Composable** | Small, well-defined units that can be combined freely |
| **Pluggable** | Integrate with any existing data stack |

### Where OpenClaw Fits in Your Stack

| Layer | OpenClaw's Role |
|-------|----------------|
| **Ingestion** | Tracks source connections and schema at capture time |
| **Storage** | Attaches lineage metadata to datasets in data lakes / warehouses |
| **Transformation** | Wraps dbt, Spark, SQL transforms and records what changed |
| **Orchestration** | Integrates with Airflow, Prefect, Dagster to add lineage context |
| **Serving** | Exposes lineage APIs so downstream consumers understand data origin |
| **Observability** | Feeds anomaly detection and data quality checks automatically |

---

## 2. Architecture Overview

OpenClaw is built around a **Directed Acyclic Graph (DAG)** model of data flow. Every dataset, transformation, and assertion is a node; every dependency is a directed edge. This graph is stored in OpenClaw's metadata store and is queryable in real time.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Source Node │───▶│  Transform   │───▶│  Assertion   │───▶│  Sink Node   │
│  (Dataset)   │    │    Node      │    │    Node      │    │  (Dataset)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Metadata Store  │
                              │  (Lineage Graph) │
                              └─────────────────┘
```

### Core Components

| Component | Description | Analogy |
|-----------|-------------|---------|
| **Claw Engine** | Central processing unit; executes pipeline plans and resolves dependency order | The CPU of OpenClaw |
| **Metadata Store** | Persistent graph database holding lineage, schemas, stats, and run history | The brain / memory |
| **Connector Layer** | Plugins that interface with external sources and targets (S3, Snowflake, Kafka...) | Hands reaching out |
| **Policy Engine** | Enforces data contracts, SLAs, and access rules | The rulebook enforcer |
| **Lineage API** | REST + GraphQL API exposing the full lineage graph to external tools | The public face |
| **CLI / SDK** | Developer interface for authoring, testing, and deploying pipelines | The toolbelt |

### Data Flow Model

Every unit of work in OpenClaw follows this pattern:

1. **Source node** declares what data exists and where
2. **Transform node** defines how data is reshaped
3. **Assertion node** validates the result against contracts
4. **Sink node** writes output and registers lineage
5. **Metadata store** updates the graph and emits events

> 🔑 **Key design insight:** Unlike traditional schedulers that care only about task success/failure, OpenClaw tracks the **data itself** — its shape, volume, freshness, and provenance — at every step.

---

## 3. Core Concepts

### 3.1 Datasets

A **Dataset** in OpenClaw is any named, versioned collection of data — a database table, a file in S3, a Kafka topic, or an in-memory DataFrame. Every dataset has:

- A unique **URN** (Universal Resource Name)
- A versioned **schema**
- **Partition metadata** (if applicable)
- **Freshness expectations** (max staleness)
- **Owner and classification tags**

```
urn:openclaw:snowflake:prod.analytics.orders   # example dataset URN
```

### 3.2 Transforms

A **Transform** is a computation that takes one or more datasets as input and produces one or more datasets as output. Transforms are first-class citizens — versioned, logged, and attached to the lineage graph automatically.

| Transform Type | Use Case |
|---------------|----------|
| **SQL Transform** | Runs a SQL query against a warehouse; captures column-level lineage |
| **Spark Transform** | Wraps a PySpark job; lineage captured via OpenClaw Spark listener |
| **dbt Transform** | Thin wrapper around dbt models; syncs dbt docs to OpenClaw graph |
| **Python Transform** | Arbitrary Python; lineage tracked via dataset decorators |
| **Stream Transform** | Kafka Streams / Flink; near-real-time lineage with offset tracking |

### 3.3 Lineage

Lineage is OpenClaw's superpower. It tracks **three levels of granularity**:

- 📦 **Dataset-level**: `orders table was derived from raw_events`
- 📊 **Column-level**: `revenue column comes from price * quantity`
- 🔬 **Row-level** *(experimental)*: trace individual records across transforms

Lineage is captured **automatically** — you do not need to manually document it. OpenClaw instruments your transforms at runtime.

### 3.4 Data Contracts

Data Contracts are explicit agreements between producers and consumers of a dataset:

| Contract Type | What It Checks |
|--------------|---------------|
| **Schema contract** | Column names, types, nullability |
| **Freshness contract** | Data must arrive within N minutes |
| **Volume contract** | Row count must be within expected range |
| **Quality contract** | Custom SQL-based assertions (e.g., no negative prices) |

> ⚠️ **Important:** Violations **block downstream transforms** by default, preventing bad data from propagating silently.

### 3.5 Partitions and Incremental Processing

OpenClaw is built for incremental processing. When a transform runs, it checks which partitions have changed and processes only those — dramatically reducing compute costs.

### 3.6 Run Context

Every execution is wrapped in a **Run Context** — a rich metadata envelope containing:

- Run ID (globally unique)
- Trigger type (scheduled, manual, event-driven)
- Input dataset snapshots at run time
- Performance metrics (rows processed, duration, resource usage)
- Assertion results and contract check outcomes

---

## 4. Beginner: Getting Started

### 4.1 Installation

OpenClaw requires **Python 3.9+** and a metadata backend.

```bash
pip install openclaw

# Local development (SQLite)
openclaw init --backend sqlite

# Production (PostgreSQL)
openclaw init --backend postgres --url postgresql://user:pass@host:5432/openclaw
```

### 4.2 Your First Pipeline

```python
from openclaw import pipeline, dataset, transform, assert_not_null

# Declare your source dataset
raw_orders = dataset('file://data/raw_orders.csv', format='csv')

# Define a transform
@transform(inputs=[raw_orders])
def clean_orders(df):
    df = df.dropna(subset=['order_id', 'amount'])
    df['amount'] = df['amount'].astype(float)
    return df

# Add quality assertions
clean_orders.assert_not_null(['order_id', 'amount'])

# Write output (lineage captured automatically)
clean_orders.write_to('file://data/clean_orders.parquet')
```

```bash
# Run it
openclaw run my_pipeline.py
```

### 4.3 Understanding the Output

After a run, OpenClaw outputs:

- ✅ Datasets read/written
- 📊 Rows processed
- 🧪 Assertion results
- 🔗 Lineage graph URL (viewable in the web UI)

Everything is logged to the metadata store automatically.

### 4.4 Exploring Lineage via CLI

```bash
# View lineage for a dataset
openclaw lineage show --dataset file://data/clean_orders.parquet

# Traverse upstream 3 hops
openclaw lineage upstream --dataset prod.analytics.orders --depth 3

# See everything downstream of a source
openclaw lineage downstream --dataset prod.raw.events
```

---

## 5. Intermediate: Building Real Pipelines

### 5.1 Connecting to Real Systems

Configure connectors in `openclaw.yaml`:

```yaml
connectors:
  snowflake:
    account: myaccount.us-east-1
    warehouse: COMPUTE_WH
    database: PROD
    auth: env:SNOWFLAKE_TOKEN

  s3:
    region: us-east-1
    bucket: my-data-lake

  kafka:
    brokers:
      - broker1:9092
      - broker2:9092
```

### 5.2 dbt Integration

```bash
# After running dbt, push lineage to OpenClaw
openclaw ingest dbt \
  --manifest target/manifest.json \
  --catalog target/catalog.json
```

This automatically:
- Creates dataset nodes for every dbt model
- Captures column-level SQL lineage
- Attaches dbt test results as contract assertions

### 5.3 Spark Integration

```python
from openclaw.spark import OpenClawListener

spark = SparkSession.builder \
    .appName('MyJob') \
    .config('spark.extraListeners', OpenClawListener.__module__) \
    .getOrCreate()

# All reads/writes are now tracked automatically
df = spark.read.parquet('s3://my-bucket/raw/')
df.write.parquet('s3://my-bucket/clean/')
```

### 5.4 Defining Data Contracts

```python
from openclaw import Dataset, Contract

orders = Dataset('prod.analytics.orders')
orders.add_contract(Contract(
    schema={
        'order_id': 'string',
        'amount':   'float',
        'ts':       'timestamp'
    },
    freshness_hours=2,
    min_rows=1000,
    max_null_fraction={'amount': 0.0},
    custom_assertions=[
        'SELECT COUNT(*) = 0 FROM {{table}} WHERE amount < 0'
    ]
))
```

### 5.5 Incremental Pipelines

```python
@transform(
    inputs=[raw_events],
    partition_by='event_date',
    incremental=True
)
def daily_aggregates(df, partition):
    # Only called for new or changed partitions
    return df.groupby('user_id').agg(...)
```

### 5.6 Airflow Integration

```python
from openclaw.airflow import OpenClawOperator

run_pipeline = OpenClawOperator(
    task_id='run_orders_pipeline',
    pipeline='pipelines/orders.py',
    dag=dag
)
```

> 📌 **Orchestrator vs OpenClaw:** The orchestrator (Airflow/Prefect) handles *when* tasks run. OpenClaw handles *what data was used*, *how it was transformed*, and *whether it's trustworthy*. They are complementary, not competing.

---

## 6. Advanced: Production Patterns

### 6.1 Column-Level Lineage Deep Dive

Column-level lineage (CLL) answers: *"Which source column feeds this target column?"*

OpenClaw captures CLL through:
- **SQL parsing** — static analysis of SELECT, JOIN, and GROUP BY clauses
- **Spark instrumentation** — tracking DataFrame column operations at runtime
- **Manual annotation** — decorators for Python code OpenClaw cannot infer

```python
from openclaw import column_lineage

@column_lineage(
    outputs={'net_revenue': ['gross_revenue', 'discount_amount']},
    logic='gross_revenue - discount_amount'
)
def compute_net_revenue(df):
    df['net_revenue'] = df['gross_revenue'] - df['discount_amount']
    return df
```

### 6.2 Multi-Hop Impact Analysis

When a source schema changes, OpenClaw immediately surfaces every affected downstream asset:

```bash
openclaw impact \
  --dataset prod.raw.events \
  --change 'rename user_id to customer_id'
```

The output is a **ranked list of affected assets with severity scores**, helping teams prioritize remediation.

### 6.3 Data Quality Monitoring

Run quality checks on a schedule — independent of pipeline runs:

```bash
openclaw monitor create \
  --dataset prod.analytics.orders \
  --check freshness    --max-age 4h \
  --check volume       --min 500 --max 100000 \
  --check null-rate    --column amount --threshold 0.01 \
  --alert slack:#data-alerts
```

### 6.4 Data Contracts as Code

Store contracts in version control alongside your pipeline code:

```yaml
# contracts/orders.yaml
dataset: prod.analytics.orders
version: "2.1"
owner: data-engineering@company.com

schema:
  order_id: { type: string,    nullable: false }
  amount:   { type: float,     nullable: false, min: 0 }
  order_ts: { type: timestamp, nullable: false }

freshness: { max_age_hours: 2 }
volume:    { min_rows: 1000, max_rows: 10000000 }
```

### 6.5 Lineage API (GraphQL)

```graphql
query GetLineage {
  dataset(urn: "urn:openclaw:snowflake:prod.analytics.orders") {
    upstreamDatasets   { urn name owner }
    downstreamDatasets { urn name owner }
    columns {
      name
      type
      upstreamColumns { dataset column }
    }
    latestRun { status rowsWritten duration }
  }
}
```

### 6.6 Scaling the Metadata Store

In production with many concurrent pipelines:

- Use **PostgreSQL with connection pooling** (PgBouncer recommended)
- Enable **async lineage emission** to avoid blocking pipeline execution
- Set **retention policies** to archive old run history
- Use **read replicas** for the Lineage API to separate read/write loads
- Enable the **Redis cache layer** for frequently-accessed lineage queries

### 6.7 Custom Connectors

```python
from openclaw.connectors import BaseConnector

class MyInternalDBConnector(BaseConnector):
    connector_type = 'my_internal_db'

    def read_dataset(self, urn, partition=None):
        # Return a DataFrame
        ...

    def write_dataset(self, df, urn, mode='overwrite'):
        # Write and register lineage
        ...

    def get_schema(self, urn):
        # Return OpenClaw Schema object
        ...
```

---

## 7. Common Patterns & Best Practices

### 7.1 Medallion Architecture with OpenClaw

| Layer | OpenClaw Pattern | Key Contracts |
|-------|-----------------|---------------|
| **🥉 Bronze (Raw)** | No transforms — ingest only, schema-on-read, full lineage from source | Source freshness, volume bounds |
| **🥈 Silver (Cleaned)** | SQL or Spark transforms; strict schema contracts enforced | No nulls on key fields, type contracts |
| **🥇 Gold (Business)** | dbt models; column-level lineage auto-captured | Business logic assertions, SLA freshness |

### 7.2 Testing Pipelines Locally

```bash
# Validate everything without writing any data
openclaw test pipelines/orders.py --sample 1000 --dry-run
```

Use `--dry-run` in CI/CD to validate transforms, contract definitions, and connector configs before deploying.

### 7.3 Handling Late-Arriving Data

```python
@transform(
    partition_by='event_date',
    late_arrival_window='2 days'   # Reprocess partitions up to 2 days old
)
def daily_events(df, partition):
    ...
```

### 7.4 Deprecating Datasets Safely

```bash
openclaw deprecate \
  --dataset prod.legacy.old_table \
  --notify-owners \
  --sunset-date 2026-06-01
```

OpenClaw will notify all downstream dataset owners automatically and block any new pipelines from reading the dataset after the sunset date.

---

## 8. Learning Path Summary

| Level | Topics to Master | Outcome |
|-------|-----------------|---------|
| 🟢 **Beginner** | Install, init, first pipeline, CLI basics, lineage concepts | Can run and inspect a local pipeline |
| 🟡 **Intermediate** | Connectors, dbt/Spark integration, contracts, incremental, scheduling | Can build production-grade pipelines |
| 🔴 **Advanced** | Column-level lineage, impact analysis, monitoring, API, custom connectors, scaling | Can architect and govern a full data platform |

### Recommended Learning Order

1. Work through the official docs quickstart (`openclaw.io/docs/quickstart`)
2. Build a local pipeline using SQLite as the metadata backend
3. Integrate with your team's existing dbt project
4. Add contracts to your three most critical datasets
5. Explore the Lineage API and connect it to your data catalog

---

## 9. CLI Command Reference

| Command | Purpose |
|---------|---------|
| `openclaw init` | Initialize a new OpenClaw project with metadata backend |
| `openclaw run <pipeline>` | Execute a pipeline and capture lineage |
| `openclaw test <pipeline>` | Test pipeline with sample data without writing output |
| `openclaw lineage show` | Display lineage graph for a dataset |
| `openclaw lineage upstream` | Traverse lineage upstream from a dataset |
| `openclaw lineage downstream` | Traverse lineage downstream from a dataset |
| `openclaw impact --change` | Assess impact of a schema change across all pipelines |
| `openclaw monitor create` | Create a standalone data quality monitor |
| `openclaw deprecate` | Safely mark a dataset for retirement |
| `openclaw ingest dbt` | Sync dbt manifest and catalog into the lineage graph |
| `openclaw serve` | Start the Lineage API server (REST + GraphQL) |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **URN** | Universal Resource Name — OpenClaw's unique identifier for every dataset, e.g. `urn:openclaw:snowflake:prod.schema.table` |
| **Lineage Graph** | Directed acyclic graph where nodes are datasets/transforms and edges are dependencies |
| **Data Contract** | Formal agreement on schema, freshness, volume, and quality for a dataset |
| **Claw Engine** | OpenClaw's core execution engine that resolves pipeline plans and runs transforms |
| **Partition** | A named subset of a dataset, usually time-based (e.g. `event_date=2026-03-01`) |
| **Run Context** | Metadata envelope for a single pipeline execution (ID, inputs, metrics, results) |
| **Column-Level Lineage** | Tracking which source columns feed each target column in a transform |
| **Impact Analysis** | Querying downstream effects of a schema or data change |
| **Assertion** | A data quality check that must pass for a transform to be considered successful |
| **Connector** | Plugin that interfaces OpenClaw with a specific data source or target system |
| **Policy Engine** | Component that enforces contracts, SLAs, and access rules at runtime |
| **Metadata Store** | Persistent storage backend for OpenClaw's lineage graph and run history |
