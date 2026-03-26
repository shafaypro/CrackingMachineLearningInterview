# Apache Iceberg – Complete Guide (2026 Edition)

**Apache Iceberg** is an open table format for huge analytic datasets. It brings SQL table semantics — ACID transactions, schema evolution, hidden partitioning, and time travel — to data lake files on S3/GCS/ADLS. In 2026, Iceberg has become the **default open table format** for the data lakehouse.

---

## Table of Contents
1. [What is Apache Iceberg?](#what-is-apache-iceberg)
2. [Iceberg vs Delta Lake vs Hudi](#iceberg-vs-delta-lake-vs-hudi)
3. [Core Architecture](#core-architecture)
4. [Getting Started](#getting-started)
5. [Core Operations](#core-operations)
6. [Schema Evolution](#schema-evolution)
7. [Partitioning & Hidden Partitioning](#partitioning--hidden-partitioning)
8. [Time Travel & Snapshots](#time-travel--snapshots)
9. [Table Maintenance](#table-maintenance)
10. [Iceberg Catalogs](#iceberg-catalogs)
11. [Iceberg in 2026](#iceberg-in-2026)
12. [Cheat Sheet](#cheat-sheet)

---

## What is Apache Iceberg?

Before open table formats, data lakes had three options:
1. **Raw Parquet/ORC files** — fast reads, but no ACID, no schema, no history
2. **Hive Metastore tables** — basic metadata, but terrible at scale
3. **Proprietary formats** — vendor lock-in

Iceberg solves this by adding a **metadata layer** on top of existing file formats:

```
Your Data (Parquet/ORC/Avro files on S3)
           +
Iceberg Metadata Layer
           =
A "proper" SQL table with ACID, schema, history, and fast queries
```

### Why Iceberg Won (vs Delta Lake vs Hudi)

Iceberg has become the industry standard because:
- **Truly open** — Apache Software Foundation, no single vendor owns it
- **Catalog-agnostic** — works with Hive Metastore, Glue, Nessie, REST, JDBC
- **Engine-agnostic** — Spark, Flink, Trino, Snowflake, Athena, DuckDB, BigQuery all read it
- **Multi-writer support** — designed for concurrent writes from multiple engines
- **Column statistics** — fine-grained stats for query pruning
- **Partition evolution** — change partition strategy without rewriting data

---

## Iceberg vs Delta Lake vs Hudi

| Feature | Apache Iceberg | Delta Lake | Apache Hudi |
|---------|---------------|------------|-------------|
| **License** | Apache 2.0 (ASF) | Apache 2.0 (Linux Foundation) | Apache 2.0 (ASF) |
| **Primary backer** | Netflix, Apple, Databricks, etc. | Databricks | Uber |
| **Engine support** | Spark, Flink, Trino, DuckDB, Snowflake, BigQuery... | Spark, Trino (read), Databricks | Spark, Flink |
| **ACID** | ✓ | ✓ | ✓ |
| **Time travel** | ✓ | ✓ | ✓ |
| **Schema evolution** | ✓ (best-in-class) | ✓ | ✓ |
| **Partition evolution** | ✓ (unique) | Partial | Partial |
| **Multi-engine writes** | ✓ | Limited | Limited |
| **Row-level deletes** | ✓ (v2: Delete Files) | ✓ | ✓ (MOR tables) |
| **2026 adoption** | **Industry standard** | Strong (Databricks) | Streaming CDC use cases |

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Iceberg Table                            │
├─────────────────────────────────────────────────────────────────┤
│ Catalog Layer   │  Hive / Glue / Nessie / REST / JDBC          │
│                 │  → stores: table location + current metadata  │
├─────────────────────────────────────────────────────────────────┤
│ Metadata Layer  │  metadata/v3.metadata.json                   │
│                 │  → schema, partition spec, snapshot history   │
├─────────────────────────────────────────────────────────────────┤
│ Manifest Layer  │  snap-123.avro  (manifest list)              │
│                 │  → list of manifest files                     │
│                 │  m1.avro, m2.avro  (manifests)               │
│                 │  → list of data files + column stats          │
├─────────────────────────────────────────────────────────────────┤
│ Data Layer      │  part-00001.parquet, part-00002.parquet, ... │
│                 │  → actual data (Parquet / ORC / Avro)         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metadata Concepts

| Concept | Description |
|---------|-------------|
| **Snapshot** | Immutable point-in-time state of the table. Every write creates a new snapshot. |
| **Manifest List** | Points to all manifests for a snapshot |
| **Manifest File** | Lists data files + their column-level statistics |
| **Data File** | The actual Parquet/ORC/Avro file with rows |
| **Delete File** | Records which rows are deleted (v2 format — avoids rewriting) |
| **Sequence Number** | Monotonically increasing version counter for ordering snapshots |

---

## Getting Started

### With PySpark

```bash
pip install pyspark pyiceberg
```

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Iceberg Demo") \
    .master("local[*]") \
    .config("spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.0") \
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/tmp/iceberg-warehouse") \
    .getOrCreate()
```

### With PyIceberg (no Spark needed for reads)

```bash
pip install pyiceberg[s3,glue,duckdb]
```

```python
from pyiceberg.catalog import load_catalog

# Load from REST catalog
catalog = load_catalog("my_catalog", **{
    "type": "rest",
    "uri": "https://my-iceberg-catalog.com",
    "credential": "token:my-token",
})

# Load from AWS Glue
catalog = load_catalog("glue", **{
    "type": "glue",
    "region_name": "us-east-1",
})

# List tables
catalog.list_namespaces()
catalog.list_tables("my_db")

# Load a table
table = catalog.load_table("my_db.orders")
```

---

## Core Operations

### Create Table

```sql
-- SQL (via Spark)
CREATE TABLE local.db.orders (
    order_id    BIGINT,
    customer_id BIGINT,
    amount      DECIMAL(18, 2),
    status      STRING,
    created_at  TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(created_at))    -- hidden partition!
TBLPROPERTIES (
    'write.format.default' = 'parquet',
    'write.parquet.compression-codec' = 'snappy'
);
```

```python
# Python (PyIceberg)
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField, LongType, DecimalType, StringType, TimestampType
)
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform

schema = Schema(
    NestedField(1, "order_id", LongType(), required=True),
    NestedField(2, "customer_id", LongType()),
    NestedField(3, "amount", DecimalType(18, 2)),
    NestedField(4, "status", StringType()),
    NestedField(5, "created_at", TimestampType()),
)

partition_spec = PartitionSpec(
    PartitionField(source_id=5, field_id=1000, transform=DayTransform(), name="created_at_day")
)

table = catalog.create_table(
    identifier="my_db.orders",
    schema=schema,
    partition_spec=partition_spec,
    location="s3://my-bucket/iceberg/orders/",
)
```

### Read and Write

```python
# Write with Spark
df = spark.createDataFrame([
    (1, 100, 99.99, "completed", "2025-01-15 10:00:00"),
    (2, 101, 149.50, "pending", "2025-01-15 11:30:00"),
], ["order_id", "customer_id", "amount", "status", "created_at"])

df.writeTo("local.db.orders").append()
df.writeTo("local.db.orders").overwritePartitions()

# Read with Spark
orders = spark.table("local.db.orders")
orders.filter("status = 'completed'").show()

# Read with PyIceberg (no Spark!)
table = catalog.load_table("my_db.orders")
scan = table.scan(row_filter="status = 'completed'", limit=100)
df = scan.to_pandas()           # → Pandas DataFrame
arrow_table = scan.to_arrow()   # → PyArrow Table
```

### Write Modes

```python
# Append — add rows
df.writeTo("local.db.orders").append()

# Overwrite matching partitions
df.writeTo("local.db.orders").overwritePartitions()

# Dynamic overwrite (replace only affected partitions)
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
df.writeTo("local.db.orders").overwrite(F.col("created_at_day") == "2025-01-15")

# Create or replace table
df.writeTo("local.db.orders").createOrReplace()
```

### MERGE (Upsert)

```sql
MERGE INTO local.db.orders AS target
USING updates AS source
ON target.order_id = source.order_id
WHEN MATCHED AND source.status = 'cancelled' THEN DELETE
WHEN MATCHED THEN UPDATE SET
    target.amount = source.amount,
    target.status = source.status
WHEN NOT MATCHED THEN INSERT *;
```

```python
# Python MERGE
from pyiceberg.expressions import EqualTo

table = catalog.load_table("my_db.orders")

# Upsert using overwrite_partitions
table.overwrite(new_df, overwrite_filter=EqualTo("status", "pending"))
```

---

## Schema Evolution

Iceberg's schema evolution is **lossless** — you can safely:
- **Add columns** (new reads get NULL for old rows)
- **Drop columns** (old files still have the data, just hidden)
- **Rename columns** — tracked by column ID, not name
- **Promote types** — int → long, float → double
- **Reorder columns** — cosmetic only

```sql
-- Add column
ALTER TABLE local.db.orders
ADD COLUMN discount DECIMAL(18, 2);

-- Rename column
ALTER TABLE local.db.orders
RENAME COLUMN discount TO discount_amount;

-- Drop column
ALTER TABLE local.db.orders
DROP COLUMN discount_amount;

-- Change type (widening only)
ALTER TABLE local.db.orders
ALTER COLUMN amount TYPE DECIMAL(24, 2);

-- Add nested column
ALTER TABLE local.db.orders
ADD COLUMN metadata.source STRING;
```

```python
# Python schema evolution
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType

table = catalog.load_table("my_db.orders")

with table.update_schema() as update:
    update.add_column("discount", DecimalType(18, 2))
    update.rename_column("status", "order_status")
```

---

## Partitioning & Hidden Partitioning

Iceberg's **hidden partitioning** is a major advantage over Hive-style partitioning.

### Hive Partitioning (old way — avoid)
```sql
-- Hive: partition column IS a real column
-- Users must write: WHERE dt = '2025-01-15'
-- Changing partition granularity requires rewriting ALL data
CREATE TABLE orders (order_id INT, amount FLOAT)
PARTITIONED BY (dt STRING);  -- user must manage this manually
```

### Iceberg Hidden Partitioning (new way)

```sql
-- Iceberg: partition is derived from existing column
-- Iceberg automatically prunes the right partitions
-- Changing granularity is non-breaking!
CREATE TABLE orders (order_id INT, amount FLOAT, created_at TIMESTAMP)
USING iceberg
PARTITIONED BY (days(created_at));  -- hidden! User just queries on created_at

-- Query: no need to know about partitions
SELECT * FROM orders WHERE created_at BETWEEN '2025-01-01' AND '2025-01-31';
-- Iceberg automatically scans only January partition files
```

### Partition Transforms

| Transform | Example | Description |
|-----------|---------|-------------|
| `identity(col)` | `identity(region)` | Direct value (like Hive) |
| `year(ts)` | `year(created_at)` | Year of timestamp |
| `month(ts)` | `month(created_at)` | Month of timestamp |
| `day(ts)` | `day(created_at)` | Day of timestamp |
| `hour(ts)` | `hour(event_time)` | Hour of timestamp |
| `bucket(N, col)` | `bucket(16, user_id)` | Hash bucket (good for high-cardinality) |
| `truncate(W, col)` | `truncate(4, zip_code)` | Prefix truncation |

### Partition Evolution

```sql
-- Change from day to hour partitioning — zero data rewrite!
ALTER TABLE local.db.orders
REPLACE PARTITION FIELD days(created_at)
WITH hours(created_at);

-- Old data stays in day partitions
-- New data goes into hour partitions
-- Queries work transparently across both
```

---

## Time Travel & Snapshots

```sql
-- View snapshot history
SELECT * FROM local.db.orders.history;

-- Query at specific snapshot ID
SELECT * FROM local.db.orders
VERSION AS OF 4654808683286204734;

-- Query at specific timestamp
SELECT * FROM local.db.orders
TIMESTAMP AS OF '2025-01-15 10:00:00';

-- View all snapshots
SELECT * FROM local.db.orders.snapshots;

-- View data files for current snapshot
SELECT * FROM local.db.orders.files;

-- View manifests
SELECT * FROM local.db.orders.manifests;
```

```python
# Python time travel
table = catalog.load_table("my_db.orders")

# List snapshots
for snapshot in table.history():
    print(snapshot.snapshot_id, snapshot.timestamp_ms)

# Scan at specific snapshot
scan = table.scan(snapshot_id=4654808683286204734)
df = scan.to_pandas()
```

---

## Table Maintenance

```sql
-- Compact small files (bin-packing)
CALL local.system.rewrite_data_files('db.orders');

-- Compact with options
CALL local.system.rewrite_data_files(
    table => 'db.orders',
    strategy => 'sort',
    sort_order => 'zorder(customer_id, created_at)'
);

-- Remove old snapshots (expire)
CALL local.system.expire_snapshots(
    table => 'db.orders',
    older_than => TIMESTAMP '2025-01-01 00:00:00',
    retain_last => 5
);

-- Remove orphan files (files not referenced by any snapshot)
CALL local.system.remove_orphan_files(
    table => 'db.orders',
    older_than => TIMESTAMP '2025-01-01 00:00:00'
);

-- Rewrite manifests for faster planning
CALL local.system.rewrite_manifests('db.orders');
```

```python
# Python maintenance
from pyiceberg.catalog import load_catalog

table = catalog.load_table("my_db.orders")

# Expire old snapshots
table.expire_snapshots().expire_older_than(
    datetime(2025, 1, 1, tzinfo=timezone.utc)
).commit()

# Compact files
table.rewrite_data_files().rewrite_all().execute()
```

---

## Iceberg Catalogs

A **catalog** tracks table locations and the current metadata pointer.

| Catalog | Description | Use Case |
|---------|-------------|---------|
| **Hive Metastore** | Traditional warehouse catalog | On-prem, legacy systems |
| **AWS Glue** | Fully managed on AWS | AWS data lakes |
| **Nessie** | Git-like version control for data | Data versioning, branching |
| **REST Catalog** | HTTP API (Tabular, Polaris) | Cloud-native, multi-engine |
| **JDBC** | Any SQL database as catalog | Simple self-hosted |
| **Unity Catalog** | Databricks unified catalog | Databricks environments |

### AWS Glue Catalog

```python
spark = SparkSession.builder \
    .config("spark.sql.catalog.glue", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.glue.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config("spark.sql.catalog.glue.warehouse", "s3://my-bucket/iceberg/") \
    .config("spark.sql.catalog.glue.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .getOrCreate()

spark.sql("CREATE TABLE glue.mydb.orders (...) USING iceberg")
```

### Nessie (Git for Data)

```python
# Nessie lets you create branches of your data!
spark = SparkSession.builder \
    .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog") \
    .config("spark.sql.catalog.nessie.uri", "http://localhost:19120/api/v1") \
    .config("spark.sql.catalog.nessie.ref", "main") \
    .getOrCreate()

# Work on a branch — isolates your changes
spark.sql("USE REFERENCE my_feature_branch IN nessie")
spark.sql("UPDATE nessie.db.orders SET status = 'v2' WHERE ...")

# Merge branch to main after validation
# (via Nessie API or CLI)
```

---

## DuckDB + Iceberg

DuckDB can query Iceberg tables **without Spark** — perfect for local development and ad-hoc queries.

```sql
-- Install extension
INSTALL iceberg;
LOAD iceberg;

-- Query Iceberg table on S3
SELECT *
FROM iceberg_scan('s3://my-bucket/iceberg/orders/')
WHERE created_at >= '2025-01-01'
LIMIT 100;

-- Time travel
SELECT * FROM iceberg_scan(
    's3://my-bucket/iceberg/orders/',
    snapshot_id = 4654808683286204734
);

-- Via catalog (REST)
ATTACH 'https://my-rest-catalog.com' AS iceberg_catalog (TYPE ICEBERG);
SELECT * FROM iceberg_catalog.my_db.orders LIMIT 10;
```

---

## Iceberg in 2026

### What Changed

| Feature | Status in 2026 |
|---------|---------------|
| **Iceberg v3** | Row lineage, default values, multi-arg transforms, variant type |
| **REST Catalog standard** | The default catalog API — all engines support it |
| **Apache Polaris** | Anthropic/Snowflake open-sourced REST catalog — industry standard |
| **Uniform read** | Snowflake, BigQuery, Athena all read Iceberg natively |
| **Delta UniForm** | Delta tables expose Iceberg metadata — bridging the gap |
| **Streaming support** | Flink + Iceberg is production-standard for streaming ETL |
| **PyIceberg 0.8+** | Native Python reads without Spark — DuckDB integration |

### Adoption in 2026

- **AWS**: S3 Tables (native Iceberg tables in S3), Athena + Glue
- **GCP**: BigQuery supports Iceberg as external tables
- **Azure**: ADLS + Iceberg via Fabric
- **Snowflake**: Iceberg tables in Snowflake (data lives in your S3)
- **Databricks**: Full Iceberg read/write support alongside Delta

---

## Cheat Sheet

```sql
-- Create
CREATE TABLE catalog.db.tbl (col1 TYPE, ...) USING iceberg
PARTITIONED BY (days(ts_col));

-- Write
INSERT INTO catalog.db.tbl VALUES (...);
INSERT INTO catalog.db.tbl SELECT * FROM other_tbl;

-- Merge/Upsert
MERGE INTO target USING source ON condition
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT *;

-- Time travel
SELECT * FROM tbl VERSION AS OF <snapshot_id>;
SELECT * FROM tbl TIMESTAMP AS OF '<timestamp>';

-- Schema
ALTER TABLE tbl ADD COLUMN new_col TYPE;
ALTER TABLE tbl RENAME COLUMN old TO new;
ALTER TABLE tbl DROP COLUMN col;

-- History / Snapshots
SELECT * FROM tbl.history;
SELECT * FROM tbl.snapshots;
SELECT * FROM tbl.files;
SELECT * FROM tbl.manifests;

-- Maintenance
CALL catalog.system.rewrite_data_files('db.tbl');
CALL catalog.system.expire_snapshots('db.tbl', TIMESTAMP '2025-01-01', 5);
CALL catalog.system.remove_orphan_files('db.tbl');

-- Partition evolution
ALTER TABLE tbl REPLACE PARTITION FIELD days(ts) WITH hours(ts);
```

```python
# PyIceberg quick reference
from pyiceberg.catalog import load_catalog

catalog = load_catalog("my_catalog", type="rest", uri="https://...")
table = catalog.load_table("db.tbl")

# Scan
scan = table.scan(row_filter="col > 0", limit=100)
df = scan.to_pandas()

# Schema evolution
with table.update_schema() as upd:
    upd.add_column("new_col", StringType())

# Snapshots
for snap in table.history():
    print(snap.snapshot_id, snap.timestamp_ms)
```
