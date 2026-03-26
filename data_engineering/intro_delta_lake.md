# Delta Lake – Complete Guide (2026 Edition)

**Delta Lake** is an open-source storage layer that brings **ACID transactions**, **schema enforcement**, **time travel**, and **scalable metadata** to data lakes on cloud object storage (S3, GCS, ADLS). It's the backbone of the **Lakehouse** architecture.

---

## What is Delta Lake?

Before Delta Lake, data lakes had serious problems:
- **No ACID transactions** — partial writes leave inconsistent data
- **No schema enforcement** — garbage data corrupts your tables
- **No updates/deletes** — append-only; can't fix mistakes
- **Slow reads** — listing millions of files on S3 is slow
- **No history** — can't see what data looked like yesterday

Delta Lake solves all of these on top of Parquet files in S3.

```
Data Lake (raw S3/GCS) + Delta Lake = Lakehouse
                                        ↓
                           ACID + Schema + Time Travel
                           + Fast reads + Streaming + ML
```

---

## Core Features

| Feature | Description |
|---------|-------------|
| **ACID Transactions** | Atomic commits — either all changes succeed or none |
| **Schema Enforcement** | Reject writes that don't match the table schema |
| **Schema Evolution** | Safely add new columns |
| **Time Travel** | Query data as it was at any point in time |
| **Upserts (MERGE)** | Update or insert based on conditions |
| **Deletes** | Delete specific rows (GDPR compliance) |
| **Streaming + Batch** | Read and write to same table from batch and stream |
| **Data Lineage** | Track what changed, when, and how |
| **Optimize/Compaction** | Combine small files, Z-order for faster queries |
| **Change Data Feed** | Read only what changed since last read |

---

## Getting Started

```bash
pip install delta-spark pyspark
```

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DeltaLake") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
```

---

## Writing and Reading

```python
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# Create a Delta table
data = [(1, "Alice", 85.0, "2025-01-01"),
        (2, "Bob", 92.5, "2025-01-01")]
df = spark.createDataFrame(data, ["id", "name", "score", "date"])

# Write as Delta
df.write.format("delta").mode("overwrite").save("/data/scores")

# Read Delta table
scores = spark.read.format("delta").load("/data/scores")
scores.show()

# Create table in metastore (SQL access)
df.write.format("delta").saveAsTable("scores")

# Or use SQL
spark.sql("""
    CREATE TABLE IF NOT EXISTS scores (
        id INT,
        name STRING,
        score DOUBLE,
        date DATE
    )
    USING DELTA
    LOCATION 's3://my-bucket/delta/scores/'
""")
```

---

## ACID Transactions

```python
# Concurrent writers are safe — Delta uses optimistic concurrency control

# Writer 1: append new records
new_data = spark.createDataFrame([(3, "Charlie", 78.3, "2025-01-02")], ["id", "name", "score", "date"])
new_data.write.format("delta").mode("append").save("/data/scores")

# Writer 2: update existing records
# Both complete successfully without corrupting data
```

---

## Schema Enforcement & Evolution

```python
# Schema enforcement — this will FAIL if schema doesn't match
try:
    bad_df = spark.createDataFrame([(4, "Dave")], ["id", "name"])
    bad_df.write.format("delta").mode("append").save("/data/scores")
except Exception as e:
    print("Schema mismatch blocked:", e)

# Schema evolution — allow adding new columns
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

new_df = spark.createDataFrame(
    [(5, "Eve", 88.0, "2025-01-03", "Math")],
    ["id", "name", "score", "date", "subject"]  # new column: subject
)
new_df.write.format("delta").mode("append") \
    .option("mergeSchema", "true") \
    .save("/data/scores")
```

---

## Time Travel

One of Delta Lake's killer features.

```python
dt = DeltaTable.forPath(spark, "/data/scores")

# View table history
dt.history().show()
# version | timestamp | operation | operationParameters | ...

# Query specific version
spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/data/scores") \
    .show()

# Query at a specific timestamp
spark.read.format("delta") \
    .option("timestampAsOf", "2025-01-01 00:00:00") \
    .load("/data/scores") \
    .show()

# SQL time travel
spark.sql("SELECT * FROM scores VERSION AS OF 0")
spark.sql("SELECT * FROM scores TIMESTAMP AS OF '2025-01-01'")
```

---

## Updates, Deletes, and MERGE (Upserts)

```python
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/data/scores")

# Update specific rows
dt.update(
    condition=F.col("name") == "Alice",
    set={"score": F.lit(95.0)}
)

# Delete rows
dt.delete(condition=F.col("score") < 60)

# MERGE (Upsert) — the most powerful operation
updates = spark.createDataFrame([
    (1, "Alice", 99.0, "2025-01-05"),   # update existing
    (6, "Frank", 81.5, "2025-01-05"),   # new record
], ["id", "name", "score", "date"])

dt.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "score": "source.score",
    "date": "source.date"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "score": "source.score",
    "date": "source.date"
}).execute()

# Full upsert in SQL
spark.sql("""
    MERGE INTO scores AS target
    USING updates AS source
    ON target.id = source.id
    WHEN MATCHED THEN
        UPDATE SET target.score = source.score
    WHEN NOT MATCHED THEN
        INSERT (id, name, score, date) VALUES (source.id, source.name, source.score, source.date)
    WHEN NOT MATCHED BY SOURCE THEN
        DELETE
""")
```

---

## Partitioning

```python
# Write with partitioning
df.write.format("delta") \
    .partitionBy("date", "region") \
    .mode("overwrite") \
    .save("/data/sales")

# Partition pruning happens automatically on filter
sales = spark.read.format("delta").load("/data/sales")
sales.filter(F.col("date") == "2025-01-01").count()   # only reads 2025-01-01 partition
```

---

## Optimize & Z-Order

```python
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/data/orders")

# Compact small files (bin-packing)
dt.optimize().executeCompaction()

# Z-Order: co-locate related data for faster filtering
dt.optimize().executeZOrderBy("customer_id", "date")

# After Z-Order: queries filtering on customer_id or date are much faster
spark.sql("OPTIMIZE orders ZORDER BY (customer_id, date)")

# Vacuum: remove old files to save storage
# (default retainDurationHours=168, i.e., 7 days for time travel)
dt.vacuum()           # keep 7 days of history
dt.vacuum(24)         # keep only 24 hours (less time travel)
```

---

## Streaming with Delta Lake

```python
# Streaming write to Delta
query = df_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoints/orders") \
    .table("orders")

# Streaming read from Delta (reads new changes as they arrive)
changes = spark.readStream \
    .format("delta") \
    .option("readChangeFeed", "true") \    # Change Data Feed
    .option("startingVersion", 0) \
    .table("orders")
```

---

## Change Data Feed (CDF)

```python
# Enable CDF on a table
spark.sql("""
    ALTER TABLE orders
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Read changes since version 5
cdf = spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 5) \
    .table("orders")

cdf.select("_change_type", "order_id", "amount").show()
# _change_type: insert / update_preimage / update_postimage / delete
```

---

## Delta Lake + dbt

```yaml
# dbt/profiles.yml (Databricks)
my_profile:
  target: prod
  outputs:
    prod:
      type: databricks
      catalog: my_catalog    # Unity Catalog
      schema: analytics
      host: "{{ env_var('DBT_DATABRICKS_HOST') }}"
      http_path: "{{ env_var('DBT_HTTP_PATH') }}"
      token: "{{ env_var('DATABRICKS_TOKEN') }}"
```

```sql
-- dbt model using Delta features
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_id',
    file_format='delta',
    post_hook="OPTIMIZE {{ this }} ZORDER BY (customer_id)"
) }}

SELECT * FROM {{ ref('stg_orders') }}
{% if is_incremental() %}
WHERE updated_at > (SELECT MAX(updated_at) FROM {{ this }})
{% endif %}
```

---

## Delta Lake Architecture (Lakehouse)

```
Raw Files (S3/GCS/ADLS)
        ↓
Delta Lake Tables (Bronze/Silver/Gold)
        ↓
┌───────────────────────────────────────┐
│         Compute Engines               │
│  Apache Spark    Apache Flink         │
│  Trino/Presto    DuckDB               │
│  Snowflake       Databricks SQL       │
└───────────────────────────────────────┘
        ↓
BI Tools / ML Models / Data Apps
```

---

## Delta Lake in 2026

| Feature | Description |
|---------|-------------|
| **Delta Lake 4.0** | Standalone library (no Spark required for reads) |
| **Liquid Clustering** | Replaces partitioning + Z-Order; auto-optimizes layout |
| **Deletion Vectors** | Soft-deletes without rewriting files (faster deletes) |
| **Variant type** | Native semi-structured data type for JSON |
| **Universal Format (UniForm)** | Delta tables readable as Iceberg and Hudi automatically |
| **Delta Kernel** | Embeddable library for building Delta connectors |
| **DuckDB + Delta** | `SELECT * FROM delta_scan('s3://...')` — no Spark needed |

### Liquid Clustering (Replaces Partitioning)

```sql
-- Create table with liquid clustering (Databricks / Delta 4.0)
CREATE TABLE orders
CLUSTER BY (customer_id, order_date)
USING DELTA;

-- Automatically maintained - no manual OPTIMIZE needed
-- Better than partitioning for high-cardinality columns
```

---

## Cheat Sheet

```python
# Read
df = spark.read.format("delta").load("path/")
df = spark.read.format("delta").option("versionAsOf", 5).load("path/")
df = spark.table("my_delta_table")

# Write
df.write.format("delta").mode("overwrite").save("path/")
df.write.format("delta").mode("append").save("path/")
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("path/")

# DeltaTable API
from delta.tables import DeltaTable
dt = DeltaTable.forPath(spark, "path/")
dt = DeltaTable.forName(spark, "my_table")

dt.history().show()                    # version history
dt.toDF().show()                       # as DataFrame
dt.update(condition, set_dict)         # update rows
dt.delete(condition)                   # delete rows
dt.merge(source, condition)            # upsert

# Maintenance
dt.optimize().executeCompaction()
dt.optimize().executeZOrderBy("col1", "col2")
dt.vacuum()
dt.vacuum(retainHours=24)

# SQL
spark.sql("DESCRIBE HISTORY my_table")
spark.sql("OPTIMIZE my_table ZORDER BY (customer_id)")
spark.sql("VACUUM my_table RETAIN 168 HOURS")
spark.sql("ALTER TABLE my_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
```
