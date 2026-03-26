# Apache Spark – Complete Guide (2026 Edition)

**Apache Spark** is the leading distributed data processing engine. It processes massive datasets in parallel across a cluster, supporting batch processing, streaming, ML, and SQL — all in one unified engine.

---

## What is Spark?

Spark processes data that doesn't fit on a single machine by splitting work across many nodes:

```
Single Machine         →         Spark Cluster
100GB CSV file                  ┌─ Worker 1: processes partition 1 ─┐
(takes hours)                   ├─ Worker 2: processes partition 2  ├→ Result in minutes
                                └─ Worker 3: processes partition 3 ─┘
```

### Spark vs SQL Databases vs Pandas

| Tool | Data Size | Use Case |
|------|-----------|---------|
| Pandas | GBs (fits in RAM) | Local data analysis, prototyping |
| SQL Database | GBs–TBs | OLTP, OLAP queries |
| **Apache Spark** | TBs–PBs | Distributed processing, ETL, ML at scale |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               Driver Program (SparkContext)           │
│  - Creates SparkContext                              │
│  - Builds DAG of transformations                    │
│  - Schedules tasks                                   │
└─────────────────────┬───────────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Cluster Manager│
              │ (YARN/K8s/Mesos)│
              └───────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Executor │   │Executor │   │Executor │
   │ (Node 1)│   │ (Node 2)│   │ (Node 3)│
   │ Tasks   │   │ Tasks   │   │ Tasks   │
   └─────────┘   └─────────┘   └─────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **RDD** | Resilient Distributed Dataset — fundamental data structure, fault-tolerant, partitioned |
| **DataFrame** | Table with named columns and schema. Built on RDDs. Use this 99% of the time. |
| **Dataset** | Type-safe DataFrame (Scala/Java). In Python = DataFrame. |
| **Partition** | A chunk of data processed by one task on one executor |
| **DAG** | Directed Acyclic Graph — Spark's execution plan |
| **Transformation** | Lazy operation (filter, select, join) — builds the DAG |
| **Action** | Triggers execution (count, show, write, collect) |
| **Catalyst** | Spark's query optimizer — optimizes your execution plan |
| **Tungsten** | Memory management and code generation engine |

---

## Getting Started

```python
from pyspark.sql import SparkSession

# Create SparkSession (entry point)
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \             # local mode: use all CPU cores
    .config("spark.sql.adaptive.enabled", "true") \  # enable AQE
    .getOrCreate()

sc = spark.sparkContext  # for low-level RDD operations (rarely needed)
```

---

## DataFrames

### Creating DataFrames

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("demo").master("local[*]").getOrCreate()

# From a list
data = [(1, "Alice", 85.0), (2, "Bob", 92.5), (3, "Charlie", 78.3)]
df = spark.createDataFrame(data, ["id", "name", "score"])

# With explicit schema
schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("name", StringType(), nullable=True),
    StructField("score", DoubleType(), nullable=True),
])
df = spark.createDataFrame(data, schema)

# From CSV
df = spark.read.csv("data/sales.csv", header=True, inferSchema=True)

# From Parquet (recommended for production)
df = spark.read.parquet("data/events/")

# From JSON
df = spark.read.json("data/logs/*.json")

# From Delta Lake
df = spark.read.format("delta").load("s3://my-bucket/delta/orders/")

# From JDBC (database)
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost/mydb") \
    .option("dbtable", "orders") \
    .option("user", "postgres") \
    .option("password", "secret") \
    .load()
```

### Inspecting DataFrames

```python
df.show(5)               # display first 5 rows
df.show(5, truncate=False)  # don't truncate long values
df.printSchema()         # column names and types
df.dtypes                # list of (column, type) tuples
df.columns               # list of column names
df.count()               # row count (action — triggers execution)
df.describe().show()     # stats: count, mean, std, min, max
df.schema                # full StructType schema
```

---

## Transformations

### Column Operations

```python
from pyspark.sql import functions as F

# Select columns
df.select("name", "score")
df.select(F.col("name"), F.col("score") * 1.1)

# Add / rename columns
df.withColumn("score_pct", F.col("score") / 100)
df.withColumnRenamed("score", "exam_score")
df.drop("unnecessary_col")

# Type casting
df.withColumn("id", F.col("id").cast("string"))

# String operations
df.withColumn("name_upper", F.upper(F.col("name")))
df.withColumn("first_name", F.split(F.col("full_name"), " ")[0])
df.withColumn("email_domain", F.regexp_extract(F.col("email"), r"@(.+)$", 1))

# Date operations
df.withColumn("year", F.year(F.col("created_at")))
df.withColumn("month", F.month(F.col("created_at")))
df.withColumn("date_trunc", F.date_trunc("month", F.col("created_at")))
df.withColumn("days_ago", F.datediff(F.current_date(), F.col("created_at")))

# Conditional
df.withColumn("grade",
    F.when(F.col("score") >= 90, "A")
    .when(F.col("score") >= 80, "B")
    .when(F.col("score") >= 70, "C")
    .otherwise("F")
)

# Null handling
df.fillna(0, subset=["score"])
df.dropna(subset=["name", "score"])
df.withColumn("score", F.coalesce(F.col("score"), F.lit(0)))
```

### Filtering

```python
df.filter(F.col("score") > 80)
df.filter("score > 80")                     # SQL string syntax
df.filter((F.col("score") > 80) & (F.col("name") != "Bob"))
df.where(F.col("status").isin(["active", "pending"]))
df.filter(F.col("name").like("%alice%"))
df.filter(F.col("email").rlike(r"@gmail\.com$"))
df.filter(F.col("score").isNotNull())
```

### Aggregations

```python
# Simple aggregation
df.agg(
    F.count("*").alias("total"),
    F.avg("score").alias("avg_score"),
    F.sum("revenue").alias("total_revenue"),
    F.max("score").alias("max_score"),
    F.countDistinct("user_id").alias("unique_users")
).show()

# Group by
df.groupBy("department") \
  .agg(
      F.count("*").alias("emp_count"),
      F.avg("salary").alias("avg_salary"),
      F.sum("revenue").alias("dept_revenue")
  ) \
  .orderBy(F.desc("dept_revenue")) \
  .show()

# Window functions
from pyspark.sql.window import Window

window = Window.partitionBy("department").orderBy(F.desc("salary"))

df.withColumn("rank", F.rank().over(window)) \
  .withColumn("dense_rank", F.dense_rank().over(window)) \
  .withColumn("row_number", F.row_number().over(window)) \
  .withColumn("running_total", F.sum("salary").over(
      Window.partitionBy("department").rowsBetween(Window.unboundedPreceding, Window.currentRow)
  ))
```

### Joins

```python
orders = spark.read.parquet("orders/")
customers = spark.read.parquet("customers/")

# Inner join (default)
result = orders.join(customers, orders.customer_id == customers.id, "inner")

# Left join
result = orders.join(customers, "customer_id", "left")   # same-name column shorthand

# Multiple conditions
result = df1.join(df2,
    (df1.order_id == df2.order_id) & (df1.date == df2.date),
    "inner"
)

# Broadcast join (for small tables — avoids shuffle)
from pyspark.sql.functions import broadcast
result = orders.join(broadcast(customers), "customer_id")
```

---

## Spark SQL

```python
# Register temp view
df.createOrReplaceTempView("orders")
customers.createOrReplaceTempView("customers")

# Run SQL
result = spark.sql("""
    SELECT
        c.name,
        COUNT(o.id) as order_count,
        SUM(o.amount) as total_revenue,
        AVG(o.amount) as avg_order_value
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    WHERE o.status = 'completed'
      AND o.created_at >= date_trunc('month', current_date())
    GROUP BY c.name
    ORDER BY total_revenue DESC
    LIMIT 20
""")
result.show()

# Global temp view (accessible across SparkSessions)
df.createOrReplaceGlobalTempView("global_orders")
spark.sql("SELECT * FROM global_temp.global_orders")
```

---

## Writing Data

```python
# Write to Parquet (columnar, compressed — recommended)
df.write.mode("overwrite").parquet("output/data/")
df.write.mode("append").partitionBy("year", "month").parquet("output/data/")

# Write to Delta Lake
df.write.format("delta").mode("overwrite").save("s3://bucket/delta/table/")

# Write to CSV
df.write.mode("overwrite").option("header", True).csv("output/csv/")

# Write to JDBC
df.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost/mydb") \
  .option("dbtable", "results") \
  .mode("overwrite") \
  .save()

# Control number of output files
df.coalesce(1).write.parquet("single_file/")   # merge to 1 file
df.repartition(10).write.parquet("10_files/")  # split to 10 files
```

---

## Structured Streaming

Spark can process **real-time streams** with the same DataFrame API.

```python
# Read from Kafka stream
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Parse JSON payload
from pyspark.sql.types import StructType, StringType, TimestampType

schema = StructType() \
    .add("user_id", StringType()) \
    .add("event", StringType()) \
    .add("timestamp", TimestampType())

events = stream_df \
    .selectExpr("CAST(value AS STRING)") \
    .select(F.from_json(F.col("value"), schema).alias("data")) \
    .select("data.*")

# Aggregate: count events per minute
windowed = events \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        F.window("timestamp", "1 minute"),
        "event"
    ) \
    .count()

# Write to Delta Lake
query = windowed.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoints/events/") \
    .start("s3://bucket/delta/event_counts/")

query.awaitTermination()
```

---

## Performance Tuning

### Partitioning

```python
# Check number of partitions
df.rdd.getNumPartitions()

# Repartition (full shuffle — for large skewed data)
df.repartition(200)
df.repartition(200, "customer_id")  # partition by column for joins

# Coalesce (reduce partitions, no full shuffle)
df.coalesce(50)
```

### Caching

```python
# Cache in memory (use when DF is used multiple times)
df.cache()
df.persist()   # same as cache()

# Control storage level
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)   # spill to disk if needed

# Unpersist when done
df.unpersist()
```

### Broadcast Variables

```python
# Broadcast small lookup tables to all executors (avoids shuffle join)
small_table = spark.read.parquet("dim_product/")
result = large_table.join(broadcast(small_table), "product_id")
```

### Adaptive Query Execution (AQE)

```python
# Enable AQE (default in Spark 3.x)
spark.conf.set("spark.sql.adaptive.enabled", "true")

# AQE automatically:
# - Adjusts partition count after shuffles
# - Converts sort-merge joins to broadcast joins when safe
# - Handles data skew
```

---

## PySpark on Databricks / Spark Clusters

```python
# On Databricks, SparkSession is already available as `spark`
# No need to create it

df = spark.read.table("catalog.schema.orders")  # Unity Catalog

# Databricks utilities
dbutils.fs.ls("dbfs:/data/")
dbutils.secrets.get(scope="my-scope", key="db-password")
dbutils.notebook.run("./other_notebook", timeout_seconds=300)
```

---

## Spark in 2026

| Feature | Status |
|---------|--------|
| **Spark Connect** | Remote Spark client — use PySpark from anywhere without running on the cluster |
| **Delta Lake 3.x** | Native Delta support in Spark, liquid clustering |
| **Python UDFs** | Arrow-optimized UDFs (Pandas UDFs) are now the default |
| **Spark on K8s** | Standard deployment method alongside YARN |
| **Unity Catalog** | Databricks' data governance layer across Delta Lake |
| **Spark 4.0** | Python 3.12+, improved structured streaming, better SQL compatibility |

---

## Cheat Sheet

```python
# SparkSession
spark = SparkSession.builder.appName("name").master("local[*]").getOrCreate()

# Read
df = spark.read.parquet("path/")
df = spark.read.csv("path.csv", header=True, inferSchema=True)
df = spark.read.format("delta").load("s3://bucket/path/")
df = spark.read.table("schema.table")

# Inspect
df.show(10)
df.printSchema()
df.count()
df.describe().show()

# Transform
df.select("col1", "col2")
df.filter(F.col("x") > 0)
df.withColumn("new", F.col("a") + F.col("b"))
df.groupBy("key").agg(F.sum("val").alias("total"))
df.join(other, "key", "left")
df.orderBy(F.desc("col"))
df.limit(100)
df.distinct()
df.dropDuplicates(["col1", "col2"])

# Write
df.write.mode("overwrite").parquet("output/")
df.write.mode("append").partitionBy("date").parquet("output/")
df.write.format("delta").mode("overwrite").save("s3://bucket/delta/")

# SQL
df.createOrReplaceTempView("my_table")
spark.sql("SELECT * FROM my_table WHERE x > 0")

# Performance
df.cache()
df.repartition(N)
df.coalesce(N)
df.join(broadcast(small_df), "key")
```
