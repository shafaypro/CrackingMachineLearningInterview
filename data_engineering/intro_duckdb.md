# DuckDB – Complete Guide (2026 Edition)

**DuckDB** is an in-process OLAP database that runs directly inside your application — no server, no setup, no cluster. It's fast, embeddable, and can query Parquet, CSV, JSON, Arrow, Iceberg, and Delta Lake files directly. In 2026, DuckDB has become the go-to tool for local data analytics and the "SQLite of OLAP."

---

## Table of Contents
1. [What is DuckDB?](#what-is-duckdb)
2. [Installation & Setup](#installation--setup)
3. [SQL Basics](#sql-basics)
4. [Reading Files Directly](#reading-files-directly)
5. [Python API](#python-api)
6. [Advanced SQL Features](#advanced-sql-features)
7. [Integrations](#integrations)
8. [DuckDB in Production](#duckdb-in-production)
9. [DuckDB vs Spark vs Pandas](#duckdb-vs-spark-vs-pandas)
10. [DuckDB in 2026](#duckdb-in-2026)
11. [Cheat Sheet](#cheat-sheet)

---

## What is DuckDB?

DuckDB is a **columnar, vectorized, in-process** analytical database.

- **In-process**: Runs inside your Python/R/Go/Java/Node app — like SQLite, but for analytics
- **Columnar**: Stores data column-by-column — perfect for aggregations over wide tables
- **Vectorized**: Processes data in vectors (batches) — massively parallel on a single machine
- **No server**: Nothing to install, configure, or maintain

```python
import duckdb
result = duckdb.sql("SELECT 42 AS answer").fetchone()
# → (42,)   — it just works, no server needed
```

### When to Use DuckDB

```
Your data fits on one machine (< ~500GB compressed)?    → DuckDB
Need SQL on Parquet/CSV/JSON files without setup?       → DuckDB
Data engineering locally / in notebooks?                → DuckDB
ETL transformations before loading to warehouse?        → DuckDB
Ad-hoc queries on S3 files?                            → DuckDB
Replace heavy Spark job for < 100GB data?               → DuckDB
Need Spark-scale (TBs, cluster)?                        → Spark
```

---

## Installation & Setup

```bash
# Python
pip install duckdb

# CLI
brew install duckdb                    # macOS
# or download from duckdb.org/docs/installation

# Node.js
npm install duckdb

# R
install.packages("duckdb")
```

### CLI

```bash
# Start interactive shell
duckdb

# Open/create a database file
duckdb my_database.duckdb

# Run a query file
duckdb my_database.duckdb < query.sql

# Run inline SQL
duckdb my_database.duckdb "SELECT 42"
```

---

## SQL Basics

### Tables

```sql
-- Create table
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount      DECIMAL(18, 2),
    status      VARCHAR,
    created_at  TIMESTAMP
);

-- Insert
INSERT INTO orders VALUES (1, 100, 99.99, 'completed', '2025-01-15 10:00:00');

-- Or insert from a query
INSERT INTO orders SELECT * FROM read_parquet('orders.parquet');

-- Create table from query
CREATE TABLE summary AS
SELECT customer_id, SUM(amount) AS total
FROM orders GROUP BY customer_id;

-- Temporary table (session-scoped)
CREATE TEMP TABLE tmp_orders AS SELECT * FROM orders WHERE status = 'pending';
```

### Querying

```sql
-- Standard SQL — DuckDB is highly SQL-compliant
SELECT
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order,
    MAX(amount) AS largest_order
FROM orders
WHERE created_at >= '2025-01-01'
  AND status IN ('completed', 'shipped')
GROUP BY customer_id
HAVING COUNT(*) > 5
ORDER BY total_revenue DESC
LIMIT 20;
```

---

## Reading Files Directly

This is DuckDB's superpower — **no import needed**.

### CSV

```sql
-- Auto-detect schema
SELECT * FROM read_csv('data/sales.csv');

-- With options
SELECT * FROM read_csv(
    'data/sales.csv',
    header = true,
    delimiter = ',',
    null_padding = true,
    ignore_errors = true
);

-- Multiple files (glob)
SELECT * FROM read_csv('data/sales_*.csv');
SELECT * FROM read_csv(['jan.csv', 'feb.csv', 'mar.csv']);

-- Create view over CSV (no copy needed)
CREATE VIEW sales AS SELECT * FROM read_csv('data/sales.csv');
```

### Parquet

```sql
-- Read single file
SELECT * FROM read_parquet('data/orders.parquet');

-- Read multiple files (partitioned directory)
SELECT * FROM read_parquet('data/orders/year=*/month=*/*.parquet');

-- Glob pattern
SELECT * FROM read_parquet('s3://my-bucket/events/**/*.parquet');

-- With HIVE partition columns
SELECT * FROM read_parquet(
    'data/orders/year=*/month=*/*.parquet',
    hive_partitioning = true
);

-- Metadata inspection
SELECT * FROM parquet_schema('data/orders.parquet');
SELECT * FROM parquet_metadata('data/orders.parquet');
```

### JSON

```sql
-- Read JSON Lines (ndjson)
SELECT * FROM read_json('data/events.jsonl');

-- Nested JSON — DuckDB handles it!
SELECT
    json_extract(payload, '$.user.id') AS user_id,
    json_extract(payload, '$.event') AS event_name
FROM read_json('data/events.json');

-- Auto-struct flattening
SELECT data.user.id, data.event
FROM read_json('data/events.json', auto_detect = true);
```

### S3 / Cloud Storage

```sql
-- Install and load httpfs extension
INSTALL httpfs;
LOAD httpfs;

-- Configure S3 credentials
SET s3_region = 'us-east-1';
SET s3_access_key_id = 'AKIA...';
SET s3_secret_access_key = 'secret...';

-- Or use IAM role (on EC2/Lambda/ECS)
SET s3_use_ssl = true;

-- Now query S3 directly
SELECT COUNT(*) FROM read_parquet('s3://my-bucket/data/orders/*.parquet');

-- Write back to S3
COPY (SELECT * FROM orders) TO 's3://my-bucket/output/orders.parquet' (FORMAT PARQUET);
```

### Delta Lake

```sql
INSTALL delta;
LOAD delta;

SELECT * FROM delta_scan('s3://my-bucket/delta/orders/');

-- Time travel
SELECT * FROM delta_scan('s3://my-bucket/delta/orders/', version = 5);
```

### Apache Iceberg

```sql
INSTALL iceberg;
LOAD iceberg;

SELECT * FROM iceberg_scan('s3://my-bucket/iceberg/orders/');

-- With snapshot
SELECT * FROM iceberg_scan(
    's3://my-bucket/iceberg/orders/',
    snapshot_id = 4654808683286204734
);
```

---

## Python API

```python
import duckdb
import pandas as pd
import polars as pl

# In-memory database (temporary)
conn = duckdb.connect()

# Persistent database
conn = duckdb.connect('my_data.duckdb')

# Execute SQL, fetch results
result = conn.execute("SELECT * FROM orders LIMIT 10")
rows = result.fetchall()          # list of tuples
df = result.df()                  # Pandas DataFrame
arrow = result.arrow()            # PyArrow Table
polars_df = result.pl()           # Polars DataFrame

# Shorthand (in-memory)
df = duckdb.sql("SELECT * FROM orders").df()
```

### Querying DataFrames Directly

```python
import duckdb
import pandas as pd

# DuckDB can query Pandas/Polars DataFrames directly — zero copy!
orders_df = pd.read_csv('orders.csv')

# Query the DataFrame by name
result = duckdb.sql("SELECT customer_id, SUM(amount) FROM orders_df GROUP BY 1").df()

# Polars works too
import polars as pl
orders_pl = pl.read_parquet('orders.parquet')
result = duckdb.sql("SELECT * FROM orders_pl WHERE amount > 100").pl()
```

### Parameterized Queries

```python
conn = duckdb.connect()

# Positional parameters
conn.execute(
    "SELECT * FROM orders WHERE customer_id = ? AND status = ?",
    [customer_id, "completed"]
).df()

# Named parameters
conn.execute(
    "SELECT * FROM orders WHERE customer_id = $id AND amount > $min_amount",
    {"id": 123, "min_amount": 50.0}
).df()
```

### Ingesting and Exporting

```python
import duckdb

conn = duckdb.connect('warehouse.duckdb')

# Load CSV
conn.execute("COPY orders FROM 'orders.csv' (FORMAT CSV, HEADER true)")

# Load Parquet
conn.execute("INSERT INTO orders SELECT * FROM read_parquet('orders.parquet')")

# Export to Parquet
conn.execute("COPY (SELECT * FROM orders) TO 'output.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY)")

# Export to CSV
conn.execute("COPY orders TO 'output.csv' (FORMAT CSV, HEADER true, DELIMITER ',')")

# Export to JSON
conn.execute("COPY orders TO 'output.json' (FORMAT JSON)")

# Pandas → DuckDB → Parquet
import pandas as pd
df = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
duckdb.sql("COPY (SELECT * FROM df) TO 'result.parquet' (FORMAT PARQUET)")
```

---

## Advanced SQL Features

### Window Functions

```sql
SELECT
    customer_id,
    order_id,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id ORDER BY created_at) AS running_total,
    RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rank_by_amount,
    LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY created_at) AS prev_amount,
    amount - LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY created_at) AS delta
FROM orders;
```

### ASOF JOIN (Time-series joins)

```sql
-- Join each event to the most recent price at that time
SELECT
    e.event_id,
    e.ts,
    p.price
FROM events e
ASOF JOIN prices p
ON e.symbol = p.symbol AND e.ts >= p.ts;
```

### Unnest / Array Operations

```sql
-- Unnest arrays
SELECT unnest([1, 2, 3]) AS n;

-- Generate series
SELECT * FROM generate_series(1, 10) AS t(n);
SELECT * FROM range('2025-01-01'::DATE, '2025-02-01'::DATE, INTERVAL 1 DAY) AS t(d);

-- List aggregation
SELECT customer_id, LIST(order_id) AS order_ids
FROM orders
GROUP BY customer_id;

-- Array functions
SELECT list_sum([1, 2, 3]);         -- 6
SELECT list_avg([1, 2, 3]);         -- 2.0
SELECT list_filter([1,2,3,4], x -> x > 2);  -- [3, 4]
SELECT list_transform([1,2,3], x -> x * 2); -- [2, 4, 6]
```

### Struct and Map Types

```sql
-- Struct
SELECT {'name': 'Alice', 'age': 30} AS person;
SELECT person.name FROM (SELECT {'name': 'Alice', 'age': 30} AS person);

-- Map
SELECT MAP(['a', 'b'], [1, 2]) AS m;
SELECT m['a'] FROM (SELECT MAP(['a', 'b'], [1, 2]) AS m);
```

### Pivot / Unpivot

```sql
-- Pivot
PIVOT orders
ON status
USING SUM(amount)
GROUP BY customer_id;

-- Unpivot
UNPIVOT monthly_sales
ON (jan, feb, mar)
INTO NAME month VALUE amount;
```

### Lateral Joins

```sql
-- Cross join with lateral unnest
SELECT o.order_id, item
FROM orders o, UNNEST(o.items) AS t(item);
```

### Sampling

```sql
-- Sample 10% of rows (fast, approximate)
SELECT * FROM orders USING SAMPLE 10%;
SELECT * FROM orders USING SAMPLE 10000 ROWS;
SELECT * FROM orders USING SAMPLE RESERVOIR(1000);
```

---

## Integrations

### dbt + DuckDB

```yaml
# profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: /tmp/dev.duckdb
      threads: 4
```

```bash
dbt run --target dev   # blazing fast local development!
```

### Pandas

```python
import duckdb, pandas as pd

# Query pandas DataFrames
df = pd.read_csv('large_file.csv')
result = duckdb.sql("SELECT region, SUM(revenue) FROM df GROUP BY region").df()
```

### Polars

```python
import duckdb, polars as pl

# DuckDB → Polars
df = duckdb.sql("SELECT * FROM read_parquet('data.parquet')").pl()

# Polars → DuckDB
polars_df = pl.read_parquet('data.parquet')
result = duckdb.sql("SELECT COUNT(*) FROM polars_df").fetchone()
```

### Apache Arrow

```python
import duckdb, pyarrow as pa

# DuckDB → Arrow (zero-copy)
arrow_table = duckdb.sql("SELECT * FROM orders").arrow()

# Arrow → DuckDB
result = duckdb.sql("SELECT SUM(amount) FROM arrow_table").fetchone()
```

### Jupyter / Notebooks

```python
# In Jupyter cells
%pip install duckdb jupysql

# Load jupysql magic
%load_ext sql
%sql duckdb:///:memory:

%%sql
SELECT customer_id, SUM(amount)
FROM read_parquet('orders.parquet')
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10
```

---

## DuckDB in Production

### MotherDuck (Managed DuckDB in the Cloud)

```python
import duckdb

# Connect to MotherDuck (cloud-hosted DuckDB)
conn = duckdb.connect('md:my_database?motherduck_token=...')

# Works exactly like local DuckDB + shares databases
conn.execute("SELECT * FROM orders LIMIT 10").df()
```

### DuckDB as ETL Engine

```python
import duckdb

def run_etl_pipeline():
    conn = duckdb.connect('warehouse.duckdb')

    # Extract + transform + load in SQL
    conn.execute("""
        CREATE OR REPLACE TABLE mart_revenue AS
        SELECT
            DATE_TRUNC('month', created_at) AS month,
            region,
            SUM(amount) AS revenue,
            COUNT(DISTINCT customer_id) AS customers,
            COUNT(*) AS orders
        FROM read_parquet('s3://bucket/raw/orders/**/*.parquet')
        WHERE status = 'completed'
        GROUP BY 1, 2
    """)

    # Export to Parquet
    conn.execute("""
        COPY mart_revenue
        TO 's3://bucket/marts/revenue/' (
            FORMAT PARQUET,
            PARTITION_BY (month),
            OVERWRITE_OR_IGNORE true
        )
    """)

    print("ETL complete!")
```

### Performance Tips

```sql
-- Use Parquet over CSV (10-100x faster for analytical queries)
-- Parquet: columnar, compressed, predicate pushdown

-- Use persistent database for repeated queries
-- (avoid re-reading files every time)
ATTACH 'warehouse.duckdb';
CREATE TABLE orders AS SELECT * FROM read_parquet('orders.parquet');
-- Now orders queries use in-memory columnar storage

-- Set memory limit
SET memory_limit = '8GB';

-- Set thread count
SET threads = 8;

-- Use approximate aggregations for huge datasets
SELECT approx_count_distinct(user_id) FROM events;  -- much faster than COUNT(DISTINCT)
SELECT percentile_disc(0.95) WITHIN GROUP (ORDER BY latency) FROM requests;
```

---

## DuckDB vs Spark vs Pandas

| Feature | DuckDB | PySpark | Pandas |
|---------|--------|---------|--------|
| **Data scale** | GBs–~500GB | TBs–PBs | MBs–GBs |
| **Setup** | Zero (pip install) | Complex (cluster) | Zero |
| **SQL** | Full SQL | SQL + Python | Limited |
| **Streaming** | No (batch) | Yes | No |
| **Speed (GB scale)** | ⚡ Very fast | Slower (overhead) | Fast |
| **Distributed** | No | Yes | No |
| **File formats** | Parquet, CSV, JSON, Delta, Iceberg | All | CSV, Parquet |
| **Cloud storage** | S3, GCS, ADLS | All | Limited |
| **Cost** | Free, no cluster | $$ (cluster compute) | Free |

**Rule of thumb:**
- `< 10GB` → Pandas (simple) or DuckDB (SQL)
- `10GB – 500GB` → **DuckDB** (fastest, easiest)
- `> 500GB` → **Spark**

---

## DuckDB in 2026

| Feature | Description |
|---------|-------------|
| **DuckDB 1.x** | Stable release, production-ready |
| **MotherDuck** | Managed cloud DuckDB — scale beyond single machine |
| **DuckDB Extensions** | Iceberg, Delta, Spatial, Substrait, Excel, MySQL, PostgreSQL |
| **WASM** | DuckDB running in the browser (duckdb-wasm) |
| **Polars integration** | Zero-copy interchange with Polars via Arrow |
| **Lakehouse queries** | Standard tool for querying Iceberg/Delta without Spark |
| **dbt-duckdb** | Fast local dbt development before deploying to warehouse |
| **Streaming** | DuckDB Appender for high-speed inserts from streaming sources |

### DuckDB Extensions Ecosystem

```sql
-- See installed extensions
SELECT * FROM duckdb_extensions();

-- Install and use
INSTALL httpfs;    LOAD httpfs;      -- S3/HTTP file reading
INSTALL iceberg;   LOAD iceberg;     -- Apache Iceberg
INSTALL delta;     LOAD delta;       -- Delta Lake
INSTALL spatial;   LOAD spatial;     -- Geospatial (PostGIS-compatible)
INSTALL postgres;  LOAD postgres;    -- PostgreSQL scanner
INSTALL mysql;     LOAD mysql;       -- MySQL scanner
INSTALL excel;     LOAD excel;       -- Read .xlsx files
INSTALL json;      LOAD json;        -- Enhanced JSON support
INSTALL tpch;      LOAD tpch;        -- TPC-H benchmark data
```

---

## Cheat Sheet

```sql
-- Read files
SELECT * FROM read_csv('file.csv');
SELECT * FROM read_parquet('file.parquet');
SELECT * FROM read_parquet('s3://bucket/path/**/*.parquet');
SELECT * FROM read_json('file.jsonl');
SELECT * FROM delta_scan('s3://bucket/delta/table/');
SELECT * FROM iceberg_scan('s3://bucket/iceberg/table/');

-- Write files
COPY (SELECT ...) TO 'output.parquet' (FORMAT PARQUET);
COPY (SELECT ...) TO 'output.csv' (FORMAT CSV, HEADER);

-- Create from query
CREATE TABLE t AS SELECT * FROM read_parquet('data.parquet');

-- Useful functions
DATE_TRUNC('month', ts)          -- truncate timestamp
DATE_DIFF('day', start, end)     -- difference in days
STRFTIME('%Y-%m', ts)            -- format timestamp
LIST_AGG(col, ',')               -- string aggregation
APPROX_COUNT_DISTINCT(col)       -- fast distinct count
PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) -- median
GENERATE_SERIES(1, 10)           -- number sequence
RANGE('2025-01-01', '2025-02-01', INTERVAL 1 DAY) -- date range

-- Extensions
INSTALL extension_name; LOAD extension_name;

-- Settings
SET memory_limit = '8GB';
SET threads = 8;
```

```python
import duckdb

# Connect
conn = duckdb.connect()              # in-memory
conn = duckdb.connect('file.duckdb') # persistent

# Query
df = conn.sql("SELECT ...").df()        # → Pandas
pl_df = conn.sql("SELECT ...").pl()     # → Polars
arrow = conn.sql("SELECT ...").arrow()  # → PyArrow
rows = conn.sql("SELECT ...").fetchall()

# Query DataFrames directly (zero-copy)
result = duckdb.sql("SELECT * FROM my_df WHERE col > 0").df()

# Parameterized
conn.execute("SELECT * FROM t WHERE id = ?", [42]).df()
```
