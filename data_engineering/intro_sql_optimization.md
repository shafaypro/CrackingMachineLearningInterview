# SQL Query Optimization

Writing correct SQL gets you through the coding screen. Making it fast on a billion rows is the Data Engineer interview, and it rests on one skill: reading an execution plan and knowing which operator is the problem.

---

## Table of Contents
1. [How a Query Actually Runs](#how-a-query-actually-runs)
2. [Reading an Execution Plan](#reading-an-execution-plan)
3. [Indexes](#indexes)
4. [Join Algorithms](#join-algorithms)
5. [Predicate Pushdown and Sargability](#predicate-pushdown-and-sargability)
6. [Statistics and Cardinality Estimation](#statistics-and-cardinality-estimation)
7. [Window Functions and Aggregation](#window-functions-and-aggregation)
8. [Partitioning and Clustering](#partitioning-and-clustering)
9. [Columnar Warehouses vs OLTP](#columnar-warehouses-vs-oltp)
10. [Common Rewrites](#common-rewrites)
11. [A Debugging Workflow](#a-debugging-workflow)
12. [Interview Q&A](#interview-qa)
13. [Common Pitfalls](#common-pitfalls)
14. [Related Topics](#related-topics)

---

## How a Query Actually Runs

SQL is declarative, so the engine decides the execution. Logical evaluation order differs from the written order, and knowing it resolves a surprising number of bugs:

```
FROM / JOIN  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  DISTINCT  →  ORDER BY  →  LIMIT
```

Two consequences that come up constantly:

- **You cannot reference a `SELECT` alias in `WHERE`**, because `WHERE` runs first. (Some engines allow it in `GROUP BY`/`ORDER BY` as a convenience.)
- **`WHERE` filters rows, `HAVING` filters groups.** Putting a non-aggregate condition in `HAVING` works but filters *after* grouping — much more expensive than filtering before.

The optimizer then rewrites this logical plan into a physical one: which index to use, which join algorithm, what order to join tables, whether to parallelize.

---

## Reading an Execution Plan

This is the skill. Everything else follows from it.

```sql
-- Postgres: ANALYZE actually runs it and reports real numbers
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT c.name, count(*)
FROM orders o JOIN customers c ON c.id = o.customer_id
WHERE o.created_at >= '2026-01-01'
GROUP BY c.name;
```

```
HashAggregate  (cost=... rows=1000 width=..) (actual time=850 rows=48200 loops=1)
  ->  Hash Join  (cost=..) (actual time=620 rows=2100000 loops=1)
        Hash Cond: (o.customer_id = c.id)
        ->  Seq Scan on orders o  (actual time=310 rows=2100000 loops=1)
              Filter: (created_at >= '2026-01-01')
              Rows Removed by Filter: 47900000
        ->  Hash  (actual time=95 rows=50000 loops=1)
              ->  Seq Scan on customers c
```

**What to look for, in priority order:**

| Signal | Meaning | Action |
|---|---|---|
| **Estimated rows ≫ or ≪ actual rows** | Bad cardinality estimate → wrong plan | Update statistics; check correlated predicates |
| **Seq Scan on a large table with a selective filter** | Missing or unusable index | Add index; check sargability |
| **Rows Removed by Filter is huge** | Reading far more than needed | Push the predicate down; partition |
| **Nested Loop with a large outer side** | Quadratic behaviour | Force hash join; fix estimates |
| **Sort spilling to disk** (`external merge`) | `work_mem` too small | Raise memory, or avoid the sort |
| **`loops=N` with N large** | Inner side executed N times | Usually a nested-loop problem |

The single most valuable habit: **compare estimated to actual rows at every node**. A plan is only as good as its estimates, and almost every catastrophically slow query traces back to an estimate that was off by orders of magnitude.

`BUFFERS` shows actual I/O — `shared hit` is cache, `shared read` is disk. A query that looks fast on a warm cache can be terrible cold.

---

## Indexes

**B-tree** is the default and handles equality, ranges, and sorted retrieval.

```sql
CREATE INDEX idx_orders_customer_created ON orders (customer_id, created_at);
```

### Composite index column order

The rule: **equality columns first, then range, then columns used only for ordering.** An index on `(a, b)` supports lookups on `a` and on `(a, b)`, but **not** on `b` alone — the same reason a phone book sorted by (surname, first name) can't find everyone named "James".

```sql
-- Supports: WHERE customer_id = 5
--           WHERE customer_id = 5 AND created_at > '2026-01-01'
-- Does NOT support: WHERE created_at > '2026-01-01'    (leading column missing)
```

### Covering indexes

If the index contains every column the query needs, the engine never touches the table — an **index-only scan**:

```sql
CREATE INDEX idx_cover ON orders (customer_id, created_at) INCLUDE (amount, status);
```

This can be an order-of-magnitude win on wide tables because it avoids the random heap lookups.

### Partial indexes

Index only the rows you query, which is smaller and faster to maintain:

```sql
CREATE INDEX idx_pending ON orders (created_at) WHERE status = 'pending';
```

Excellent when a status column is heavily skewed — indexing 0.1% of rows instead of all of them.

### Other index types

| Type | For |
|---|---|
| **Hash** | Equality only; rarely worth it over B-tree |
| **GIN** | Arrays, JSONB, full-text search |
| **GiST / SP-GiST** | Geometric, range, nearest-neighbour |
| **BRIN** | Very large tables with natural physical ordering (append-only time series) — tiny index, coarse filtering |
| **Bitmap** (warehouses) | Low-cardinality columns |

### The cost of indexes

Every index slows `INSERT`, `UPDATE`, and `DELETE`, consumes storage, and must be maintained. Unused indexes are pure overhead — check `pg_stat_user_indexes` for `idx_scan = 0` and drop them. "Add an index" is not a free answer, and saying so unprompted is a good signal.

---

## Join Algorithms

| Algorithm | Cost | Chosen when |
|---|---|---|
| **Nested Loop** | `O(n · m)`, or `O(n · log m)` with an index on the inner side | Small outer input, indexed inner |
| **Hash Join** | `O(n + m)` | Large unsorted inputs, equality join, hash table fits memory |
| **Merge Join** | `O(n log n + m log m)`, or `O(n + m)` if pre-sorted | Both inputs already sorted on the key; range joins |

**Nested loop is the dangerous one.** It's optimal when the outer side has 10 rows and the inner has an index. It's catastrophic when the optimizer *estimated* 10 rows and there are actually 10 million — the same plan becomes 10 million index lookups. That mis-estimate is the most common cause of a query that ran in 50 ms yesterday and 50 minutes today.

**Hash join** builds a hash table on the smaller side and probes with the larger. If the build side doesn't fit in memory it spills to disk in batches, which is much slower — watch for that in the plan.

**Join order matters enormously.** With `n` tables there are factorially many orders; optimizers search heuristically and give up on very large joins. The practical rule: filter early, join the most selective tables first, so intermediate results stay small.

---

## Predicate Pushdown and Sargability

**Sargable** (Search ARGument able) means the predicate can use an index. Wrapping an indexed column in a function destroys that:

```sql
-- NOT sargable: function on the column
WHERE DATE(created_at) = '2026-01-01'
WHERE UPPER(email) = 'A@B.COM'
WHERE amount * 2 > 100
WHERE created_at::text LIKE '2026%'

-- Sargable: column stays bare, transformation moves to the constant
WHERE created_at >= '2026-01-01' AND created_at < '2026-01-02'
WHERE email = LOWER('A@B.COM')       -- with a lower(email) expression index
WHERE amount > 50
```

This is the most common single cause of "I added an index and nothing got faster", and it comes up in interviews constantly.

**Expression indexes** rescue cases where the function is genuinely needed:

```sql
CREATE INDEX idx_lower_email ON users (LOWER(email));
-- now WHERE LOWER(email) = '...' can use it
```

**Leading wildcards** also defeat B-trees — `LIKE '%term'` cannot use an index because the prefix is unknown. Use a trigram index (`pg_trgm`) or full-text search instead.

**`OR` across different columns** often prevents index usage; rewriting as `UNION ALL` of two indexed queries can be dramatically faster.

---

## Statistics and Cardinality Estimation

The optimizer picks plans from estimated row counts, derived from sampled statistics. Bad statistics produce bad plans regardless of indexing.

```sql
ANALYZE orders;                                  -- refresh statistics
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 1000;   -- finer histogram
```

**Correlated columns** are the classic estimation failure. The optimizer assumes independence, so for `WHERE city = 'Paris' AND country = 'France'` it multiplies the two selectivities and estimates far too few rows — when in reality the predicates are nearly redundant. Postgres addresses this with extended statistics:

```sql
CREATE STATISTICS stat_city_country (dependencies, ndistinct)
  ON city, country FROM addresses;
```

**Skew** is the other estimation trap. If 95% of orders have `status = 'complete'`, an index on `status` helps for the rare values and is useless for the common one — and the optimizer, seeing average selectivity, may choose wrongly for both.

---

## Window Functions and Aggregation

Window functions avoid self-joins and are usually far faster:

```sql
-- Slow: correlated subquery, executed per row
SELECT o.*, (SELECT count(*) FROM orders x WHERE x.customer_id = o.customer_id)
FROM orders o;

-- Fast: single pass with a window
SELECT o.*, count(*) OVER (PARTITION BY customer_id)
FROM orders o;
```

**Reuse the window definition** so the engine sorts once rather than repeatedly:

```sql
SELECT
  customer_id,
  row_number() OVER w  AS rn,
  sum(amount)  OVER w  AS running_total,
  lag(amount)  OVER w  AS prev_amount
FROM orders
WINDOW w AS (PARTITION BY customer_id ORDER BY created_at);
```

**Frame clauses** change both semantics and cost. `ROWS BETWEEN` is cheaper than `RANGE BETWEEN`, and the default frame for an ordered window is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` — which surprises people expecting `ROWS`, particularly with ties.

**Top-N per group** — the perennial interview question, with a performance angle:

```sql
-- Portable
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY customer_id ORDER BY amount DESC) rn
  FROM orders
) t WHERE rn <= 3;

-- Postgres: DISTINCT ON is faster for top-1
SELECT DISTINCT ON (customer_id) *
FROM orders ORDER BY customer_id, amount DESC;

-- Best when there are few groups: LATERAL with an index seek per group
SELECT c.id, o.*
FROM customers c
CROSS JOIN LATERAL (
  SELECT * FROM orders o WHERE o.customer_id = c.id
  ORDER BY amount DESC LIMIT 3
) o;
```

The `LATERAL` form wins when groups are few and there's an index on `(customer_id, amount DESC)` — it does a small index seek per group instead of sorting the whole table.

---

## Partitioning and Clustering

**Partitioning** splits a table physically so queries can skip whole partitions (*partition pruning*):

```sql
CREATE TABLE events (id bigint, created_at timestamptz, payload jsonb)
PARTITION BY RANGE (created_at);

CREATE TABLE events_2026_01 PARTITION OF events
  FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

Pruning only works if the query filters on the **partition key**. A query filtering on `user_id` when partitioned by date scans everything — so the partition key must match the dominant access pattern.

The other big benefit: dropping an old partition is instant metadata work, versus a `DELETE` that has to touch every row and bloat the table.

**Over-partitioning is a real failure mode.** Thousands of tiny partitions make planning slower and each scan less efficient; aim for partitions large enough to be worth the bookkeeping.

**Clustering / sort keys** control physical row order within storage, which drives how much data warehouses can skip:

```sql
-- Snowflake
ALTER TABLE events CLUSTER BY (created_at, customer_id);
-- BigQuery
CREATE TABLE t PARTITION BY DATE(created_at) CLUSTER BY customer_id AS ...;
```

---

## Columnar Warehouses vs OLTP

The optimization mindset differs, and interviewers check that you know which world you're in.

| | OLTP (Postgres, MySQL) | Columnar (Snowflake, BigQuery, Redshift) |
|---|---|---|
| Storage | Row-oriented | Column-oriented |
| Optimize for | Point lookups, small transactions | Large scans and aggregations |
| Main lever | **Indexes** | **Partition pruning + clustering** |
| `SELECT *` | Mildly wasteful | **Very expensive** — reads every column |
| Row count filters | Index seek | File/block skipping via min-max stats |
| Joins | Index nested loop common | Hash joins, broadcast vs shuffle |
| Cost model | I/O and CPU | **Bytes scanned** (often literally billed) |

In a warehouse, **column selection is the primary optimization** — `SELECT *` on a 200-column table reads 200 columns' worth of data when you needed three. Partition pruning and clustering replace indexes as the mechanism for reading less.

**Broadcast vs shuffle join** in distributed engines: if one side is small it's broadcast to every node (cheap); otherwise both sides are shuffled across the network by join key (expensive). Data **skew** — one key with a disproportionate share of rows — makes one task run far longer than the rest, which is the classic Spark/warehouse performance problem. Salting the key is the standard fix.

---

## Common Rewrites

```sql
-- 1. Existence check: EXISTS short-circuits, COUNT scans everything
-- Slow
SELECT * FROM customers c WHERE (SELECT count(*) FROM orders o WHERE o.customer_id = c.id) > 0;
-- Fast
SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

-- 2. NOT IN with nullable columns is both a correctness AND performance trap
-- WRONG: returns zero rows if the subquery contains any NULL
SELECT * FROM customers WHERE id NOT IN (SELECT customer_id FROM orders);
-- Correct and faster
SELECT * FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

-- 3. Aggregate before joining when the join fans out
SELECT c.id, c.name, o.total
FROM customers c
JOIN (SELECT customer_id, sum(amount) total FROM orders GROUP BY customer_id) o
  ON o.customer_id = c.id;

-- 4. Deep OFFSET is O(offset) — use keyset pagination
-- Slow at page 10,000
SELECT * FROM orders ORDER BY id LIMIT 20 OFFSET 200000;
-- Fast: seek directly
SELECT * FROM orders WHERE id > :last_seen_id ORDER BY id LIMIT 20;

-- 5. UNION deduplicates (sort/hash); UNION ALL does not
SELECT a FROM t1 UNION ALL SELECT a FROM t2;   -- when duplicates are fine or impossible
```

**The `NOT IN` NULL trap** is worth internalizing: if the subquery returns any NULL, `x NOT IN (...)` evaluates to UNKNOWN for every row and the result is empty. It's a correctness bug that looks like a data problem, and `NOT EXISTS` avoids it entirely while usually planning better.

---

## A Debugging Workflow

1. **Reproduce and measure.** Get the actual runtime, on warm and cold cache.
2. **`EXPLAIN ANALYZE`.** Find the node consuming the most actual time.
3. **Check estimates against actuals** at that node. Off by 100×? Statistics problem.
4. **Ask what it's reading.** Large `Rows Removed by Filter`, a Seq Scan on a big table, or a spilling sort each point somewhere specific.
5. **Check sargability.** Is a function wrapping an indexed column?
6. **Consider the index.** Does one exist, is the column order right, would a covering or partial index help?
7. **Consider a rewrite.** `EXISTS` instead of `COUNT`, aggregate before join, window instead of correlated subquery.
8. **Re-measure.** Then confirm the plan actually changed — sometimes it doesn't.

Optimize the **plan node that dominates actual time**, not the part of the SQL that looks ugliest.

---

## Interview Q&A

#### What does it mean for a predicate to be sargable?

Sargable means the predicate can use an index to seek rather than forcing a scan. The rule is that the indexed column must appear bare on one side of the comparison — the moment you wrap it in a function or arithmetic, the engine can no longer map the predicate onto the index's sort order.

So `WHERE DATE(created_at) = '2026-01-01'` cannot use an index on `created_at`, while the equivalent `WHERE created_at >= '2026-01-01' AND created_at < '2026-01-02'` can. Same for `UPPER(email) = ...` versus comparing against a lowered constant, and `amount * 2 > 100` versus `amount > 50`.

This is the most common reason someone adds an index and sees no improvement. When the function is genuinely required, an expression index on `LOWER(email)` restores index usage. Leading wildcards (`LIKE '%foo'`) are the related case — no known prefix means no B-tree seek, so you need trigram or full-text indexing.

#### Explain the three join algorithms and when each is chosen.

**Nested loop**: for each row of the outer input, look up matches in the inner. `O(n·m)` naively, but `O(n·log m)` when the inner side has an index. Chosen when the outer input is small.

**Hash join**: build a hash table on the smaller input, probe with the larger. `O(n+m)`, the workhorse for large equality joins. Degrades sharply if the build side doesn't fit in memory and has to spill to disk in batches.

**Merge join**: sort both inputs on the key and walk them together. `O(n log n + m log m)`, but `O(n+m)` if the inputs are already sorted — so it's the natural choice when an index provides the ordering for free, and it also handles range joins that hash join can't.

The dangerous one is nested loop, because its cost depends entirely on the outer row count being small. If the optimizer estimates 10 rows and there are actually 10 million, the same plan turns into 10 million lookups. That estimate error is the most common cause of a query that was fast yesterday and is unusable today.

#### A query got 100× slower with no code change. What happened?

Almost always the plan changed, and almost always because estimates changed.

I'd start with `EXPLAIN ANALYZE` and compare estimated to actual rows at each node. The usual causes: **stale statistics** after a large data load, so the optimizer is planning against an old distribution — fixed by `ANALYZE`. **Data growth crossing a threshold**, where a nested loop that was correct at 1,000 rows is catastrophic at 10 million. **Skew**, where a previously uniform column became heavily concentrated. **Parameter sniffing**, where a cached plan built for a selective parameter value is reused for a non-selective one.

I'd also check the boring possibilities: an index was dropped, a cold cache (compare `shared hit` versus `shared read` in `BUFFERS`), or resource contention from something else on the box. But statistics and estimate drift account for most of these, and `ANALYZE` plus a plan comparison usually identifies it in minutes.

#### Why can't an index on `(a, b)` serve a query filtering only on `b`?

Because a composite index is sorted lexicographically by `a` first, then `b` within each `a`. Rows with a given `b` value are scattered across the whole index rather than contiguous, so there's no range to seek to.

The phone book analogy makes it concrete: a directory sorted by surname then first name lets you find "Smith", or "Smith, James", but finding everyone named James means reading the entire book.

Practically: put equality columns first, then range columns, then ordering columns. If you frequently filter on `b` alone, you need a separate index — though some engines can do an "index skip scan" when `a` has very few distinct values, which partly rescues the case.

#### How is optimizing a columnar warehouse different from optimizing Postgres?

Different levers, because the storage model is different.

In row-store OLTP, the win is finding the few rows you want without touching the rest, so **indexes** are the primary tool and point lookups are the target workload.

In a columnar warehouse, data is stored and compressed per column, and queries scan large ranges. There typically are no indexes in the OLTP sense. The levers are **reading fewer columns** — `SELECT *` on a 200-column table is genuinely expensive because it reads all 200 — and **reading fewer blocks**, achieved through partition pruning on the partition key and clustering/sort keys so min-max statistics let the engine skip files entirely.

The cost model also differs: warehouses often bill by **bytes scanned**, so optimization is directly a cost exercise rather than only a latency one. And in distributed engines you additionally care about join strategy (broadcast the small side versus shuffling both) and **data skew**, where one hot key makes a single task run far longer than the rest — usually addressed by salting the key.

#### What's wrong with `NOT IN` on a subquery?

Two things, one of them a silent correctness bug.

**Correctness**: if the subquery returns even one NULL, `x NOT IN (subquery)` evaluates to UNKNOWN for every row, and the query returns zero rows. Three-valued logic means `x <> NULL` is never true, so the "not in" can never be satisfied. This looks like missing data rather than a bug, which is why it's so pernicious.

**Performance**: `NOT IN` with a subquery often can't be transformed into an anti-join, so the engine may materialize and re-check the whole subquery.

`NOT EXISTS` fixes both — it's NULL-safe by construction and typically plans as a proper anti-join. `LEFT JOIN ... WHERE right.key IS NULL` is the third equivalent form and sometimes plans better still.

#### How would you paginate through 10 million rows efficiently?

Not with `OFFSET`. `LIMIT 20 OFFSET 200000` requires the engine to generate and discard 200,000 rows before returning anything, so cost grows linearly with page depth — the last pages become unusable.

**Keyset (seek) pagination** instead: remember the sort key of the last row returned and use it as a filter — `WHERE id > :last_seen ORDER BY id LIMIT 20`. With an index on the sort key this is an index seek, so every page costs the same regardless of depth.

The trade-offs worth stating: you can't jump to an arbitrary page number, only forward and backward; the sort key must be unique or you need a tiebreaker (`(created_at, id)`) or rows can be skipped or duplicated; and it's a different API contract, so it's a design decision rather than a drop-in change.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Function wrapping an indexed column | Predicate becomes non-sargable; index unused | Rewrite as a range; or add an expression index |
| Assuming `(a, b)` index helps `WHERE b = ...` | Leading column missing; no seek possible | Separate index, or reorder columns |
| `NOT IN` on a nullable subquery | Returns zero rows silently | `NOT EXISTS` |
| Deep `OFFSET` pagination | Cost grows with page depth | Keyset pagination |
| `SELECT *` in a columnar warehouse | Reads every column; billed on bytes scanned | Select only needed columns |
| Trusting `EXPLAIN` without `ANALYZE` | Estimates, not reality | `EXPLAIN ANALYZE` and compare est. vs actual |
| Adding indexes without measuring | Slows writes, wastes storage | Check `idx_scan`; drop unused indexes |
| Partitioning on a column you don't filter by | No pruning; full scan anyway | Partition key = dominant filter |
| Thousands of tiny partitions | Planning overhead exceeds the benefit | Fewer, larger partitions |
| Filtering in `HAVING` instead of `WHERE` | Filters after grouping; far more work | Non-aggregate conditions go in `WHERE` |
| Correlated subquery per row | Executed once per outer row | Window function or join |
| Ignoring skew in distributed joins | One task dominates runtime | Salt the hot key; broadcast small sides |
| `UNION` when duplicates are impossible | Pays for a needless dedup sort | `UNION ALL` |

---

## Related Topics

- [SQL Coding Challenges](../coding_challenges/sql_coding_challenges.md)
- [Data Modeling](./data-modeling.md)
- [Data Architecture](./data-architecture.md)
- [DuckDB](./intro_duckdb.md)
- [Apache Spark](./intro_apache_spark.md)
- [Apache Iceberg](./intro_apache_iceberg.md)
- [Delta Lake](./intro_delta_lake.md)
- [dbt](./intro_dbt.md)
- [Pandas and NumPy Challenges](../coding_challenges/pandas_numpy_challenges.md)
- [Backend System Design Interview Guide](../system_design/backend_system_design_interview_guide.md)
