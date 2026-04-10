# Coding Challenges for ML, AI, Python, and SQL Interviews

This track is for interview rounds where you are asked to code, not just explain concepts.

It complements the rest of the repo by focusing on:

- Python problem solving under time pressure
- SQL query writing for analytics and data workflows
- algorithm selection and complexity trade-offs
- practice topics that show up in ML engineer, data engineer, analytics engineer, and AI platform interviews

---

## What to Practice First

### Python coding rounds

Prioritize these topics in order:

1. Arrays, strings, hash maps, sets
2. Sorting, heaps, binary search
3. Stacks, queues, recursion
4. Trees and graphs with BFS and DFS
5. Sliding window, two pointers, prefix sums
6. Dynamic programming basics
7. Matrix and tabular manipulation patterns
8. Practical data-processing utilities in Python

### SQL coding rounds

Prioritize these topics in order:

1. `SELECT`, `WHERE`, `GROUP BY`, `HAVING`
2. Joins and null handling
3. Window functions
4. CTEs and layered query design
5. Deduplication and ranking
6. Time-series aggregation and date bucketing
7. Funnel, retention, and cohort patterns
8. Performance reasoning and data-model awareness

---

## How This Connects to the Rest of the Repo

Use this track with these concept-heavy guides:

- [Statistics & Probability Guide](../classical_ml/intro_statistics_probability.md)
- [Classical ML Overview](../classical_ml/README.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [DuckDB Complete Guide](../data_engineering/intro_duckdb.md)
- [dbt Interview Q&A](../data_engineering/interview_dbt.md)
- [Backend System Design Interview Guide](../system_design/backend_system_design_interview_guide.md)

Why these matter:

- Python challenges often test the same thinking you need for feature engineering, preprocessing, metrics, and pipeline code.
- SQL challenges often mirror warehouse analytics, dbt modeling, experimentation analysis, and event data work.
- Backend interview loops regularly combine algorithmic reasoning with system trade-offs.

---

## Interview Patterns by Role

| Role | Likely coding focus | What to master |
|---|---|---|
| ML Engineer | Python, arrays, metrics, data transforms | hash maps, heaps, matrices, feature logic, complexity |
| AI / Applied AI Engineer | Python, APIs, parsing, async workflows | strings, recursion, queues, graph traversal, data shaping |
| Data Engineer | SQL and Python scripting | joins, windows, CTEs, dedup, batching, file transforms |
| Analytics Engineer | SQL-heavy | aggregations, windows, date logic, dbt-style layered queries |
| MLOps / Platform | Python and systems-flavored coding | queues, retries, parsing, state handling, complexity trade-offs |

---

## Practice Strategy

### 30-minute Python block

1. Solve one easy-to-medium array or string problem
2. Solve one medium graph, heap, or interval problem
3. Review complexity and edge cases out loud

### 30-minute SQL block

1. Write one aggregation query
2. Write one window-function query
3. Rewrite one query with a cleaner CTE structure

### Weekly review

- revisit mistakes
- classify them as logic, syntax, edge case, or complexity mistakes
- write the shortest correct explanation you would give in an interview

---

## What Good Interview Performance Looks Like

You do not need the fanciest solution first. You need to show:

- a correct baseline
- awareness of time and space complexity
- clarity on edge cases
- ability to improve the solution step by step
- readable Python or SQL under pressure

In practice:

- start with the brute-force version if needed
- state its complexity clearly
- then optimize

---

## Recommended Order in This Track

1. [Coding Challenges Overview](./README.md)
2. [Python Coding Challenges](./python_coding_challenges.md)
3. [SQL Coding Challenges](./sql_coding_challenges.md)

---

## Quick Self-Assessment

You are in good shape if you can do the following without much hesitation:

- use a dictionary or set immediately when lookup speed matters
- explain when to choose sorting versus a heap
- write BFS and DFS without searching for syntax
- use `row_number()`, `rank()`, `lag()`, and running aggregates in SQL
- deduplicate rows with a window function
- reason about time-bucketed analytics queries

If not, start with the two practice guides in this track and follow the linked concept guides.
