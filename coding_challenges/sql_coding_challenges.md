# SQL Coding Challenges for Interviews

This guide focuses on SQL interview rounds for ML, data, analytics, and platform roles.

Most real interview SQL is not about obscure syntax. It is about whether you can turn a business question into a correct query with clean logic.

---

## What Interviewers Usually Test

- correct joins and aggregation logic
- clean step-by-step query decomposition with CTEs
- window functions
- deduplication
- time-based analytics
- practical reasoning about data grain and nulls

---

## 1. Core Aggregations

### Concepts to know

- `GROUP BY`
- `HAVING`
- conditional aggregation with `CASE WHEN`
- choosing the correct grain before you aggregate

### Typical prompts

- revenue by customer
- average order value by region
- count active users by day
- number of failed jobs by service

### Example

```sql
SELECT
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING COUNT(*) >= 3;
```

### What to watch

- Are you aggregating at the right grain?
- Does `WHERE` filter rows before or after grouping?
- Should the filter belong in `HAVING` instead?

---

## 2. Joins and Null Handling

### Concepts to know

- inner join
- left join
- join cardinality
- `COALESCE`
- avoiding accidental row multiplication

### Typical prompts

- users with no orders
- orders with customer metadata
- products never purchased
- combine fact and dimension tables

### Example

```sql
SELECT
    c.customer_id,
    c.customer_name,
    COALESCE(SUM(o.amount), 0) AS lifetime_value
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
GROUP BY c.customer_id, c.customer_name;
```

### What to say in the interview

Mention the join grain explicitly:

- `customers` is one row per customer
- `orders` is many rows per customer
- left join preserves customers with zero orders

---

## 3. Deduplication and Latest-Row Logic

### Concepts to know

- `row_number()`
- partitioning by business key
- ordering by update timestamp

### Typical prompts

- keep the latest status per user
- remove duplicate transactions
- select the most recent model run

### Example

```sql
WITH ranked AS (
    SELECT
        user_id,
        status,
        updated_at,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY updated_at DESC
        ) AS rn
    FROM user_status_history
)
SELECT
    user_id,
    status,
    updated_at
FROM ranked
WHERE rn = 1;
```

### Why it matters

This is one of the most common warehouse and dbt patterns.

Related repo reading:

- [dbt Interview Q&A](../data_engineering/interview_dbt.md)

---

## 4. Ranking and Top-N Per Group

### Concepts to know

- `row_number()`
- `rank()`
- `dense_rank()`
- top-N within each partition

### Typical prompts

- top 3 products per category
- highest revenue user per country
- top model per experiment

### Example

```sql
WITH ranked_products AS (
    SELECT
        category,
        product_id,
        revenue,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY revenue DESC
        ) AS rn
    FROM product_revenue
)
SELECT
    category,
    product_id,
    revenue
FROM ranked_products
WHERE rn <= 3;
```

---

## 5. Window Functions

### Concepts to know

- running totals
- moving averages
- `lag()` and `lead()`
- partitioned calculations

### Typical prompts

- running revenue by day
- compare each event to the previous event
- rolling 7-day average
- detect changes in user behavior

### Example

```sql
SELECT
    event_date,
    daily_signups,
    SUM(daily_signups) OVER (
        ORDER BY event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_signups,
    AVG(daily_signups) OVER (
        ORDER BY event_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg
FROM daily_signup_metrics;
```

### Related repo reading

- [DuckDB Complete Guide](../data_engineering/intro_duckdb.md)

---

## 6. Time Bucketing and Cohort Logic

### Concepts to know

- `date_trunc`
- grouping by day, week, or month
- cohort assignment
- retention windows

### Typical prompts

- monthly active users
- week-1 retention
- cohort revenue by signup month
- daily model inference counts

### Example: monthly active users

```sql
SELECT
    DATE_TRUNC('month', event_time) AS month_start,
    COUNT(DISTINCT user_id) AS monthly_active_users
FROM user_events
WHERE event_name = 'session_start'
GROUP BY DATE_TRUNC('month', event_time)
ORDER BY month_start;
```

### Example: next-event gap

```sql
SELECT
    user_id,
    event_time,
    LAG(event_time) OVER (
        PARTITION BY user_id
        ORDER BY event_time
    ) AS previous_event_time
FROM user_events;
```

---

## 7. Funnel and Conversion Queries

### Concepts to know

- staged CTEs
- one row per user or session before final aggregation
- conditional counting

### Typical prompts

- view -> click -> purchase funnel
- signup -> activation -> subscription
- experiment exposure -> conversion

### Example

```sql
WITH user_steps AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_name = 'view_product' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS added_to_cart,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM events
    GROUP BY user_id
)
SELECT
    SUM(viewed) AS users_viewed,
    SUM(added_to_cart) AS users_added_to_cart,
    SUM(purchased) AS users_purchased
FROM user_steps;
```

---

## 8. Clean Query Design with CTEs

### Concepts to know

- one transformation per CTE
- naming CTEs by purpose
- separate filtering, enrichment, and final aggregation

### Typical prompts

- messy analytics question with multiple joins
- retention question with several business rules
- interviewers asking you to "make it cleaner"

### Example structure

```sql
WITH filtered_orders AS (
    SELECT *
    FROM orders
    WHERE status = 'completed'
),
customer_revenue AS (
    SELECT
        customer_id,
        SUM(amount) AS revenue
    FROM filtered_orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    revenue
FROM customer_revenue
ORDER BY revenue DESC;
```

### What interviewers like

- readable stages
- correct naming
- clear grain in each step

---

## Practice Checklist

### Easy

- aggregation by one dimension
- left join with null handling
- distinct counts
- simple date filters

### Medium

- dedup with `row_number()`
- top-N per group
- running totals
- lag/lead comparisons
- clean CTE decomposition

### Harder but high value

- retention
- funnels
- sessionization
- cohort analysis
- performance reasoning on large tables

---

## Common Mistakes

- using `COUNT(*)` when `COUNT(DISTINCT user_id)` is needed
- filtering after a left join in a way that turns it into an inner join
- forgetting the data grain before aggregation
- using `rank()` when `row_number()` is required
- mixing event-level rows with user-level metrics without a normalization step
- ordering windows incorrectly

---

## Good Interview Habits

1. State the table grain first.
2. State the output grain next.
3. Say whether you need a join, a dedup step, a window, or a CTE pipeline.
4. Write the query in layers.
5. Validate it with one small example in words.

---

## Repo Concepts to Pair With Practice

- [DuckDB Complete Guide](../data_engineering/intro_duckdb.md)
- [dbt Interview Q&A](../data_engineering/interview_dbt.md)
- [Data Engineering for AI](../data_engineering/intro_data_engineering_for_ai.md)
- [Statistics & Probability Guide](../classical_ml/intro_statistics_probability.md)

If you can solve medium SQL questions quickly and explain the underlying grain, windows, and business logic, you are already strong for many ML-adjacent interview loops.
