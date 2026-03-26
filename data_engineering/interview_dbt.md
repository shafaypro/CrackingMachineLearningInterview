# dbt Interview Q&A — Mid-level · Senior · Lead

> Complete questions and answers for Mid-Senior and Lead-level dbt interviews.
> Covers design, Jinja/macros, performance, testing, CI/CD, and scenario questions.

---

## Table of Contents

- [Mid-level / Engineer — Design & Modelling](#mid-level--engineer--design--modelling)
- [Mid–Senior — Jinja, Macros & Packages](#midsenior--jinja-macros--packages)
- [Senior / Lead — Performance, Testing & CI/CD](#senior--lead--performance-testing--cicd)
- [Scenario / Situational Questions](#scenario--situational-questions)

---

## Mid-level / Engineer — Design & Modelling

---

### Q1. How do incremental models work? What strategies exist and how do you choose?

**Answer:**

An incremental model tells dbt: "on the first run, build the full table; on every subsequent run, only process new or changed rows and merge them in." This avoids a full table rebuild every time, which is critical for large datasets.

The model uses the `is_incremental()` macro to apply a filter during incremental runs:

```sql
{{ config(
    materialized = 'incremental',
    unique_key   = 'event_id'
) }}

select
    event_id,
    user_id,
    event_type,
    cast(occurred_at as timestamp) as occurred_at
from {{ source('raw', 'events') }}

{% if is_incremental() %}
    where occurred_at > (select max(occurred_at) from {{ this }})
{% endif %}
```

**The four incremental strategies:**

| Strategy | What it does | Best for |
|---|---|---|
| `append` | Insert new rows only — no deduplication | Immutable event logs where duplicates are impossible |
| `merge` | Upsert: update existing rows, insert new ones using `unique_key` | Most use cases — orders, users, any mutable entity |
| `delete+insert` | Delete all rows matching `unique_key`, re-insert them | Warehouses that don't support `MERGE` natively |
| `insert_overwrite` | Replace entire partitions | BigQuery/Spark partitioned tables |

**How to choose:**
- Data never changes after insert → `append`
- Rows can be updated → `merge`
- Warehouse is Redshift (no native MERGE) → `delete+insert`
- BigQuery with date partitions → `insert_overwrite`

**Force a full rebuild:**
```bash
dbt run -s my_model --full-refresh
```

---

### Q2. Append vs merge vs delete+insert — what are the trade-offs?

**Answer:**

**`append`**
- Fastest — just inserts new rows, no lookups
- No deduplication — if the source resends a row, you get a duplicate
- Only safe when data is truly immutable (e.g., raw click events that never change)

**`merge`**
- The most common strategy. Uses a `unique_key` to match existing rows
- Updates changed rows, inserts new ones in a single atomic operation
- Slightly slower than append due to the match step
- Requires the warehouse to support `MERGE` (Snowflake, BigQuery, Databricks do; Redshift does not)

```sql
{{ config(
    materialized         = 'incremental',
    unique_key           = 'order_id',
    incremental_strategy = 'merge'
) }}
```

**`delete+insert`**
- Deletes all rows where `unique_key` matches, then re-inserts
- Semantically equivalent to merge but works on warehouses without `MERGE`
- Slightly less efficient than merge because delete + insert = 2 operations
- Good fallback for Redshift

**Key trade-off:** append is fastest but dangerous if source data is mutable. Merge is the right default unless you have a specific reason not to use it.

---

### Q3. How do you handle late-arriving data in an incremental model?

**Answer:**

Late-arriving data is the most common production problem with incremental models. It happens when events are recorded in the source *after* they actually occurred — e.g., a mobile event buffered offline, a Kafka consumer catching up, or a source DB replication lag.

**The problem:**
```sql
-- This filter misses events that arrived late:
where occurred_at > (select max(occurred_at) from {{ this }})
```

If an event from yesterday arrives today, `occurred_at` is yesterday — the filter excludes it — and it never gets loaded.

**The fix — lookback window:**
```sql
{% if is_incremental() %}
    where occurred_at > (
        select dateadd(day, -3, max(occurred_at))   -- look back 3 days
        from {{ this }}
    )
{% endif %}
```

This reprocesses the last 3 days on every run, catching anything that arrived late. The `merge` strategy with `unique_key` ensures rows are upserted rather than duplicated.

**How to choose the window size:**
- Look at your source system's SLA — how late can data arrive?
- Check your historical data for the maximum observed lag
- Balance reprocessing cost vs. risk of missing rows

**For extreme cases**, run a periodic `--full-refresh` (weekly/monthly) as a safety net to catch anything the lookback window missed.

---

### Q4. What are snapshots? How do they implement SCD Type 2?

**Answer:**

A snapshot captures the state of a row at a point in time and tracks how it changes. This is the standard implementation of **Slowly Changing Dimension Type 2 (SCD2)** — instead of overwriting a row when it changes, you keep all historical versions with validity timestamps.

dbt adds four columns automatically:

| Column | Meaning |
|---|---|
| `dbt_scd_id` | Unique key for each snapshot record |
| `dbt_updated_at` | When this snapshot record was created |
| `dbt_valid_from` | When this version of the row became active |
| `dbt_valid_to` | When this version expired (`null` = currently active) |

**Timestamp strategy** (use when source has a reliable `updated_at`):
```sql
{% snapshot orders_snapshot %}
{{
  config(
    target_schema = 'snapshots',
    unique_key    = 'order_id',
    strategy      = 'timestamp',
    updated_at    = 'updated_at'
  )
}}
select * from {{ source('raw', 'orders') }}
{% endsnapshot %}
```

**Check strategy** (use when there's no `updated_at` — dbt hashes the columns you specify):
```sql
{% snapshot customers_snapshot %}
{{
  config(
    target_schema = 'snapshots',
    unique_key    = 'customer_id',
    strategy      = 'check',
    check_cols    = ['email', 'plan_type', 'country']
  )
}}
select * from {{ source('raw', 'customers') }}
{% endsnapshot %}
```

**Resulting data:**
```
customer_id | plan    | dbt_valid_from      | dbt_valid_to
1           | free    | 2024-01-01 00:00:00 | 2024-06-01 00:00:00
1           | pro     | 2024-06-01 00:00:00 | null              ← current
```

Run with:
```bash
dbt snapshot
```

---

### Q5. How would you structure a dbt project for a team of 10+ engineers?

**Answer:**

At scale, the biggest problems are naming collisions, unclear ownership, slow CI runs, and models no one knows the purpose of. Structure solves all of these.

**Folder structure:**
```
models/
├── staging/          # 1:1 with source, owned by data engineers
│   ├── shopify/
│   │   ├── _sources.yml
│   │   ├── _stg_shopify.yml
│   │   ├── stg_shopify__orders.sql
│   │   └── stg_shopify__customers.sql
│   └── stripe/
│       ├── _sources.yml
│       └── stg_stripe__payments.sql
├── intermediate/     # joins, business logic, not exposed to BI
│   └── int_orders_with_payments.sql
└── marts/            # owned by domain teams
    ├── finance/
    │   ├── _finance.yml
    │   └── fct_revenue.sql
    └── product/
        ├── _product.yml
        └── fct_events.sql
```

**Key conventions at scale:**
- **Naming:** `stg_[source]__[table]`, `int_[entity]_[verb]`, `fct_[event]`, `dim_[entity]`
- **One owner per folder** — declare in `schema.yml` using `owner.team`
- **Tags per domain** — `+tags: ['finance']` lets CI run only the affected domain
- **No raw source references in marts** — marts must go through staging
- **Required tests on every primary key** — enforce with `dbt_project_evaluator` package
- **Descriptions mandatory** — fail CI if models/columns have no description

**dbt_project.yml defaults:**
```yaml
models:
  my_project:
    staging:
      +materialized: view
      +tags: ['staging']
    intermediate:
      +materialized: view
      +tags: ['intermediate']
    marts:
      +materialized: table
      finance:
        +tags: ['finance']
        +meta:
          owner: finance-team
```

---

### Q6. How do custom schemas work? What does `generate_schema_name` do?

**Answer:**

By default, all models land in the schema defined in `profiles.yml`. You override this with `schema` in a config block:

```sql
{{ config(schema='finance') }}
```

But dbt doesn't just use `'finance'` directly. It calls the `generate_schema_name` macro which, by default, **concatenates** your target schema and the custom schema:

- In dev: `dbt_yourname_finance`
- In prod: `prod_finance`

This is intentional — it prevents dev models from colliding with prod schemas.

**Overriding the macro** (common in real projects — you usually want clean schema names in prod):

```sql
-- macros/generate_schema_name.sql
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {%- if target.name == 'prod' -%}
            {{ custom_schema_name | trim }}
        {%- else -%}
            {{ default_schema }}_{{ custom_schema_name | trim }}
        {%- endif -%}
    {%- endif -%}
{%- endmacro %}
```

Result:
- Dev: `dbt_alice_finance` (isolated, no collision)
- Prod: `finance` (clean, predictable)

---

### Q7. What is the difference between `dbt run`, `dbt build`, and `dbt compile`?

**Answer:**

| Command | What it does |
|---|---|
| `dbt compile` | Resolves all Jinja (`ref()`, macros, vars) and writes plain SQL to `target/compiled/`. Nothing runs in the warehouse. Use this to inspect generated SQL or debug macros. |
| `dbt run` | Executes compiled SQL in the warehouse to create/update tables and views. Does not run tests or seeds. |
| `dbt test` | Runs all tests (generic + singular). Does not build models. |
| `dbt build` | Runs everything in DAG order: seeds → snapshots → run → test. If a test fails, downstream models are skipped. This is the recommended command for CI. |

**When to use each:**
- `dbt compile -s my_model` → debug Jinja output
- `dbt run -s +my_model` → rebuild a model and its upstream
- `dbt test -s my_model --store-failures` → investigate a test failure
- `dbt build` → full pipeline in CI

---

### Q8. Explain node selectors — `+`, `@`, `tag:`, `state:`.

**Answer:**

Node selectors let you run a subset of your DAG rather than everything.

```bash
# basics
dbt run -s stg_orders              # exact model
dbt run -s staging.*               # all models in staging/ folder
dbt run -s tag:finance             # all models tagged 'finance'
dbt run -s path:models/marts       # all models in the marts/ path

# graph operators
dbt run -s +orders                 # orders + ALL upstream dependencies
dbt run -s orders+                 # orders + ALL downstream dependents
dbt run -s +orders+                # orders + upstream + downstream
dbt run -s 1+orders                # orders + 1 level upstream only
dbt run -s orders+1                # orders + 1 level downstream only

# @ operator — upstream + all downstream of upstream
dbt run -s @orders                 # useful for testing: build everything orders touches

# state selector — only changed models (requires saved manifest)
dbt run -s state:modified          # models whose SQL changed vs saved manifest
dbt run -s state:modified+         # changed models + their downstream
```

The `state:modified+` selector is the foundation of **slim CI** — you only build and test models affected by a PR, not the entire project.

---

## Mid–Senior — Jinja, Macros & Packages

---

### Q9. What is Jinja and why does dbt use it?

**Answer:**

Jinja is a Python templating language. dbt uses it to make SQL dynamic — you can write conditionals, loops, variables, and reusable functions (macros) inside `.sql` files.

When you run any dbt command, Jinja is compiled first. The warehouse only ever receives plain SQL — it never sees `{{ }}` or `{% %}`.

**Three Jinja delimiters:**

| Syntax | Purpose | Example |
|---|---|---|
| `{{ }}` | Expression — outputs a value | `{{ ref('orders') }}`, `{{ var('date') }}` |
| `{% %}` | Statement — control flow | `{% if %}`, `{% for %}`, `{% set %}` |
| `{# #}` | Comment — stripped before SQL | `{# TODO: remove this #}` |

**Why SQL alone isn't enough:**
- You can't parameterise environment (`dev` vs `prod`) in plain SQL
- You'd copy-paste the same `CASE WHEN` across 20 models
- You can't loop to generate columns dynamically
- You can't reference other models without hardcoding schema names

Jinja solves all of these.

---

### Q10. Write a macro from scratch. What can a macro return?

**Answer:**

A macro is a Jinja function stored in `macros/*.sql`. It can return a **string** (which gets rendered into SQL), a **list**, a **dict**, or any Python-compatible value via `return()`.

**Example — a safe division macro:**

```sql
-- macros/safe_divide.sql
{% macro safe_divide(numerator, denominator) %}
    case
        when {{ denominator }} is null or {{ denominator }} = 0 then null
        else {{ numerator }} / cast({{ denominator }} as float)
    end
{% endmacro %}
```

Usage:
```sql
select
    order_id,
    {{ safe_divide('revenue', 'num_items') }}   as revenue_per_item
from {{ ref('orders') }}
```

Compiled output:
```sql
select
    order_id,
    case
        when num_items is null or num_items = 0 then null
        else revenue / cast(num_items as float)
    end   as revenue_per_item
from prod.orders
```

**A macro that returns a list:**
```sql
{% macro get_payment_methods() %}
    {% set query %}
        select distinct payment_method from {{ source('raw', 'payments') }} order by 1
    {% endset %}
    {% if execute %}
        {% set results = run_query(query) %}
        {{ return(results.columns[0].values()) }}
    {% else %}
        {{ return([]) }}
    {% endif %}
{% endmacro %}
```

**What macros can return:**
- A rendered SQL string (default — just write the SQL in the macro body)
- An explicit value via `{{ return(value) }}` — list, dict, string, number

---

### Q11. What is `run_query` and why do you need to guard it with `{% if execute %}`?

**Answer:**

`run_query` executes a SQL statement **at compile time** and returns the result as an Agate table you can loop over in Jinja. It's used to generate dynamic SQL based on data that exists in the warehouse.

```sql
{% macro get_active_countries() %}
    {% set query %}
        select distinct country_code
        from {{ ref('stg_customers') }}
        where is_active = true
        order by 1
    {% endset %}

    {% if execute %}
        {% set results = run_query(query) %}
        {{ return(results.columns[0].values()) }}
    {% else %}
        {{ return([]) }}
    {% endif %}
{% endmacro %}
```

**Why `{% if execute %}` is mandatory:**

dbt processes your Jinja in two phases:
1. **Parse phase** (`dbt parse`, `dbt ls`, dependency resolution) — Jinja is evaluated but `execute = False`. No SQL runs.
2. **Execute phase** (`dbt run`, `dbt build`) — `execute = True`. SQL actually runs.

During the parse phase, `run_query` returns `None`. If you call `.columns[0].values()` on `None`, you get a crash. The `{% if execute %}` guard returns an empty list during parse and the real result during execution.

**Without the guard:**
```
AttributeError: 'NoneType' object has no attribute 'columns'
```

---

### Q12. What is `dbt_utils`? Name five macros you've used from it.

**Answer:**

`dbt_utils` is the official dbt Labs package of utility macros. Install via `packages.yml`:

```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
```

**Five essential macros:**

**1. `generate_surrogate_key` — stable primary key from multiple columns:**
```sql
select
    {{ dbt_utils.generate_surrogate_key(['order_id', 'line_item_id']) }} as sk,
    order_id,
    line_item_id
from {{ ref('stg_order_items') }}
```

**2. `pivot` — turn row values into columns:**
```sql
select
    order_id,
    {{ dbt_utils.pivot('payment_method', dbt_utils.get_column_values(ref('payments'), 'payment_method')) }}
from {{ ref('payments') }}
group by 1
```

**3. `union_relations` — union multiple tables with schema reconciliation:**
```sql
{{ dbt_utils.union_relations(
    relations=[ref('orders_2022'), ref('orders_2023'), ref('orders_2024')]
) }}
```

**4. `date_spine` — generate one row per date (great for gap-filling):**
```sql
{{ dbt_utils.date_spine(
    datepart   = 'day',
    start_date = "cast('2024-01-01' as date)",
    end_date   = "cast('2024-12-31' as date)"
) }}
```

**5. `get_column_values` — get distinct values from a column at compile time:**
```sql
{% set statuses = dbt_utils.get_column_values(ref('stg_orders'), 'status') %}
{% for status in statuses %}
    sum(case when status = '{{ status }}' then 1 else 0 end) as {{ status }}_count,
{% endfor %}
```

---

### Q13. What are `pre-hook` and `post-hook`? Give a real use case.

**Answer:**

Hooks are SQL statements (or macro calls) that dbt runs before (`pre-hook`) or after (`post-hook`) a model executes. They're declared in `dbt_project.yml` or in a model's `config()` block.

**Common use cases:**

**1. Grant permissions after a model builds (post-hook):**
```sql
-- macros/grant_access.sql
{% macro grant_access(table, role='reporter') %}
    grant select on {{ table }} to role {{ role }};
{% endmacro %}
```

```yaml
# dbt_project.yml
models:
  my_project:
    marts:
      +post-hook: "{{ grant_access(this) }}"
```

**2. Drop a temp table before building (pre-hook):**
```sql
{{ config(
    pre_hook  = "drop table if exists {{ this }}_staging",
    post_hook = "analyze {{ this }}"
) }}
```

**3. Log model run metadata to an audit table:**
```yaml
+post-hook: >
  insert into audit.model_runs (model_name, run_at)
  values ('{{ this.name }}', current_timestamp)
```

**Key points:**
- `{{ this }}` refers to the model's own fully-qualified table name
- Multiple hooks can be passed as a list: `+post-hook: ["grant ...", "analyze ..."]`
- Hooks run inside the same transaction as the model where supported

---

### Q14. When is `execute` false in dbt? Why does it matter?

**Answer:**

`execute` is a global Jinja variable that is `True` only during the **execution phase** of a dbt command (i.e., when SQL is actually being sent to the warehouse).

It is `False` during:
- `dbt parse` — dbt reads all models to build the DAG
- `dbt ls` — dbt lists models without running them
- The **dependency resolution phase** at the start of `dbt run` / `dbt build`
- Any macro call that happens during graph compilation

**Why it matters:**

Any macro that calls `run_query`, reads from the warehouse, or manipulates query results must be guarded:

```sql
{% set results = run_query("select distinct status from raw.orders") %}

{% if execute %}
    {# results is a real Agate table here #}
    {% for row in results.rows %}
        '{{ row[0] }}'{% if not loop.last %},{% endif %}
    {% endfor %}
{% else %}
    {# results is None here — don't touch it #}
    'placeholder'
{% endif %}
```

Without the guard, `dbt parse` and `dbt ls` will crash with a `NoneType` error, making your entire project unusable for listing or compiling.

---

## Senior / Lead — Performance, Testing & CI/CD

---

### Q15. A dbt model is running slowly. Walk me through how you'd optimise it.

**Answer:**

Diagnose before optimising. The answer depends entirely on *why* it's slow.

**Step 1 — Identify where the time is spent:**
```bash
dbt run -s slow_model --debug    # see exact SQL and timing
```
Check your warehouse's query history/profiler (Snowflake Query Profile, BigQuery Execution Details, Redshift SVL_QUERY_REPORT).

**Step 2 — Common causes and fixes:**

**Materialization wrong:**
The model is a `view` and BI tools query it directly — every query re-runs all the SQL.
```sql
{{ config(materialized='table') }}  -- or incremental for large tables
```

**Full table scan on a large table:**
No partition pruning, no cluster key. The warehouse reads every row.
```sql
{{ config(
    materialized  = 'incremental',
    unique_key    = 'event_id',
    cluster_by    = ['occurred_date']   -- Snowflake
    -- partition_by = {'field': 'occurred_date', 'data_type': 'date'}  -- BigQuery
) }}
```

**Incremental model reprocessing too much:**
The lookback window is too wide, or `unique_key` forces an expensive merge across billions of rows.
Fix: tighten the filter, reduce lookback window, consider `insert_overwrite` on partitions.

**View chain — views stacked on views:**
Each downstream query re-executes the entire chain from the raw table.
Fix: materialize the base of the chain as a table.

**Inefficient SQL — `select *`, non-sargable filters, cartesian joins:**
Review the compiled SQL (`target/compiled/`). Look for:
- `select *` pulling 200 columns when you need 5
- `where cast(id as varchar) = '123'` (prevents index use)
- Implicit cross joins from missing join conditions

**Step 3 — Measure the improvement:**
Run before/after, record bytes scanned and execution time. A good optimisation reduces bytes scanned by 10x+, not just seconds.

---

### Q16. How do you set up CI/CD for dbt? What is slim CI?

**Answer:**

**Basic CI pipeline (GitHub Actions):**
```yaml
# .github/workflows/dbt_ci.yml
name: dbt CI

on:
  pull_request:
    branches: [main]

jobs:
  dbt-ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dbt
        run: pip install dbt-snowflake==1.7.0

      - name: dbt deps
        run: dbt deps

      - name: dbt build
        run: dbt build --target ci
        env:
          DBT_USER:     ${{ secrets.DBT_USER }}
          DBT_PASSWORD: ${{ secrets.DBT_PASSWORD }}
```

**The problem with `dbt build` in CI:**
On a large project with 500+ models, `dbt build` rebuilds everything on every PR — wasteful and slow (can take hours).

**Slim CI — only build what changed:**

Slim CI uses the `state:modified+` selector with a saved production manifest to build only the models changed in the PR and their downstream dependents.

```yaml
- name: Download prod manifest
  run: |
    # fetch the latest production manifest.json
    aws s3 cp s3://my-dbt-artifacts/manifest.json ./prod-manifest/manifest.json

- name: dbt build — changed models only
  run: |
    dbt build \
      --select state:modified+ \
      --defer \
      --state ./prod-manifest
```

**Key flags:**
- `--select state:modified+` — only models whose SQL/config changed vs the prod manifest, plus their downstream
- `--defer` — for models that haven't changed, use the prod version instead of rebuilding from scratch
- `--state ./prod-manifest` — path to the saved `manifest.json` from the last prod run

**Result:** A PR that changes 3 models only builds those 3 models + their children, not all 500. CI drops from 60 minutes to 5.

---

### Q17. What is test severity? How do you use `warn` vs `error`?

**Answer:**

By default, a failing dbt test exits with a non-zero code — it fails the CI run and blocks downstream models in `dbt build`. Severity lets you make some failures **warnings** (logged but non-blocking) rather than hard errors.

**Setting severity in YAML:**
```yaml
models:
  - name: orders
    columns:
      - name: order_id
        tests:
          - unique:
              severity: error       # blocks CI — primary key must be unique
          - not_null:
              severity: error

      - name: amount
        tests:
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100000
              severity: warn        # logs a warning but doesn't block the pipeline
```

**Thresholds — fail only if more than N rows fail:**
```yaml
- not_null:
    severity: warn
    warn_if: ">10"      # warn only if more than 10 rows fail
    error_if: ">100"    # error only if more than 100 rows fail
```

This is valuable for:
- **New tests on existing models** — you suspect there are existing data quality issues but don't want to block prod immediately. Start with `warn`, investigate, fix data, then promote to `error`.
- **Non-critical fields** — a missing `phone_number` is worth logging but shouldn't stop the pipeline.
- **Gradual rollout** — set a threshold (`error_if: ">0"`) that tightens over time.

---

### Q18. What are exposures and how do you use them for impact analysis?

**Answer:**

Exposures document what *uses* your dbt models outside of dbt — dashboards, ML models, reverse ETL pipelines, APIs. They appear as nodes in the lineage graph, letting you see the full blast radius of a model change.

```yaml
# models/marts/exposures.yml
version: 2

exposures:
  - name: executive_orders_dashboard
    type: dashboard
    maturity: high
    url: https://looker.company.com/dashboards/42
    description: >
      Executive-facing daily orders summary. Refreshed every morning at 6am.
      Breaking this will affect C-suite reporting.
    depends_on:
      - ref('orders')
      - ref('customers')
    owner:
      name: Analytics Team
      email: analytics@company.com

  - name: churn_prediction_model
    type: ml
    maturity: medium
    description: "Weekly propensity-to-churn model. Reads customer feature table."
    depends_on:
      - ref('customers')
      - ref('fct_events')
    owner:
      name: ML Team
      email: ml@company.com
```

**Impact analysis workflow:**

Before changing `orders.sql`, run:
```bash
dbt ls -s orders+    # see everything downstream, including exposures
```

The output includes `exposure:executive_orders_dashboard` — you immediately know that breaking `orders` will affect the executive dashboard and you need to notify the owner before deploying.

**Types:** `dashboard`, `notebook`, `analysis`, `ml`, `application`
**Maturity:** `low`, `medium`, `high` — signals how critical the downstream consumer is

---

### Q19. What is the dbt semantic layer? What problem does MetricFlow solve?

**Answer:**

The semantic layer solves the **metric consistency problem**: when five different teams define "monthly revenue" five different ways in five different dashboards, the business loses trust in its data.

MetricFlow (dbt's semantic layer engine) lets you define metrics **once** in dbt and have any BI tool query them consistently.

**Define a semantic model:**
```yaml
# models/marts/semantic/orders.yml
semantic_models:
  - name: orders
    model: ref('orders')
    defaults:
      agg_time_dimension: ordered_at
    entities:
      - name: order
        type: primary
        expr: order_id
      - name: customer
        type: foreign
        expr: customer_id
    dimensions:
      - name: ordered_at
        type: time
        type_params:
          time_granularity: day
      - name: order_tier
        type: categorical
    measures:
      - name: order_count
        agg: count
        expr: order_id
      - name: total_revenue
        agg: sum
        expr: amount

metrics:
  - name: total_revenue
    type: simple
    label: "Total Revenue"
    type_params:
      measure: total_revenue

  - name: average_order_value
    type: ratio
    label: "Average Order Value"
    type_params:
      numerator: total_revenue
      denominator: order_count
```

**Query via dbt Cloud or the CLI:**
```bash
dbt sl query --metrics total_revenue --group-by ordered_at__month
```

**What problem it solves:**
- One definition of `total_revenue` — no more "which revenue is correct?"
- BI tools query the semantic layer, not raw SQL — consistent results everywhere
- Business logic lives in dbt, not scattered across Looker LookML, Tableau calcs, and Jupyter notebooks

---

### Q20. How do you manage dev / staging / prod environments in dbt?

**Answer:**

Environments are controlled by **targets** in `profiles.yml`. Each target points to a different schema (and optionally warehouse, database, or role).

```yaml
# ~/.dbt/profiles.yml
my_project:
  target: dev
  outputs:

    dev:
      type: snowflake
      schema: dbt_{{ env_var('DBT_USERNAME') }}   # personal schema per developer
      warehouse: dev_wh
      role: transformer_dev
      threads: 4
      # ... connection details

    ci:
      type: snowflake
      schema: dbt_ci_{{ env_var('PR_NUMBER') }}   # isolated schema per PR
      warehouse: ci_wh
      role: transformer_ci
      threads: 8

    prod:
      type: snowflake
      schema: prod
      warehouse: prod_wh
      role: transformer_prod
      threads: 16
```

**Key isolation patterns:**

| Environment | Schema naming | Purpose |
|---|---|---|
| Dev | `dbt_alice`, `dbt_bob` | Each engineer has a personal sandbox — changes don't affect others |
| CI | `dbt_ci_pr123` | Each PR gets an isolated schema — torn down after the PR closes |
| Staging | `staging` | Mirror of prod — used for integration tests and stakeholder review |
| Prod | `prod` (or domain name) | What BI tools read — only merged, tested code runs here |

**Environment-aware SQL using `target`:**
```sql
{% if target.name != 'prod' %}
limit 10000   -- fast dev iterations
{% endif %}
```

**dbt Cloud** manages this more explicitly with Environment → Job → Run scoping, where each environment has its own connection credentials and deployment triggers.

---

### Q21. What are dbt contracts and how do they enforce data quality at the model level?

**Answer:**

dbt contracts (introduced in dbt 1.5) let you declare the **expected schema** of a model — column names, data types, and constraints — and have dbt enforce them at build time. If the model's output doesn't match the contract, the run fails before the table is swapped in.

```yaml
# models/marts/_marts.yml
models:
  - name: orders
    config:
      contract:
        enforced: true    # schema must exactly match what's declared below
    columns:
      - name: order_id
        data_type: varchar
        constraints:
          - type: not_null
          - type: primary_key
      - name: ordered_at
        data_type: timestamp
        constraints:
          - type: not_null
      - name: amount
        data_type: float
      - name: customer_id
        data_type: varchar
        constraints:
          - type: not_null
          - type: foreign_key
            expression: "prod.customers (customer_id)"
```

**What contracts enforce:**
- Column names must match exactly (extra or missing columns = failure)
- Data types must match (dbt casts and validates)
- Constraints like `not_null`, `primary_key`, `foreign_key` are applied at the DB level where supported

**Why they matter at senior/lead level:**
Contracts are the foundation of **data contracts** between producer and consumer teams. The team owning `orders` promises its schema to the 5 teams consuming it. Any breaking change (dropping a column, changing a type) fails CI before it reaches prod — the producer can't silently break downstream consumers.

---

### Q22. How do you version dbt models and handle breaking changes safely?

**Answer:**

dbt 1.5+ introduced **model versioning** to handle breaking changes without breaking downstream consumers immediately.

**Defining versions:**
```yaml
# models/marts/_marts.yml
models:
  - name: orders
    latest_version: 2
    versions:
      - v: 1
        defined_in: orders_v1    # → models/marts/orders_v1.sql
      - v: 2
        defined_in: orders       # → models/marts/orders.sql (current)
```

**Downstream consumers pin to a version:**
```sql
-- still using v1 — not broken by the v2 changes
select * from {{ ref('orders', v=1) }}

-- opted in to v2
select * from {{ ref('orders', v=2) }}
-- or just:
select * from {{ ref('orders') }}    -- always resolves to latest_version
```

**Safe breaking change workflow:**
1. Create `v2` with the breaking change (renamed column, dropped column, new type)
2. Deploy both `v1` and `v2` to prod — consumers still use `v1`
3. Communicate the migration window to downstream teams
4. Consumers migrate their models to `ref('orders', v=2)` one by one
5. Once no models reference `v1`, deprecate and delete it

**For non-breaking additive changes** (new columns, new rows), no versioning needed — just add the column and document it.

---

## Scenario / Situational Questions

---

### Q23. A `not_null` test fails in prod on your orders model. What do you do?

**Answer:**

**Step 1 — Don't panic, assess scope:**
```bash
dbt test -s orders --store-failures
```
`--store-failures` writes the failing rows to a table in your warehouse (`dbt_test__audit.not_null_orders_order_id`). Query it:
```sql
select * from dbt_test__audit.not_null_orders_order_id limit 100;
```
How many rows? Is this 5 rows or 500,000? That determines urgency.

**Step 2 — Trace the null back to its source:**
```sql
-- is the null coming from the source, or introduced in transformation?
select count(*) from raw.orders where id is null;
select count(*) from dev.stg_orders where order_id is null;
select count(*) from prod.orders where order_id is null;
```
Walk back through the layers until you find where the null originates.

**Step 3 — Determine the cause:**
- **Source sent nulls** → data quality issue upstream. Fix the source or add a `where id is not null` filter in staging
- **Join introduced nulls** → a `LEFT JOIN` in the mart is producing unmatched rows that become null
- **Logic error in a CASE/COALESCE** → review the SQL

**Step 4 — Fix and validate:**
Fix the SQL, run `dbt build -s +orders` locally, confirm the test passes.

**Step 5 — Prevent recurrence:**
- If it was a source issue, add a source freshness test and a `not_null` test on the source itself
- Consider setting `severity: warn` with a threshold on non-critical columns so you're alerted before it becomes a hard failure

---

### Q24. You need to add a column to a mart used by 5 dashboards. What's your process?

**Answer:**

**Additive change (new column)** — low risk, straightforward:
1. Add the column to the model SQL
2. Add it to `schema.yml` with a description and tests
3. Run `dbt build -s orders` in dev and confirm tests pass
4. Open a PR, CI runs `state:modified+` — only `orders` and anything downstream in dbt gets rebuilt
5. Communicate to dashboard owners that a new column is available
6. Merge and deploy

The dashboards are unaffected — they select specific columns and a new one doesn't break anything.

**Breaking change (rename/drop column)** — high risk, needs a plan:

1. **Audit exposure** first:
   ```bash
   dbt ls -s orders+
   ```
   This shows all dbt models and exposures downstream of `orders`.

2. **Use model versioning** if the column is referenced by many consumers. Create `orders_v2` with the new schema; let consumers migrate gradually.

3. **If versioning is overkill** (small team, fast migration):
   - Add the new column name *alongside* the old one temporarily
   - Communicate the deprecation window
   - Update all downstream dbt models to use the new column name
   - Update dashboards
   - Remove the old column once all references are gone

4. **Never rename a column in prod on a Friday.**

---

### Q25. How would you migrate a 500-line stored procedure into dbt?

**Answer:**

A stored procedure is usually a monolith — one giant block doing ten things. The goal is to decompose it into a graph of small, testable, named models.

**Step 1 — Read and map the procedure:**
Annotate every logical step: "lines 1–50 join customers and orders, lines 51–100 calculate revenue tiers, lines 101–200 pivot by region…"

**Step 2 — Identify layers:**
- Steps that clean/rename raw tables → `staging` models
- Steps that join or enrich → `intermediate` models
- The final output → a `mart` model

**Step 3 — Build bottom-up:**
Start with staging models (simplest), then intermediate, then the mart. Each step should be independently runnable and testable.

**Step 4 — Validate against the existing procedure:**
```sql
-- compare row counts and key metrics between old and new
select 'old' as src, count(*), sum(revenue) from old_proc_output
union all
select 'new' as src, count(*), sum(revenue) from {{ ref('new_mart') }}
```
Numbers should match. If they don't, the procedure had bugs — document whether you're fixing them or replicating them.

**Step 5 — Add tests:**
Every primary key gets `unique` + `not_null`. Key metrics get range tests. This gives you a regression harness before you decommission the procedure.

**Step 6 — Decommission the procedure:**
Once the new models are in prod and consumers are validated, remove the procedure. Don't keep both running in parallel indefinitely.

---

### Q26. Your incremental model is producing duplicate rows. What are the causes?

**Answer:**

Duplicates in an incremental model are one of the most common production bugs. Here are the causes in order of likelihood:

**Cause 1 — No `unique_key` set, using `append` strategy:**
Without a `unique_key`, dbt just appends rows. If the same row is processed twice (e.g., a rerun after failure), it gets inserted twice.

Fix: set a `unique_key` and use `merge` strategy:
```sql
{{ config(materialized='incremental', unique_key='order_id') }}
```

**Cause 2 — `unique_key` is not actually unique in the source:**
If the source table has duplicate `order_id` values, dbt's merge will try to update a single target row but finds multiple source matches — result depends on warehouse behaviour, often inserts all of them.

Fix: deduplicate before the merge:
```sql
with deduped as (
    select *,
        row_number() over (partition by order_id order by updated_at desc) as rn
    from {{ source('raw', 'orders') }}
)
select * from deduped where rn = 1
```

**Cause 3 — Lookback window overlaps with already-processed rows:**
A 3-day lookback reprocesses rows that were already correctly in the table, and `append` adds them again.

Fix: ensure `unique_key` is set so `merge` upserts rather than appends.

**Cause 4 — `--full-refresh` was run without truncating downstream:**
A full-refresh rebuilds the incremental model from scratch, but if a downstream model already had data derived from the old version, a partial rerun can produce inconsistency.

Fix: run `--full-refresh` on the full lineage: `dbt build -s +my_model+ --full-refresh`

**Cause 5 — Composite `unique_key` not granular enough:**
```sql
{{ config(unique_key=['order_id', 'event_date']) }}
```
If the actual grain is `order_id + event_date + event_type`, the composite key is too broad and multiple rows survive.

---

### Q27. How would you test a macro you've written?

**Answer:**

Macros don't have a native unit test framework in dbt core (though `dbt-unit-testing` package adds one), so the standard approaches are:

**1. `dbt compile` — inspect the rendered SQL:**
```bash
dbt compile -s model_that_uses_my_macro
```
Open `target/compiled/.../model_that_uses_my_macro.sql` and verify the output is what you expect. This is the fastest feedback loop.

**2. Write a model that exercises the macro:**
```sql
-- models/tests/test_safe_divide.sql (materialized as view, not exposed to BI)
{{ config(materialized='view', tags=['macro_tests']) }}
select
    {{ safe_divide('10', '2') }}    as normal_case,       -- expect 5.0
    {{ safe_divide('10', '0') }}    as zero_denominator,  -- expect null
    {{ safe_divide('10', null) }}   as null_denominator   -- expect null
```

Then run a singular test that asserts the values:
```sql
-- tests/assert_safe_divide_returns_null_on_zero.sql
select 1
from {{ ref('test_safe_divide') }}
where zero_denominator is not null    -- fails if null wasn't returned
   or null_denominator is not null
```

**3. Use `dbt-unit-testing` package:**
```yaml
# tests/unit/test_safe_divide.yml
unit_tests:
  - name: test_safe_divide_zero
    model: model_that_uses_my_macro
    given:
      - input: ref('orders')
        rows:
          - {order_id: 1, revenue: 10, num_items: 0}
    expect:
      rows:
        - {order_id: 1, revenue_per_item: null}
```

**4. Test edge cases deliberately:**
- Null inputs
- Zero denominators
- Empty lists in loop macros
- Both `execute=True` and `execute=False` paths for `run_query` macros

---

*This document covers Mid-level through Lead-level dbt interview questions and answers.*
*For fundamentals (Junior/Analyst level), see the main dbt reference guide.*
