# dbt (Data Build Tool) — Complete Reference Guide

> A complete, beginner-to-advanced reference for dbt concepts. Click any section to expand it.

---

## Table of Contents

- [What is dbt?](#what-is-dbt)
- [How dbt fits in the Modern Data Stack](#how-dbt-fits-in-the-modern-data-stack)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
  - [Models](#models)
  - [Sources](#sources)
  - [Tests](#tests)
  - [Documentation](#documentation)
  - [Macros](#macros)
  - [Seeds](#seeds)
  - [Snapshots](#snapshots)
  - [The DAG](#the-dag)
- [The Layered Data Model](#the-layered-data-model)
- [Materializations](#materializations)
  - [View](#view-materialization)
  - [Table](#table-materialization)
  - [Incremental](#incremental-materialization)
  - [Ephemeral](#ephemeral-materialization)
- [Jinja & Macros](#jinja--macros-in-depth)
- [Packages](#packages)
- [Variables](#variables)
- [Environments & Profiles](#environments--profiles)
- [Advanced Topics](#advanced-topics)
  - [Snapshots (SCDs)](#snapshots-slowly-changing-dimensions)
  - [Exposures](#exposures)
  - [Semantic Layer & Metrics](#semantic-layer--metrics)
  - [Custom Schemas](#custom-schemas)
  - [CI/CD with dbt](#cicd-with-dbt)
- [Key Commands](#key-commands)
- [Quick Reference Cheatsheet](#quick-reference-cheatsheet)

---

## What is dbt?

**dbt (data build tool)** is a transformation framework that lets data analysts and engineers write SQL `SELECT` statements to define transformations, and dbt handles the `CREATE TABLE / VIEW` logic, dependency resolution, testing, and documentation automatically.

dbt is the **T in ELT** — it runs entirely inside your data warehouse. No data ever leaves.

**Key principles:**
- Write `SELECT` — never `CREATE`, `DROP`, or `INSERT`
- Every `.sql` file = one table or view in your warehouse
- Dependencies are declared with `ref()` — dbt resolves run order automatically
- Everything is version-controlled, tested, and documented

---

## How dbt fits in the Modern Data Stack

```
┌─────────────┐     Load      ┌──────────────────┐   Transform   ┌──────────────┐   Query   ┌──────────────┐
│   Sources   │ ────────────► │  Data Warehouse   │ ────────────► │     dbt      │ ────────► │  BI / ML     │
│  DBs, APIs  │               │ Snowflake·BigQuery│               │  SQL + tests │           │ Looker·Metab.│
│  CSVs, etc. │               │ Redshift·DuckDB   │               │  docs + DAG  │           │ Tableau etc. │
└─────────────┘               └──────────────────┘               └──────────────┘           └──────────────┘
                 ▲                                                       ▲
                 │                                                       │
            EL tools                                              dbt lives here
         (Fivetran, Airbyte)                               runs SQL inside your warehouse
```

---

## Project Structure

```
my_dbt_project/
├── models/                        ← your SQL lives here
│   ├── staging/                   ← light cleaning of raw tables
│   │   ├── stg_orders.sql
│   │   ├── stg_customers.sql
│   │   ├── _sources.yml           ← declare raw source tables
│   │   └── _stg_models.yml        ← tests + docs for staging models
│   ├── intermediate/              ← optional join/logic layer (not exposed to BI)
│   │   └── int_orders_with_items.sql
│   └── marts/                     ← business-level models (BI reads these)
│       ├── orders.sql
│       ├── customers.sql
│       └── _marts.yml
├── macros/                        ← reusable Jinja functions
│   └── clean_string.sql
├── seeds/                         ← CSV files loaded as warehouse tables
│   └── country_codes.csv
├── tests/                         ← custom singular test SQL files
│   └── assert_orders_positive_amount.sql
├── snapshots/                     ← slowly changing dimension configs
│   └── orders_snapshot.sql
├── analyses/                      ← ad-hoc SQL (compiled but not run)
├── dbt_project.yml                ← project config (name, paths, materializations)
└── profiles.yml                   ← warehouse connection (usually ~/.dbt/profiles.yml)
```

---

## Core Concepts

### Models

<details>
<summary><strong>What is a model?</strong></summary>

A model is a single `.sql` file containing a `SELECT` statement. dbt executes it and creates a table or view in your warehouse. The filename becomes the table name.

```sql
-- models/staging/stg_orders.sql
select
    id              as order_id,
    customer_id,
    cast(created_at as timestamp) as ordered_at,
    amount_cents / 100.0          as amount,
    lower(trim(status))           as status
from {{ source('raw', 'orders') }}
where id is not null
```

dbt turns this into:
```sql
create or replace view dev.stg_orders as (
    select
        id              as order_id,
        ...
    from raw.orders
    where id is not null
);
```

</details>

<details>
<summary><strong>The ref() function — the most important concept in dbt</strong></summary>

`ref()` is how you reference another model. It does two things:

1. **Resolves the correct table path** for your environment (`dev.stg_orders` in dev, `prod.stg_orders` in prod)
2. **Registers a dependency** so dbt builds models in the correct order

```sql
-- models/marts/orders.sql
select
    o.order_id,
    o.ordered_at,
    o.amount,
    c.full_name    as customer_name,
    c.email        as customer_email
from {{ ref('stg_orders') }} o          -- dependency on stg_orders
left join {{ ref('stg_customers') }} c  -- dependency on stg_customers
    on o.customer_id = c.customer_id
```

dbt sees the `ref()` calls and knows: run `stg_orders` and `stg_customers` before `orders`. Always. Automatically.

</details>

<details>
<summary><strong>Inline config block</strong></summary>

Add a `config()` block at the top of any model to override project-level settings:

```sql
{{
  config(
    materialized = 'table',
    tags         = ['marts', 'finance'],
    schema       = 'reporting'
  )
}}

select * from {{ ref('stg_orders') }}
```

Config options include: `materialized`, `tags`, `schema`, `database`, `alias`, `unique_key`, `partition_by`, `cluster_by`, `indexes`.

</details>

<details>
<summary><strong>A complete staging model example</strong></summary>

```sql
-- models/staging/stg_orders.sql
{{
  config(
    materialized = 'view',
    tags         = ['staging', 'orders']
  )
}}

with source as (
    select * from {{ source('raw', 'orders') }}
),

renamed as (
    select
        id                                   as order_id,
        customer_id,
        cast(created_at  as timestamp)       as ordered_at,
        cast(updated_at  as timestamp)       as updated_at,
        coalesce(amount_cents, 0) / 100.0    as amount,
        lower(trim(status))                  as status,
        _fivetran_deleted                    as is_deleted
    from source
    where id is not null
      and _fivetran_deleted = false
)

select * from renamed
```

</details>

<details>
<summary><strong>A complete mart model example</strong></summary>

```sql
-- models/marts/orders.sql
{{
  config(
    materialized = 'table',
    tags         = ['marts', 'core']
  )
}}

with orders as (
    select * from {{ ref('stg_orders') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

final as (
    select
        o.order_id,
        o.ordered_at,
        o.amount,
        o.status,
        o.customer_id,
        c.full_name                          as customer_name,
        c.email                              as customer_email,
        c.country_code,
        date_trunc('month', o.ordered_at)    as order_month,
        case
            when o.amount >= 100 then 'high_value'
            when o.amount >= 50  then 'mid_value'
            else 'low_value'
        end                                  as order_tier
    from orders o
    left join customers c on o.customer_id = c.customer_id
)

select * from final
```

</details>

---

### Sources

<details>
<summary><strong>What are sources?</strong></summary>

Sources are raw tables that dbt doesn't own — they're loaded by your EL tool (Fivetran, Airbyte, etc.). You declare them in a YAML file so dbt knows about them and can run freshness checks.

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw                          # logical name used in {{ source() }}
    database: my_database              # optional — defaults to profile database
    schema: raw                        # actual schema in your warehouse
    tables:
      - name: orders
        description: "Raw orders loaded by Fivetran from the production DB"
        loaded_at_field: _fivetran_synced
        freshness:
          warn_after:  {count: 12, period: hour}
          error_after: {count: 24, period: hour}

      - name: customers
        description: "Raw customer records from the production DB"
        loaded_at_field: _fivetran_synced
        freshness:
          warn_after:  {count: 24, period: hour}
          error_after: {count: 48, period: hour}
```

Reference a source in a model:
```sql
select * from {{ source('raw', 'orders') }}
--                         ▲       ▲
--                    source name  table name (from YAML)
```

Check freshness:
```bash
dbt source freshness
```

</details>

---

### Tests

<details>
<summary><strong>Generic tests (built-in)</strong></summary>

Declared in YAML alongside your model. dbt ships with four built-in generic tests:

| Test | What it checks |
|------|----------------|
| `unique` | Every value in the column is distinct |
| `not_null` | No `NULL` values exist |
| `accepted_values` | All values are in a predefined list |
| `relationships` | Every value exists in another model's column (FK integrity) |

```yaml
# models/marts/_marts.yml
version: 2

models:
  - name: orders
    columns:
      - name: order_id
        tests:
          - unique
          - not_null

      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_customers')
              field: customer_id

      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'shipped', 'delivered', 'cancelled']

      - name: amount
        tests:
          - not_null
```

</details>

<details>
<summary><strong>Singular tests (custom SQL)</strong></summary>

A singular test is a `.sql` file in the `tests/` folder. It should return **zero rows** to pass — any row returned is a failure.

```sql
-- tests/assert_orders_positive_amount.sql
-- Fails if any order has a negative or zero amount
select order_id
from {{ ref('orders') }}
where amount <= 0
```

```sql
-- tests/assert_no_orphaned_orders.sql
-- Fails if any order references a non-existent customer
select o.order_id
from {{ ref('orders') }} o
left join {{ ref('customers') }} c on o.customer_id = c.customer_id
where c.customer_id is null
```

</details>

<details>
<summary><strong>Running tests</strong></summary>

```bash
dbt test                        # run all tests
dbt test -s orders              # test one model
dbt test -s tag:marts           # test all models tagged 'marts'
dbt test --store-failures       # save failing rows to a table for inspection
dbt build -s +orders            # run + test orders and all upstream models
```

</details>

---

### Documentation

<details>
<summary><strong>schema.yml — tests + docs in one place</strong></summary>

```yaml
# models/marts/_marts.yml
version: 2

models:
  - name: orders
    description: >
      One row per order. Joins stg_orders with stg_customers.
      Source of truth for all order-level reporting.
    columns:
      - name: order_id
        description: "Primary key. Unique identifier for each order."
      - name: ordered_at
        description: "Timestamp when the order was placed (UTC)."
      - name: amount
        description: "Order total in USD."
      - name: order_tier
        description: "high_value (≥$100), mid_value ($50–$99), low_value (<$50)."
```

</details>

<details>
<summary><strong>Generating and serving docs</strong></summary>

```bash
dbt docs generate    # compile docs into target/catalog.json
dbt docs serve       # open a local documentation site at localhost:8080
```

The docs site includes:
- Searchable model + column descriptions
- Auto-generated DAG / lineage graph
- Source freshness status
- Test results

</details>

<details>
<summary><strong>doc() blocks — reusable descriptions</strong></summary>

For long descriptions, write them in a `.md` file and reference them:

```markdown
<!-- models/marts/docs.md -->
{% docs order_tier %}
Bucketed order value:
- **high_value**: order total ≥ $100
- **mid_value**: order total $50–$99
- **low_value**: order total < $50
{% enddocs %}
```

```yaml
# Reference in schema.yml
- name: order_tier
  description: "{{ doc('order_tier') }}"
```

</details>

---

### Macros

<details>
<summary><strong>What are macros?</strong></summary>

Macros are Jinja-templated functions stored in `macros/`. They let you write reusable SQL logic — think of them as functions in a programming language.

```sql
-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name) %}
    coalesce({{ column_name }}, 0) / 100.0
{% endmacro %}
```

Use it in any model:
```sql
select
    id              as order_id,
    {{ cents_to_dollars('amount_cents') }}  as amount
from {{ source('raw', 'orders') }}
```

</details>

<details>
<summary><strong>Macro with arguments and defaults</strong></summary>

```sql
-- macros/date_spine.sql
{% macro clean_string(column_name, replacement='unknown') %}
    nullif(lower(trim({{ column_name }})), '')
{% endmacro %}
```

```sql
-- macros/generate_surrogate_key.sql
{% macro generate_surrogate_key(field_list) %}
    md5(
        concat_ws('-',
            {% for field in field_list %}
                cast({{ field }} as varchar)
                {% if not loop.last %}, {% endif %}
            {% endfor %}
        )
    )
{% endmacro %}
```

Usage:
```sql
select
    {{ generate_surrogate_key(['order_id', 'line_item_id']) }} as surrogate_key
from {{ ref('stg_order_items') }}
```

</details>

---

### Seeds

<details>
<summary><strong>What are seeds?</strong></summary>

Seeds are CSV files in the `seeds/` directory that dbt loads as tables in your warehouse. Useful for static lookup data — country codes, mapping tables, cost rates, etc.

```csv
-- seeds/country_codes.csv
country_code,country_name,region
US,United States,North America
GB,United Kingdom,Europe
DE,Germany,Europe
JP,Japan,Asia Pacific
```

Load them:
```bash
dbt seed                    # load all seeds
dbt seed -s country_codes   # load one seed
```

Reference in a model:
```sql
select
    o.*,
    cc.country_name,
    cc.region
from {{ ref('orders') }} o
left join {{ ref('country_codes') }} cc on o.country_code = cc.country_code
```

Configure in `dbt_project.yml`:
```yaml
seeds:
  my_project:
    country_codes:
      +column_types:
        country_code: varchar(2)
```

</details>

---

### Snapshots

<details>
<summary><strong>What are snapshots?</strong></summary>

Snapshots track changes to rows over time — useful for slowly changing dimensions (SCDs). dbt adds `dbt_valid_from` and `dbt_valid_to` columns to record when each version of a row was active.

```sql
-- snapshots/orders_snapshot.sql
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

Run snapshots:
```bash
dbt snapshot
```

Result table includes:
| order_id | status | dbt_valid_from | dbt_valid_to |
|----------|--------|----------------|--------------|
| 1 | pending | 2024-01-01 | 2024-01-03 |
| 1 | shipped | 2024-01-03 | null |

</details>

---

### The DAG

<details>
<summary><strong>How the DAG works</strong></summary>

dbt automatically builds a **Directed Acyclic Graph** from all your `ref()` and `source()` calls. It guarantees models run in the correct order — you never have to manage this manually.

```
raw.orders ──────► stg_orders ──────┐
                                    ├──► orders (mart)
raw.customers ───► stg_customers ───┘
```

View the DAG:
```bash
dbt docs generate && dbt docs serve
# navigate to the lineage graph in the UI
```

Select models using graph operators:
```bash
dbt run -s stg_orders          # just this model
dbt run -s +orders             # orders + everything upstream
dbt run -s orders+             # orders + everything downstream
dbt run -s +orders+            # orders + all upstream + all downstream
dbt run -s stg_orders+1        # stg_orders + one level downstream only
```

</details>

---

## The Layered Data Model

The industry-standard pattern for organising a dbt project:

```
Raw (sources)
    │
    ▼
Staging  ──  1:1 with source tables. Rename, cast, clean. No joins. No business logic.
    │         Materialized as: VIEW
    ▼
Intermediate  ──  Join staging models. Begin applying business logic.
    │              Not exposed to BI tools.
    │              Materialized as: VIEW (or ephemeral)
    ▼
Marts  ──  Business entities: Orders, Customers, Revenue.
           BI tools, ML models, and analysts read these.
           Materialized as: TABLE
```

**Naming conventions:**
- `stg_[source]__[table].sql` — staging (e.g. `stg_shopify__orders.sql`)
- `int_[entity]_[verb].sql` — intermediate (e.g. `int_orders_joined.sql`)
- `[entity].sql` or `fct_[entity].sql` / `dim_[entity].sql` — marts

---

## Materializations

### View Materialization

<details>
<summary><strong>How view materialization works</strong></summary>

dbt executes `CREATE OR REPLACE VIEW`. No data is stored — the SQL runs live every time something queries the view.

```sql
{{ config(materialized='view') }}

with source as (
    select * from {{ source('raw', 'orders') }}
),
renamed as (
    select
        id          as order_id,
        customer_id,
        cast(created_at as timestamp) as ordered_at,
        amount_cents / 100.0          as amount
    from source
)
select * from renamed
```

**Use when:** staging layer, small tables, models not queried directly by BI tools.

**Avoid when:** the table is large (slow queries), or you have deep chains of views.

</details>

---

### Table Materialization

<details>
<summary><strong>How table materialization works</strong></summary>

dbt runs `CREATE TABLE AS SELECT`. Data is physically stored. On every run, dbt drops and recreates the full table.

```sql
{{ config(materialized='table') }}

select
    o.order_id,
    o.ordered_at,
    o.amount,
    c.full_name as customer_name
from {{ ref('stg_orders') }} o
left join {{ ref('stg_customers') }} c on o.customer_id = c.customer_id
```

**Use when:** marts, models queried by BI tools, large datasets where query speed matters.

**Avoid when:** the table is extremely large and a full rebuild is too slow (use incremental instead).

</details>

---

### Incremental Materialization

<details>
<summary><strong>How incremental materialization works</strong></summary>

On the first run, dbt builds the full table. On subsequent runs, it only processes new or changed rows and appends/upserts them.

```sql
{{
  config(
    materialized = 'incremental',
    unique_key   = 'order_id'
  )
}}

select
    order_id,
    customer_id,
    ordered_at,
    amount,
    status
from {{ ref('stg_orders') }}

{% if is_incremental() %}
  -- only process rows newer than the most recent row already in the table
  where ordered_at > (select max(ordered_at) from {{ this }})
{% endif %}
```

**Incremental strategies:**

| Strategy | Behaviour | Supported by |
|----------|-----------|--------------|
| `append` | Insert only new rows (no dedup) | All warehouses |
| `merge` | Upsert rows by `unique_key` | Snowflake, BigQuery, Databricks |
| `delete+insert` | Delete matching rows, re-insert | Snowflake, Redshift |
| `insert_overwrite` | Overwrite entire partitions | BigQuery, Spark |

```sql
{{ config(
    materialized = 'incremental',
    unique_key   = 'order_id',
    incremental_strategy = 'merge'
) }}
```

Force a full refresh:
```bash
dbt run -s orders --full-refresh
```

**Use when:** large event tables, logs, any table where rebuilding everything each run is too slow/expensive.

</details>

---

### Ephemeral Materialization

<details>
<summary><strong>How ephemeral materialization works</strong></summary>

An ephemeral model is never written to the warehouse. Instead, dbt injects it as a CTE into any model that references it.

```sql
{{ config(materialized='ephemeral') }}

select
    order_id,
    amount,
    case
        when amount >= 100 then 'high_value'
        when amount >= 50  then 'mid_value'
        else 'low_value'
    end as order_tier
from {{ ref('stg_orders') }}
```

When a downstream model calls `ref('orders_with_tier')`, dbt inlines the SQL as a CTE:

```sql
-- compiled output of downstream model
with orders_with_tier as (
    -- ephemeral model SQL injected here
    select order_id, amount, case ... end as order_tier
    from dev.stg_orders
),
final as (
    select * from orders_with_tier where order_tier = 'high_value'
)
select * from final
```

**Use when:** small helper transformations you don't want cluttering your warehouse schema.

**Avoid when:** multiple models reference it — the SQL gets duplicated into each.

</details>

---

### Setting materializations globally in dbt_project.yml

```yaml
# dbt_project.yml
name: my_project
version: '1.0.0'
config-version: 2

models:
  my_project:
    staging:
      +materialized: view       # all models in staging/ → view
    intermediate:
      +materialized: view       # all models in intermediate/ → view
    marts:
      +materialized: table      # all models in marts/ → table
      finance:
        +materialized: table
        +tags: ['finance']
```

The `+` prefix applies the config to the folder and all subfolders. An inline `config()` block in a model file always overrides the `dbt_project.yml` default.

---

## Jinja & Macros in Depth

<details>
<summary><strong>Jinja basics — if, for, variables</strong></summary>

dbt uses Jinja2 templating. Anything inside `{{ }}` is rendered; `{% %}` is control flow.

```sql
-- if statement
{% if target.name == 'prod' %}
    where is_deleted = false
{% else %}
    where true   -- include everything in dev
{% endif %}

-- for loop
select
    {% for payment_method in ['credit_card', 'bank_transfer', 'voucher'] %}
        sum(case when payment_method = '{{ payment_method }}'
                 then amount else 0 end)
            as {{ payment_method }}_amount
        {% if not loop.last %},{% endif %}
    {% endfor %}
from {{ ref('payments') }}

-- set a variable
{% set my_date = '2024-01-01' %}
where ordered_at >= '{{ my_date }}'
```

</details>

<details>
<summary><strong>Writing and calling macros</strong></summary>

```sql
-- macros/safe_divide.sql
{% macro safe_divide(numerator, denominator) %}
    case
        when {{ denominator }} = 0 or {{ denominator }} is null then null
        else {{ numerator }} / {{ denominator }}
    end
{% endmacro %}
```

```sql
-- usage in a model
select
    order_id,
    {{ safe_divide('revenue', 'items_count') }} as revenue_per_item
from {{ ref('orders') }}
```

</details>

<details>
<summary><strong>run_query — execute SQL inside a macro</strong></summary>

```sql
{% macro get_column_values(table, column) %}
    {% set query %}
        select distinct {{ column }}
        from {{ table }}
        order by 1
    {% endset %}
    {% set results = run_query(query) %}
    {% if execute %}
        {{ return(results.columns[0].values()) }}
    {% endif %}
{% endmacro %}
```

</details>

---

## Packages

<details>
<summary><strong>Installing and using packages</strong></summary>

dbt packages are shared libraries of macros and models. Add them to `packages.yml`:

```yaml
# packages.yml
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
  - package: calogica/dbt_expectations
    version: 0.10.0
  - package: dbt-labs/dbt_project_evaluator
    version: 0.8.0
```

Install:
```bash
dbt deps
```

**dbt_utils — most-used macros:**

```sql
-- surrogate key from multiple columns
{{ dbt_utils.generate_surrogate_key(['order_id', 'line_item_id']) }}

-- pivot rows to columns
{{ dbt_utils.pivot('payment_method', dbt_utils.get_column_values(ref('payments'), 'payment_method')) }}

-- union multiple tables
{{ dbt_utils.union_relations(relations=[ref('orders_2022'), ref('orders_2023')]) }}

-- date spine (generate a row for every date in a range)
{{ dbt_utils.date_spine(
    datepart   = 'day',
    start_date = "'2024-01-01'",
    end_date   = "'2024-12-31'"
) }}
```

**dbt_expectations — Great Expectations-style tests:**

```yaml
- name: amount
  tests:
    - dbt_expectations.expect_column_values_to_be_between:
        min_value: 0
        max_value: 100000
    - dbt_expectations.expect_column_to_exist
    - dbt_expectations.expect_column_values_to_not_be_null
```

</details>

---

## Variables

<details>
<summary><strong>Defining and using variables</strong></summary>

Variables let you pass dynamic values into your models.

**Define in `dbt_project.yml`:**
```yaml
vars:
  start_date: '2024-01-01'
  payment_methods: ['credit_card', 'bank_transfer']
  is_test_run: false
```

**Use in a model:**
```sql
where ordered_at >= '{{ var("start_date") }}'

{% if var("is_test_run") %}
    limit 100
{% endif %}
```

**Override at runtime:**
```bash
dbt run --vars '{"start_date": "2023-01-01", "is_test_run": true}'
```

**Variable with default fallback:**
```sql
{{ var('start_date', '2020-01-01') }}
-- uses 2020-01-01 if start_date is not set anywhere
```

</details>

---

## Environments & Profiles

<details>
<summary><strong>profiles.yml — warehouse connections</strong></summary>

Located at `~/.dbt/profiles.yml` (outside the project, so credentials aren't checked into git):

```yaml
# ~/.dbt/profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: xy12345.us-east-1
      user: "{{ env_var('DBT_USER') }}"
      password: "{{ env_var('DBT_PASSWORD') }}"
      role: transformer
      database: analytics
      warehouse: dev_wh
      schema: dbt_yourname        # personal dev schema
      threads: 4

    prod:
      type: snowflake
      account: xy12345.us-east-1
      user: "{{ env_var('DBT_PROD_USER') }}"
      password: "{{ env_var('DBT_PROD_PASSWORD') }}"
      role: transformer_prod
      database: analytics
      warehouse: prod_wh
      schema: prod
      threads: 8
```

Run against a specific target:
```bash
dbt run --target prod
dbt run --target dev    # default
```

</details>

<details>
<summary><strong>target object — environment-aware SQL</strong></summary>

The `target` object exposes the current environment in Jinja:

```sql
-- limit rows in dev to keep things fast
select * from {{ ref('events') }}
{% if target.name != 'prod' %}
    limit 1000
{% endif %}

-- use a cheaper warehouse in dev
{{ config(
    snowflake_warehouse = 'prod_wh' if target.name == 'prod' else 'dev_wh'
) }}
```

</details>

---

## Advanced Topics

### Snapshots (Slowly Changing Dimensions)

<details>
<summary><strong>Timestamp strategy</strong></summary>

Use when your source table has a reliable `updated_at` column:

```sql
{% snapshot customers_snapshot %}
{{
  config(
    target_schema = 'snapshots',
    unique_key    = 'customer_id',
    strategy      = 'timestamp',
    updated_at    = 'updated_at'
  )
}}
select * from {{ source('raw', 'customers') }}
{% endsnapshot %}
```

</details>

<details>
<summary><strong>Check strategy</strong></summary>

Use when there's no `updated_at` — dbt hashes specified columns to detect changes:

```sql
{% snapshot orders_snapshot %}
{{
  config(
    target_schema = 'snapshots',
    unique_key    = 'order_id',
    strategy      = 'check',
    check_cols    = ['status', 'amount']
  )
}}
select * from {{ source('raw', 'orders') }}
{% endsnapshot %}
```

</details>

---

### Exposures

<details>
<summary><strong>Documenting downstream consumers</strong></summary>

Exposures document what uses your dbt models — dashboards, ML models, reverse ETL, etc. They appear in the lineage graph so you can see the blast radius of a model change.

```yaml
# models/marts/exposures.yml
version: 2

exposures:
  - name: orders_dashboard
    type: dashboard
    maturity: high
    url: https://looker.mycompany.com/dashboards/42
    description: "Executive orders dashboard — refreshed daily"
    depends_on:
      - ref('orders')
      - ref('customers')
    owner:
      name: Data Team
      email: data@mycompany.com

  - name: churn_prediction_model
    type: ml
    maturity: medium
    description: "Propensity-to-churn model trained weekly"
    depends_on:
      - ref('customers')
    owner:
      name: ML Team
      email: ml@mycompany.com
```

</details>

---

### Semantic Layer & Metrics

<details>
<summary><strong>Defining metrics with MetricFlow</strong></summary>

dbt's semantic layer (powered by MetricFlow) lets you define metrics once and query them consistently across any BI tool.

```yaml
# models/marts/metrics.yml
semantic_models:
  - name: orders
    defaults:
      agg_time_dimension: ordered_at
    model: ref('orders')
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

</details>

---

### Custom Schemas

<details>
<summary><strong>Controlling where models land in the warehouse</strong></summary>

By default, all models land in the schema defined in `profiles.yml`. Override with config:

```sql
{{ config(schema='finance') }}
-- lands in: analytics.dbt_yourname_finance (dev) or analytics.finance (prod)
```

Customise the schema-naming logic by overriding `generate_schema_name` in `macros/`:

```sql
-- macros/generate_schema_name.sql
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {%- if target.name == 'prod' -%}
            {{ custom_schema_name | trim }}           -- prod: just the custom name
        {%- else -%}
            {{ default_schema }}_{{ custom_schema_name | trim }}  -- dev: prefix with dev schema
        {%- endif -%}
    {%- endif -%}
{%- endmacro %}
```

</details>

---

### CI/CD with dbt

<details>
<summary><strong>GitHub Actions workflow</strong></summary>

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

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dbt
        run: pip install dbt-snowflake

      - name: dbt deps
        run: dbt deps

      - name: dbt build (slim CI — only changed models + downstream)
        run: |
          dbt build \
            --select state:modified+ \
            --defer \
            --state ./prod-artifacts
        env:
          DBT_USER:     ${{ secrets.DBT_USER }}
          DBT_PASSWORD: ${{ secrets.DBT_PASSWORD }}
```

**Slim CI** with `state:modified+` only builds models that changed in the PR and their downstream dependencies — much faster than rebuilding everything.

</details>

---

## Key Commands

```bash
# ── Project setup ──────────────────────────────────────────────
dbt init my_project          # scaffold a new project
dbt debug                    # test warehouse connection
dbt deps                     # install packages from packages.yml

# ── Running models ─────────────────────────────────────────────
dbt run                      # build all models
dbt run -s stg_orders        # build one model
dbt run -s +orders           # build orders + all upstream
dbt run -s orders+           # build orders + all downstream
dbt run -s staging.*         # build all models in the staging/ folder
dbt run -s tag:finance       # build all models tagged 'finance'
dbt run --target prod        # run against production
dbt run --full-refresh       # rebuild incremental models from scratch

# ── Testing ────────────────────────────────────────────────────
dbt test                     # run all tests
dbt test -s orders           # test one model
dbt test --store-failures    # save failing rows to a table
dbt source freshness         # check source table freshness

# ── Building (run + test + seed + snapshot) ────────────────────
dbt build                    # run everything
dbt build -s +orders         # build and test orders + all upstream

# ── Docs ───────────────────────────────────────────────────────
dbt docs generate            # compile documentation
dbt docs serve               # serve docs locally at localhost:8080

# ── Utilities ──────────────────────────────────────────────────
dbt compile -s orders        # compile SQL without running (writes to target/)
dbt seed                     # load CSV seeds into warehouse
dbt snapshot                 # run snapshots (SCD tracking)
dbt clean                    # delete target/ and dbt_packages/
dbt ls -s tag:staging        # list models matching a selector
```

---

## Quick Reference Cheatsheet

| Concept | Purpose | File location |
|---------|---------|---------------|
| `model` | A SQL SELECT → table/view in warehouse | `models/**/*.sql` |
| `source()` | Reference a raw table you don't own | `models/**/_sources.yml` |
| `ref()` | Reference another dbt model | Inside any `.sql` model |
| `config()` | Set materialization, tags, schema per model | Top of `.sql` file |
| `schema.yml` | Declare tests + descriptions | `models/**/_*.yml` |
| `macro` | Reusable Jinja function | `macros/*.sql` |
| `seed` | Load a CSV as a warehouse table | `seeds/*.csv` |
| `snapshot` | Track row history (SCD type 2) | `snapshots/*.sql` |
| `exposure` | Document downstream consumers | `models/**/*.yml` |
| `packages.yml` | Install third-party macro libraries | Project root |
| `dbt_project.yml` | Project-level config + folder materializations | Project root |
| `profiles.yml` | Warehouse connection credentials | `~/.dbt/` |

| Materialization | Stores data? | Rebuilt how? | Best for |
|-----------------|-------------|--------------|----------|
| `view` | No | SQL re-runs on every query | Staging, light transforms |
| `table` | Yes | Full drop + recreate each run | Marts, BI-facing models |
| `incremental` | Yes | Append/upsert new rows only | Large event tables, logs |
| `ephemeral` | No | Injected as CTE into downstream | Small helper logic |

---

## Resources

- [dbt Documentation](https://docs.getdbt.com)
- [dbt Discourse (community forum)](https://discourse.getdbt.com)
- [dbt Slack community](https://community.getdbt.com)
- [dbt Package Hub](https://hub.getdbt.com)
- [dbt_utils package](https://github.com/dbt-labs/dbt-utils)
- [dbt_expectations package](https://github.com/calogica/dbt-expectations)
- [MetricFlow (semantic layer)](https://docs.getdbt.com/docs/build/about-metricflow)

---
