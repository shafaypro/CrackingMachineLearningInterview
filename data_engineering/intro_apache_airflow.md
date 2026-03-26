# Apache Airflow – Complete Guide (2026 Edition)

**Apache Airflow** is the most widely used workflow orchestration platform. It lets you define, schedule, and monitor data pipelines as code using Python.

---

## What is Airflow?

Airflow orchestrates the **when** and **how** of running tasks:

```
Every day at 2am:
  1. Extract data from Salesforce API     (3 min)
  2. Load to S3                           (2 min)
         ↓
  3a. Run dbt models (staging)            (5 min) ─┐
  3b. Run data quality checks             (2 min) ─┤
         ↓                                          │
  4. Run dbt models (marts)               (8 min) ←┘
  5. Notify Slack on completion           (instant)
```

Airflow defines this as a **DAG** (Directed Acyclic Graph) in Python.

### Airflow vs Alternatives

| Tool | Style | Best For |
|------|-------|---------|
| **Airflow** | Python code, task-centric | Complex pipelines, ML workflows |
| **Prefect** | Python code, flow-centric | Developer experience, dynamic workflows |
| **Dagster** | Asset-centric, typed | Data assets, lineage, testing |
| **Mage** | Notebook-style | Data engineers who love notebooks |
| **Temporal** | Workflow engine | Long-running, stateful workflows |

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **DAG** | Directed Acyclic Graph — a workflow defined in Python |
| **Task** | A unit of work within a DAG |
| **Operator** | Template for a task type (PythonOperator, BashOperator, etc.) |
| **TaskFlow API** | Modern way to write DAGs using `@task` decorators |
| **Dependency** | `task_a >> task_b` means task_b runs after task_a |
| **Schedule** | When the DAG runs (cron, timedelta, or @daily) |
| **DAG Run** | A specific execution of a DAG |
| **XCom** | Cross-communication: pass data between tasks |
| **Connection** | Stored credentials for external systems |
| **Variable** | Key-value store for DAG configuration |
| **Pool** | Limit concurrency for resource-intensive tasks |

---

## Installation

```bash
# Local (virtualenv)
pip install apache-airflow==2.10.0

# Initialize database
airflow db init

# Start webserver + scheduler
airflow webserver --port 8080 &
airflow scheduler &

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Docker Compose (recommended)

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d
```

---

## Writing DAGs

### TaskFlow API (modern, recommended)

```python
# dags/my_pipeline.py
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests
import json

@dag(
    dag_id="daily_sales_pipeline",
    description="Extract sales data, transform, load to warehouse",
    schedule="0 2 * * *",          # 2am daily (cron)
    start_date=days_ago(1),
    catchup=False,                  # don't backfill past runs
    max_active_runs=1,              # prevent concurrent runs
    tags=["sales", "etl"],
    default_args={
        "owner": "data-team",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
        "email": ["data-alerts@company.com"],
    }
)
def daily_sales_pipeline():

    @task
    def extract_sales() -> list[dict]:
        """Extract sales from API."""
        response = requests.get(
            "https://api.company.com/sales",
            params={"date": "{{ ds }}"},     # Jinja template: execution date
            headers={"Authorization": "Bearer ..."}
        )
        return response.json()["sales"]

    @task
    def validate_data(sales: list[dict]) -> list[dict]:
        """Validate and clean data."""
        valid = [s for s in sales if s.get("amount", 0) > 0]
        if len(valid) == 0:
            raise ValueError("No valid sales records found!")
        print(f"Validated {len(valid)} / {len(sales)} records")
        return valid

    @task
    def load_to_warehouse(sales: list[dict]) -> str:
        """Load to Snowflake."""
        import snowflake.connector
        conn = snowflake.connector.connect(
            account=Variable.get("snowflake_account"),
            user=Variable.get("snowflake_user"),
            password=Variable.get("snowflake_password"),
        )
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO raw.sales VALUES (%s, %s, %s)",
            [(s["id"], s["amount"], s["date"]) for s in sales]
        )
        return f"Loaded {len(sales)} records"

    @task
    def run_dbt_models(load_result: str):
        """Run dbt transformations."""
        import subprocess
        result = subprocess.run(
            ["dbt", "run", "--models", "tag:sales", "--profiles-dir", "/etc/dbt"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise Exception(f"dbt failed: {result.stderr}")
        print(result.stdout)

    @task
    def notify_success(dbt_result):
        """Send Slack notification."""
        import requests
        requests.post(
            Variable.get("slack_webhook_url"),
            json={"text": "✅ Daily sales pipeline completed successfully!"}
        )

    # Define dependencies using >> operator
    sales = extract_sales()
    validated = validate_data(sales)
    loaded = load_to_warehouse(validated)
    dbt_result = run_dbt_models(loaded)
    notify_success(dbt_result)

# Instantiate the DAG
daily_sales_pipeline()
```

### Classic Operators (older style, still widely used)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.dates import days_ago

def my_python_fn(**context):
    execution_date = context["ds"]
    print(f"Running for date: {execution_date}")

with DAG(
    "classic_dag",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
) as dag:

    extract = SimpleHttpOperator(
        task_id="extract",
        http_conn_id="sales_api",
        endpoint="/api/sales",
        method="GET",
        response_filter=lambda r: r.json(),
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=my_python_fn,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="dbt run --models tag:sales --profiles-dir /etc/dbt",
    )

    # Dependencies
    extract >> transform >> dbt_run
```

---

## Common Operators

```python
# Python
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.decorators import task

# Bash
from airflow.operators.bash import BashOperator

# dbt
from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig  # astronomer-cosmos

# Email
from airflow.operators.email import EmailOperator

# Trigger another DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Sensors (wait for something to be true)
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.providers.http.sensors.http import HttpSensor

# Data warehouse
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.providers.amazon.aws.operators.redshift_sql import RedshiftSQLOperator

# Cloud
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
```

---

## Advanced DAG Patterns

### Dynamic Task Mapping

```python
@dag(schedule="@daily", start_date=days_ago(1))
def parallel_processing():

    @task
    def get_tables() -> list[str]:
        return ["orders", "customers", "products", "returns"]

    @task
    def process_table(table: str) -> str:
        # This will create one task per table, running in parallel
        print(f"Processing table: {table}")
        subprocess.run(["dbt", "run", "--models", table])
        return f"Done: {table}"

    @task
    def summarize(results: list[str]):
        print(f"All done! Processed: {results}")

    tables = get_tables()
    processed = process_table.expand(table=tables)   # dynamic task mapping!
    summarize(processed)

parallel_processing()
```

### Branching

```python
@dag(schedule="@daily", start_date=days_ago(1))
def branching_dag():

    @task.branch
    def decide_branch(**context) -> str:
        day = context["logical_date"].weekday()
        if day == 6:  # Sunday
            return "full_load"
        return "incremental_load"

    @task
    def full_load():
        print("Running full load (Sunday)")

    @task
    def incremental_load():
        print("Running incremental load")

    branch = decide_branch()
    branch >> [full_load(), incremental_load()]

branching_dag()
```

### Sensors

```python
from airflow.sensors.filesystem import FileSensor

wait_for_file = FileSensor(
    task_id="wait_for_file",
    filepath="/data/input/{{ ds }}/sales.csv",
    poke_interval=300,   # check every 5 minutes
    timeout=3600,        # fail after 1 hour
    mode="reschedule",   # don't hold a worker slot while waiting
)
```

### XComs (cross-task communication)

```python
@task
def extract() -> dict:
    return {"count": 1234, "source": "salesforce"}

@task
def report(stats: dict):
    # stats is the return value of extract()
    print(f"Extracted {stats['count']} records from {stats['source']}")

# In TaskFlow, return values are automatically XComs
stats = extract()
report(stats)
```

---

## dbt + Airflow Integration

### Using Astronomer Cosmos

```python
from cosmos import DbtDag, ProjectConfig, ProfileConfig, ExecutionConfig, RenderConfig
from cosmos.profiles import SnowflakeUserPasswordProfileMapping
from pathlib import Path

dbt_dag = DbtDag(
    dag_id="dbt_pipeline",
    project_config=ProjectConfig(
        dbt_project_path=Path("/usr/local/airflow/dbt"),
    ),
    profile_config=ProfileConfig(
        profile_name="my_profile",
        target_name="prod",
        profile_mapping=SnowflakeUserPasswordProfileMapping(
            conn_id="snowflake_conn",
            profile_args={"database": "PROD", "schema": "ANALYTICS"},
        ),
    ),
    execution_config=ExecutionConfig(
        dbt_executable_path="/usr/local/bin/dbt",
    ),
    render_config=RenderConfig(
        select=["tag:daily"],     # only run daily-tagged models
    ),
    schedule="@daily",
    start_date=days_ago(1),
)
```

---

## Variables & Connections

```python
from airflow.models import Variable

# In DAG code
my_var = Variable.get("my_variable")
json_var = Variable.get("my_json_var", deserialize_json=True)

# With default
api_key = Variable.get("api_key", default_var="fallback_key")
```

```bash
# CLI
airflow variables set my_variable "hello"
airflow variables get my_variable

# Connections
airflow connections add snowflake_prod \
  --conn-type snowflake \
  --host myaccount.snowflakecomputing.com \
  --login myuser \
  --password secret
```

---

## Monitoring & Alerts

```python
from airflow.utils.email import send_email

def on_failure_callback(context):
    dag_id = context["dag"].dag_id
    task_id = context["task"].task_id
    exec_date = context["ds"]
    exception = context.get("exception")

    # Send Slack alert
    import requests
    requests.post(
        Variable.get("slack_webhook"),
        json={
            "text": f"❌ Task Failed!\nDAG: {dag_id}\nTask: {task_id}\nDate: {exec_date}\nError: {exception}"
        }
    )

# Use in DAG default_args
default_args = {
    "on_failure_callback": on_failure_callback,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}
```

---

## Deployment (Kubernetes)

```bash
# Install Airflow via Helm on Kubernetes
helm repo add apache-airflow https://airflow.apache.org
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --create-namespace \
  -f values.yaml

# values.yaml key settings:
# executor: KubernetesExecutor   ← each task in its own pod
# dags.gitSync.enabled: true     ← sync DAGs from Git
```

---

## Airflow in 2026

| Feature | Description |
|---------|-------------|
| **Airflow 3.0** | Major release: asset-based scheduling, improved UI, edge labels, decoupled scheduler |
| **AIP-72 (Assets)** | Schedule DAGs based on data readiness, not just time |
| **Airflow ObjectStore** | Built-in artifact storage |
| **Task SDK** | Separate package for writing tasks without full Airflow dependency |
| **Astronomer** | Managed Airflow cloud service (most popular) |
| **MWAA** | AWS Managed Workflows for Apache Airflow |
| **Cloud Composer** | GCP's managed Airflow |

### Asset-based Scheduling (Airflow 3.0)

```python
from airflow.sdk import Asset

sales_asset = Asset("s3://bucket/sales/{{ ds }}/")

@dag(schedule=sales_asset)   # runs when asset is updated
def process_sales():
    ...

@dag(schedule="@daily")
def extract_sales():
    @task(outlets=[sales_asset])  # marks asset as updated on completion
    def extract():
        ...
```

---

## Cheat Sheet

```bash
# CLI
airflow dags list
airflow dags trigger my_dag
airflow dags trigger my_dag --exec-date 2025-01-01
airflow dags backfill my_dag --start-date 2025-01-01 --end-date 2025-01-31
airflow tasks test my_dag my_task 2025-01-01
airflow dags pause my_dag
airflow dags unpause my_dag

# Variables
airflow variables set KEY VALUE
airflow variables get KEY
airflow variables import vars.json
airflow variables export vars.json
```

```python
# Schedule strings
schedule = "0 2 * * *"      # 2am daily
schedule = "@daily"          # midnight daily
schedule = "@weekly"         # midnight Sunday
schedule = "@monthly"        # midnight 1st of month
schedule = "@hourly"         # top of every hour
schedule = None              # manual trigger only
schedule = timedelta(hours=6) # every 6 hours

# Task dependencies
a >> b           # a then b
a >> [b, c]      # a then b and c in parallel
[a, b] >> c      # both a and b must complete before c
a.set_downstream(b)   # same as a >> b
```
