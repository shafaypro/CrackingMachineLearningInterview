# Data Quality & Validation

Data quality issues are responsible for the majority of ML model failures in production. This guide covers how to detect, validate, and monitor data quality using Great Expectations, data contracts, and drift detection frameworks.

---

## Table of Contents
1. [Types of Data Quality Issues](#types-of-data-quality-issues)
2. [Great Expectations](#great-expectations)
3. [Data Contracts](#data-contracts)
4. [Pandera for DataFrame Validation](#pandera-for-dataframe-validation)
5. [Data Drift vs Concept Drift](#data-drift-vs-concept-drift)
6. [Drift Detection with Evidently AI](#drift-detection-with-evidently-ai)
7. [dbt Tests for Data Quality](#dbt-tests-for-data-quality)
8. [Interview Q&A](#interview-qa)
9. [Common Pitfalls](#common-pitfalls)
10. [Related Topics](#related-topics)

---

## Types of Data Quality Issues

| Issue | Description | Detection |
|-------|-------------|-----------|
| **Missing values** | Nulls/NaNs in required fields | Null rate monitoring |
| **Schema drift** | Column added/removed/renamed/type changed | Schema validation |
| **Distribution drift** | Feature distribution shifts over time | Statistical tests (KS, PSI) |
| **Concept drift** | Relationship between features and target changes | Model performance monitoring |
| **Outliers** | Values outside expected range | Z-score, IQR |
| **Duplicates** | Duplicate rows or keys | Uniqueness checks |
| **Referential integrity** | Foreign key values don't exist in parent table | Relationship tests |
| **Business rule violations** | Values violate domain rules (e.g., age < 0) | Custom constraints |
| **Data freshness** | Data not updated within expected window | Timestamp checks |

---

## Great Expectations

Great Expectations (GX) is the most popular open-source data quality framework. Define "expectations" (assertions about your data), run validations, and generate Data Docs reports.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Expectation** | A declarative assertion about data (e.g., column not null, values in range) |
| **Expectation Suite** | A collection of expectations for a dataset |
| **Validation Definition** | Pairs an expectation suite with a batch definition (the data to check) |
| **Checkpoint** | Runs one or more validation definitions and triggers actions (alerts, Data Docs) |
| **Data Docs** | Auto-generated HTML report of validation results |
| **Data Context** | The entry point: manages expectations, stores, and checkpoints |

### Installation & Setup

```bash
pip install great_expectations
```

The examples use GX Core 1.x. The `great_expectations init` CLI, `context.sources`, validators and `add_or_update_*` methods belong to the 0.x API and were removed in 1.0; the project context is now created from Python with `gx.get_context()`.

### Defining Expectations

```python
import great_expectations as gx
import pandas as pd

# Load data
df = pd.read_csv("users.csv")

# Create a data context
context = gx.get_context()

# Connect data: data source -> data asset -> batch definition
data_source = context.data_sources.add_pandas("users_datasource")
data_asset = data_source.add_dataframe_asset(name="users_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe("users_batch")

# Create an expectation suite (expectations are classes in GX 1.x)
suite = context.suites.add(gx.ExpectationSuite(name="users_suite"))
E = gx.expectations

# Define expectations
suite.add_expectation(E.ExpectColumnToExist(column="user_id"))
suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(E.ExpectColumnValuesToBeUnique(column="user_id"))
suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="email"))
suite.add_expectation(E.ExpectColumnValuesToMatchRegex(
    column="email", regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"))

# Numeric expectations
suite.add_expectation(E.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120))
suite.add_expectation(E.ExpectColumnValuesToBeBetween(column="salary", min_value=0, max_value=10_000_000))

# Distribution expectations
suite.add_expectation(E.ExpectColumnMeanToBeBetween(column="age", min_value=20, max_value=60))
suite.add_expectation(E.ExpectColumnStdevToBeBetween(column="salary", min_value=1000, max_value=100000))

# Categorical expectations
suite.add_expectation(E.ExpectColumnValuesToBeInSet(column="status", value_set=["active", "inactive", "pending"]))
suite.add_expectation(E.ExpectColumnProportionOfUniqueValuesToBeBetween(column="country", min_value=0.01, max_value=1.0))

# Null rate expectations
suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="age", mostly=0.95))  # Allow 5% nulls

# Quick interactive check against one batch
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
print(batch.validate(suite).success)
```

### Running Validations in a Pipeline

```python
import great_expectations as gx

context = gx.get_context()

# Pair the suite with the data it should validate
validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(name="users_validation", data=batch_definition, suite=suite)
)

# Create a checkpoint that runs the validation and updates Data Docs
checkpoint = context.checkpoints.add(gx.Checkpoint(
    name="users_checkpoint",
    validation_definitions=[validation_definition],
    actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
))

# Run the checkpoint on today's dataframe
result = checkpoint.run(batch_parameters={"dataframe": df})

# Check if all expectations passed
if result.success:
    print("All data quality checks passed")
else:
    # Get failed expectations
    for validation_result in result.run_results.values():
        failed = [r for r in validation_result.results if not r.success]
        for f in failed:
            print(f"FAILED: {f.expectation_config.type} on {f.expectation_config.kwargs}")

# Build and open Data Docs (HTML report)
context.build_data_docs()
```

---

## Data Contracts

A data contract is a formal agreement between data producers and consumers about the schema, quality, and SLAs of a dataset. It prevents "silent breaking changes" that cause downstream pipeline failures.

```yaml
# data_contract_users.yaml
name: users
version: "2.1.0"
owner: "data-platform-team"
description: "User profile data from the authentication service"
updated_at: "2026-01-15"

schema:
  fields:
    - name: user_id
      type: bigint
      nullable: false
      unique: true
      description: "Primary key"

    - name: email
      type: string
      nullable: false
      pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"

    - name: age
      type: integer
      nullable: true
      minimum: 0
      maximum: 120

    - name: signup_date
      type: date
      nullable: false

quality:
  null_rate_threshold: 0.02        # Max 2% nulls in any required column
  duplicate_rate_threshold: 0.001   # Max 0.1% duplicates
  freshness_threshold_hours: 24     # Data must be updated within 24 hours

sla:
  availability: "99.9%"
  latency_p99_ms: 500
```

```python
import yaml
import pandas as pd
from datetime import datetime, timedelta

def validate_contract(df: pd.DataFrame, contract_path: str) -> dict:
    """Validate a DataFrame against a data contract."""
    with open(contract_path) as f:
        contract = yaml.safe_load(f)

    violations = []

    for field in contract['schema']['fields']:
        col = field['name']

        # Check column exists
        if col not in df.columns:
            violations.append(f"Missing column: {col}")
            continue

        # Null check
        if not field.get('nullable', True):
            null_rate = df[col].isnull().mean()
            threshold = contract['quality']['null_rate_threshold']
            if null_rate > threshold:
                violations.append(f"{col}: null rate {null_rate:.3f} exceeds threshold {threshold}")

        # Range check
        if 'minimum' in field and pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] < field['minimum']).any():
                violations.append(f"{col}: values below minimum {field['minimum']}")

    return {"success": len(violations) == 0, "violations": violations}
```

---

## Pandera for DataFrame Validation

Pandera provides a lightweight, Pythonic way to validate DataFrames inline in code.

```python
import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd

# Schema definition
class UserSchema(pa.DataFrameModel):
    user_id: Series[int] = pa.Field(gt=0, unique=True)
    email: Series[str] = pa.Field(str_matches=r'^[\w.-]+@[\w.-]+\.\w+$')
    age: Series[float] = pa.Field(ge=0, le=120, nullable=True)
    salary: Series[float] = pa.Field(ge=0, nullable=True)
    status: Series[str] = pa.Field(isin=['active', 'inactive', 'pending'])

    class Config:
        coerce = True  # Auto-convert types if possible

# Validate
df = pd.read_csv("users.csv")

try:
    UserSchema.validate(df, lazy=True)  # lazy=True: collect all errors
    print("Validation passed")
except pa.errors.SchemaErrors as e:
    print(f"Validation failed:\n{e.failure_cases}")

# Use as decorator in pipeline functions
@pa.check_types
def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 120],
                              labels=['young', 'adult', 'senior', 'elderly'])
    return df
```

---

## Data Drift vs Concept Drift

| Type | Definition | Detection | Impact |
|------|-----------|-----------|--------|
| **Data/Covariate drift** | Input feature distribution P(X) changes | Statistical tests on features | Model predictions shift |
| **Label drift** | Target distribution P(Y) changes | Monitor prediction distribution | Class imbalance changes |
| **Concept drift** | Relationship P(Y\|X) changes | Monitor model accuracy over time | Model becomes stale |
| **Upstream data drift** | Source data schema or business logic changes | Schema monitoring | Pipeline breaks |

```
Data drift:  P_train(age) ≠ P_serve(age)   → features shifted
Concept drift: P_train(churn|features) ≠ P_serve(churn|features) → model is wrong
```

---

## Drift Detection with Evidently AI

```python
import pandas as pd
from evidently import Report  # Evidently 0.7+ API; the old one lives in evidently.legacy
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import DriftedColumnsCount, ValueDrift

# Reference data (training distribution) vs current data (production)
reference_df = pd.read_parquet("reference_data.parquet")
current_df = pd.read_parquet("current_data.parquet")

# Full data drift report
report = Report([
    DataDriftPreset(),         # Drift for all features
    DataSummaryPreset(),       # Null rates, distribution stats
])

snapshot = report.run(current_data=current_df, reference_data=reference_df)
snapshot.save_html("drift_report.html")

# Check results programmatically (first metric of the preset is DriftedColumnsCount)
result = snapshot.dict()
drift_share = result['metrics'][0]['value']['share']
print(f"Fraction of drifted columns: {drift_share:.2%}")

# Per-column drift
col_report = Report([
    ValueDrift(column="age"),
    ValueDrift(column="salary"),
    DriftedColumnsCount(drift_share=0.3),  # Dataset counts as drifted if >30% columns drift
])
col_snapshot = col_report.run(current_data=current_df, reference_data=reference_df)
```

### Statistical Tests for Drift

```python
from scipy import stats
import numpy as np

def detect_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Run multiple drift detection tests."""
    results = {}

    # Kolmogorov-Smirnov test (continuous features)
    ks_stat, ks_pval = stats.ks_2samp(reference, current)
    results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval, 'drifted': ks_pval < 0.05}

    # Population Stability Index (PSI)
    def psi(expected, actual, buckets=10):
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + 1e-10
        actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + 1e-10
        return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    psi_score = psi(reference, current)
    results['psi'] = {
        'score': psi_score,
        'drifted': psi_score > 0.2,  # PSI: < 0.1 no change, 0.1-0.2 slight, > 0.2 significant
        'severity': 'none' if psi_score < 0.1 else 'slight' if psi_score < 0.2 else 'significant'
    }

    return results
```

---

## dbt Tests for Data Quality

dbt (data build tool) includes built-in and custom data tests that run after transformations:

```yaml
# models/schema.yml
models:
  - name: users
    description: "Cleaned user data"
    columns:
      - name: user_id
        description: "Primary key"
        tests:
          - not_null
          - unique

      - name: email
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "email LIKE '%@%.%'"

      - name: age
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 120

      - name: status
        tests:
          - accepted_values:
              values: ['active', 'inactive', 'pending']

    tests:
      # Table-level tests
      - dbt_utils.recency:
          datepart: hour
          field: updated_at
          interval: 24   # Alert if no data in last 24 hours
```

---

## Interview Q&A

**Q1: What is the difference between data drift and concept drift?**
Data drift (covariate shift) is when the input feature distribution P(X) changes: e.g., users skew older, average transaction amounts increase. Concept drift is when the relationship between features and target P(Y|X) changes: the model's learned patterns no longer hold. Data drift can be detected by comparing feature distributions to a reference. Concept drift requires monitoring model accuracy and prediction distributions over time.

**Q2: How would you monitor data quality in a production ML pipeline?**
1. Schema validation at pipeline ingestion (Pandera, Great Expectations)
2. Statistical drift monitoring on features (KS test, PSI, Evidently)
3. Model performance monitoring (accuracy, F1 against labeled holdout)
4. Prediction distribution monitoring (alert if prediction mean shifts significantly)
5. Data freshness checks (alert if no data in expected window)
6. Automated alerts to Slack/PagerDuty with severity levels

**Q3: What is PSI (Population Stability Index) and how do you interpret it?**
PSI measures how much a distribution has shifted relative to a reference. Compute by bucketing both distributions, then PSI = Σ (actual% - reference%) × ln(actual% / reference%). Interpretation: < 0.1 no significant change; 0.1-0.2 slight change, monitor; > 0.2 significant drift, investigate and likely retrain.

**Q4: What are data contracts and why are they important?**
A data contract is a formal SLA between data producers (e.g., an engineering team) and data consumers (e.g., ML team or analytics). It specifies schema, quality expectations (null rates, uniqueness), freshness SLAs, and versioning. Without contracts, producers change schemas silently, breaking downstream ML pipelines, often discovered only when models produce wrong predictions.

**Q5: How do you handle schema drift in an ML pipeline?**
1. Schema validation at ingestion (fail fast, alert immediately)
2. Backward-compatible changes (adding nullable columns): handle in feature engineering with defaults
3. Breaking changes (column removal, type change): trigger pipeline pause + alert
4. Use Avro/Protobuf schemas with registries (Confluent Schema Registry for Kafka) to enforce compatibility
5. Version your feature engineering code alongside schema versions

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| No schema validation at ingestion | Silent data corruption propagates | Add schema validation as first pipeline step |
| Comparing to wrong reference data | False drift alerts or missed drift | Use stable reference period (30-day rolling baseline) |
| Alert fatigue from too many tests | Team ignores alerts | Prioritize critical expectations; use severity levels |
| No freshness monitoring | Stale data used for predictions | Add timestamp checks on all critical datasets |
| Testing only at training time | Production data quality not monitored | Run GX or Pandera in the serving pipeline too |
| No data lineage | Can't trace quality issues to source | Use tools like DataHub, Amundsen, or dbt lineage |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [MLflow](./intro_mlflow.md) | Log data quality metrics alongside model metrics |
| [Feature Stores](./intro_feature_stores.md) | Feature stores need quality validation before materialization |
| [dbt Introduction](../data_engineering/intro_dbt.md) | dbt tests are a primary tool for data quality in warehouses |
| [LLMOps](../ai_genai/intro_llmops.md) | LLM input/output quality monitoring |
| [Study Pattern](../docs/study-pattern.md) | Data Quality is an Intermediate (🟡) MLOps topic |
