# Model Monitoring Guide

A comprehensive guide to monitoring ML models in production — detecting drift, measuring performance, and knowing when to retrain.

---

## Table of Contents

1. [Types of Drift](#types-of-drift)
2. [Statistical Tests for Drift Detection](#statistical-tests-for-drift-detection)
3. [Monitoring Metrics](#monitoring-metrics)
4. [Monitoring Tools](#monitoring-tools)
5. [Code Examples with Evidently AI](#code-examples-with-evidently-ai)
6. [When to Retrain](#when-to-retrain)
7. [Interview Q&A](#interview-qa)
8. [References](#references)

---

## Types of Drift

### Data Drift (Covariate Shift)

The distribution of input features changes while the relationship between inputs and outputs remains the same.

```
P_train(X) ≠ P_prod(X)
P_train(Y|X) = P_prod(Y|X)
```

**Example:** A loan approval model trained on pre-pandemic data sees different income and debt distributions post-pandemic.

**Detection:** Statistical tests on feature distributions (KS test, PSI).

### Concept Drift

The relationship between inputs and outputs changes.

```
P_train(Y|X) ≠ P_prod(Y|X)
```

**Example:** Fraud patterns evolve — transactions that were previously legitimate are now fraudulent.

**Detection:** Requires ground truth labels (delayed feedback). Monitor proxy metrics or use windowed holdout sets.

**Types of concept drift:**
- **Sudden drift:** Abrupt change (e.g., system update, rule change)
- **Gradual drift:** Slow shift over time
- **Recurring drift:** Seasonal patterns (e.g., holiday fraud patterns)
- **Incremental drift:** Monotonic progression

### Model Drift / Performance Drift

The downstream effect of data or concept drift — the model's predictions become less accurate or useful over time.

**Detection:** Monitor business KPIs, accuracy metrics (when labels available), and prediction score distributions.

### Label Shift (Prior Probability Shift)

The distribution of the target variable changes while the class-conditional distribution stays the same.

```
P_train(Y) ≠ P_prod(Y)
P_train(X|Y) = P_prod(X|Y)
```

**Example:** The base rate of fraud increases during an economic crisis.

---

## Statistical Tests for Drift Detection

### Kolmogorov-Smirnov (KS) Test

Tests whether two continuous distributions are drawn from the same distribution.

**Null hypothesis:** The two distributions are identical.

```python
from scipy import stats
import numpy as np

# Reference distribution (training data)
reference = np.random.normal(0, 1, 1000)

# Current distribution (production data)
current = np.random.normal(0.5, 1.2, 500)

ks_stat, p_value = stats.ks_2samp(reference, current)

print(f"KS Statistic: {ks_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("DRIFT DETECTED: distributions are significantly different")
else:
    print("No significant drift detected")
```

**Interpretation:**
- KS statistic: 0 = identical distributions, 1 = completely different
- p-value < 0.05: reject null hypothesis → drift detected

### Population Stability Index (PSI)

Measures how much a distribution has shifted from a reference distribution. Widely used in banking/credit risk.

```python
import numpy as np

def calculate_psi(reference, current, buckets=10):
    """Calculate PSI between reference and current distributions."""
    # Create bins based on reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to proportions, avoid division by zero
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

    psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi = np.sum(psi_values)
    return psi

psi = calculate_psi(reference_data, production_data)

if psi < 0.1:
    status = "No significant change"
elif psi < 0.25:
    status = "Moderate change — investigate"
else:
    status = "Significant change — retrain required"

print(f"PSI: {psi:.4f} — {status}")
```

### Chi-Squared Test (Categorical Features)

```python
from scipy.stats import chi2_contingency
import numpy as np

# For categorical features
reference_counts = np.array([150, 200, 100, 50])
current_counts = np.array([120, 250, 80, 30])

# Build contingency table
contingency = np.array([reference_counts, current_counts])

chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"Chi-squared: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("DRIFT DETECTED: categorical distribution has shifted")
```

### Jensen-Shannon Divergence

```python
from scipy.spatial.distance import jensenshannon
import numpy as np

def js_divergence(p, q, bins=50):
    """Jensen-Shannon divergence between two distributions."""
    hist_p, bin_edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

    # Normalize
    hist_p = hist_p / hist_p.sum() + 1e-10
    hist_q = hist_q / hist_q.sum() + 1e-10

    return jensenshannon(hist_p, hist_q)

js = js_divergence(reference, current)
print(f"JS Divergence: {js:.4f}")  # 0 = identical, 1 = completely different
```

---

## Monitoring Metrics

### Infrastructure Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Latency (p95) | 95th percentile response time | > 200ms |
| Latency (p99) | 99th percentile response time | > 500ms |
| Throughput | Requests per second | Drop > 20% |
| Error rate | % of requests returning 5xx | > 1% |
| Memory usage | Memory consumed by serving | > 80% |
| GPU utilization | GPU usage for inference | < 20% (underutilized) |

### Data Quality Metrics

| Metric | Description |
|--------|-------------|
| Missing rate | % of null values per feature |
| Schema violations | Type mismatches, unexpected columns |
| Out-of-range values | Values outside expected min/max |
| Feature drift (PSI) | PSI score per feature |

### Model Performance Metrics

| Metric | Type | Requires Labels |
|--------|------|----------------|
| Accuracy / F1 / AUC | Classification | Yes |
| MAE / RMSE | Regression | Yes |
| Prediction score distribution | Any | No |
| Calibration | Classification | Yes |
| Business KPIs | Domain-specific | Proxy available |

---

## Monitoring Tools

| Tool | Strengths | Best For |
|------|-----------|----------|
| **Evidently AI** | Open source, rich reports, drift tests | Quick setup, data scientists |
| **WhyLogs** | Lightweight, streaming, Apache license | High-volume pipelines |
| **Arize AI** | Enterprise, model observability platform | Production ML teams |
| **Fiddler AI** | Explainability + monitoring | Regulated industries |
| **Grafana + Prometheus** | Infrastructure monitoring, customizable | Engineering-focused teams |
| **Seldon Alibi Detect** | Statistical drift detection library | Custom monitoring pipelines |

---

## Code Examples with Evidently AI

### Installation

```bash
pip install evidently
```

### Data Drift Report

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

# Reference data (training distribution)
reference_data = pd.read_csv("train_data.csv")

# Current data (recent production data)
current_data = pd.read_csv("production_data.csv")

# Create a drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save HTML report
report.save_html("drift_report.html")

# Get results as dict
results = report.as_dict()
drift_detected = results["metrics"][0]["result"]["dataset_drift"]
print(f"Dataset drift detected: {drift_detected}")
```

### Column-Level Drift

```python
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric

# Check specific columns
report = Report(metrics=[
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="income"),
    ColumnDriftMetric(column_name="credit_score"),
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

result = report.as_dict()
for metric in result["metrics"]:
    col = metric["result"]["column_name"]
    drift = metric["result"]["drift_detected"]
    score = metric["result"]["drift_score"]
    print(f"{col}: drift={drift}, score={score:.4f}")
```

### Model Performance Monitoring

```python
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

# When you have ground truth labels (delayed feedback)
report = Report(metrics=[
    ClassificationPreset(),
])

report.run(
    reference_data=reference_with_labels,
    current_data=current_with_labels,
    column_mapping={"target": "label", "prediction": "prediction"}
)
report.save_html("classification_report.html")
```

### Continuous Monitoring Pipeline

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns

def run_monitoring_check(reference_data, current_data, threshold=0.3):
    """Run drift checks and return alert status."""

    test_suite = TestSuite(tests=[
        TestNumberOfDriftedColumns(lt=3),  # Alert if more than 3 columns drift
        TestShareOfDriftedColumns(lt=threshold),  # Alert if > 30% of columns drift
    ])

    test_suite.run(
        reference_data=reference_data,
        current_data=current_data
    )

    results = test_suite.as_dict()
    all_passed = all(
        test["status"] == "SUCCESS"
        for test in results["tests"]
    )

    return all_passed, results

passed, results = run_monitoring_check(reference_data, production_data)
if not passed:
    print("ALERT: Model monitoring check failed — investigate drift")
    # Send alert to Slack, PagerDuty, etc.
```

---

## When to Retrain

### Trigger Conditions

| Trigger | Description | Priority |
|---------|-------------|----------|
| PSI ≥ 0.25 on key features | Significant data drift | High |
| Model accuracy drops > 5% | Performance degradation | High |
| Business KPI anomaly | Conversion/revenue drop | High |
| PSI 0.1–0.25 on multiple features | Moderate drift | Medium |
| Scheduled interval | Weekly/monthly safeguard | Low |
| New data available (large batch) | Proactive retraining | Low |

### Retraining Strategies

**Full retraining:** Train from scratch on all available data. Safe but slow.

**Incremental/online learning:** Update model weights with new data only. Fast but requires algorithms that support it.

**Windowed retraining:** Train on a sliding window of recent data. Balances historical and recent patterns.

```python
# Example: Triggering retraining based on PSI
def check_and_trigger_retrain(psi_scores, threshold=0.25):
    """Check PSI scores and trigger retraining if needed."""
    drifted_features = [
        feature for feature, psi in psi_scores.items()
        if psi >= threshold
    ]

    if drifted_features:
        print(f"Drift detected in: {drifted_features}")
        print("Triggering retraining pipeline...")
        # Trigger Airflow DAG, Kubeflow pipeline, etc.
        trigger_training_pipeline(
            reason="drift_detected",
            affected_features=drifted_features
        )
        return True
    return False
```

---

## Interview Q&A

**Q1: What is the difference between data drift and concept drift?** 🟡 Intermediate

Data drift (covariate shift) is when the input feature distribution changes `P(X)` while the relationship `P(Y|X)` remains the same. Concept drift is when the relationship between inputs and outputs changes `P(Y|X)`. Both lead to model degradation but require different interventions. Data drift can often be handled by retraining on new data; concept drift may require rethinking features or model architecture.

---

**Q2: What statistical tests would you use for drift detection?** 🟡 Intermediate

For continuous features: KS test (returns p-value and D-statistic) or PSI (interpretable thresholds). For categorical features: Chi-squared test. For probability distributions: Jensen-Shannon divergence or Wasserstein distance. The choice depends on feature type and interpretability requirements. PSI is preferred in banking because of its clear thresholds (< 0.1, 0.1-0.25, > 0.25).

---

**Q3: How do you monitor a model when you don't have ground truth labels immediately?** 🔴 Advanced

When labels are delayed (e.g., fraud confirmed days later, loan default confirmed months later):

1. Monitor input feature distributions using statistical tests (proxy for data drift)
2. Monitor prediction score distributions — shifts indicate potential concept drift
3. Monitor business proxy metrics (transaction decline rates, user complaints)
4. Use a held-out labeled set for periodic evaluation
5. Use model confidence scores as a proxy metric
6. Once labels arrive, run retrospective evaluation and trigger retraining

---

**Q4: What is PSI and what are its thresholds?** 🟡 Intermediate

PSI (Population Stability Index) measures distribution shift: `PSI = Σ (Actual% - Expected%) × ln(Actual%/Expected%)`. Thresholds: PSI < 0.1 (no significant change), 0.1 ≤ PSI < 0.25 (moderate change, investigate), PSI ≥ 0.25 (significant change, retrain). PSI is widely used in credit risk because it is interpretable and standardized.

---

**Q5: What monitoring would you set up for a fraud detection model?** 🔴 Advanced

- **Feature drift:** Monitor all input features (transaction amount, merchant category, time of day) using PSI and KS tests
- **Score distribution:** Watch for shifts in the fraud probability score distribution
- **Alert rate:** Monitor % of transactions flagged as fraud — anomalies indicate drift
- **Business metrics:** Track false positive rate (customer friction) and false negative rate (missed fraud)
- **Latency:** Ensure real-time scoring meets SLA (< 50ms)
- **Feedback loop:** When fraud confirmed, update training data and check if model would have caught it

---

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [WhyLogs Documentation](https://whylogs.readthedocs.io/)
- [Arize AI Blog on Model Monitoring](https://arize.com/blog/)
- [Chip Huyen — ML Monitoring (book chapter)](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)
- [PSI and CSI: Understanding Population Stability Index](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)
