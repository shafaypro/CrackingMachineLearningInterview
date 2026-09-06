# CI/CD for Machine Learning

Shipping a model is not shipping code. A traditional pipeline validates one artifact — the binary — against tests that are deterministic and fast. An ML pipeline has three coupled artifacts (code, data, model), tests that are statistical rather than binary, and a deploy whose correctness cannot be fully established before real traffic touches it. This guide covers the pipeline, the testing pyramid, the model registry and promotion gates, deployment strategies, and rollback.

---

## Table of Contents
1. [Why ML CI/CD Is Different](#why-ml-cicd-is-different)
2. [The Pipeline End to End](#the-pipeline-end-to-end)
3. [The ML Testing Pyramid](#the-ml-testing-pyramid)
4. [Model Registry and Promotion Gates](#model-registry-and-promotion-gates)
5. [Deployment Strategies](#deployment-strategies)
6. [Rollback](#rollback)
7. [Continuous Training](#continuous-training)
8. [Reproducibility Requirements](#reproducibility-requirements)
9. [A Worked GitHub Actions Pipeline](#a-worked-github-actions-pipeline)
10. [Maturity Levels](#maturity-levels)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## Why ML CI/CD Is Different

| Dimension | Traditional software | Machine learning |
|---|---|---|
| Versioned artifacts | Code | Code **+ data + model + features** |
| Test outcome | Pass / fail, deterministic | Statistical — "AUC ≥ 0.82", threshold-dependent |
| CI duration | Seconds to minutes | Minutes to hours (training) |
| Correct at deploy time? | Provable by tests | Only observable under real traffic |
| Degrades while idle? | No | **Yes** — data drift moves the world under a static model |
| Rollback unit | Previous binary | Previous model **and** its feature pipeline |
| Triggers | Code commit | Code commit, **new data**, drift alert, schedule |

The two rows that generate most interview follow-ups: models degrade without anyone touching them, so the pipeline needs triggers other than commits; and a rollback must revert the model *together with* the feature transformations it expects, or you get a fresh skew bug on top of the original problem.

---

## The Pipeline End to End

```
   commit / new data / drift alert / schedule
              │
              ▼
   ┌──────────────────────┐
   │ CI: code + data      │  lint, unit tests, schema + distribution checks
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Train                │  pinned data version, seeded, tracked
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Evaluate             │  holdout + slices + baseline comparison
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Register (staging)   │  model + metrics + lineage + signature
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Gates                │  quality, fairness, latency, size, skew check
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Deploy: shadow →     │  progressive exposure with automated rollback
   │ canary → full        │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Monitor              │  performance, drift, latency → feeds triggers
   └──────────────────────┘
```

The loop closing back on triggers is the part that distinguishes an ML pipeline from a code pipeline. Monitoring is not a dashboard at the end; it is an input to the next run.

---

## The ML Testing Pyramid

Fast and cheap at the bottom, slow and expensive at the top. Run the bottom on every commit.

| Layer | Tests | Runtime | Runs on |
|---|---|---|---|
| **Code** | Unit tests on transforms, feature functions, serving handlers | Seconds | Every commit |
| **Data** | Schema, nullability, ranges, cardinality, volume, freshness, distribution vs baseline | Seconds–minutes | Every commit + every pipeline run |
| **Model** | Trains without error, beats a baseline, meets thresholds overall **and per slice**, behavioral tests | Minutes–hours | Every model change |
| **Integration** | Feature pipeline → model → API contract; train/serve parity | Minutes | Pre-deploy |
| **Deployment** | Smoke test, latency under load, shadow comparison | Minutes | Every deploy |

### Data tests

```python
import pandas as pd

def validate_training_data(df: pd.DataFrame, baseline_stats: dict) -> list[str]:
    """Return a list of failures. Empty list means the data is safe to train on."""
    failures = []

    expected = {"user_id": "int64", "amount": "float64", "category": "object"}
    for col, dtype in expected.items():
        if col not in df.columns:
            failures.append(f"missing column: {col}")
        elif str(df[col].dtype) != dtype:
            failures.append(f"{col}: expected {dtype}, got {df[col].dtype}")

    if len(df) < baseline_stats["min_rows"]:
        failures.append(f"row count {len(df)} below floor {baseline_stats['min_rows']}")

    for col, max_null in baseline_stats["max_null_rate"].items():
        rate = df[col].isna().mean()
        if rate > max_null:
            failures.append(f"{col}: null rate {rate:.3f} exceeds {max_null}")

    # Distribution shift on numeric features — catches semantic changes that pass schema checks
    for col, ref in baseline_stats["psi_reference"].items():
        psi = population_stability_index(df[col], ref)
        if psi > 0.25:
            failures.append(f"{col}: PSI {psi:.3f} indicates significant shift")

    return failures
```

A partial load is the sneakiest failure here: the schema is valid, every value is plausible, and only the row count reveals that half the data is missing. Compare volume against the same weekday historically, not a fixed constant, or seasonality generates constant false alarms.

### Model tests beyond an aggregate metric

An aggregate threshold hides the failure that matters. Three additions:

```python
def evaluate_model(model, X_test, y_test, slices, baseline_metrics, thresholds):
    results = {"overall": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}

    # 1. Per-slice metrics — an overall gain can mask a large regression on a segment
    for name, mask in slices.items():
        if mask.sum() >= 100:                       # ignore slices too small to be meaningful
            results[name] = roc_auc_score(y_test[mask], model.predict_proba(X_test[mask])[:, 1])

    # 2. Never ship a regression against what is already in production
    regressions = [k for k, v in results.items()
                   if k in baseline_metrics and v < baseline_metrics[k] - thresholds["tolerance"]]

    # 3. Behavioral tests — invariances and directional expectations, not just accuracy
    behavioral = run_behavioral_tests(model)        # e.g. raising income must not raise default risk
    return results, regressions, behavioral
```

**Behavioral tests** are the ML analogue of unit tests: assert an invariance (prediction unchanged when an irrelevant field changes), a directional expectation (raising a known-protective feature moves the score the right way), or a minimum-functionality case (an obvious fraud pattern is caught). They catch classes of bug that any aggregate metric passes straight over.

---

## Model Registry and Promotion Gates

The registry is the system of record: a versioned model binary plus everything needed to trust, reproduce, and roll it back.

| Stored with every version | Why |
|---|---|
| Model artifact + framework version | Reproduce the runtime |
| Training data version / snapshot ID | Reproduce the inputs |
| Code commit SHA | Reproduce the logic |
| Hyperparameters + random seeds | Reproduce the run |
| Evaluation metrics, overall and per slice | Compare against candidates and production |
| Input/output signature and schema | Validate at serving; detect contract breaks |
| Feature pipeline version | **Roll back together with the model** |
| Stage (`staging` / `production` / `archived`) | Drive promotion |

```python
import mlflow

with mlflow.start_run() as run:
    mlflow.log_params({"max_depth": 6, "learning_rate": 0.05, "seed": 42})
    mlflow.log_metrics({"auc": auc, "pr_auc": pr_auc, **slice_metrics})
    mlflow.log_dict({"data_version": data_version, "feature_pipeline": fp_version}, "lineage.json")
    mlflow.sklearn.log_model(model, "model", signature=signature, registered_model_name="churn")

client = mlflow.MlflowClient()
if passes_all_gates(metrics, baseline):
    client.transition_model_version_stage("churn", version, stage="Staging")
```

**Promotion gates** — every one automated, and every one able to block:

1. **Quality**: beats the production model on the primary metric, and does not regress any tracked slice beyond tolerance.
2. **Baseline sanity**: beats a trivial baseline (majority class, last value, simple heuristic). A model that cannot is not worth operating.
3. **Fairness**: metric parity across protected segments within policy.
4. **Operational**: p99 inference latency, memory footprint, artifact size within limits.
5. **Skew check**: features computed by the serving path match the training path for the same entities.
6. **Reproducibility**: lineage is complete — a missing data version fails the gate.

Automate the gates; keep a human approval for the final production transition in regulated or high-stakes settings. The gates are what make that approval a review rather than a rubber stamp.

---

## Deployment Strategies

| Strategy | Mechanism | Risk | Cost | Gives you |
|---|---|---|---|---|
| **Recreate** | Stop old, start new | High | Low | Nothing — downtime and no fallback |
| **Blue/green** | Two full environments, flip traffic | Low | 2x infra | Instant rollback |
| **Canary** | 1% → 5% → 25% → 100%, watching metrics | Low | Low | Graduated real-traffic exposure |
| **Shadow** | New model scores real traffic; output discarded | **None** | Extra compute | Real-traffic validation with zero user risk |
| **A/B test** | Random split, measure business metrics | Medium | Low | Causal read on the business outcome |
| **Multi-armed bandit** | Adaptive traffic to the better arm | Medium | Medium | Faster convergence, less regret |

The sequence that works for a meaningful model change is **shadow → canary → A/B → full**:

- **Shadow** first, because it is the only step that validates on real production traffic at zero user risk. It catches the failures offline evaluation cannot: train/serve skew, latency under real load, unexpected input distributions, and serialization or dependency problems. Compare prediction distributions, not just aggregate metrics.
- **Canary** next, to expose a small slice of users with automated rollback wired to guardrail metrics.
- **A/B** if the question is whether the model improves a *business* outcome, which offline metrics only proxy.

```yaml
# Canary with automated rollback (Argo Rollouts style)
strategy:
  canary:
    steps:
      - setWeight: 5
      - pause: {duration: 15m}
      - analysis:
          templates: [{templateName: model-health}]   # error rate, p99 latency, prediction drift
      - setWeight: 25
      - pause: {duration: 1h}
      - analysis:
          templates: [{templateName: model-health}]
      - setWeight: 100
```

---

## Rollback

Rollback must be **fast, tested, and complete**.

- **Fast**: a config or traffic-weight change, not a rebuild. Keep the previous version warm — a rollback that requires a 20-minute container build is not a rollback, it is an outage with a plan.
- **Tested**: rehearse it. An untested rollback path fails exactly when you need it.
- **Complete**: revert the model **and** its feature pipeline, preprocessing, and configuration together. Reverting only the model artifact while the new feature transformations remain live produces a skew bug that is harder to diagnose than the original problem. This is why the registry stores the feature pipeline version alongside the model.

Automate the trigger where the signal is unambiguous — error rate above threshold, p99 latency breach, prediction distribution shift beyond a bound, or a guardrail business metric dropping. Keep a manual kill switch regardless, since not every failure is one a rule anticipated.

---

## Continuous Training

Retraining is triggered by one of four things, and choosing among them is a real design decision:

| Trigger | When it fits | Watch out for |
|---|---|---|
| **Scheduled** | Steady, predictable drift | Retrains when nothing changed; wastes compute |
| **Performance-based** | Labels arrive fast enough to measure degradation | Useless when labels are delayed weeks |
| **Drift-based** | Input distribution shift is measurable immediately | Drift does not always mean degradation — verify before acting |
| **Data-volume** | Enough new labeled data has accumulated | Can retrain on a non-representative recent slice |

Whatever the trigger, an automatically retrained model **goes through the same gates as a manual one**. Auto-retrain plus auto-deploy without gates is how a corrupted upstream table silently becomes a production model. The retraining pipeline should also guard against feedback loops: if the model's own decisions shape which labels you observe (only approved loans get repayment outcomes), naive retraining amplifies its existing bias.

---

## Reproducibility Requirements

An interviewer's favorite probe: *can you reproduce the model that is in production right now?*

Requirements: pinned data snapshot or table version (time-travel via Delta/Iceberg, or an immutable partition); pinned code commit; pinned dependency versions (a lockfile, not a range — a minor scikit-learn bump can change defaults); seeded RNGs for Python, NumPy, and the framework; recorded hardware and framework versions where GPU nondeterminism matters; and the environment captured as a container image digest, not a tag.

Bitwise reproducibility on GPU requires deterministic kernels and costs throughput, so the practical standard is *statistical* reproducibility — retraining yields a model within a small tolerance on the eval set — with full lineage recorded. Say which one you mean; conflating them is a common slip.

---

## A Worked GitHub Actions Pipeline

```yaml
name: ML Pipeline

on:
  push: {branches: [main]}
  schedule: [{cron: '0 2 * * 1'}]          # weekly retrain
  workflow_dispatch:                        # manual + drift-alert webhook

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11', cache: pip}
      - run: pip install -r requirements-lock.txt   # locked, not ranged
      - run: ruff check . && mypy src/
      - run: pytest tests/unit tests/data -v        # fast layers on every commit

  train:
    needs: test
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: python -m src.train --config configs/prod.yaml --seed 42
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      - run: python -m src.evaluate --run-id "$MLFLOW_RUN_ID" --compare-to production
      # evaluate exits non-zero if any gate fails, which fails the job

  deploy_staging:
    needs: train
    steps:
      - run: python -m src.promote --stage Staging
      - run: python -m src.deploy --env staging --mode shadow --duration 24h
      - run: python -m src.compare_shadow --fail-on-divergence

  deploy_production:
    needs: deploy_staging
    environment: production        # requires human approval in GitHub
    steps:
      - run: python -m src.deploy --env prod --strategy canary --auto-rollback
```

Two details worth calling out in an interview: training runs on a **self-hosted GPU runner** because hosted runners lack GPUs and time out; and the `environment: production` key is what inserts a human approval gate without scripting one.

---

## Maturity Levels

| Level | Training | Deployment | Retraining | Typical signal |
|---|---|---|---|---|
| **0 — Manual** | Notebook | Manual copy | Ad hoc, when someone notices | No versioning; "it works on my machine" |
| **1 — Automated training** | Pipeline script | Manual | Scheduled | Experiments tracked; deploys still hand-run |
| **2 — Automated deployment** | Pipeline | CI/CD with gates | Triggered | Registry, gates, canary, rollback |
| **3 — Full CT/CD** | Pipeline | Automated | Drift/performance-triggered | Monitoring feeds retraining automatically |

Most teams sit at level 1 and benefit most from reaching level 2 — the registry, gates, and a tested rollback. Level 3 pays off only with high data velocity and reliable, fast labels; adopting it earlier mostly automates the propagation of mistakes.

---

## Interview Q&A

#### How does CI/CD for ML differ from CI/CD for regular software?

Three structural differences. First, there are **three coupled artifacts** — code, data, and model — so versioning one is insufficient; reproducing a production model requires the data snapshot and feature pipeline version too. Second, **tests are statistical**: instead of assert-equals you have "AUC at least 0.82 and no slice regressing more than 2 points," which means defining thresholds and tolerating variance. Third, **models decay without any change** — data drift degrades a static model, so the pipeline needs triggers beyond commits: schedules, drift alerts, and performance thresholds.

There is also a validation gap: you cannot fully establish a model is correct before real traffic touches it, which is why shadow and canary deployment matter far more here than for a typical service.

#### What gates would you put between training and production?

Automated, and each able to block: (1) the candidate beats the current production model on the primary metric; (2) no tracked slice regresses beyond tolerance — an overall gain can hide a large regression on a segment; (3) it beats a trivial baseline, because a model that cannot is not worth the operational cost; (4) fairness metrics across protected segments stay within policy; (5) operational limits — p99 latency, memory, artifact size; (6) a train/serve skew check comparing features from both paths for the same entities; (7) complete lineage, so a missing data version fails the gate.

In regulated or high-stakes settings I'd keep a human approval for the final production transition, but the automated gates are what turn that approval into a real review rather than a formality.

#### Explain shadow deployment and why it's valuable.

The new model receives a copy of real production traffic and produces predictions that are logged but never returned to users; the existing model continues serving. It is the only validation step with genuinely zero user risk.

Its value is catching what offline evaluation structurally cannot: **train/serve skew** — features computed differently in the serving path; **latency under real load** rather than benchmark conditions; **real input distributions**, including the malformed and edge-case requests that never appear in a clean test set; and **serialization or dependency failures** in the production runtime.

I'd compare prediction *distributions* between old and new, not just aggregate metrics — a shifted distribution with similar aggregate accuracy is a strong signal something is wrong. The cost is running inference twice, which is why it is usually time-boxed rather than permanent.

#### Your newly deployed model is degrading. Walk me through the rollback.

First **stop the bleeding**: shift traffic back to the previous version by config or weight change — seconds, not a rebuild. This is why the previous version stays warm.

Crucially, revert **the model and its feature pipeline together**. Reverting only the artifact while new feature transformations stay live creates a skew bug on top of the original failure, and it is much harder to diagnose. The registry stores the feature pipeline version with the model precisely so this is one atomic action.

Then **diagnose** with the traffic already safe: compare input distributions before and after, check whether an upstream schema changed, look at per-slice metrics to see whether degradation is uniform or concentrated in a segment, and check whether the deployed artifact matches what passed the gates. Finally, **close the loop** — add whatever would have caught this as a gate or monitor, since the same class of failure otherwise recurs.

#### When should retraining be automatic, and when should a human be in the loop?

Automatic retraining fits when data velocity is high, labels arrive quickly and reliably, drift is a routine and well-understood phenomenon, and the gates are strong enough to catch a bad model. Recommendation and demand forecasting typically qualify.

Keep a human in the loop when labels are delayed or noisy (you would be retraining on a signal you cannot yet verify), in regulated domains requiring documented review, when the model's own decisions shape future labels — a feedback loop where naive retraining amplifies existing bias — and when the cost of a bad model is severe relative to the cost of staleness.

The invariant either way: an automatically retrained model passes the **same gates** as a manual one. Auto-retrain with auto-deploy and no gates is how a corrupted upstream table becomes a production model overnight.

#### How do you version data alongside code?

Layered. **Immutable raw data**, append-only and partitioned by ingest date, never mutated — this is the foundation, and mutating raw data destroys reproducibility permanently. **Versioned table formats** (Delta Lake, Iceberg) for curated tables, so "train on the data as of March 1" is a time-travel query rather than an archaeology project. **Snapshot references recorded in the model artifact**, so every training run records exactly which table versions, feature definitions, and code commit produced it. DVC or LakeFS cover file-based assets where table formats do not apply.

The test of whether it actually works: can you reproduce a model trained six months ago from its recorded metadata alone, without asking anyone what they ran?

#### Why aren't aggregate metrics enough as a promotion gate?

Because they average away the failures that matter. A model can improve overall AUC by 2 points while regressing badly on a segment — a minority language, a small region, new users — and the aggregate never shows it, especially when that segment is a small share of the data. That is both a quality failure and, for protected attributes, a fairness one.

So I'd gate on per-slice metrics with a regression tolerance, on comparison against the current production model rather than an absolute threshold, and on **behavioral tests** — invariances and directional expectations that assert properties rather than accuracy. Those catch bug classes any aggregate metric passes straight over, like a model that becomes sensitive to a field that should be irrelevant.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Versioning code but not data | Production models cannot be reproduced | Pin data snapshots; record versions in the artifact |
| Rolling back the model but not the feature pipeline | Creates a skew bug on top of the original failure | Store and revert them together |
| Aggregate metrics as the only gate | Hides large regressions on a segment | Per-slice gates with regression tolerance |
| Auto-retrain with auto-deploy and no gates | A corrupted upstream table becomes a production model | Same gates for automated and manual runs |
| Dependency ranges instead of a lockfile | A minor library bump silently changes model behavior | Lockfile + container image digest |
| Rollback path never rehearsed | It fails exactly when it is needed | Test rollback as part of the deploy pipeline |
| Cold previous version | "Rollback" becomes a 20-minute rebuild | Keep the previous version warm |
| Fixed row-count thresholds for data checks | Weekly seasonality causes constant false alarms | Compare against the same weekday historically |
| Training on hosted CI runners | No GPU, and jobs time out | Self-hosted GPU runners for the training job |
| Deploying straight to 100% | No safe way to observe a regression | Shadow, then canary with automated rollback |
| Treating drift as automatically requiring retraining | Drift does not always mean degradation | Verify performance impact before triggering |

---

## Related Topics

- [MLOps Overview](./README.md)
- [MLflow](./intro_mlflow.md)
- [Model Serving](./intro_model_serving.md)
- [Model Monitoring](./intro_model_monitoring.md)
- [Data Quality](./intro_data_quality.md)
- [A/B Testing](./intro_ab_testing.md)
- [Feature Stores](./intro_feature_stores.md)
- [GitHub Actions](../devops/intro_github_actions.md)
- [Docker](../devops/intro_docker.md)
- [Kubernetes](../devops/intro_kubernetes.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
