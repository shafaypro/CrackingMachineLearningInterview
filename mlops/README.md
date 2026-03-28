# MLOps Overview

A comprehensive guide to Machine Learning Operations (MLOps) — the practices, tools, and workflows for deploying and maintaining ML models in production.

---

## Table of Contents

1. [What is MLOps and Why It Matters](#what-is-mlops-and-why-it-matters)
2. [MLOps Lifecycle](#mlops-lifecycle)
3. [Tool Comparison](#tool-comparison)
4. [Model Drift](#model-drift)
5. [A/B Testing and Deployment Strategies](#ab-testing-and-deployment-strategies)
6. [Feature Stores](#feature-stores)
7. [CI/CD for ML](#cicd-for-ml)
8. [Interview Q&A](#interview-qa)
9. [References](#references)

---

## What is MLOps and Why It Matters

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

**Why it matters:**
- 87% of ML projects never make it to production (VentureBeat, 2019)
- Models degrade over time due to data and concept drift
- Reproducibility and auditability are required in regulated industries
- Manual deployment and monitoring doesn't scale

**Core principles:**
- Automation of the ML pipeline
- Continuous training (CT), integration (CI), and delivery (CD)
- Monitoring and observability
- Reproducibility and versioning
- Collaboration between data scientists and engineers

---

## MLOps Lifecycle

```
Data Collection → Data Validation → Feature Engineering → Model Training
       ↑                                                         ↓
  Retraining ←── Monitoring ←── Production Serving ←── Evaluation & Registry
```

### Stages

| Stage | Description | Key Tools |
|-------|-------------|-----------|
| **Data** | Ingestion, validation, versioning | DVC, Delta Lake, Great Expectations |
| **Feature Engineering** | Transform raw data into model features | Feast, Tecton, Hopsworks |
| **Training** | Experiment tracking, hyperparameter tuning | MLflow, W&B, Optuna |
| **Evaluation** | Model validation, bias/fairness checks | Evidently AI, Deepchecks |
| **Registry** | Versioned model storage and staging | MLflow Registry, W&B Artifacts |
| **Deployment** | Serving, canary, blue-green, shadow mode | BentoML, Seldon, KServe, FastAPI |
| **Monitoring** | Drift detection, alerting, dashboards | Evidently AI, Grafana, Arize |
| **Retraining** | Trigger-based or scheduled retraining | Airflow, Kubeflow Pipelines, ZenML |

---

## Tool Comparison

| Tool | Primary Use | Strengths | Weaknesses | Best For |
|------|-------------|-----------|------------|----------|
| **MLflow** | Experiment tracking + registry | Open source, simple API, wide adoption | UI is basic, no native orchestration | Teams wanting self-hosted tracking |
| **Weights & Biases (W&B)** | Experiment tracking + collaboration | Rich UI, sweeps, artifacts, reports | Paid for enterprise | Research teams, collaborations |
| **DVC** | Data + model versioning | Git-native, storage-agnostic | Learning curve for pipelines | Data versioning alongside Git |
| **ZenML** | ML pipeline orchestration | Framework-agnostic, stack concept | Newer, smaller community | Portable ML pipelines |
| **Kubeflow** | Kubernetes-native ML platform | Scalable, full lifecycle | Complex setup and maintenance | Large teams with K8s infra |

---

## Model Drift

### Types of Drift

**Data Drift (Covariate Shift):** The distribution of input features changes, but the relationship between inputs and outputs remains the same.

```
P_train(X) ≠ P_prod(X)
```

Example: A model trained on summer traffic patterns sees winter patterns in production.

**Concept Drift:** The relationship between inputs and outputs changes.

```
P_train(Y|X) ≠ P_prod(Y|X)
```

Example: Customer behavior changes after an economic event; click-through rate patterns shift.

**Model Drift / Performance Drift:** The model's accuracy metrics degrade over time due to data or concept drift.

### Detecting Drift

| Test | Use Case | Type |
|------|----------|------|
| Kolmogorov-Smirnov (KS) test | Continuous features | Statistical |
| Population Stability Index (PSI) | Feature and score distributions | Statistical |
| Chi-squared test | Categorical features | Statistical |
| Jensen-Shannon Divergence | Probability distributions | Information theory |
| CUSUM | Sequential drift detection | Time-series |

**PSI Formula:**
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
PSI < 0.1: No significant change
0.1 ≤ PSI < 0.25: Moderate change
PSI ≥ 0.25: Significant change — retrain required
```

---

## A/B Testing and Deployment Strategies

### A/B Testing

Split traffic between two model versions to compare performance using real user interactions.

```
Users → Load Balancer → Model A (50%) → Metrics
                     → Model B (50%) → Metrics
Compare: conversion rate, CTR, revenue, latency
```

**Requirements:** Statistical significance, enough traffic, consistent user assignment, guard rails.

### Canary Deployment

Gradually roll out a new model version to a small percentage of traffic before full deployment.

```
Phase 1: 5% traffic → new model
Phase 2: 20% traffic → new model  (if metrics OK)
Phase 3: 50% traffic → new model  (if metrics OK)
Phase 4: 100% traffic → new model
```

### Shadow Mode

The new model runs in parallel with the production model but its predictions are not served to users. Used to validate behavior before switching.

```
Request → Production Model → User sees response
        → Shadow Model    → Logs only (not served)
Compare predictions offline
```

### Blue-Green Deployment

Two identical production environments. "Blue" is live; "Green" is the new version. Switch traffic instantly.

```
Current state: All traffic → Blue (v1)
Deploy to:     Green (v2) — fully tested
Switch:        All traffic → Green (v2)
Rollback:      All traffic → Blue (v1) — instant
```

---

## Feature Stores

A feature store is a centralized repository for storing, versioning, and serving ML features. See [Feature Store Guide](./intro_feature_store.md) for details.

**Key concepts:**
- **Online store:** Low-latency feature retrieval for real-time inference (Redis, DynamoDB)
- **Offline store:** High-throughput feature retrieval for training (S3, BigQuery, Hive)
- **Feature pipeline:** Transforms raw data into features with consistent logic

---

## CI/CD for ML

### Traditional CI/CD vs ML CI/CD

| Step | Traditional CI/CD | ML CI/CD |
|------|-------------------|----------|
| Build | Compile code | Train model, run feature pipeline |
| Test | Unit/integration tests | Model validation, data validation |
| Deploy | Push artifact to server | Register model, deploy endpoint |
| Monitor | Application logs | Model metrics, data drift |

### ML Pipeline Stages

```yaml
# Example GitHub Actions pipeline
stages:
  - data_validation      # Great Expectations / Pandera
  - feature_engineering  # Run feature pipeline
  - model_training       # Train with tracked parameters
  - model_evaluation     # Compare against baseline
  - model_registry       # Push to MLflow/W&B if better
  - deployment           # Deploy to staging/prod
  - smoke_test           # Hit endpoint with test requests
  - monitoring_setup     # Configure drift alerts
```

---

## Interview Q&A

**Q1: What is the difference between MLOps and DevOps?** 🟢 Beginner

DevOps focuses on software deployment and CI/CD for traditional applications. MLOps extends these practices to handle ML-specific challenges: data versioning, experiment tracking, model training pipelines, model drift monitoring, and retraining triggers. MLOps adds a continuous training (CT) component to the traditional CI/CD loop.

---

**Q2: What are the main challenges in deploying ML models to production?** 🟢 Beginner

- Data pipeline failures (schema drift, missing data)
- Model performance degradation over time (drift)
- Reproducibility (environment, data, code)
- Latency requirements for real-time serving
- Version management of data, code, and models
- Monitoring and alerting
- Regulatory compliance and explainability

---

**Q3: Explain the difference between data drift and concept drift.** 🟡 Intermediate

Data drift (covariate shift) occurs when the distribution of input features changes — `P(X)` changes but `P(Y|X)` stays the same. For example, a model trained on users from one region sees users from a different region.

Concept drift occurs when the relationship between inputs and outputs changes — `P(Y|X)` changes. For example, fraud patterns evolve and what used to be flagged as fraudulent is no longer fraudulent.

Both lead to model degradation but require different responses.

---

**Q4: How would you detect that a model needs to be retrained?** 🟡 Intermediate

1. **Statistical drift tests:** Monitor PSI or KS test on feature distributions
2. **Performance monitoring:** Track accuracy, F1, or business metrics if labels are available
3. **Prediction distribution shift:** Watch for changes in the model's output score distribution
4. **Business metric degradation:** Conversion rate drops, revenue anomalies
5. **Scheduled retraining:** Retrain on a fixed schedule (daily, weekly) as a safeguard
6. **Reference window comparison:** Compare current distribution to training distribution

---

**Q5: What is the difference between online and batch inference?** 🟢 Beginner

- **Batch inference:** Process a large dataset at a scheduled time. Predictions are stored and used later. Lower latency requirements. Example: generating personalized recommendations nightly.
- **Online (real-time) inference:** Serve predictions in response to individual requests with low latency. Example: fraud detection at transaction time, search ranking.

---

**Q6: What is a model registry and why is it important?** 🟡 Intermediate

A model registry is a centralized store for versioned model artifacts, metadata, and lifecycle stages (Staging, Production, Archived). It enables:
- Versioned model storage with metadata (metrics, parameters, data version)
- Controlled promotion workflows (Staging → Production)
- Rollback capability
- Audit trails for compliance
- Team collaboration on model lifecycle management

MLflow Model Registry and W&B Model Registry are common implementations.

---

**Q7: Explain the canary deployment strategy for ML models.** 🟡 Intermediate

Canary deployment gradually routes a small percentage of traffic (e.g., 5%) to the new model while the remaining traffic uses the old model. Metrics are monitored closely. If the new model performs well, traffic is incrementally increased (5% → 20% → 50% → 100%). If metrics degrade, traffic is instantly rolled back. This reduces risk compared to a hard cutover while allowing real-traffic validation.

---

**Q8: What is shadow mode testing and when would you use it?** 🟡 Intermediate

In shadow mode, the new model receives the same requests as the production model but its responses are not served to users. Both models run in parallel, and the shadow model's predictions are logged for offline comparison.

Use shadow mode when:
- You need to validate a new model's behavior on real production traffic
- The model makes decisions in a high-stakes domain (fraud, medical)
- You want to compare predictions without any user impact risk

---

**Q9: What statistical tests would you use to detect feature drift?** 🔴 Advanced

- **KS Test (Kolmogorov-Smirnov):** For continuous features; tests whether two distributions are the same. Returns p-value and D-statistic.
- **PSI (Population Stability Index):** Measures how much a distribution has shifted. PSI ≥ 0.25 signals significant shift.
- **Chi-squared test:** For categorical features; tests independence of distributions.
- **Jensen-Shannon Divergence:** Symmetric version of KL divergence; measures similarity between two probability distributions.
- **Wasserstein distance (Earth Mover's Distance):** Measures the distance between distributions; robust to outliers.

Choice depends on feature type (continuous vs categorical) and interpretability requirements.

---

**Q10: How do you handle model rollbacks in production?** 🟡 Intermediate

1. **Model Registry versioning:** Keep previous model versions in the registry with their artifacts
2. **Blue-Green deployment:** Instantly redirect traffic back to the previous environment
3. **Feature flags:** Toggle model versions at the application layer
4. **Traffic routing rules:** Update load balancer weights to redirect to the previous model endpoint
5. **Automated rollback triggers:** Set thresholds (e.g., error rate > 5%) that automatically rollback

---

**Q11: What is the difference between model retraining and model rebuilding?** 🟡 Intermediate

- **Retraining:** Use the same model architecture, features, and hyperparameters, but train on newer data. Fast and safe; addresses data drift.
- **Rebuilding:** Start from scratch — re-evaluate the problem, explore new features, test different architectures. Necessary for significant concept drift or when the existing model architecture is fundamentally limited.

---

**Q12: What metrics would you monitor for an ML model in production?** 🟡 Intermediate

**Technical metrics:**
- Latency (p50, p95, p99)
- Throughput (requests per second)
- Error rate (5xx errors)
- Model prediction distribution

**Data quality metrics:**
- Feature missing rates
- Feature drift (PSI, KS)
- Schema violations

**Model performance metrics:**
- Accuracy/F1/AUC (if labels available)
- Business KPIs (conversion rate, revenue)
- Prediction confidence distribution

---

**Q13: Explain the concept of a feature store and its role in MLOps.** 🔴 Advanced

A feature store solves the problem of consistent feature computation between training and serving. Without it, data scientists compute features differently during experimentation vs production, leading to training-serving skew.

A feature store provides:
- **Unified feature definitions:** Compute once, use everywhere
- **Offline store:** Point-in-time correct feature retrieval for training (prevents data leakage)
- **Online store:** Low-latency feature retrieval for real-time serving
- **Feature versioning:** Track changes to feature logic over time
- **Reusability:** Multiple models share the same features

---

**Q14: What is training-serving skew and how do you prevent it?** 🔴 Advanced

Training-serving skew occurs when the features used during training are computed differently from how they are computed at serving time. This causes the model to see a different data distribution in production than it was trained on.

**Causes:**
- Different preprocessing code paths for training and serving
- Data leakage in training (using future data)
- Different feature computation timing

**Prevention:**
- Use a feature store with unified feature definitions
- Share preprocessing code between training and serving pipelines
- Validate feature distributions at serving time against training distribution
- Use point-in-time correct joins in offline feature retrieval

---

**Q15: How would you design a CI/CD pipeline for an ML model?** 🔴 Advanced

```
1. Code commit → GitHub Actions triggered
2. Data validation: Run Great Expectations on new training data
3. Feature pipeline: Rebuild feature set with updated data
4. Model training: Train with tracked parameters in MLflow
5. Model evaluation: Compare metrics against registered baseline
6. Automated gate: If metrics improve → proceed; else → fail pipeline
7. Model registration: Push to MLflow Registry with metadata
8. Integration tests: Deploy to staging, run smoke tests
9. Canary deployment: Route 5% prod traffic to new model
10. Monitor: Watch metrics for 24–48 hours
11. Full rollout: 100% traffic if metrics are stable
12. Alerting: PagerDuty alerts for degradation
```

---

## References

- [Google MLOps: Continuous delivery and automation in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015)](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Feature Store Comparison — Feast, Tecton, Hopsworks](https://www.featurestore.org/)
- [Chip Huyen — Designing Machine Learning Systems (O'Reilly)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
