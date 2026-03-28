# Fraud Detection System Design

A complete ML system design for real-time fraud detection at scale.

---

## Table of Contents

1. [Problem Definition and Requirements](#problem-definition-and-requirements)
2. [Data Sources and Features](#data-sources-and-features)
3. [Model Selection](#model-selection)
4. [Real-Time vs Batch Scoring](#real-time-vs-batch-scoring)
5. [Handling Class Imbalance](#handling-class-imbalance)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Production Architecture](#production-architecture)
8. [Monitoring and Retraining Strategy](#monitoring-and-retraining-strategy)

---

## Problem Definition and Requirements

### Business Context

A payment processing company wants to automatically detect fraudulent transactions in real-time to prevent financial loss while minimizing false positives that frustrate legitimate customers.

### Functional Requirements

- Score every transaction as fraud or legitimate
- Real-time decision before transaction authorization (< 100ms)
- Support rule overrides by fraud analysts
- Explainable decisions for regulatory compliance
- Human review queue for borderline cases

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Latency (p99) | < 100ms |
| Throughput | 10,000 TPS (transactions per second) |
| Availability | 99.99% (≤ 52 minutes downtime/year) |
| False Positive Rate | < 0.5% (keep customer friction low) |
| Fraud Detection Rate | > 80% of fraud dollars caught |
| Scale | 500M transactions/day |

### ML Objective

Binary classification: `fraud (1) vs legitimate (0)`

Business-aware objective: Maximize fraud dollars caught while limiting false positive rate to < 0.5%.

---

## Data Sources and Features

### Data Sources

```
Transaction Stream (Kafka)
├── transaction_id
├── user_id
├── merchant_id
├── amount
├── currency
├── timestamp
├── device_fingerprint
├── ip_address
├── card_type
└── payment_method

User History (Data Warehouse)
├── account_age_days
├── total_transactions_30d
├── avg_transaction_amount_30d
├── countries_transacted_30d
└── dispute_history

Merchant Database
├── merchant_risk_score
├── merchant_category_code (MCC)
├── merchant_country
├── chargeback_rate_30d
└── merchant_age_days

Device Intelligence (Third-party API)
├── device_type
├── is_known_device
├── device_age_days
├── vpn_detected
└── tor_detected

IP Intelligence (MaxMind/Third-party)
├── ip_country
├── ip_city
├── ip_isp
├── ip_proxy_type
└── ip_risk_score
```

### Feature Engineering

```python
# Transaction-level features
features = {
    # Amount features
    "amount": transaction.amount,
    "log_amount": np.log1p(transaction.amount),
    "amount_deviation": (amount - user_avg_amount) / user_std_amount,
    "is_round_number": amount % 10 == 0,

    # Velocity features (aggregated over sliding windows)
    "txn_count_1h": count_transactions(user_id, window="1h"),
    "txn_count_24h": count_transactions(user_id, window="24h"),
    "txn_count_7d": count_transactions(user_id, window="7d"),
    "unique_merchants_24h": count_unique(user_id, "merchant_id", window="24h"),
    "unique_countries_7d": count_unique(user_id, "country", window="7d"),

    # Amount velocity
    "total_amount_1h": sum_amount(user_id, window="1h"),
    "max_amount_7d": max_amount(user_id, window="7d"),

    # Behavioral anomalies
    "unusual_hour": 1 if transaction.hour in [2, 3, 4, 5] else 0,
    "new_country": transaction.country not in user_frequent_countries,
    "new_device": not is_known_device(transaction.device_fingerprint),

    # Merchant risk
    "merchant_risk_score": merchant.risk_score,
    "merchant_chargeback_rate": merchant.chargeback_rate_30d,
    "high_risk_mcc": merchant.mcc in HIGH_RISK_MCC_CODES,

    # Network features
    "ip_risk_score": ip_lookup(transaction.ip),
    "vpn_detected": ip_info.vpn_detected,
    "country_mismatch": transaction.ip_country != user.home_country,
    "ip_velocity": count_unique_users_from_ip(transaction.ip, window="1h"),
}
```

---

## Model Selection

### Why Gradient Boosting (XGBoost/LightGBM)

| Factor | Justification |
|--------|---------------|
| Performance | State-of-the-art on tabular data |
| Latency | Inference < 5ms on CPU |
| Interpretability | SHAP values for feature importance |
| Class imbalance | Built-in `scale_pos_weight` parameter |
| Handles missing | Native handling |

### Model Architecture

```
Hybrid: Rules Engine + ML Model

1. Hard Rules (< 1ms):
   - Block: Known fraud IP, blacklisted cards, sanctioned countries
   - Allow: Trusted merchant + known device + low amount

2. ML Model (< 10ms):
   - LightGBM with 500 trees, depth 6
   - Input: ~100 features
   - Output: Fraud probability (0-1)

3. Decision Logic:
   - P > 0.9: Auto-decline
   - 0.5 < P < 0.9: Hold for review / 3DS challenge
   - P < 0.5: Auto-approve (with continued monitoring)
   - Threshold tuned for < 0.5% FPR

4. Fallback:
   - If ML model unavailable: rules engine only
   - If < 500ms: allow but flag for review
```

```python
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Heavily imbalanced — 0.1% fraud rate
# scale_pos_weight compensates: ~ 999 for 1:999 ratio
fraud_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    scale_pos_weight=fraud_weight,   # Handle imbalance
    max_depth=6,
    min_child_samples=100,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

---

## Real-Time vs Batch Scoring

### Real-Time Scoring (Primary Path)

```
Transaction Event → Kafka → Feature Service → ML Model → Decision
         ↓                        ↓
   < 100ms total         Online Feature Store (Redis)
                         Latest velocity features
```

**Requirements:**
- Feature retrieval: < 20ms (Redis lookup)
- Model inference: < 5ms (LightGBM on CPU)
- Total end-to-end: < 100ms (SLA)

### Batch Scoring (Supplementary)

Used for:
- Historical analysis of past transactions
- Model training data preparation
- Backdating fraud labels when chargebacks arrive (delayed feedback)
- Generating risk scores for inactive accounts

```
S3 Transaction Files → Spark Job → Feature Join → Model Scoring → Risk Database
(daily batch)        (hourly)     (warehouse)    (millions/hr)   (for analytics)
```

---

## Handling Class Imbalance

| Technique | Implementation | Notes |
|-----------|---------------|-------|
| Class weights | `scale_pos_weight=999` | Most important, built-in |
| Threshold tuning | Set P > 0.5 based on cost-benefit | Tune on validation set |
| SMOTE | Oversample minority class | Use carefully — may generate unrealistic fraud |
| Undersampling | Downsample legitimate transactions | Risk losing signal |
| Cost-sensitive learning | Weight misclassification by fraud amount | Custom loss function |
| Stratified splitting | Ensure fraud rate consistent across folds | Critical for cross-validation |

**Practical recommendation:** Use `scale_pos_weight` + careful threshold tuning. Avoid SMOTE — synthetic fraud patterns may not reflect real fraud behavior.

---

## Evaluation Metrics

### Why Not Accuracy?

With 0.1% fraud rate, a model predicting "no fraud" for everything achieves 99.9% accuracy but catches zero fraud. Useless.

### Primary Metrics

| Metric | Formula | Target | Why |
|--------|---------|--------|-----|
| **Recall@FPR0.5%** | Fraud caught at 0.5% false positives | > 80% | Maximize catch rate at acceptable customer friction |
| **Fraud Dollar Recall** | Fraud $ caught / Total fraud $ | > 85% | High-value fraud disproportionately important |
| **Precision** | TP / (TP + FP) | > 30% at review threshold | Review queue capacity |
| **AUC-ROC** | Area under ROC curve | > 0.97 | Model discrimination ability |
| **AUC-PR** | Area under Precision-Recall | > 0.50 | Better for imbalanced data |

### Business Metrics

```
Financial metrics:
- Fraud loss prevented ($)
- False positive customer impact (declined legitimate transactions)
- Review queue cost (analyst hours)
- Cost per fraud case reviewed

Operational metrics:
- Model latency (p99)
- Model availability
- False alert rate for high-value merchants
```

---

## Production Architecture

```
                        FRAUD DETECTION SYSTEM
═══════════════════════════════════════════════════════════════════════

  Payment API        Feature Store              ML Service
  ─────────────      ──────────────             ─────────────
  POST /authorize  ──→  Redis Online Store  ──→  LightGBM Model
       │               (velocity features)       │
       │               ──────────────            │
       │             User Profile Service         │
       │               (account age,              │
       │                history)                  │
       │                                          │
       ↓                                          ↓
   Transaction                               Risk Score (0-1)
   Event → Kafka                                  │
       │                                          ↓
       │                               Decision Engine
       │                               ├── Rules: Block/Allow
       │                               ├── P > 0.9: Decline
       │                               ├── 0.5-0.9: Review
       │                               └── P < 0.5: Approve
       │                                          │
       ↓                                          ↓
  Offline Store (S3)              ─→  Case Management System
  Training Data                      (Human Review Queue)
  Labels from Chargebacks

═══════════════════════════════════════════════════════════════════════

Offline Training Pipeline (Daily/Weekly):
  S3 Transactions → Spark Feature Engineering → Training Dataset
                 → LightGBM Training (MLflow tracking)
                 → Model Evaluation (compare vs champion)
                 → Deploy if better → Canary (5% traffic)
                 → Full rollout if metrics OK
```

### Infrastructure

```yaml
services:
  ml-service:
    replicas: 20           # Auto-scale to 50 under load
    resources:
      cpu: "4"
      memory: "8Gi"
    model: lgb-fraud-v12   # Load model into memory at startup

  feature-service:
    replicas: 10
    redis:
      cluster: true
      nodes: 6
      memory: "50Gi"       # Stores 10M user velocity features

  kafka:
    partitions: 100        # High throughput
    replication: 3
    retention: "7d"
```

---

## Monitoring and Retraining Strategy

### What to Monitor

```
Real-time dashboards (refresh every 5 minutes):
├── Alert rate (% transactions declined)
├── Auto-decline rate
├── Review queue length
├── Model latency (p50, p95, p99)
└── Feature compute failures

Daily reports:
├── Fraud detection rate (when chargebacks arrive ~30 days later)
├── False positive rate (from user feedback)
├── Feature drift (PSI on top features)
├── Score distribution shift
└── New fraud patterns identified by analysts

Alerts:
├── Alert rate spikes > 2x baseline → possible fraud attack or model bug
├── Alert rate drops > 50% → possible model degradation
├── Latency p99 > 150ms → scaling issue
└── Kafka consumer lag > 10K → processing bottleneck
```

### Retraining Strategy

**Trigger conditions:**
1. Fraud detection rate drops > 5% (weekly measurement)
2. PSI > 0.25 on top 10 features (monthly)
3. New fraud pattern identified by analysts
4. Scheduled monthly retraining (safety net)

**Training data strategy:**
```
Training window: Rolling 90 days
├── Include recent fraud confirmed via chargebacks (30-60 day lag)
├── Balance: ~2% fraud rate in training set via class weights
├── Temporal split: train on oldest 80%, validate on most recent 20%
└── Never shuffle — time ordering is critical (prevents leakage)
```

**Deployment process:**
```
1. Train new model on recent data
2. Backtest on historical holdout period
3. Shadow mode: run new model alongside production (no impact)
4. Canary: 5% traffic → new model (monitor for 48 hours)
5. Ramp: 20% → 50% → 100% (if metrics stable)
6. Rollback available instantly via feature flag
```

---

## Key Interview Points

1. **Why gradient boosting over deep learning?** Gradient boosting trains faster, has lower latency inference, is more interpretable (SHAP), and performs better on tabular data with engineered features.

2. **How do you handle the label delay problem?** Fraud is only confirmed when chargebacks arrive (30-60 days later). Solution: train on data where labels are confirmed; accept that recent data cannot be used for training until labels arrive.

3. **How do you prevent data leakage?** Use temporal splits (train on past, test on future). Use point-in-time correct feature values. Never include information available only after the transaction.

4. **How would you scale to 10x traffic?** Horizontal scaling of ML service pods, scale Redis cluster, increase Kafka partitions, consider moving to online feature computation with lower latency.

5. **How do you handle adversarial fraud?** Fraudsters adapt. Monitor distribution shifts, use anomaly detection layers on top of the model, use behavioral analytics, and include analyst feedback loops for emerging patterns.
