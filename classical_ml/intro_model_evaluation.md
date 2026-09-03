# Model Evaluation and Metrics

Choosing the wrong metric is the most expensive mistake in applied ML, and interviewers probe it constantly. A model with 99% accuracy on a 1% fraud rate has learned nothing. This guide covers classification and regression metrics, ROC vs PR curves, probability calibration, threshold selection, and validation strategies that survive contact with production.

---

## Table of Contents
1. [The Confusion Matrix](#the-confusion-matrix)
2. [Classification Metrics](#classification-metrics)
3. [ROC-AUC vs PR-AUC](#roc-auc-vs-pr-auc)
4. [Threshold Selection](#threshold-selection)
5. [Probability Calibration](#probability-calibration)
6. [Regression Metrics](#regression-metrics)
7. [Ranking and Recommendation Metrics](#ranking-and-recommendation-metrics)
8. [Cross-Validation Strategies](#cross-validation-strategies)
9. [Offline vs Online Metrics](#offline-vs-online-metrics)
10. [Metric Selection Cheat Sheet](#metric-selection-cheat-sheet)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## The Confusion Matrix

Every classification metric is a ratio of these four cells:

```
                  Predicted
                Neg      Pos
Actual  Neg  |  TN   |   FP   |   ← FP = false alarm (Type I error)
        Pos  |  FN   |   TP   |   ← FN = miss (Type II error)
```

```python
from sklearn.metrics import confusion_matrix, classification_report

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(classification_report(y_true, y_pred, digits=3))
```

Ask in every interview: **what does a false positive cost, and what does a false negative cost?** The answer determines the metric, and stating it unprompted is a strong signal.

---

## Classification Metrics

| Metric | Formula | Answers | Use when |
|---|---|---|---|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | Overall correctness | Balanced classes, symmetric costs |
| Precision | `TP/(TP+FP)` | "Of my positive calls, how many were right?" | False positives are costly |
| Recall (Sensitivity, TPR) | `TP/(TP+FN)` | "Of all real positives, how many did I catch?" | Misses are costly |
| Specificity (TNR) | `TN/(TN+FP)` | "Of all real negatives, how many did I clear?" | Screening, ROC's x-axis |
| F1 | `2PR/(P+R)` | Harmonic mean of P and R | Need one number, imbalanced data |
| Fβ | `(1+β²)PR/(β²P+R)` | Weighted P/R tradeoff | β=2 favors recall, β=0.5 favors precision |
| Balanced accuracy | `(TPR+TNR)/2` | Accuracy corrected for imbalance | Imbalanced, both classes matter |
| MCC | correlation of predictions and labels | Single balanced score in [-1, 1] | Imbalanced, want one honest number |
| Log loss | `-Σ[y log p + (1-y) log(1-p)]` | Probability quality | You need calibrated probabilities |
| Brier score | `mean((p - y)²)` | Squared probability error | Calibration + sharpness together |

**Why the harmonic mean in F1?** It punishes imbalance between P and R. A model with P=1.0 and R=0.01 has arithmetic mean 0.505 but F1 = 0.0198 — the harmonic mean refuses to reward a model that gets one number by destroying the other.

**Macro vs micro vs weighted averaging** (multiclass):
- **Macro**: unweighted mean over classes — every class counts equally, so rare classes matter. Use when rare classes are important.
- **Micro**: pool all TP/FP/FN, then compute — equals accuracy in single-label multiclass. Dominated by frequent classes.
- **Weighted**: mean weighted by class support — a compromise, but hides rare-class failure.

---

## ROC-AUC vs PR-AUC

Both summarize a ranking across all thresholds, but they answer different questions.

**ROC** plots TPR vs FPR. **ROC-AUC** = probability that a random positive is ranked above a random negative. It is **invariant to class balance** — which is exactly its strength and its trap.

**PR curve** plots Precision vs Recall. **PR-AUC (average precision)** depends on prevalence and reflects what an operator actually experiences on an imbalanced problem.

```python
from sklearn.metrics import roc_auc_score, average_precision_score

y_proba = model.predict_proba(X_val)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
print(f"PR-AUC : {average_precision_score(y_val, y_proba):.4f}")
print(f"Baseline PR-AUC (prevalence): {y_val.mean():.4f}")
```

Always print the prevalence baseline. A PR-AUC of 0.30 is excellent at 1% prevalence and terrible at 40%.

### The worked example every interviewer wants

100,000 transactions, 100 are fraud (0.1%). A model flags 1,000 transactions and catches 90 frauds.

- Recall = 90/100 = **90%**
- Precision = 90/1000 = **9%**
- FPR = 910/99,900 = **0.9%** → ROC looks superb
- Accuracy = 99.0% — *worse than predicting "never fraud"* (99.9%)

ROC-AUC will read ~0.98 here because FPR is normalized by the huge negative class. PR-AUC honestly reports that 91% of the analyst's review queue is noise. **On heavy imbalance, report PR-AUC; use ROC-AUC only as a secondary ranking check.**

---

## Threshold Selection

Models output scores; decisions need a cutoff. `0.5` is a default, not an answer.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

# Option A: maximize F1
f1 = 2 * precision * recall / (precision + recall + 1e-12)
best_f1_threshold = thresholds[np.argmax(f1[:-1])]

# Option B: highest recall subject to a precision floor the business set
mask = precision[:-1] >= 0.80
best_recall_threshold = thresholds[mask][np.argmax(recall[:-1][mask])] if mask.any() else None

# Option C: minimize expected cost — the most defensible in an interview
COST_FN, COST_FP = 500, 10     # e.g. missed fraud vs wasted analyst review

def expected_cost(threshold):
    predicted = y_proba >= threshold
    false_negatives = ((~predicted) & (y_val == 1)).sum()
    false_positives = ((predicted) & (y_val == 0)).sum()
    return false_negatives * COST_FN + false_positives * COST_FP

best_cost_threshold = thresholds[int(np.argmin([expected_cost(t) for t in thresholds]))]
```

Option C is the strongest answer: express the decision as expected cost, and the optimal threshold falls out of the cost ratio rather than a convention. Also account for **capacity**: if analysts can review 500 cases a day, the threshold is whatever produces 500 alerts, regardless of what F1 prefers.

Tune the threshold on a **validation** set and report final numbers on a **test** set. Choosing the threshold on the test set is a subtle but real form of overfitting.

---

## Probability Calibration

A model is calibrated if, among predictions of 0.7, roughly 70% are positive. Ranking quality (AUC) and calibration are independent: a model can rank perfectly and still be systematically overconfident.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10, strategy='quantile')
# Perfect calibration lies on y = x; plot and look at the deviation.

calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_cal, y_cal)      # a held-out calibration set, not the training set
print(f"Brier before: {brier_score_loss(y_val, y_proba):.4f}")
print(f"Brier after : {brier_score_loss(y_val, calibrated.predict_proba(X_val)[:,1]):.4f}")
```

| Method | Shape assumed | Data needed | Notes |
|---|---|---|---|
| **Platt scaling** (sigmoid) | Sigmoid distortion | Small (~hundreds) | Good for SVMs and small calibration sets |
| **Isotonic regression** | Any monotonic | Larger (~1000+) | More flexible, overfits on small sets |
| **Temperature scaling** | Single scalar on logits | Very small | Standard for neural networks, preserves argmax |

Which models need calibration? SVMs and naive Bayes are badly calibrated by construction. Boosted trees are pushed toward 0 and 1 by the loss. Random Forests are compressed toward the middle (averaging pulls away from extremes). Logistic regression trained with log loss is usually close to calibrated already. Anything trained with class reweighting or on a resampled dataset is miscalibrated by design and must be corrected before its scores feed a downstream cost calculation.

Calibration matters when the score is **used as a number** — expected-value decisions, pricing, ranking against a cost threshold, or feeding another model. If you only threshold once, ranking is enough.

---

## Regression Metrics

| Metric | Formula | Units | Sensitivity to outliers | Use when |
|---|---|---|---|---|
| MAE | `mean|y - ŷ|` | Target units | Low | Outliers are noise; you want the median behavior |
| MSE | `mean(y - ŷ)²` | Squared units | High | Large errors are disproportionately bad |
| RMSE | `√MSE` | Target units | High | Same as MSE but readable |
| MAPE | `mean|y-ŷ|/|y|` | Percent | Medium | Comparing across scales — breaks near `y=0` |
| SMAPE | symmetric variant | Percent | Medium | MAPE with bounded blow-up |
| R² | `1 - SSE/SST` | Unitless | High | Explaining variance vs the mean baseline |
| Quantile / pinch loss | asymmetric | Target units | Tunable | Over- and under-prediction cost differently |
| Huber | quadratic then linear | Target units | Medium | Want MSE's gradients with MAE's robustness |

Two things worth saying out loud in an interview:

- **MAE optimizes the median, MSE optimizes the mean.** If you train with MSE on a right-skewed target (revenue, latency), the model systematically over-predicts the typical case. That's a modeling choice, not a bug — but it should be deliberate.
- **R² can be negative** (worse than predicting the mean) and always increases when you add features, so use adjusted R² when comparing models with different feature counts. R² is also not comparable across datasets with different target variance.

For skewed positive targets, training on `log1p(y)` and reporting RMSE in log space (RMSLE) penalizes under-prediction more than over-prediction and stops a few huge values from dominating the loss.

---

## Ranking and Recommendation Metrics

| Metric | What it captures |
|---|---|
| **Precision@k** | Fraction of the top-k that are relevant — matches a fixed-size UI slot |
| **Recall@k** | Fraction of all relevant items appearing in top-k — the standard retrieval metric |
| **MRR** | `mean(1 / rank of first relevant)` — right for "one correct answer" tasks |
| **MAP** | Mean average precision — rewards relevant items ranked higher |
| **NDCG@k** | Discounted cumulative gain, normalized — handles graded relevance and position discount |
| **Hit rate@k** | Any relevant item in top-k — coarse but interpretable |

NDCG is the default for search and recommendation because it handles both *graded* relevance (a 4-star match beats a 2-star match) and *position discount* (rank 1 is worth more than rank 10):

```
DCG@k = Σ (2^rel_i - 1) / log₂(i + 1)      NDCG@k = DCG@k / IDCG@k
```

These same metrics apply to RAG retrieval evaluation — recall@k on the retriever is usually the first thing to measure when a RAG system gives wrong answers.

---

## Cross-Validation Strategies

| Strategy | When | Why |
|---|---|---|
| **K-Fold** | i.i.d. data, plenty of it | Standard, low bias |
| **Stratified K-Fold** | Classification, especially imbalanced | Preserves class ratio per fold |
| **TimeSeriesSplit** | Any temporal data | Train always precedes validation |
| **GroupKFold** | Repeated entities (users, patients, sessions) | Keeps a group entirely in one fold |
| **StratifiedGroupKFold** | Both of the above | Rare in libraries, common in reality |
| **Nested CV** | Reporting a performance estimate *and* tuning | Outer loop estimates, inner loop tunes |
| **LOOCV** | Very small datasets | Nearly unbiased, high variance, expensive |

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, cross_val_score

# Temporal: fold i trains on [0..i] and validates on [i+1]
tscv = TimeSeriesSplit(n_splits=5, gap=7)   # `gap` guards against label leakage windows

# Grouped: no user appears in both train and validation
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=user_ids, cv=gkf, scoring='roc_auc')
print(f"AUC {scores.mean():.4f} ± {scores.std():.4f}")
```

Always report the **standard deviation** across folds, not just the mean. A model scoring 0.82 ± 0.01 and one scoring 0.84 ± 0.06 are not clearly different, and saying so is a strong interview signal.

**Nested CV** exists because tuning on the same folds you report from is optimistically biased: the hyperparameter search has already seen every validation fold. Outer loop for the estimate, inner loop for tuning.

---

## Offline vs Online Metrics

Offline metrics are proxies. The model exists to move a business number.

| Layer | Examples | Measured by |
|---|---|---|
| **Model metrics** | AUC, RMSE, NDCG | Offline holdout |
| **System metrics** | p99 latency, error rate, cost per 1k predictions | Production telemetry |
| **Product metrics** | CTR, conversion, session length, task completion | A/B test |
| **Business metrics** | Revenue, retention, fraud loss, support cost | A/B test + finance |

Proxies diverge from outcomes. A recommender optimizing CTR learns clickbait; a fraud model optimizing AUC may improve on the frauds you already catch. The standard answer: pick one primary online metric, define **guardrail metrics** that must not regress (latency, complaint rate, revenue per session), and validate offline gains with an A/B test before you believe them.

Include a **degradation plan** too: metrics drift as the input distribution moves, so define what a regression looks like and what triggers a rollback.

---

## Metric Selection Cheat Sheet

| Problem | Primary metric | Why |
|---|---|---|
| Balanced binary classification | ROC-AUC + accuracy | Both meaningful when classes are balanced |
| Fraud / rare-event detection | PR-AUC, recall @ fixed precision | ROC hides the false-positive burden |
| Medical screening | Recall at a specificity floor | Misses are far costlier than follow-ups |
| Spam filtering | Precision at a recall floor | A blocked legitimate email is expensive |
| Credit scoring | AUC + calibration (Brier) | Scores feed pricing, so they must be real probabilities |
| Demand forecasting | MAE or quantile loss | Robust to spikes; over/under stock costs differ |
| Revenue prediction | RMSLE or MAE | Skewed target, relative error matters |
| Search / recommendation | NDCG@k, recall@k | Position and graded relevance matter |
| RAG retrieval | Recall@k, MRR | Generation cannot fix what retrieval missed |
| LLM generation | Task success rate, LLM-as-judge + human review | No single automatic metric is sufficient |

---

## Interview Q&A

#### Your model has 99% accuracy. Is it good?

Unanswerable without the class balance and the cost structure. If 99% of examples are negative, predicting "always negative" scores 99% and has zero value — accuracy is measuring the prior, not the model. I'd ask for prevalence, then report PR-AUC against the prevalence baseline, plus recall at whatever precision the business can tolerate. I'd also ask what a false positive and a false negative each cost, because that determines both the metric and the operating threshold.

#### When would you use ROC-AUC over PR-AUC, and vice versa?

ROC-AUC when classes are roughly balanced, when both classes matter symmetrically, or when I need a prevalence-invariant number to compare the same model across populations with different base rates. PR-AUC when positives are rare and I care about the quality of the positive predictions specifically.

The mechanism: ROC's x-axis is FPR = FP/(FP+TN). With a huge negative class, thousands of false positives barely move FPR, so ROC stays optimistic while precision collapses. PR-AUC uses FP/(TP+FP), which has no large denominator to hide behind.

#### What is probability calibration, and when do you need it?

Calibration means predicted probabilities match observed frequencies — of the cases scored 0.7, about 70% should be positive. You need it whenever the score is consumed as a number rather than as a rank: expected-value decisions, risk pricing, thresholds derived from costs, or feeding the score into a downstream model or human judgment.

You don't strictly need it when you only rank (recommendations shown in order) or threshold once at an empirically tuned cutoff. Check with a reliability diagram and the Brier score; fix with Platt scaling (small data), isotonic regression (larger data), or temperature scaling (neural networks) — always fit on a held-out calibration set.

#### Why is a random train/test split wrong for time series?

It leaks the future into the past. A random split lets the model train on Tuesday and Thursday while validating on Wednesday, so it can exploit information no production model would have — trends, same-day correlated events, or explicit lag features computed over the whole series. Validation scores look great and production degrades immediately.

Use a forward-chaining split (`TimeSeriesSplit`), evaluate on strictly later data than training, and add a `gap` when the label takes time to materialize — e.g. a 30-day churn label means the last 30 days of training data would not have been labeled yet in production.

#### Explain the precision-recall tradeoff and how you pick a threshold.

Both derive from one score and one cutoff. Lower the threshold and you predict positive more often: recall rises, precision falls. The curve is a property of the model's ranking; the threshold is a business decision on top of it.

I pick it by expected cost when costs are known: choose the threshold minimizing `COST_FN × FN + COST_FP × FP` on validation data. When costs aren't quantified, I anchor on an operational constraint — reviewer capacity, or a precision floor stakeholders will accept — and take the best recall subject to it. F1-optimal is a fallback for when no business context exists at all, which is rarer than people assume.

#### What's the difference between MAE and RMSE, and when does the choice matter?

RMSE squares errors before averaging, so a single error of 10 contributes as much as 100 errors of 1. MAE weighs every error linearly. Consequently RMSE ≥ MAE always, and the gap grows with error variance.

The choice matters when outliers exist and their treatment is a real decision. Training on MSE fits the conditional mean and chases outliers; training on MAE fits the conditional median and ignores them. For delivery-time prediction, a rare 3-hour delay may genuinely be catastrophic (RMSE), while for sensor data the same spike is measurement noise (MAE, or Huber for a middle ground).

#### How do you evaluate a model when labels arrive weeks later?

Three moves, used together:
1. **Proxy metrics available immediately** — prediction distribution shift, feature drift (PSI/KL), and model confidence, monitored for anomalies that predict a quality drop.
2. **A delayed evaluation pipeline** — join predictions to labels as they arrive and backfill true performance on a rolling basis, accepting that today's number describes the model from weeks ago.
3. **A fast-label subsample** — for some domains you can buy or manually label a small sample quickly for a same-week signal.

I'd also make sure the training setup respects the delay: features must be as-of prediction time and the CV gap must equal the label latency, or offline scores will be unreachable in production.

#### Your offline AUC improved but the A/B test is flat. What happened?

The usual suspects, in the order I'd check them:
- **Train/serve skew**: features computed differently offline and online — the most common cause by a wide margin.
- **The offline metric is the wrong proxy**: AUC improved on easy examples that don't change any decision, or the threshold means the ranking improvement never crosses into a different action.
- **The improvement is inside noise**: the A/B test may be underpowered for an effect that size; check the minimum detectable effect before concluding "flat".
- **Leakage offline**: the offline gain was never real.
- **System effects**: latency added by the new model reduced engagement enough to cancel the quality gain.
- **Position/feedback effects**: the model's own past outputs shaped the training data, so offline replay overestimates the gain.

#### Why report standard deviation across CV folds?

Because a single mean hides instability. Fold variance tells you whether a difference between two models is real, whether the dataset has heterogeneous subgroups (one fold much worse usually means a distinct segment the model fails on), and whether the model is sensitive to which rows it sees. If model A is 0.84 ± 0.06 and model B is 0.82 ± 0.01, B may be the better production choice — the paired difference across folds, not the raw means, is what to test.

#### What is data leakage, and how do you detect it?

Leakage is any information in training features that would not be available at prediction time in production. Common forms: a feature computed after the label event, target encoding fit before the split, scaler or imputer fit on the full dataset, duplicate rows spanning the split, and IDs correlated with the label.

Detection signals: implausibly high scores, one feature dominating importance, a large gap between offline and online performance, and a metric that collapses when the split is made temporal instead of random. The structural fix is to build every transformation inside a `Pipeline` fit only on the training fold, and to write features with explicit as-of timestamps.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Accuracy on imbalanced data | The majority-class baseline already scores high | PR-AUC, recall at fixed precision, MCC |
| Fitting the scaler/imputer before splitting | Test statistics leak into training | Fit inside a `Pipeline` per fold |
| Tuning the threshold on the test set | Test performance becomes optimistic | Tune on validation, report on test |
| Comparing R² across different datasets | R² depends on target variance | Compare RMSE/MAE in the same units |
| Ignoring calibration when scores drive decisions | Expected-value math is wrong even with great AUC | Reliability diagram + isotonic/Platt |
| Reporting only the CV mean | Hides instability and subgroup failure | Report mean ± std, inspect worst fold |
| Random split with repeated users/sessions | The same entity in train and test inflates scores | `GroupKFold` on the entity |
| MAPE on targets near zero | Division by ~0 explodes the metric | SMAPE, MAE, or a relative-error variant |
| Optimizing a proxy without guardrails | CTR up, satisfaction down | Primary metric + explicit guardrail metrics |
| One aggregate metric for all segments | Masks failure on a minority segment | Slice metrics by segment and report the worst |

---

## Related Topics

- [Ensemble Methods and Gradient Boosting](./intro_ensemble_methods.md)
- [Statistics and Probability](./intro_statistics_probability.md)
- [Feature Engineering](./intro_feature_engineering.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md)
- [Classical ML Overview](./README.md)
