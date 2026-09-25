# Anomaly Detection

Anomaly detection is where the usual supervised playbook breaks down: you rarely have labels, the positive class is a fraction of a percent, "normal" drifts over time, and the cost of a miss is wildly different from the cost of a false alarm. It underpins fraud, intrusion detection, predictive maintenance, and data quality monitoring, and interviewers use it to test whether you can reason without a clean training signal.

---

## Table of Contents
1. [Three Kinds of Anomaly](#three-kinds-of-anomaly)
2. [Choosing a Problem Framing](#choosing-a-problem-framing)
3. [Statistical Methods](#statistical-methods)
4. [Distance and Density Methods](#distance-and-density-methods)
5. [Isolation Forest](#isolation-forest)
6. [One-Class SVM](#one-class-svm)
7. [Autoencoders](#autoencoders)
8. [Time Series Anomalies](#time-series-anomalies)
9. [Evaluation Without Labels](#evaluation-without-labels)
10. [Thresholding and Alert Budgets](#thresholding-and-alert-budgets)
11. [Production Concerns](#production-concerns)
12. [Method Selection](#method-selection)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Three Kinds of Anomaly

Naming which one you're dealing with is the first move in any interview answer, because the method follows from it.

| Type | Definition | Example | Approach |
|---|---|---|---|
| **Point** | A single record is abnormal on its own | A $50,000 charge on a card averaging $40 | Density, distance, isolation |
| **Contextual** | Normal in general, abnormal *in context* | 30°C is fine in July, anomalous in January | Model conditioned on context; residual-based |
| **Collective** | Each point is fine; the *sequence* is not | A steady, slow data exfiltration | Sequence models, windowed features |

The classic interview trap is applying a point method to a contextual problem. A transaction of $500 is unremarkable globally and highly unusual for a specific user at 3am in a new country, so the useful features are *relative* to context (deviation from the user's own baseline), not absolute.

---

## Choosing a Problem Framing

Before picking an algorithm, settle what labels you actually have. This determines everything:

| Situation | Framing | Notes |
|---|---|---|
| No labels at all | **Unsupervised** | Assume anomalies are rare and different; most common starting point |
| Only clean "normal" data | **Semi-supervised (novelty detection)** | Model normality, flag deviations. Often the strongest setup |
| A few labelled anomalies | **Supervised, heavily imbalanced** | Use them: a supervised model on 200 labelled frauds usually beats any unsupervised method |
| Labels arrive later | **Hybrid** | Unsupervised for cold start, supervised as labels accumulate |

The answer interviewers want: **if you have labels, use them.** Unsupervised anomaly detection finds what is *statistically unusual*, which is not the same as what is *bad*. A user's first-ever large purchase is unusual and legitimate; a $4 test charge before a fraud spree is usual-looking and malicious. Supervised learning targets the thing you care about; unsupervised targets a proxy.

The realistic production design is both: an unsupervised layer for novel patterns nobody has labelled yet, and a supervised model for known fraud patterns, with the union feeding a review queue.

---

## Statistical Methods

Start here. They are fast, explainable, and often sufficient.

**Z-score**: assumes roughly Gaussian data: `z = (x - μ) / σ`, flag `|z| > 3`. Its weakness is that the mean and standard deviation are themselves corrupted by the outliers you're hunting.

**Modified Z-score**: uses the median and MAD instead, which are robust:

```python
import numpy as np

def modified_zscore(x, threshold=3.5):
    """Robust to outliers: median and MAD are not dragged by extreme values."""
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:                                  # degenerate: >50% identical values
        mad = np.mean(np.abs(x - median)) * 1.253 or 1e-9
    scores = 0.6745 * (x - median) / mad          # 0.6745 makes MAD comparable to σ
    return np.abs(scores) > threshold, scores
```

**IQR**: flag outside `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`. Distribution-free and the basis of the box plot.

**Extreme value theory (POT)**: instead of assuming a distribution for all the data, model only the *tail* above a high quantile with a generalized Pareto distribution. This is the principled way to set a threshold for "how extreme is too extreme" and is what a strong candidate mentions when asked how to pick a cutoff on a heavy-tailed metric.

**Why robust statistics matter here**: with 5% contamination, the sample mean and standard deviation shift enough that genuine outliers fall inside `3σ`: the outliers mask themselves. Median and MAD have a 50% breakdown point and don't.

---

## Distance and Density Methods

**k-NN distance**: score each point by its distance to its k-th nearest neighbour. Simple, effective, `O(n²)` without an index.

**Local Outlier Factor (LOF)**: the key idea is that density is *local*. A point in a sparse cluster is normal for that cluster even if it's far from everything else in absolute terms. LOF compares a point's local density to that of its neighbours:

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
labels = lof.fit_predict(X)            # -1 = outlier, 1 = inlier
scores = -lof.negative_outlier_factor_  # higher = more anomalous
```

LOF is the right answer when the data has clusters of differing density: a global distance threshold either floods you with false positives from the sparse cluster or misses outliers near the dense one. Its cost is `O(n²)` and it doesn't naturally handle new points (use `novelty=True` to enable `predict` on unseen data).

---

## Isolation Forest

The most practical default for tabular anomaly detection, and the one most likely to come up.

**The insight is inverted from other methods**: rather than modelling what normal looks like and measuring deviation, it exploits the fact that anomalies are *easy to isolate*. Build random trees by picking a random feature and a random split value; anomalies, being few and different, get separated into their own leaf in very few splits. The anomaly score is the average path length across the forest: short path means anomalous.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,
    max_samples=256,        # subsampling is deliberate: see below
    contamination=0.01,     # expected anomaly rate; sets the decision threshold
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train)                       # ideally clean-ish data
scores = -iso.score_samples(X_test)    # higher = more anomalous
labels = iso.predict(X_test)           # -1 anomaly, 1 normal
```

Two details worth knowing:

- **`max_samples=256` is not a performance shortcut.** Small subsamples *improve* accuracy by reducing swamping (normal points misjudged because the sample is crowded) and masking (a cluster of anomalies looking normal because there are enough of them to form their own dense region). The original paper found 256 works well across datasets.
- **`contamination` only sets the threshold**, not the model. The scores are unchanged; it just picks where to cut. If you don't know the rate, keep the continuous scores and threshold on your own operational criterion.

Strengths: linear time, no distance computation, no scaling needed, handles high dimensions reasonably. Weakness: it splits on axis-aligned boundaries, so it struggles with anomalies defined by a *combination* of correlated features: a point that is normal in `x` and normal in `y` but abnormal in `x/y`. Feature engineering (add the ratio) or Extended Isolation Forest addresses that.

---

## One-Class SVM

Learns a boundary enclosing the normal data in a kernel-induced feature space; anything outside is anomalous.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_train)   # scaling is REQUIRED here
oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)  # nu ≈ expected outlier fraction
oc.fit(X_scaled)
```

It works best in the semi-supervised setting where training data is clean. It is sensitive to `gamma` and `nu`, scales poorly (`O(n²)` to `O(n³)`), and requires standardized features. In practice Isolation Forest has largely displaced it for tabular data: mention it as the classical option and say why you'd usually pick something else.

---

## Autoencoders

Train a neural network to reconstruct its input through a bottleneck. Trained only on normal data, it learns to reconstruct normal patterns well and fails on anomalies; reconstruction error becomes the anomaly score.

```python
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_hidden=32, d_latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def anomaly_scores(model, X):
    model.eval()
    with torch.no_grad():
        recon = model(X)
        return ((X - recon) ** 2).mean(dim=1)    # per-sample reconstruction error
```

The bottleneck is the whole mechanism: if the latent dimension is large enough, the network learns the identity function and reconstructs anomalies perfectly too, scoring everything as normal. That is the number-one autoencoder failure, and naming it is a strong signal.

Use an autoencoder when the data is high-dimensional with rich structure (images, sensor arrays, network traffic embeddings), where simpler methods can't capture what "normal" means. For a 20-column tabular table, Isolation Forest is faster, needs no tuning, and usually wins.

Variants worth naming: **VAE** (probabilistic, gives a likelihood rather than a raw error), **denoising autoencoder** (harder to learn identity), and **LSTM autoencoder** for sequences.

---

## Time Series Anomalies

Most real anomaly detection is temporal, and the framing changes.

**Decompose first.** Split into trend, seasonality, and residual, then detect on the residual, otherwise every Monday morning peak is an "anomaly":

```python
from statsmodels.tsa.seasonal import STL

stl = STL(series, period=24*7, robust=True).fit()   # robust=True downweights outliers
resid = stl.resid
flags, _ = modified_zscore(resid, threshold=4.0)     # detect on the residual, not the raw series
```

**Forecast-and-compare.** Predict the next value with any forecaster (ARIMA, Prophet, gradient boosting on lags); a large prediction interval violation is an anomaly. This handles contextual anomalies naturally because the forecast *is* the context.

**Changepoint detection** answers a different question, not "is this point weird" but "did the underlying regime shift". CUSUM and Bayesian online changepoint detection are the standard tools, and they're what you want for drift monitoring rather than spike detection.

**Windowed features** turn a collective anomaly into a point anomaly: compute rolling mean, variance, rate of change, and entropy over a window, then run any point detector on the window features. This is the simplest way to catch "slow exfiltration" patterns.

Two temporal-specific mistakes to avoid: computing normalization statistics over the whole series (leaks the future into the past), and evaluating with a random split rather than a forward-chaining one.

---

## Evaluation Without Labels

The hardest part, and where interviews go deep.

**If you have any labels**, even a few hundred, use them properly: PR-AUC (not ROC-AUC, see the [model evaluation guide](./intro_model_evaluation.md) for why prevalence breaks ROC), and precision at the alert volume you can actually review.

**If you have none**, you still have options and should name several:
- **Synthetic anomaly injection**: perturb real records in ways that mimic real anomalies (swap a field, scale a value, shuffle a sequence) and measure recall on the injected set. Imperfect, but it gives a number.
- **Analyst feedback loop**: sample flagged cases for human review, measure precision on that sample, and feed the labels back. This is how real systems bootstrap.
- **Stability**: score distributions and flagged-rate should be stable day over day; a sudden jump means the data changed, not that fraud tripled.
- **Consensus**: agreement between independent methods is weak evidence, but disagreement is a useful triage signal.
- **Downstream proxy**: chargebacks, incident tickets, or machine failures arriving weeks later.

**Time-aware evaluation for series**: point-wise metrics badly misrepresent event detection. If an anomalous event spans 60 minutes and you flag one minute of it, point-wise recall says 1/60 while operationally you caught it. Use event-level or range-based (point-adjusted) metrics, and say so: it's a detail that distinguishes people who have actually shipped this.

---

## Thresholding and Alert Budgets

A detector outputs a score; the threshold is a business decision, and it's usually the real question behind "how would you deploy this?"

Anchor on **review capacity**. If analysts can process 200 alerts a day, the threshold is whatever produces 200 alerts a day: a percentile of the score distribution, recomputed on a rolling basis. This automatically adapts as volume grows, and it's a far more defensible answer than picking a fixed score.

Other practical mechanics:
- **Dynamic thresholds** per segment: a global cutoff over-alerts on the highest-volume merchant and under-alerts on the smallest.
- **Alert deduplication and grouping**: one incident should be one alert, not 400.
- **Hysteresis**: require N consecutive breaches to fire, and a lower threshold to clear, so a flapping metric doesn't generate a storm.
- **Severity tiers**: auto-block at extreme scores, queue for review in the middle, log only at the bottom.

---

## Production Concerns

**Concept drift is guaranteed.** Normal behaviour shifts: seasonally, and because the world changes. A model trained on last year's traffic flags this year's normal. Retrain on a rolling window, and monitor the flagged rate as a first-class metric: a jump usually means drift, not an attack.

**Adversarial adaptation.** In fraud and security, the anomalies are generated by people who adapt to your detector. Anything you deploy has a shelf life, attackers probe for the threshold, and a detector that's public knowledge stops working. This argues for ensembles, randomized thresholds, and not exposing the score.

**Feedback loops.** If you block everything the model flags, you never learn whether those cases were truly bad, and the model's own decisions shape its future training data. Hold out a small random unblocked control group to keep an unbiased signal: the same idea as an exploration arm in a bandit.

**Explainability is not optional.** An analyst can't action "score 0.87". Ship the top contributing features per alert (SHAP works on Isolation Forest, or use per-feature deviation from baseline), plus the entity's recent history. Alert quality is judged on how fast a human can decide.

**Cold start.** New users have no baseline. Fall back to population-level or segment-level norms and taper toward personal baselines as history accumulates.

---

## Method Selection

| Situation | Method | Why |
|---|---|---|
| Tabular, no labels, want a default | **Isolation Forest** | Fast, minimal tuning, scales linearly |
| Clusters of varying density | **LOF** | Local density comparison |
| Clean training data available | **One-Class SVM** or autoencoder | Semi-supervised is stronger when it applies |
| High-dimensional, structured (images, sensors) | **Autoencoder / VAE** | Learns the manifold of normal |
| Time series with seasonality | **STL + residual detection** or forecast-based | Removes the pattern before detecting |
| Regime change, not spikes | **Changepoint (CUSUM, BOCPD)** | Different question entirely |
| Some labels exist | **Supervised (gradient boosting)** | Targets what's bad, not just what's rare |
| Need interpretability for analysts | **Statistical / rules / IF + SHAP** | Explainable per alert |
| Streaming, low latency | **Robust z-score, HalfSpaceTrees** | Constant memory, online update |

---

## Interview Q&A

#### How do you detect anomalies when you have no labels at all?

Start by defining the anomaly type (point, contextual, or collective), because that decides the method. Then establish a baseline of "normal": for tabular data I'd start with Isolation Forest for a fast unsupervised score, plus simple robust-statistical rules for interpretability. For anything temporal, decompose out trend and seasonality first and detect on the residual.

The part people miss is **evaluation**. Without labels I'd inject synthetic anomalies to get a recall estimate, sample flagged cases for analyst review to get precision, monitor score-distribution stability, and set the threshold by review capacity rather than an arbitrary cutoff. I'd also build the feedback loop from day one, so that analyst dispositions accumulate into a labelled set: within a few months you can usually switch to a supervised model, which targets what's actually harmful rather than what's merely rare.

#### Explain Isolation Forest. Why does subsampling improve it?

It inverts the usual approach: instead of modelling normality and measuring deviation, it exploits that anomalies are easy to isolate. Random trees split on a random feature at a random threshold; because anomalies are few and different, they end up alone after very few splits. The score is average path length across the forest: shorter means more anomalous.

Subsampling (`max_samples=256`) improves accuracy rather than just speed, because of **swamping** and **masking**. Swamping is when normal points near an anomalous region get misjudged in a crowded sample; masking is when a cluster of anomalies is dense enough to look like a legitimate mode. Small subsamples make anomalies sparser relative to the sample, so both effects shrink. It's a rare case where less data per estimator is better.

#### Why is ROC-AUC misleading for anomaly detection?

Because anomalies are rare and ROC's x-axis is FPR = FP/(FP+TN). With 0.1% anomalies, the negative class is enormous, so thousands of false positives move FPR almost imperceptibly: you can report ROC-AUC 0.98 while 95% of the analyst's queue is noise.

PR-AUC uses FP/(TP+FP), which has no large denominator to hide behind, so it reflects what an operator actually experiences. I'd report PR-AUC against the prevalence baseline, plus precision at the alert volume the team can review, which is the number that determines whether the system is usable.

#### A point-wise metric says your time series detector has 2% recall, but operators say it catches everything. What's going on?

Point-wise metrics mismatch how anomalies actually occur. An incident spans a range (say 60 minutes), and if you flag the first 2 minutes, point-wise recall is 2/60 ≈ 3%, while operationally you detected the event promptly and completely.

The fix is **event-level or range-based evaluation**: an event counts as detected if any point within it is flagged (optionally weighted by how early), and false positives are counted per distinct alert rather than per timestamp. I'd also report detection latency, since catching an incident 2 minutes in versus 50 minutes in matters enormously and no point-wise metric captures it.

#### When would you use a supervised model instead of an unsupervised detector?

Whenever I have labels, because they target a different quantity. Unsupervised methods find what is statistically *unusual*; supervised models find what is *bad*. Those diverge constantly: a customer's first big purchase is unusual and fine, while a small card-testing charge looks perfectly ordinary and is malicious.

Even a few hundred labelled positives typically beat any unsupervised method, using class weighting and PR-AUC. The production design I'd propose is both layers: supervised for known patterns, unsupervised for novel ones nobody has labelled yet, with a feedback loop turning reviewed alerts into new labels.

#### How do you set the threshold?

From operational constraints, not from the score distribution in isolation. The most defensible version is **capacity-based**: if analysts review 200 cases a day, take the rolling percentile of the score that yields 200 alerts a day. It adapts automatically as volume grows.

If costs are quantified, minimize expected cost (`COST_FN × misses + COST_FP × false alarms`), which puts the threshold where the cost ratio says it belongs. For heavy-tailed metrics, extreme value theory (peaks-over-threshold with a generalized Pareto fit on the tail) gives a principled extremeness cutoff rather than a guessed one.

Then the operational layer: per-segment thresholds so one high-volume entity doesn't dominate, hysteresis so a flapping metric doesn't storm, deduplication so one incident is one alert, and severity tiers so extreme scores auto-action while mid-range ones queue.

#### Your autoencoder reconstructs anomalies perfectly and flags nothing. Why?

The bottleneck is too wide, so the network learned something close to the identity function: it reconstructs everything, including anomalies, and reconstruction error carries no signal.

Fixes: shrink the latent dimension until reconstruction of normal data is good but not perfect; add noise (denoising autoencoder), which makes identity useless; add regularization (sparse or contractive penalties); and verify the training data is actually clean: if anomalies are in the training set the model learns to reconstruct them by design. I'd also sanity-check by plotting reconstruction-error distributions for known-normal versus injected-anomalous data; if they overlap completely, capacity is the first suspect.

#### How do you handle seasonality and drift?

Seasonality: never detect on the raw series. Decompose with STL (using `robust=True` so existing outliers don't distort the fit) and detect on the residual, or use a forecaster whose prediction interval encodes the expected seasonal pattern: a violation is then a contextual anomaly by construction.

Drift: assume normal moves. Retrain on a rolling window, and monitor the **flagged rate** as a first-class metric: a sudden jump almost always means the input distribution shifted rather than that the anomaly rate really tripled. I'd separate "the world changed" from "something bad happened" by checking whether the shift is broad across features (drift) or concentrated in a few entities (real anomaly), and I'd run changepoint detection on the monitoring metrics themselves.

#### What's different about anomaly detection in an adversarial setting?

The data-generating process fights back. Attackers probe to find your threshold, then operate just under it; they mimic normal behaviour; and any detector's effectiveness decays from the moment it ships.

Practical consequences: don't expose scores or reasons to the user, randomize thresholds slightly so probing is noisy, ensemble diverse detectors so evading one isn't enough, retrain frequently on recent data, and hold out an unblocked control group so you keep an unbiased read on the true rate. It also changes evaluation: offline metrics on historical data overstate future performance, because that history predates the adaptation your deployment will cause.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Using mean/σ with contaminated data | Outliers inflate σ and mask themselves | Median and MAD, or trimmed statistics |
| Point method on a contextual problem | Misses "normal value, wrong context" anomalies | Model conditioned on context; detect on residuals |
| Detecting on a raw seasonal series | Every recurring peak is flagged | STL decompose, detect on the residual |
| ROC-AUC as the headline metric | Hides a false-positive flood at low prevalence | PR-AUC + precision at alert volume |
| Point-wise metrics on ranged events | Understates real detection performance | Event-level / range-based metrics + latency |
| Autoencoder latent dimension too large | Learns identity; flags nothing | Shrink bottleneck; denoising; verify clean training data |
| Training on data containing anomalies | The model learns them as normal | Clean the training set, or use a robust method |
| Fixed threshold forever | Volume grows, alerts flood or vanish | Rolling percentile tied to review capacity |
| Normalizing over the whole time series | Leaks future information into the past | Fit scalers on the training window only |
| Blocking everything flagged | Destroys the unbiased label signal | Keep a small unblocked control group |
| Score with no explanation | Analysts can't action it | Per-alert top contributing features + entity history |

---

## Related Topics

- [Model Evaluation and Metrics](./intro_model_evaluation.md)
- [Time Series Analysis & Forecasting](./intro_time_series.md)
- [Clustering Algorithms](./intro_clustering.md)
- [Ensemble Methods](./intro_ensemble_methods.md)
- [Statistics & Probability](./intro_statistics_probability.md)
- [Fraud Detection System Design](../system_design/fraud_detection.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [Data Quality & Validation](../mlops/intro_data_quality.md)
- [Classical ML Overview](./README.md)
