# ML Interview Cheat Sheet

The night-before quick reference: formulas, tables and the numbers interviewers expect you to know cold. No derivations, no narrative. Every section ends with a link to the guide that explains it properly.

Notation: `n` = samples, `d` = features, `k` = classes/clusters/top-k, `p` = predicted probability, `y` = label, `ŷ` = prediction, `η` = learning rate, `log` = natural log unless stated.

**Jump to:** [Classification](#classification-metrics) · [Regression](#regression-metrics) · [Ranking](#ranking-metrics) · [Losses](#loss-functions) · [Prob & Stats](#probability-and-statistics) · [Lin Alg & Calculus](#linear-algebra-and-calculus) · [Optimization](#optimization) · [Algorithms](#algorithm-cheat-table) · [Bias-Variance](#bias-variance-diagnostics) · [Neural Nets](#neural-network-essentials) · [LLM Numbers](#llm-numbers) · [System Numbers](#system-design-numbers) · [Top 20 Traps](#top-20-traps)

---

## Classification Metrics

```
                  Predicted
                Neg      Pos
Actual  Neg  |  TN   |   FP   |   FP = false alarm (Type I)
        Pos  |  FN   |   TP   |   FN = miss (Type II)
```

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | `(TP+TN) / (TP+TN+FP+FN)` | Balanced classes, symmetric costs |
| Precision (PPV) | `TP / (TP+FP)` | False positives are expensive (spam filter, alerts) |
| Recall (TPR, sensitivity) | `TP / (TP+FN)` | Misses are expensive (cancer, fraud) |
| Specificity (TNR) | `TN / (TN+FP)` | Clearing negatives matters; `FPR = 1 - TNR` |
| F1 | `2PR / (P+R)` = `2TP / (2TP+FP+FN)` | One number, imbalanced data |
| F-beta | `(1+β²)·P·R / (β²·P + R)` | `β>1` weights recall, `β<1` weights precision |
| Balanced accuracy | `(TPR + TNR) / 2` | Imbalanced, both classes matter |
| MCC | `(TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Honest single score in `[-1, 1]` |
| Log loss (BCE) | `-(1/n) Σ [y log p + (1-y) log(1-p)]` | Probabilities feed decisions; punishes confident errors |
| Brier score | `(1/n) Σ (p - y)²` | Calibration + sharpness; bounded `[0, 1]` |

- **Baselines to quote:** always predicting `0.5` gives Brier `0.25` and log loss `log 2 ≈ 0.693`. Predicting the prevalence `π` gives log loss `-[π log π + (1-π) log(1-π)]`.
- **Averaging (multiclass):** macro = unweighted mean over classes (rare classes count); micro = pool counts (equals accuracy for single-label multiclass); weighted = by support (hides rare-class failure).

| | ROC-AUC | PR-AUC (average precision) |
|---|---|---|
| Axes | TPR vs FPR | Precision vs Recall |
| Meaning | `P(score(random pos) > score(random neg))` | Precision averaged over recall levels |
| Random baseline | `0.5` | Prevalence `π` (e.g. `0.01` at 1% positives) |
| Class balance | Invariant to prevalence | Sensitive to prevalence |
| Prefer when | Balanced data; care about ranking both classes | Rare positives; you only act on the top of the list |

**Which metric when:**
- Rare-event detection (fraud, disease): PR-AUC, recall at fixed precision, precision@k.
- Threshold will be tuned later: ROC-AUC / PR-AUC (threshold-free).
- Probabilities are consumed (bidding, risk pricing, expected value): log loss + calibration plot / Brier.
- Costs are known: pick the threshold minimising `C_FP·FP + C_FN·FN`, not `0.5`.
- Stakeholder report: precision and recall at the operating threshold, never accuracy alone.

**Read more:** [Model Evaluation](../classical_ml/intro_model_evaluation.md) · [Anomaly Detection](../classical_ml/intro_anomaly_detection.md)

---

## Regression Metrics

| Metric | Formula | Optimal constant / notes |
|---|---|---|
| MSE | `(1/n) Σ (y - ŷ)²` | Minimised by the **mean**; outlier-sensitive |
| RMSE | `√MSE` | Same units as `y` |
| MAE | `(1/n) Σ \|y - ŷ\|` | Minimised by the **median**; robust |
| MAPE | `(100/n) Σ \|(y - ŷ) / y\|` | Undefined at `y=0`; penalises over-forecasts more (unbounded) than under-forecasts (capped at 100%) |
| sMAPE | `(100/n) Σ \|y - ŷ\| / ((\|y\| + \|ŷ\|)/2)` | Bounded `[0, 200%]`; still unstable near 0 |
| RMSLE | `√((1/n) Σ (log(1+ŷ) - log(1+y))²)` | Relative errors; penalises under-prediction more |
| R² | `1 - SS_res / SS_tot` | Fraction of variance explained; can be **negative** on test data |
| Adjusted R² | `1 - (1-R²)(n-1)/(n-p-1)` | Penalises adding features (`p` = #predictors) |
| Pinball (quantile τ) | `mean(max(τ·(y-ŷ), (τ-1)·(y-ŷ)))` | Minimised by the τ-quantile; prediction intervals |
| MASE | `MAE / MAE of naive (seasonal) forecast` | Time series; scale-free; `< 1` beats naive |

- `SS_res = Σ(y - ŷ)²`, `SS_tot = Σ(y - ȳ)²`. R² of a model that predicts `ȳ` is `0`.
- Huber loss as a metric: MSE near zero, MAE in the tails.

**Read more:** [Model Evaluation](../classical_ml/intro_model_evaluation.md) · [Time Series](../classical_ml/intro_time_series.md)

---

## Ranking Metrics

| Metric | Formula | Notes |
|---|---|---|
| Precision@k | `(# relevant in top k) / k` | Ignores order within top k |
| Recall@k | `(# relevant in top k) / (total # relevant)` | Retrieval stage of a recsys/RAG |
| Hit rate@k | `1 if any relevant in top k else 0`, averaged | One correct answer per query |
| MRR | `(1/\|Q\|) Σ_q 1 / rank_q` | `rank_q` = position of **first** relevant item |
| AP (per query) | `(1/R) Σ_{i=1..k} P@i · rel_i` | `R` = # relevant items; `rel_i ∈ {0,1}` |
| MAP | mean of AP over queries | Binary relevance, rewards putting all hits early |
| DCG@k | `Σ_{i=1..k} (2^rel_i - 1) / log2(i + 1)` | Graded relevance; some use gain `rel_i` instead of `2^rel_i - 1` |
| NDCG@k | `DCG@k / IDCG@k` | IDCG = DCG of the ideal ordering; in `[0, 1]` |

- Position 1 discount is `1/log2(2) = 1`; position 2 is `1/log2(3) ≈ 0.63`; position 3 is `0.5`.
- Offline ranking gains often fail to move online metrics (CTR, dwell, revenue). Always pair with an A/B test.

**Read more:** [Model Evaluation](../classical_ml/intro_model_evaluation.md) · [Recommender Systems](../classical_ml/intro_recommender_systems.md) · [Search Ranking System](../system_design/search_ranking_system.md)

---

## Loss Functions

| Loss | Formula | When used |
|---|---|---|
| MSE / L2 | `(1/n) Σ (y - ŷ)²` | Regression, Gaussian noise; gradient `∝ (ŷ - y)` |
| MAE / L1 | `(1/n) Σ \|y - ŷ\|` | Regression with outliers, Laplace noise; gradient is ±1 (non-smooth at 0) |
| Huber (δ) | `½r²` if `\|r\| ≤ δ`, else `δ(\|r\| - ½δ)`, `r = y - ŷ` | Robust regression; smooth version of L1 |
| Binary cross-entropy | `-[y log p + (1-y) log(1-p)]`, `p = σ(z)` | Binary / multi-label classification |
| Categorical CE | `-Σ_c y_c log p_c`, `p = softmax(z)` | Multiclass, LM next-token prediction |
| Label-smoothed CE | CE with target `(1-ε)·y + ε/K` | Reduce over-confidence; classification, MT |
| Focal | `-α_t (1 - p_t)^γ log p_t` | Heavy class imbalance (dense detection); `γ≈2` down-weights easy examples |
| Hinge | `max(0, 1 - y·f(x))`, `y ∈ {-1, +1}` | SVMs; margin-based, not probabilistic |
| Contrastive (pairwise) | `y·d² + (1-y)·max(0, m - d)²` | Siamese nets; `y=1` similar pair, `d` = distance, `m` = margin |
| InfoNCE | `-log( exp(s(q,k⁺)/τ) / Σ_j exp(s(q,k_j)/τ) )` | CLIP, SimCLR, dense retrieval; in-batch negatives, temperature `τ` |
| Triplet | `max(0, d(a,p) - d(a,n) + m)` | Face ID, metric learning; needs hard-negative mining |
| KL divergence | `KL(P‖Q) = Σ P(x) log(P(x)/Q(x))` | Distillation, VAEs, RLHF penalty; `≥ 0`, **not symmetric** |

- `CrossEntropy(P, Q) = H(P) + KL(P‖Q)`: with fixed labels, minimising CE = minimising KL.
- Maximum likelihood view: MSE = Gaussian NLL, MAE = Laplace NLL, BCE = Bernoulli NLL.
- Forward KL `KL(P‖Q)` is mean-seeking (covers all modes); reverse KL `KL(Q‖P)` is mode-seeking.

**Read more:** [Neural Network Training](../deep_learning/intro_neural_network_training.md) · [Embeddings](../ai_genai/intro_embeddings.md) · [Model Compression](../deep_learning/intro_model_compression.md)

---

## Probability and Statistics

**Core rules**
- Bayes: `P(A|B) = P(B|A)·P(A) / P(B)`, with `P(B) = Σ_i P(B|A_i)·P(A_i)`.
- Classic: prevalence 1%, sensitivity 99%, FPR 5% → `P(D|+) = 0.0099 / (0.0099 + 0.0495) ≈ 16.7%`. Base rates dominate.
- `E[aX + bY] = aE[X] + bE[Y]` always. `Var(aX + b) = a²Var(X)`. `Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)`.
- MAP with Gaussian prior on weights = L2 (ridge); Laplace prior = L1 (lasso).

| Distribution | Mean | Variance | Typical use |
|---|---|---|---|
| Bernoulli(p) | `p` | `p(1-p)` | Single click / conversion |
| Binomial(n, p) | `np` | `np(1-p)` | # successes in `n` trials |
| Geometric(p) (trials to 1st success) | `1/p` | `(1-p)/p²` | Attempts until success |
| Poisson(λ) | `λ` | `λ` | Counts per interval (arrivals, events) |
| Uniform(a, b) | `(a+b)/2` | `(b-a)²/12` | Random init, priors |
| Normal(μ, σ²) | `μ` | `σ²` | Sums/means (CLT), noise |
| Exponential(λ) | `1/λ` | `1/λ²` | Waiting time; memoryless |
| Gamma(k, θ) (shape, scale) | `kθ` | `kθ²` | Sum of `k` exponentials; positive skew |
| Beta(α, β) | `α/(α+β)` | `αβ / ((α+β)²(α+β+1))` | Prior on a rate (Thompson sampling) |

- Normal: 68 / 95 / 99.7% within 1 / 2 / 3 σ. `z = 1.645` (90%, two-sided), `1.96` (95%), `2.576` (99%).

**Inference**
- **CLT:** for i.i.d. samples with finite variance, `x̄ ≈ N(μ, σ²/n)` for large `n`. Standard error `SE = σ/√n`.
- **95% CI (mean):** `x̄ ± 1.96·s/√n` (use `t_{n-1}` for small `n`). **Proportion:** `p̂ ± 1.96·√(p̂(1-p̂)/n)`.
- A 95% CI means: 95% of intervals built this way contain the true value. Not "95% probability the value is in this interval".
- **p-value:** `P(data at least this extreme | H0 true)`. It is **not** `P(H0 | data)` and says nothing about effect size.
- `α` = Type I rate (false positive), `β` = Type II rate (false negative), power = `1 - β`.

| Test | Use when |
|---|---|
| z-test | Means with known σ or large `n`; comparing two proportions (CTR, conversion) |
| t-test (one/two-sample) | Means, unknown σ, small-to-moderate `n`; **Welch** if variances differ |
| Paired t-test | Same units measured twice (before/after, same queries under two models) |
| Chi-square | Categorical counts: independence (contingency table) or goodness of fit; expected counts `≥ 5` |
| Mann-Whitney U | Two groups, non-normal / ordinal, no mean assumption |
| ANOVA | Means across `> 2` groups |
| Bootstrap | Any statistic (median, AUC, ratio metrics) without a closed-form SE |

**A/B test sample size (per arm, two-sided):**
```
n ≈ (z_{1-α/2} + z_{1-β})² · 2σ² / δ²
α = 0.05, power = 0.8  →  (1.96 + 0.84)² ≈ 7.85  →  n ≈ 16·σ² / δ²
binary metric: σ² = p(1-p)
```
- Example: baseline conversion 10%, detect +1pp absolute → `16·0.09 / 0.0001 = 14,400` per arm.
- Halving the detectable effect `δ` quadruples `n`.

| Multiple-testing correction | Controls | Rule (`m` tests) |
|---|---|---|
| Bonferroni | FWER | Reject if `p < α/m`; simple, conservative |
| Holm (step-down) | FWER | Sort p ascending; reject while `p_(i) ≤ α/(m - i + 1)`; always at least as powerful as Bonferroni |
| Benjamini-Hochberg | FDR | Sort ascending; largest `i` with `p_(i) ≤ (i/m)·α`; reject hypotheses `1..i` |

**Read more:** [Statistics and Probability](../classical_ml/intro_statistics_probability.md) · [A/B Testing](../mlops/intro_ab_testing.md) · [Causal Inference](../classical_ml/intro_causal_inference.md)

---

## Linear Algebra and Calculus

**Derivatives to know**

| Function | Derivative |
|---|---|
| Sigmoid `σ(x) = 1/(1 + e^{-x})` | `σ(x)·(1 - σ(x))`, max `0.25` at `x = 0` |
| `tanh(x)` | `1 - tanh²(x)`, max `1` at `x = 0` |
| ReLU `max(0, x)` | `1` if `x > 0`, else `0` |
| Softmax `p_i = e^{z_i} / Σ_j e^{z_j}` | `∂p_i/∂z_j = p_i(δ_ij - p_j)` |
| Softmax + CE loss | `∂L/∂z = p - y` |
| Sigmoid + BCE loss | `∂L/∂z = p - y` |
| MSE for linear model `Xw` | `∇_w = (2/n)·Xᵀ(Xw - y)` |
| `aᵀx` | `a` |
| `xᵀAx` | `(A + Aᵀ)x` (= `2Ax` if `A` symmetric) |
| `‖x‖²` | `2x` |

**Shapes and costs**
- `(m×n)·(n×p) → (m×p)`, costing `m·n·p` multiply-adds (`2mnp` FLOPs).
- Dense layer: `X (batch×d_in) · W (d_in×d_out) + b (d_out) → (batch×d_out)`.
- Attention scores: `Q (n×d_k) · Kᵀ (d_k×n) → (n×n)`: quadratic in sequence length.
- Normal equation: `w = (XᵀX)⁻¹Xᵀy`, cost `O(nd² + d³)`; ridge: `w = (XᵀX + λI)⁻¹Xᵀy` (always invertible for `λ > 0`).

**Decompositions**
- Eigen: `Av = λv`. Symmetric matrices (covariance) have real eigenvalues and orthogonal eigenvectors.
- SVD: `A = UΣVᵀ` for any `A (m×n)`; `U (m×m)`, `V (n×n)` orthogonal; `Σ` diagonal with singular values `σ_1 ≥ σ_2 ≥ ... ≥ 0`.
- Truncated SVD (top `k`) is the best rank-`k` approximation in Frobenius and spectral norm (Eckart-Young).
- PCA = SVD of **centred** `X`: components are columns of `V`; variance along component `i` = `σ_i² / (n-1)`.
- Rank = number of non-zero singular values. Condition number = `σ_max / σ_min` (large → numerically unstable).
- Numerical stability: compute softmax/log-sum-exp as `m + log Σ e^{z_i - m}` with `m = max(z)`.

**Read more:** [Math for ML](../classical_ml/intro_math_for_ml.md) · [Dimensionality Reduction](../classical_ml/intro_dimensionality_reduction.md)

---

## Optimization

| Method | Update rule (`g = ∇L(θ)`) | Notes |
|---|---|---|
| GD / SGD | `θ ← θ - η·g` | SGD uses a mini-batch estimate of `g` |
| Momentum | `v ← μ·v + g`; `θ ← θ - η·v` | `μ ≈ 0.9`; damps oscillation, speeds up ravines |
| Nesterov | gradient evaluated at look-ahead `θ - η·μ·v` | Slightly better convergence than plain momentum |
| AdaGrad | `s ← s + g²`; `θ ← θ - η·g / (√s + ε)` | Per-parameter LR; LR decays forever (sparse features) |
| RMSProp | `s ← ρ·s + (1-ρ)·g²`; `θ ← θ - η·g / (√s + ε)` | Fixes AdaGrad's decay; `ρ ≈ 0.9` |
| Adam | `m ← β₁m + (1-β₁)g`; `v ← β₂v + (1-β₂)g²`; `m̂ = m/(1-β₁ᵗ)`; `v̂ = v/(1-β₂ᵗ)`; `θ ← θ - η·m̂/(√v̂ + ε)` | Defaults `β₁=0.9, β₂=0.999, ε=1e-8`; bias correction matters early |
| AdamW | Adam step, plus `θ ← θ - η·λ·θ` applied separately | Decoupled weight decay; default for transformers |

**Weight decay vs L2**
- L2 regularisation adds `(λ/2)‖θ‖²` to the loss → adds `λθ` to the gradient.
- With plain SGD this is identical to weight decay (`θ ← (1 - ηλ)θ - η·g`).
- With Adam it is **not**: the `λθ` term gets divided by `√v̂`, so parameters with large gradients are barely regularised. AdamW decouples it.
- Usually exclude biases and norm gains from weight decay.

| LR schedule | Formula / shape | Used for |
|---|---|---|
| Step decay | multiply by `γ` (e.g. 0.1) every `s` epochs | Classic CNNs |
| Exponential | `η_t = η_0·γᵗ` | Simple decay |
| Cosine | `η_t = η_min + ½(η_max - η_min)(1 + cos(π·t/T))` | Default for modern training |
| Linear warmup | `η_t = η_max·t/T_warm` for first steps | Transformers, large batches (Adam's `v` is noisy early) |
| Inverse sqrt | `η_t ∝ d_model^{-0.5}·min(t^{-0.5}, t·T_warm^{-1.5})` | Original transformer |
| One-cycle | ramp up then anneal down | Fast convergence on CNNs |
| Reduce-on-plateau | cut LR when val metric stalls | Small projects, fine-tuning |

- Gradient clipping by global norm (e.g. `1.0`) guards against spikes, essential for RNNs and LLMs.
- Batch size scaling (heuristic): linear LR scaling for SGD (`k×` batch → `k×` LR, with warmup); roughly `√k` for Adam.

**Read more:** [Neural Network Training](../deep_learning/intro_neural_network_training.md) · [Math for ML](../classical_ml/intro_math_for_ml.md)

---

## Algorithm Cheat Table

`n` = samples, `d` = features, `k` = clusters/neighbours, `T` = trees, `L` = tree depth, `n_sv` = support vectors.

| Algorithm | Type | Train / predict (per sample) | Key hyperparams | Scale features? | Missing values? | Pros / cons |
|---|---|---|---|---|---|---|
| Linear regression (OLS/ridge/lasso) | Supervised, regression | `O(nd² + d³)` closed form, `O(nd)` per SGD epoch / `O(d)` | `α` (L1/L2 strength) | Yes if regularised | No | + fast, interpretable / − linear only, sensitive to outliers and collinearity |
| Logistic regression | Supervised, classification | `O(nd · iters)` / `O(d)` | `C` (=1/λ), penalty | Yes | No | + calibrated-ish probabilities, strong baseline / − linear boundary |
| Naive Bayes | Supervised, classification | `O(nd)` / `O(dK)` | smoothing `α`, variant | No | Can skip features in principle; not in sklearn | + tiny data, text / − independence assumption, poor probabilities |
| k-NN | Supervised, lazy | `O(1)` (store) or `O(nd log n)` tree / `O(nd)` brute | `k`, distance, weights | Yes | No | + no training, flexible / − slow predict, curse of dimensionality |
| SVM (kernel) | Supervised | `O(n²)` to `O(n³)` / `O(n_sv·d)` | `C`, kernel, `γ` | Yes | No | + strong on small/medium, high-dim data / − doesn't scale past ~100k rows, no native probabilities |
| Decision tree | Supervised | `O(nd log n)` / `O(L)` | `max_depth`, `min_samples_leaf` | No | Impl-dependent (sklearn ≥ 1.3 yes) | + interpretable, nonlinear, mixed types / − high variance |
| Random forest | Ensemble (bagging) | `O(T·n·d' log n)`, `d'` = features tried per split / `O(T·L)` | `n_estimators`, `max_features`, depth | No | Impl-dependent (sklearn ≥ 1.4 yes) | + robust, little tuning, OOB estimate / − large models, weak extrapolation |
| Gradient boosting (XGBoost/LightGBM/CatBoost) | Ensemble (boosting) | `O(T·n·d)` with histograms / `O(T·L)` | `learning_rate`, `n_estimators`, `max_depth`/`num_leaves`, subsample | No | Yes (learned default direction) | + best on tabular data / − sequential training, overfits without early stopping |
| k-means | Unsupervised, clustering | `O(n·k·d·iters)` / `O(kd)` | `k`, init (k-means++) | Yes | No | + simple, fast / − spherical equal-size clusters, needs `k`, local optima |
| DBSCAN | Unsupervised, clustering | `O(n log n)` with index, `O(n²)` worst / no native predict | `eps`, `min_samples` | Yes | No | + arbitrary shapes, finds noise / − struggles with varying density, high `d` |
| Gaussian mixture (EM) | Unsupervised, density | `O(n·k·d²·iters)` full cov / `O(k·d²)` | `k`, covariance type | Yes | No | + soft assignments, elliptical clusters / − local optima, sensitive to init |
| PCA | Unsupervised, dim. reduction | `O(nd² + d³)` full, `O(ndk)` randomised / `O(dk)` | `n_components` | Yes (standardise) | No | + decorrelates, denoises / − linear, components hard to interpret |

- Tree models are invariant to monotonic feature transforms; distance, margin and gradient-based models are not.
- Default tabular pipeline: logistic regression baseline → gradient boosting → tune → only then consider deep learning.

**Read more:** [Classical ML Overview](../classical_ml/README.md) · [Ensemble Methods](../classical_ml/intro_ensemble_methods.md) · [Clustering](../classical_ml/intro_clustering.md) · [Dimensionality Reduction](../classical_ml/intro_dimensionality_reduction.md)

---

## Bias-Variance Diagnostics

`Expected test error = bias² + variance + irreducible noise`

| Symptom | Diagnosis | Fixes |
|---|---|---|
| Train error high, val error ≈ train error | High bias (underfitting) | Bigger model, more/better features, less regularisation, train longer, boosting |
| Train error low, val error much higher | High variance (overfitting) | More data, augmentation, regularisation (L1/L2, dropout), simpler model, early stopping, bagging |
| Both high, big gap | Both | Fix bias first, then variance |
| Val good, test/prod bad | Leakage or distribution shift | Audit splits and features; check drift, training-serving skew |
| Val error noisy across folds | Small or heterogeneous val set | More folds, repeated CV, stratify, report CI |

**Learning curves (error vs training-set size)**
- Curves converge at a high error → high bias; more data will **not** help.
- Large persistent gap that shrinks with more data → high variance; more data **will** help.

**Knobs and their direction**

| Knob | ↑ increases |
|---|---|
| Tree depth, `num_leaves`, polynomial degree, hidden units | Variance |
| `k` in k-NN | Bias |
| Regularisation `λ` (or ↓ `C` in SVM/logreg) | Bias |
| Number of trees in random forest | Neither much (variance ↓, plateaus) |
| Boosting rounds (without early stopping) | Variance |

- Double descent: very over-parameterised nets can see test error fall again past the interpolation threshold.

**Read more:** [Model Evaluation](../classical_ml/intro_model_evaluation.md) · [Neural Network Training](../deep_learning/intro_neural_network_training.md) · [Ensemble Methods](../classical_ml/intro_ensemble_methods.md)

---

## Neural Network Essentials

| Activation | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | `1/(1 + e^{-x})` | `(0, 1)` | Output for binary / multi-label; saturates, gradient ≤ 0.25 |
| Tanh | `(eˣ - e^{-x})/(eˣ + e^{-x})` | `(-1, 1)` | Zero-centred; RNN gates/state; still saturates |
| ReLU | `max(0, x)` | `[0, ∞)` | Default for CNNs/MLPs; "dying ReLU" |
| Leaky ReLU | `max(αx, x)`, `α ≈ 0.01` | `(-∞, ∞)` | Fixes dying ReLU |
| GELU | `x·Φ(x)` | `≈ (-0.17, ∞)` | BERT, GPT |
| SiLU / Swish | `x·σ(x)` | `≈ (-0.28, ∞)` | SwiGLU FFN in LLaMA-family models |
| Softmax | `e^{z_i}/Σ_j e^{z_j}` | simplex | Multiclass output, attention weights |

| Init | Weight variance | Pair with |
|---|---|---|
| Xavier / Glorot | `Var(W) = 2/(fan_in + fan_out)` (uniform bound `√(6/(fan_in + fan_out))`) | tanh, sigmoid |
| He / Kaiming | `Var(W) = 2/fan_in` | ReLU family |
| LeCun | `Var(W) = 1/fan_in` | SELU, linear |

- Goal of all three: keep activation/gradient variance roughly constant across layers. Never initialise all weights equal (symmetry never breaks).

| Normalization | Normalizes over | Where |
|---|---|---|
| BatchNorm | Batch (+ spatial) dims, per channel | CNNs; running stats at eval; breaks with tiny batches |
| LayerNorm | Features of each sample | Transformers, RNNs; same at train and eval |
| RMSNorm | Features, divides by RMS only (no mean subtraction) | LLaMA-style LLMs; cheaper than LayerNorm |
| GroupNorm | Groups of channels per sample | Detection/segmentation with small batches |
| InstanceNorm | Each channel of each sample | Style transfer |

- All compute `y = γ·(x - μ)/√(σ² + ε) + β` (RMSNorm: `y = γ·x/RMS(x)`); they differ only in which axes give `μ`, `σ`.
- **Dropout:** at train, zero each unit with prob `p` and scale survivors by `1/(1-p)` (inverted dropout); at eval, identity. Call `model.eval()`.

**Parameter counts**

| Layer | Params | Notes |
|---|---|---|
| Linear `d_in → d_out` | `d_in·d_out + d_out` | Drop `+ d_out` if no bias |
| Conv2d | `(K_h·K_w·C_in + 1)·C_out` | `groups = 1`; FLOPs ≈ `2·K_h·K_w·C_in·C_out·H_out·W_out` |
| Conv output size | `floor((H + 2P - K)/S) + 1` | `P` padding, `S` stride |
| Embedding | `V·d` | Often tied with output layer |
| Multi-head attention | `4·d² + 4d` | Q, K, V, O projections; independent of head count |
| Transformer block (FFN 4×) | `≈ 12·d²` | `4d²` attention + `8d²` FFN, ignoring biases/norms |
| Decoder-only LLM | `N ≈ 12·L·d² + V·d` | `L` layers; e.g. `L=32, d=4096` → `≈ 6.4B` + embeddings |
| LSTM layer | `4·(h·(h + x) + h)` | Four gates; PyTorch keeps two bias vectors: `4·(h·(h + x) + 2h)` |

**Attention**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```
- `√d_k` keeps dot-product variance ≈ 1 so softmax doesn't saturate.
- Cost: `O(n²·d)` time, `O(n²)` memory for the score matrix (FlashAttention avoids materialising it, same FLOPs).
- Causal mask sets future positions to `-∞` before softmax. Multi-head: `h` heads of size `d/h`, concatenated then projected.

**Compute rules of thumb**
- Training FLOPs `≈ 6·N·D` (`N` params, `D` training tokens): `2ND` forward + `4ND` backward.
- Inference FLOPs `≈ 2·N` per generated token (ignoring attention over the context).
- Chinchilla compute-optimal: `D ≈ 20·N` tokens (approximate).

**Read more:** [Neural Network Training](../deep_learning/intro_neural_network_training.md) · [Transformers](../deep_learning/intro_transformers.md) · [Computer Vision](../deep_learning/intro_computer_vision.md) · [Sequence Models](../deep_learning/intro_sequence_models.md)

---

## LLM Numbers

**Memory per parameter**

| dtype | Bytes/param | 7B model weights |
|---|---|---|
| fp32 | 4 | 28 GB |
| fp16 / bf16 | 2 | 14 GB |
| fp8 / int8 | 1 | 7 GB |
| int4 (e.g. GPTQ, AWQ, NF4) | 0.5 | 3.5 GB (+ small overhead for scales) |

- Full fine-tuning with Adam, mixed precision: `≈ 16 bytes/param` (2 weights + 2 grads + 4 fp32 master + 4 `m` + 4 `v`), before activations. 7B → `~112 GB`.
- LoRA adds `r·(d_in + d_out)` params per adapted matrix; only those get gradients and optimizer state.

**KV cache**
```
KV bytes = 2 (K and V) × n_layers × n_kv_heads × d_head × seq_len × batch × bytes_per_elem
```
- Llama-2-7B (32 layers, 32 KV heads, `d_head = 128`, fp16): `2·32·32·128·2 = 512 KB/token` → `~2 GB` for one 4k-token sequence.
- GQA/MQA shrink `n_kv_heads` (Llama-3-8B: 8 KV heads → `128 KB/token`). KV cache, not weights, often limits batch size.

**Rules of thumb (approximate)**
- English: `1 token ≈ 0.75 words ≈ 4 characters`; 1,000 tokens ≈ 750 words. Code and non-English text use more tokens per word.
- Prefill is compute-bound; decode is memory-bandwidth-bound.
- Batch-1 decode ceiling: `tokens/s ≈ memory bandwidth / model bytes` (e.g. 14 GB model on ~2 TB/s HBM → ~140 tok/s upper bound).
- Cost/latency levers: quantisation, KV-cache reuse (prefix caching), continuous batching, speculative decoding, shorter prompts.

**Read more:** [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md) · [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md) · [Model Compression](../deep_learning/intro_model_compression.md) · [Fine-Tuning](../deep_learning/intro_fine_tuning.md)

---

## System Design Numbers

All values are **approximate orders of magnitude**; hardware varies. Use them for back-of-envelope estimates, not SLAs.

| Operation | Approx. latency |
|---|---|
| L1 cache reference | ~1 ns |
| Branch mispredict | ~5 ns |
| Main memory reference | ~100 ns |
| Compress 1 KB (fast codec) | ~2-3 µs |
| Read 1 MB sequentially from RAM | ~10-100 µs |
| SSD (NVMe) random read | ~10-100 µs |
| Round trip within a datacenter | ~0.5 ms |
| Read 1 MB sequentially from SSD | ~0.1-1 ms |
| Cache / KV-store lookup over network (Redis) | ~0.5-1 ms |
| HDD seek | ~10 ms |
| Read 1 MB sequentially from HDD | ~5-20 ms |
| Cross-continent round trip | ~100-150 ms |

| Useful quantity | Approx. value |
|---|---|
| Seconds per day | `86,400 ≈ 10⁵` |
| Seconds per year | `≈ 3.15 × 10⁷` |
| QPS from daily volume | `requests/day ÷ 86,400`; plan peak at 2-3× average |
| 1M float32 vectors × 768 dims | `≈ 3 GB` (`10⁶ · 768 · 4 bytes`) |
| Interactive latency budget | Users notice > ~100 ms; recsys/ads end-to-end often 50-200 ms |
| Online feature-store read (p99) | Single-digit to ~10 ms |
| GBDT inference, one row on CPU | ~µs to ~1 ms (depends on trees/depth) |
| GPU HBM bandwidth (A100 / H100) | ~2 / ~3.35 TB/s |
| Dense bf16 throughput (A100 / H100 SXM) | ~312 / ~990 TFLOPS |
| PCIe 4.0 x16 | ~32 GB/s per direction |

**Read more:** [ML System Design Patterns](../system_design/ml_system_design_patterns.md) · [Model Serving](../mlops/intro_model_serving.md) · [Backend System Design Guide](../system_design/backend_system_design_interview_guide.md)

---

## Top 20 Traps

1. **Target leakage**: a feature that is only known after the label (e.g. "refund issued" to predict fraud). Ask "would I have this at prediction time?"
2. **Fitting preprocessing on all data**: scalers, imputers, target encoders or feature selection fit before the split. Put them inside a `Pipeline` fit per fold.
3. **Random split on time series**: trains on the future. Use time-based splits / forward-chaining CV with a gap.
4. **Entity leakage**: same user, patient or session in train and test. Use `GroupKFold`.
5. **Accuracy on imbalanced data**: 99% accuracy at 1% prevalence is a constant predictor. Use PR-AUC, recall@precision, F1.
6. **ROC-AUC on rare events**: looks great while precision is terrible. Report PR-AUC too.
7. **Oversampling (SMOTE) before splitting**: synthetic copies of test points leak into training. Resample only the training fold.
8. **Tuning on the test set**: every peek makes it a validation set. Hold out a final test set touched once.
9. **Default 0.5 threshold**: choose the threshold from costs or a precision/recall target on validation data.
10. **Uncalibrated probabilities**: after class weights, undersampling or boosting, scores are not probabilities. Recalibrate (Platt, isotonic) or correct for the sampling rate.
11. **Training-serving skew**: features computed differently offline vs online. Share feature code, log served features.
12. **Duplicates across splits**: near-duplicate images/texts, or benchmark contamination in LLM pretraining data. Dedupe before splitting.
13. **Peeking at A/B tests**: stopping when `p < 0.05` inflates false positives. Fix sample size up front or use sequential tests.
14. **Multiple comparisons**: 20 metrics at `α = 0.05` gives ~1 false win by chance. Pre-register a primary metric; correct (Holm, BH).
15. **Sample ratio mismatch**: a 50/50 test landing 52/48 means broken assignment/logging. Check with a chi-square test before reading results.
16. **Correlation ≠ causation**: confounding and Simpson's paradox. Feature importance is not a causal effect.
17. **Missing feature scaling**: k-NN, SVM, k-means, PCA and regularised linear models all depend on scale; trees don't.
18. **Forgetting `model.eval()`**: dropout stays on and BatchNorm uses batch stats at inference.
19. **Bad metric for the target**: MAPE when `y` can be ~0, RMSE on heavy-tailed targets, R² compared across datasets.
20. **Feedback loops and selection bias**: training only on data your model chose to show or approve (clicks on shown items, approved loans). Log propensities, keep exploration traffic.

**Read more:** [Feature Engineering](../classical_ml/intro_feature_engineering.md) · [Time Series](../classical_ml/intro_time_series.md) · [Model Evaluation](../classical_ml/intro_model_evaluation.md) · [A/B Testing](../mlops/intro_ab_testing.md) · [Model Monitoring](../mlops/intro_model_monitoring.md) · [Glossary](./glossary.md)

---

## Related Topics

- [Glossary](./glossary.md): every term in one or two sentences
- [2026 Interview Questions](./interview_questions_2026.md)
- [2026 Interview Roadmap](./2026-interview-roadmap.md)
- [Study Pattern](./study-pattern.md)
- [Repository Home](../README.md)
