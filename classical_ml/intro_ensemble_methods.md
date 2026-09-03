# Ensemble Methods and Gradient Boosting

Ensemble methods combine many weak learners into one strong predictor. They dominate tabular-data interviews and tabular-data production systems: on structured data, a well-tuned gradient boosted tree still beats a neural network most of the time, and interviewers know it. Expect questions on bagging vs boosting, why Random Forest reduces variance, and how XGBoost differs from LightGBM.

---

## Table of Contents
1. [Why Ensembles Work](#why-ensembles-work)
2. [Bagging and Random Forest](#bagging-and-random-forest)
3. [Boosting](#boosting)
4. [Gradient Boosting from First Principles](#gradient-boosting-from-first-principles)
5. [XGBoost vs LightGBM vs CatBoost](#xgboost-vs-lightgbm-vs-catboost)
6. [Stacking and Blending](#stacking-and-blending)
7. [Hyperparameter Tuning Guide](#hyperparameter-tuning-guide)
8. [Feature Importance and Interpretation](#feature-importance-and-interpretation)
9. [Choosing the Right Ensemble](#choosing-the-right-ensemble)
10. [Interview Q&A](#interview-qa)
11. [Common Pitfalls](#common-pitfalls)
12. [Related Topics](#related-topics)

---

## Why Ensembles Work

Expected prediction error decomposes into three parts:

```
Error = Bias² + Variance + Irreducible noise
```

Ensembles attack different terms:

| Family | Base learner | Attacks | How |
|---|---|---|---|
| **Bagging** | Low-bias, high-variance (deep trees) | Variance | Average many decorrelated models |
| **Boosting** | High-bias, low-variance (shallow trees) | Bias | Fit each new model to the previous errors |
| **Stacking** | Any mix | Both | Learn how to weight heterogeneous models |

The bagging intuition: averaging `n` i.i.d. estimators each with variance `σ²` gives variance `σ²/n`. Real models are not independent, so with pairwise correlation `ρ` the variance becomes:

```
Var(average) = ρσ² + (1 - ρ) σ² / n
```

That `ρσ²` floor is why **decorrelating the trees matters more than adding trees**. Random Forest's feature subsampling exists precisely to push `ρ` down.

---

## Bagging and Random Forest

**Bagging** (bootstrap aggregating): train `n` models on `n` bootstrap samples (sampled with replacement, same size as the original), then average (regression) or majority-vote (classification).

**Random Forest** = bagging with decision trees **plus** a second source of randomness: at each split, only a random subset of features (`max_features`) is considered.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=5000, n_features=30, n_informative=10, random_state=42)

rf = RandomForestClassifier(
    n_estimators=500,        # more is never worse for accuracy, only for latency
    max_features='sqrt',     # decorrelates trees; 'sqrt' for classification, ~1/3 for regression
    min_samples_leaf=1,      # raise to 5-20 on noisy data to regularize
    n_jobs=-1,
    oob_score=True,          # free validation estimate from out-of-bag samples
    random_state=42,
)
rf.fit(X, y)
print(f"OOB accuracy: {rf.oob_score_:.3f}")
print(f"CV accuracy:  {cross_val_score(rf, X, y, cv=5).mean():.3f}")
```

### Out-of-bag (OOB) evaluation

Each bootstrap sample leaves out roughly `1/e ≈ 36.8%` of rows. Those out-of-bag rows act as a built-in validation set, so you get an unbiased-ish generalization estimate without a separate holdout. Useful when data is scarce; not a replacement for a proper time-based split on temporal data.

### Extremely Randomized Trees (Extra Trees)

Extra Trees goes further: split thresholds are drawn at random rather than optimized. This increases bias slightly but cuts variance and training time. It often matches Random Forest on noisy data and trains 2–5x faster.

---

## Boosting

Boosting trains models **sequentially**, where each model focuses on what the ensemble got wrong so far.

### AdaBoost

Reweights misclassified samples upward each round and weights each learner by its accuracy.

```
w_i ← w_i · exp(α_t · 1[y_i ≠ h_t(x_i)])      where α_t = ½ ln((1 - err_t) / err_t)
```

AdaBoost minimizes exponential loss, which makes it sensitive to label noise and outliers — a permanently mislabeled point gets ever-larger weight.

### Gradient Boosting

Generalizes boosting to any differentiable loss. Instead of reweighting samples, each new tree fits the **negative gradient** of the loss with respect to the current predictions (the "pseudo-residuals").

---

## Gradient Boosting from First Principles

For squared error, the negative gradient *is* the residual, which makes the algorithm easy to see:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class TinyGradientBoosting:
    """Minimal gradient boosting for squared error — the interview whiteboard version."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Step 0: start from the best constant prediction (mean minimizes squared error)
        self.init_pred = y.mean()
        pred = np.full(len(y), self.init_pred)

        for _ in range(self.n_estimators):
            residual = y - pred                      # negative gradient of ½(y - p)²
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)                    # fit the errors, not the labels
            pred += self.learning_rate * tree.predict(X)   # shrunk step
            self.trees.append(tree)
        return self

    def predict(self, X):
        pred = np.full(len(X), self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
```

Three ideas carry over to every production implementation:

1. **Shrinkage** (`learning_rate`): small steps generalize better. Lower learning rate needs more trees — they trade off roughly inversely.
2. **Shallow trees**: depth 3–8. Each tree only needs to capture a bit of remaining signal.
3. **Additive, sequential**: cannot be parallelized across trees (only within a tree's split search), unlike bagging.

For non-squared losses (log loss, Huber, ranking objectives), the residual is replaced by the loss gradient and the leaf values by a Newton step using the second derivative — that second-order step is XGBoost's core contribution.

---

## XGBoost vs LightGBM vs CatBoost

| Dimension | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (depth-first balanced) | **Leaf-wise** (splits the highest-gain leaf) | Symmetric / oblivious trees |
| Speed on large data | Fast | **Fastest** (histogram + GOSS + EFB) | Moderate |
| Small data (< ~10k rows) | Good | Overfits easily leaf-wise | **Best**, ordered boosting resists overfitting |
| Categorical features | Must encode yourself | Native (`categorical_feature`) | **Native and best-in-class** (ordered target statistics) |
| Missing values | Learns a default direction per split | Learns a default direction | Handled |
| Key regularizer | `lambda`, `alpha`, `max_depth` | `num_leaves`, `min_data_in_leaf` | `depth`, `l2_leaf_reg` |
| Typical pick | Safe default, huge ecosystem | Wide/large datasets, tight training budget | Many categorical columns, small/medium data |

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=2000,          # set high; early stopping picks the real number
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,              # row sampling per tree — stochastic gradient boosting
    colsample_bytree=0.8,       # column sampling per tree
    reg_lambda=1.0,             # L2 on leaf weights
    min_child_weight=5,         # minimum summed hessian in a leaf — the main anti-overfit knob
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}, best AUC: {model.best_score:.4f}")
```

### Why leaf-wise growth is both LightGBM's strength and its trap

Level-wise growth splits every node at a depth before going deeper — balanced but wasteful. Leaf-wise always splits whichever leaf reduces loss most, so it reaches lower training loss with the same number of leaves. On small datasets it will happily grow a deep, narrow branch that memorizes 40 rows. Control it with `num_leaves` (keep `num_leaves < 2^max_depth`) and `min_data_in_leaf`, not with `n_estimators`.

---

## Stacking and Blending

Stacking trains a **meta-learner** on the out-of-fold predictions of several base models.

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)),
        ('svm', SVC(probability=True)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,             # CRITICAL: base predictions must be out-of-fold
    passthrough=False,
    n_jobs=-1,
)
stack.fit(X_tr, y_tr)
```

The `cv=5` is the whole trick. If you train base models and the meta-learner on the same rows, the base models' in-sample predictions are near-perfect, the meta-learner learns to trust them completely, and the stack collapses at inference time. This is the single most common stacking bug.

Stacking wins Kaggle competitions and rarely survives production review: 3–8x the inference cost and 3–8x the models to monitor, for perhaps 0.3% AUC. Bring it up in interviews as a tradeoff you can articulate, not a default.

---

## Hyperparameter Tuning Guide

Tune in this order — later parameters matter less than earlier ones.

| Order | Parameter | Range | Effect |
|---|---|---|---|
| 1 | `learning_rate` | 0.01–0.1 | Lower = better generalization, more trees needed |
| 2 | `n_estimators` | Use early stopping | Let validation decide, don't grid-search it |
| 3 | `max_depth` / `num_leaves` | 3–10 / 15–255 | Main capacity control |
| 4 | `min_child_weight` / `min_data_in_leaf` | 1–100 | Prevents leaves fit to a handful of rows |
| 5 | `subsample`, `colsample_bytree` | 0.6–1.0 | Adds randomness, reduces variance |
| 6 | `reg_lambda`, `reg_alpha` | 0–10 | L2 / L1 on leaf weights |

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 2000,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
    }
    m = xgb.XGBClassifier(**params, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m.best_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
```

Bayesian search (Optuna, Hyperopt) beats grid search badly here: the space is 6+ dimensional and most dimensions are unimportant, so random/Bayesian sampling covers the important axes far better per trial.

---

## Feature Importance and Interpretation

Tree "importance" has three common definitions, and they disagree:

| Method | What it measures | Bias |
|---|---|---|
| **Gain** (default in XGBoost) | Total loss reduction from splits on the feature | Inflates features used in few high-impact splits |
| **Split count / weight** | How often the feature is used | Inflates high-cardinality continuous features |
| **Permutation importance** | Score drop when the feature is shuffled | Honest but misleading with correlated features |
| **SHAP** | Per-prediction attribution with additive guarantees | Slower; the defensible choice for stakeholders |

```python
from sklearn.inspection import permutation_importance
import shap

# Permutation importance — measured on held-out data, which is the point
perm = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

# SHAP — exact and fast for trees
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

The interview trap: impurity-based importance is computed **on training data** and systematically favors high-cardinality features (a random ID column can rank in your top 5). Always say "permutation or SHAP on a holdout set" when asked how you'd rank features.

---

## Choosing the Right Ensemble

| Situation | Choose | Why |
|---|---|---|
| Tabular data, accuracy matters | Gradient boosting (XGBoost/LightGBM) | Consistently strongest on structured data |
| Many categorical columns | CatBoost | Native ordered target encoding, no leakage |
| Need a fast, no-tuning baseline | Random Forest | Works well near-default, hard to overfit |
| Very noisy labels | Random Forest / Extra Trees | Boosting chases noise |
| Millions of rows, tight training budget | LightGBM | Histogram binning + GOSS |
| Latency budget under ~1 ms | Single shallow tree / logistic regression | Ensembles cost `n_trees` traversals |
| Need calibrated probabilities | RF + isotonic, or boosting with log loss | See [Model Evaluation](./intro_model_evaluation.md) |
| Images, audio, text | Neural networks, not ensembles | Trees cannot exploit spatial/sequential structure |

---

## Interview Q&A

#### What is the difference between bagging and boosting?

Bagging trains models **in parallel** on bootstrap resamples and averages them; it reduces **variance** and needs low-bias, high-variance base learners (deep trees). Boosting trains models **sequentially**, each fitting the errors of the current ensemble; it reduces **bias** and needs high-bias base learners (shallow trees).

The practical consequences: bagging is embarrassingly parallel and nearly impossible to overfit by adding more trees; boosting is sequential and *will* overfit with too many rounds, which is why early stopping is mandatory. Bagging is robust to label noise; boosting amplifies it, because a permanently mislabeled point keeps producing a large residual that later trees keep chasing.

#### Why does Random Forest sample features at each split, not just rows?

Row bootstrapping alone leaves the trees highly correlated: if one feature is strongly predictive, every tree splits on it first and the trees look alike. Since `Var(average) = ρσ² + (1-ρ)σ²/n`, correlated trees leave a variance floor of `ρσ²` that no number of trees removes. Feature subsampling forces different trees to use different features, lowering `ρ` and therefore the floor. It costs a little bias per tree and buys much more variance reduction.

#### Can adding more trees overfit a Random Forest?

Essentially no. Each tree is trained independently, so the average converges as `n → ∞`; more trees reduce the Monte Carlo variance of the average and then plateau. What overfits a Random Forest is **individual tree depth** (`min_samples_leaf=1` on noisy data), not the count. The only cost of more trees is memory and inference latency.

Boosting is the opposite: each additional tree reduces training loss and eventually starts fitting noise, so `n_estimators` is a genuine regularization parameter that must be chosen by early stopping.

#### How does XGBoost differ from vanilla gradient boosting?

Four substantive differences:
1. **Second-order optimization** — it uses both gradient and Hessian for a Newton step, giving better leaf values and a principled split-gain formula.
2. **Regularization in the objective** — an explicit penalty on the number of leaves (`gamma`) and the L2 norm of leaf weights (`lambda`), so pruning falls out of the objective rather than being a heuristic.
3. **Sparsity-aware split finding** — it learns a default direction for missing values instead of requiring imputation.
4. **Systems engineering** — column blocks for cache-efficient split search, approximate quantile sketch for candidate splits, out-of-core training.

#### Your gradient boosting model has training AUC 0.99 and validation AUC 0.72. What do you do?

Severe overfitting. In order of impact:
1. Confirm it isn't **leakage** first — a validation AUC that collapsed only after adding a feature usually means that feature encodes the target or the future. Check the top SHAP features for anything computed after the label event.
2. Lower `learning_rate` and re-fit with early stopping on a proper validation split.
3. Reduce capacity: shallower `max_depth` / fewer `num_leaves`, raise `min_child_weight`.
4. Add randomness: `subsample=0.8`, `colsample_bytree=0.8`.
5. Raise `reg_lambda`.
6. If the gap persists and the dataset is small, switch to Random Forest or CatBoost, which resist overfitting better at this size.

Also verify the split itself is valid: with temporal data, a random split leaks the future into training and produces exactly this pattern.

#### When would you *not* use gradient boosting?

When inputs have structure trees cannot represent — images, audio, raw text, long sequences — a neural network exploiting that structure wins decisively. When the latency budget is sub-millisecond and you cannot afford hundreds of tree traversals. When you need a genuinely interpretable model for regulatory reasons and SHAP is not accepted. When you must extrapolate beyond the training range (trees output constants outside the observed feature range, so a linear model handles trends better). And when the dataset has a few hundred rows, where a regularized linear model is more honest.

#### How do you handle class imbalance with tree ensembles?

Start with the metric, not the model: accuracy is meaningless at 1% positives; use PR-AUC or recall at a fixed precision. Then in rough order of preference:
- `scale_pos_weight = n_negative / n_positive` in XGBoost (or `class_weight='balanced'` in sklearn) — reweights the loss, costs nothing.
- **Threshold tuning** on the validation set — the model's ranking is often fine and only the 0.5 cutoff is wrong. This alone fixes most "the model predicts all zeros" complaints.
- Resampling (SMOTE, undersampling) — sometimes helps, but SMOTE interpolates in feature space and can create implausible synthetic points with categorical or high-dimensional data. Always resample **inside** the CV fold, never before splitting.

Note that reweighting distorts predicted probabilities; if you need calibrated probabilities, recalibrate afterwards.

#### What is stacking, and what is the main way it goes wrong?

Stacking trains a meta-learner on the predictions of several base models, letting it learn which model to trust in which region of feature space. The dominant failure is **training the meta-learner on in-sample base predictions**: base models nearly memorize their training rows, so the meta-features look far more accurate at fit time than at inference time, and the meta-learner over-trusts them. The fix is out-of-fold predictions — train base models on K-1 folds, predict the held-out fold, assemble those out-of-fold predictions as the meta-training set.

#### Why does a lower learning rate usually generalize better?

Each tree takes a smaller step toward the gradient direction, so the ensemble explores a smoother path and no single tree can dominate the prediction. It's the boosting analogue of small steps in SGD: more, smaller corrections average out the noise in each individual fit. The cost is more rounds — roughly, halving the learning rate doubles the trees needed — so the practical rule is to pick the smallest learning rate your training budget tolerates and let early stopping choose the tree count.

#### How do you make an ensemble fast enough for real-time serving?

Measure first: latency is `n_trees × average_depth` memory-bound traversals. Then, in order:
- Reduce `n_estimators` — the accuracy/latency curve is steeply diminishing, and 200 trees is often within 0.2% of 2000.
- Compile the model: Treelite, ONNX Runtime, or `xgboost`'s native inplace prediction give 2–10x over Python-loop scoring.
- Batch requests so the traversal amortizes across rows.
- Cache predictions for repeated entities.
- As a last resort, distill the ensemble into a single shallow tree or a small neural network trained on the ensemble's outputs.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Grid-searching `n_estimators` for boosting | Wastes the search budget on a parameter early stopping solves for free | Set `n_estimators` high, use `early_stopping_rounds` |
| Trusting default impurity importance | Inflates high-cardinality features; a random ID can rank top-5 | Permutation importance or SHAP on held-out data |
| Random CV split on temporal data | Leaks the future into training; validation looks great, production fails | Time-based split or `TimeSeriesSplit` |
| SMOTE before the train/test split | Synthetic points derived from test rows leak across the split | Resample inside the CV fold only |
| `num_leaves` set to `2^max_depth` in LightGBM | Leaf-wise growth then overfits aggressively | Keep `num_leaves` well below `2^max_depth` |
| Stacking on in-sample base predictions | Meta-learner over-trusts base models | Out-of-fold predictions (`cv=` in `StackingClassifier`) |
| Scaling features for trees | Wasted work — trees are invariant to monotonic transforms | Skip scaling; spend the effort on feature construction |
| Reading `predict_proba` as a real probability | Boosting with reweighting produces distorted scores | Calibrate (Platt/isotonic) and check a reliability curve |
| Adding trees to fix underfitting from a tiny `max_depth` | Depth-1 stumps cannot express interactions no matter how many | Raise depth to capture interactions, then re-tune |

---

## Related Topics

- [Model Evaluation and Metrics](./intro_model_evaluation.md)
- [Feature Engineering](./intro_feature_engineering.md)
- [Statistics and Probability](./intro_statistics_probability.md)
- [Dimensionality Reduction](./intro_dimensionality_reduction.md)
- [Model Explainability](../mlops/intro_model_explainability.md)
- [Classical ML Overview](./README.md)
