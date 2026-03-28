# Classical Machine Learning

A comprehensive reference for classical ML algorithms, evaluation metrics, and fundamental concepts.

---

## Table of Contents

1. [Algorithm Comparison Table](#algorithm-comparison-table)
2. [Supervised Learning Algorithms](#supervised-learning-algorithms)
3. [Unsupervised Learning Algorithms](#unsupervised-learning-algorithms)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Feature Engineering Techniques](#feature-engineering-techniques)
6. [Interview Q&A](#interview-qa)
7. [References](#references)

---

## Algorithm Comparison Table

| Algorithm | Type | Pros | Cons | Best Use Case |
|-----------|------|------|------|---------------|
| Linear Regression | Supervised / Regression | Fast, interpretable, no hyperparams | Assumes linearity, sensitive to outliers | Continuous target, few features |
| Logistic Regression | Supervised / Classification | Fast, interpretable, probabilistic output | Assumes linearity, poor with non-linear data | Binary classification, baselines |
| Decision Tree | Supervised / Both | Interpretable, handles non-linearity | Overfits, unstable | Explainable ML, rule extraction |
| Random Forest | Supervised / Both | Robust, handles missing data, feature importance | Slow inference, large memory | Tabular data, general purpose |
| Gradient Boosting (XGBoost/LightGBM/CatBoost) | Supervised / Both | State-of-the-art tabular performance | Hyperparameter tuning needed | Kaggle, tabular competitions |
| SVM | Supervised / Both | Effective in high-dim spaces, kernel trick | Slow on large data, no probability by default | Text classification, small datasets |
| K-Nearest Neighbors | Supervised / Both | Simple, no training, non-parametric | Slow inference, high memory, curse of dimensionality | Small datasets, recommendation |
| Naive Bayes | Supervised / Classification | Very fast, works well with small data | Strong independence assumption | Text classification, spam detection |
| K-Means | Unsupervised / Clustering | Simple, scalable | Requires k, assumes spherical clusters | Customer segmentation, quantization |
| DBSCAN | Unsupervised / Clustering | No k needed, finds arbitrary shapes, handles noise | Sensitive to epsilon parameter | Anomaly detection, spatial data |
| Hierarchical Clustering | Unsupervised / Clustering | No k needed, dendrogram | O(n²) complexity | Small datasets, biology |
| PCA | Unsupervised / Dim reduction | Linear, fast, interpretable | Linear only, loses non-linear structure | Feature reduction, preprocessing |
| t-SNE | Unsupervised / Dim reduction | Preserves local structure, great visualization | Slow, non-parametric, not for new data | Visualization of embeddings |
| UMAP | Unsupervised / Dim reduction | Faster than t-SNE, preserves global + local | Non-deterministic | Visualization, preprocessing |

---

## Supervised Learning Algorithms

### Linear Regression

Models a linear relationship between features and a continuous target.

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
Loss: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
Optimization: Normal equation or Gradient Descent
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"Coefficients: {model.coef_}")
```

### Logistic Regression

Classification using the sigmoid function to predict probabilities.

```
P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
σ(z) = 1 / (1 + e^(-z))
Loss: Binary Cross-Entropy = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
```

### Decision Trees

Recursively splits data using the feature that maximizes information gain or minimizes Gini impurity.

```
Gini Impurity: G = 1 - Σ pᵢ²
Entropy: H = -Σ pᵢ · log₂(pᵢ)
Information Gain = H(parent) - weighted average H(children)
```

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(
    max_depth=5,           # Prevent overfitting
    min_samples_leaf=10,   # Minimum samples per leaf
    criterion="gini",      # or "entropy"
    random_state=42
)
model.fit(X_train, y_train)
print(export_text(model, feature_names=feature_names))
```

### Random Forest

Ensemble of decision trees trained on bootstrap samples with random feature subsets.

```
Output = mode(tree₁(x), tree₂(x), ..., treeN(x))  # Classification
Output = mean(tree₁(x), tree₂(x), ..., treeN(x))  # Regression
```

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",   # Standard for classification
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.head(10))
```

### XGBoost / LightGBM / CatBoost

Gradient boosting frameworks — typically the best for tabular data.

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)

# LightGBM (faster, better for large datasets)
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

# CatBoost (best for categorical features without encoding)
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    cat_features=categorical_feature_indices,  # No encoding needed!
    random_seed=42
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
```

### Support Vector Machine (SVM)

Finds the maximum-margin hyperplane separating classes.

```
Objective: Maximize 2/||w|| (margin)
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i
Kernel trick: K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM requires feature scaling!
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,               # Regularization
        gamma="scale",       # Kernel coefficient
        probability=True     # Enable predict_proba (slower)
    ))
])
pipeline.fit(X_train, y_train)
```

---

## Unsupervised Learning Algorithms

### K-Means Clustering

Partitions data into k clusters by minimizing within-cluster variance.

```
Algorithm:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to cluster means
4. Repeat until convergence

Objective: minimize Σ Σ ||xᵢ - μₖ||² for all clusters k
```

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Find optimal k using elbow method
inertias = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, "bx-")
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow Method")
plt.show()

# Silhouette score to evaluate clustering
km = KMeans(n_clusters=5, random_state=42)
labels = km.fit_predict(X)
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")  # Range: -1 to 1, higher is better
```

### DBSCAN

Density-based clustering — finds clusters of arbitrary shape and identifies noise.

```
Parameters:
- epsilon (ε): neighborhood radius
- min_samples: minimum points to form a dense region

Point types:
- Core point: ≥ min_samples neighbors within ε
- Border point: < min_samples neighbors, but within ε of core point
- Noise point: not core or border
```

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

### PCA

Finds directions of maximum variance in the data.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardize first
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Find optimal components
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum = pca_full.explained_variance_ratio_.cumsum()
n_components_95 = (cumsum < 0.95).sum() + 1
print(f"Components for 95% variance: {n_components_95}")
```

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Accuracy
acc = accuracy_score(y_true, y_pred)

# Precision: TP / (TP + FP) — how many predicted positives are actually positive
precision = precision_score(y_true, y_pred, average="binary")

# Recall (Sensitivity): TP / (TP + FN) — how many actual positives were detected
recall = recall_score(y_true, y_pred, average="binary")

# F1 Score: harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred, average="binary")

# AUC-ROC: Area Under the ROC Curve
auc = roc_auc_score(y_true, y_prob)  # y_prob is probability of positive class

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=class_names))
```

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### Metric Selection Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classification | Accuracy, F1 |
| Imbalanced classification (e.g., fraud) | F1, AUC-ROC, Recall |
| When FP costly (e.g., spam filter) | Precision |
| When FN costly (e.g., cancer detection) | Recall |
| Regression (general) | RMSE, MAE, R² |
| Regression (outlier-sensitive) | RMSE |
| Regression (outlier-robust) | MAE |
| Ranking/recommendations | NDCG, MAP |

---

## Feature Engineering Techniques

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean/median imputation
imputer = SimpleImputer(strategy="mean")  # or "median", "most_frequent"
X_imputed = imputer.fit_transform(X)

# KNN imputation (uses similar samples)
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X)

# Missing indicator
import pandas as pd
df["income_missing"] = df["income"].isna().astype(int)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label encoding (ordinal or tree-based models)
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])

# One-hot encoding (linear models)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = ohe.fit_transform(df[["category"]])

# Target encoding (powerful but risks leakage — use cross-val)
from category_encoders import TargetEncoder
te = TargetEncoder()
X_te = te.fit_transform(X_train["category"], y_train)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance — required for SVM, logistic regression, KNN
ss = StandardScaler()
X_std = ss.fit_transform(X_train)

# MinMaxScaler: scales to [0,1] — required for neural networks sometimes
mm = MinMaxScaler()
X_mm = mm.fit_transform(X_train)

# RobustScaler: uses median and IQR — robust to outliers
rs = RobustScaler()
X_rs = rs.fit_transform(X_train)
```

---

## Interview Q&A

**Q1: What is the bias-variance tradeoff?** 🟢 Beginner

Bias is the error from wrong assumptions in the model (underfitting). Variance is the error from sensitivity to small fluctuations in training data (overfitting). High bias models (linear regression on non-linear data) have consistent but systematically wrong predictions. High variance models (deep decision trees) fit training data perfectly but generalize poorly. The goal is to find the sweet spot using cross-validation, regularization, and ensemble methods.

---

**Q2: When would you use Random Forest vs XGBoost?** 🟡 Intermediate

**Random Forest:** Better default performance out-of-the-box, less sensitive to hyperparameters, parallelizable, good for noisy datasets. Use when you need a strong baseline quickly.

**XGBoost/LightGBM/CatBoost:** Higher peak performance, handles class imbalance better, built-in regularization, early stopping. Use when you need maximum performance and are willing to tune hyperparameters. LightGBM is faster than XGBoost on large datasets due to histogram-based splitting.

---

**Q3: Explain the difference between bagging and boosting.** 🟡 Intermediate

**Bagging (Bootstrap Aggregating):** Trains multiple independent models on random bootstrap samples of training data, then averages/votes their predictions. Reduces variance. Example: Random Forest. Models are trained in parallel.

**Boosting:** Trains models sequentially, where each model focuses on the errors of the previous one. Reduces bias and variance. Example: XGBoost, AdaBoost, Gradient Boosting. Models are trained sequentially and cannot be parallelized across rounds.

---

**Q4: How does gradient boosting work?** 🔴 Advanced

Gradient boosting fits an ensemble of weak learners (usually shallow decision trees) sequentially. At each step, it fits a new tree to the **negative gradient of the loss function** with respect to the current ensemble's predictions. For MSE loss, this gradient equals the residuals (actual − predicted). Each new tree corrects the errors of all previous trees. A learning rate scales each tree's contribution, preventing overfitting.

```
F₀(x) = argmin Σ L(yᵢ, γ)   (initial prediction)
Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x) (add scaled tree)
where hₘ is fit to -∂L/∂Fₘ₋₁ (negative gradient)
```

---

**Q5: What is the kernel trick in SVM?** 🔴 Advanced

The kernel trick allows SVMs to find non-linear decision boundaries without explicitly computing high-dimensional feature transformations. Instead of mapping x to φ(x) and computing φ(xᵢ)·φ(xⱼ), the SVM only needs K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) (the dot product in the transformed space). This can be computed efficiently even when φ maps to infinite-dimensional space. Common kernels: RBF/Gaussian, polynomial, sigmoid.

---

**Q6: How do you handle class imbalance?** 🟡 Intermediate

1. **Resampling:** Oversample minority class (SMOTE — synthetic minority oversampling) or undersample majority class
2. **Class weights:** Set `class_weight="balanced"` in sklearn to give higher penalty to minority class errors
3. **Threshold adjustment:** Lower the classification threshold to increase recall for minority class
4. **Evaluation:** Use F1-score, AUC-ROC, precision-recall curves instead of accuracy
5. **Collect more data:** Get more minority class examples
6. **Algorithm choice:** Tree-based methods handle imbalance better than logistic regression

---

**Q7: What is regularization and when do you need it?** 🟢 Beginner

Regularization adds a penalty for model complexity to the loss function to prevent overfitting. L1 (Lasso) adds `λ·|w|` — drives some weights to exactly zero, performing feature selection. L2 (Ridge) adds `λ·||w||²` — shrinks weights uniformly, keeps all features. Elastic Net combines both. You need regularization when: training accuracy >> validation accuracy (overfitting), when you have many features relative to samples, or when features are highly correlated.

---

**Q8: Explain PCA — how does it work and what are its limitations?** 🟡 Intermediate

PCA finds orthogonal directions (principal components) of maximum variance in the data. It computes the covariance matrix, finds its eigenvectors (principal components) and eigenvalues (variance captured). Projects data onto the top k eigenvectors.

Limitations: (1) Linear only — cannot capture non-linear structure, (2) assumes the directions of maximum variance are the most informative (not always true for classification), (3) loses interpretability — principal components are linear combinations of all features, (4) sensitive to feature scale — requires standardization first.

---

**Q9: What is cross-validation and why do we use it?** 🟢 Beginner

Cross-validation is a model evaluation technique that trains and tests on different subsets of data to estimate generalization performance. K-fold CV splits data into k folds, trains on k-1, tests on 1, repeating k times. We use it because: single train-test split gives noisy estimates (depends on the specific split), CV uses all data for both training and testing, and it helps with hyperparameter tuning without overfitting to the test set.

---

**Q10: What is the curse of dimensionality?** 🟡 Intermediate

As dimensionality increases, the volume of the feature space grows exponentially. Key effects: (1) distances between points become increasingly similar (all points equidistant), making KNN-type methods fail, (2) exponentially more data needed to maintain sample density, (3) models overfit with many features relative to samples. Solutions: dimensionality reduction (PCA, UMAP), feature selection, regularization.

---

**Q11: What is SMOTE and when would you use it?** 🟡 Intermediate

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic minority class examples by interpolating between existing minority examples and their k-nearest neighbors. Unlike random oversampling (which duplicates), SMOTE creates new synthetic examples along the line segments connecting minority class points. Use it for classification problems with significant class imbalance (e.g., 1:10 or worse) when you want to avoid losing majority class information through undersampling. Apply SMOTE only on the training set, never the test set.

---

**Q12: How does KNN work and what are its trade-offs?** 🟢 Beginner

KNN classifies a new point by finding its k nearest training examples (by distance) and voting on the class. No training phase — just stores training data. Advantages: simple, no assumptions about data distribution, naturally handles multi-class. Disadvantages: O(n) inference (must compare to all training points), fails with high-dimensional data (curse of dimensionality), sensitive to feature scale (requires normalization), high memory.

---

**Q13: What is the difference between Gini impurity and entropy for decision trees?** 🟡 Intermediate

Both measure node impurity (how mixed the classes are). Gini impurity: `1 - Σpᵢ²`, computationally cheaper (no log). Entropy: `-Σpᵢlog₂(pᵢ)`, information-theoretic measure. In practice, they give very similar results. Gini tends to isolate the most frequent class into its own branch; entropy tends to produce more balanced trees. Default in sklearn is Gini. The difference in model quality is negligible — choose based on computational preference.

---

**Q14: What is L1 vs L2 regularization and which would you use for feature selection?** 🟡 Intermediate

L1 (Lasso) penalty: `λ·Σ|wᵢ|` — drives less important feature weights to exactly zero, performing automatic feature selection. L2 (Ridge) penalty: `λ·Σwᵢ²` — shrinks all weights toward zero but rarely to exactly zero. Use L1 when you believe many features are irrelevant (want sparse models), use L2 when you believe all features are somewhat relevant (want stable coefficients). Elastic Net combines both: `λ₁·Σ|wᵢ| + λ₂·Σwᵢ²`.

---

**Q15: What is the difference between supervised and unsupervised learning?** 🟢 Beginner

Supervised learning trains on labeled data (inputs + known outputs) to learn a mapping function. Examples: classification, regression. Unsupervised learning finds patterns in unlabeled data without predefined outputs. Examples: clustering, dimensionality reduction, anomaly detection. Semi-supervised learning uses a small labeled dataset alongside a large unlabeled dataset.

---

**Q16: When would you choose DBSCAN over K-Means?** 🟡 Intermediate

Choose DBSCAN when: (1) clusters have arbitrary shapes (not spherical), (2) you don't know the number of clusters k, (3) you need outlier/noise detection, (4) cluster densities vary significantly. Use K-Means when: (1) clusters are roughly spherical, (2) you know k or can determine it via elbow method, (3) dataset is very large (K-Means scales to millions of points), (4) you need a simple, interpretable solution.

---

**Q17: What is the difference between Type I and Type II errors?** 🟢 Beginner

**Type I error (False Positive):** Predicted positive, actually negative. Example: flagging a legitimate email as spam. **Type II error (False Negative):** Predicted negative, actually positive. Example: missing a fraudulent transaction. The tradeoff is controlled via the decision threshold. For fraud detection, minimize FN (Type II) — missing fraud is more costly. For spam filtering, minimize FP (Type I) — blocking legitimate emails is costly.

---

**Q18: What is feature importance in tree-based models?** 🟡 Intermediate

Feature importance measures how much each feature contributes to reducing impurity across all splits in all trees. Computed as the weighted average reduction in Gini/entropy across all splits on that feature, weighted by the number of samples passing through. Limitations: (1) biased toward high-cardinality features, (2) correlated features split importance between them, (3) not causal. More reliable alternatives: permutation importance, SHAP values.

---

**Q19: Explain the Random Forest algorithm step by step.** 🟡 Intermediate

1. Draw n bootstrap samples (with replacement) from training data
2. For each bootstrap sample, train a decision tree with a twist: at each split, consider only a random subset of features (typically sqrt(features) for classification, features/3 for regression)
3. Grow each tree to maximum depth (no pruning)
4. For prediction: classify by majority vote (classification) or average (regression) across all trees

Key ideas: bootstrap sampling creates diversity, random feature subsets prevent correlated trees, combining uncorrelated trees reduces variance without increasing bias.

---

**Q20: What is the difference between accuracy and F1 score, and when should you prefer F1?** 🟢 Beginner

Accuracy = (TP + TN) / Total — works well for balanced classes. F1 = 2·(Precision·Recall)/(Precision+Recall) — balances precision and recall. Prefer F1 (or precision-recall separately) when: (1) classes are imbalanced (a model predicting all-majority gets high accuracy but F1≈0), (2) the cost of FP and FN differ, (3) the business cares about detection rate (recall) or precision of alerts. Example: 1% fraud rate — a model predicting no-fraud gets 99% accuracy but 0% F1.

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [The Elements of Statistical Learning — Hastie, Tibshirani, Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Pattern Recognition and Machine Learning — Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Hands-On Machine Learning — Aurélien Géron (O'Reilly)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
