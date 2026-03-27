# Feature Engineering & Selection

Feature engineering is the process of transforming raw data into meaningful inputs for ML models. It is consistently cited as the highest-impact activity in applied ML — better features beat better algorithms.

---

## Table of Contents
1. [Missing Data](#missing-data)
2. [Encoding Categorical Variables](#encoding-categorical-variables)
3. [Feature Scaling](#feature-scaling)
4. [Feature Crosses & Interactions](#feature-crosses--interactions)
5. [Date & Time Features](#date--time-features)
6. [Text Features](#text-features)
7. [Feature Selection](#feature-selection)
8. [Target Leakage](#target-leakage)
9. [Interview Q&A](#interview-qa)
10. [Common Pitfalls](#common-pitfalls)
11. [Related Topics](#related-topics)

---

## Missing Data

### Types of Missingness

| Type | Description | Example | Implication |
|------|-------------|---------|-------------|
| **MCAR** | Missing Completely At Random | Sensor randomly fails | Safe to drop rows or impute |
| **MAR** | Missing At Random (given other variables) | Income missing more for young people (age known) | Conditional imputation works |
| **MNAR** | Missing Not At Random | High earners skip income question | Most dangerous; biased if dropped/imputed |

### Imputation Strategies

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

X = pd.DataFrame({
    'age': [25, np.nan, 35, 28, np.nan],
    'salary': [50000, 60000, np.nan, 45000, 55000],
    'category': ['A', 'B', np.nan, 'A', 'B']
})

# 1. Simple: mean/median/mode
num_imputer = SimpleImputer(strategy='median')  # Median is robust to outliers
cat_imputer = SimpleImputer(strategy='most_frequent')

X['age'] = num_imputer.fit_transform(X[['age']])
X['category'] = cat_imputer.fit_transform(X[['category']]).ravel()

# 2. KNN Imputation: use k nearest neighbors to fill gaps
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X[['age', 'salary']])

# 3. Iterative (MICE): model each feature as function of others
iterative_imp = IterativeImputer(max_iter=10, random_state=42)
X_mice = iterative_imp.fit_transform(X[['age', 'salary']])

# 4. Add a missingness indicator (preserves information about missingness)
X['age_missing'] = X['age'].isna().astype(int)
X['age'] = X['age'].fillna(X['age'].median())
```

---

## Encoding Categorical Variables

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Pandas get_dummies (quick)
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
encoded = pd.get_dummies(df, drop_first=True)  # Drop first to avoid multicollinearity

# Sklearn OneHotEncoder (pipeline-compatible)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(df[['color']])
```

**Use when**: unordered categories, low cardinality (< ~20 unique values)

### Label / Ordinal Encoding

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Ordinal: preserves order
size_order = [['S', 'M', 'L', 'XL']]
oe = OrdinalEncoder(categories=size_order)
df['size_encoded'] = oe.fit_transform(df[['size']])
```

**Use when**: categories have a natural order (low/medium/high, S/M/L/XL)

### Target Encoding (Mean Encoding)

```python
# Replace category with mean of target for that category
def target_encode(train_df, valid_df, col, target, smoothing=10):
    """Target encode with smoothing to reduce overfitting on rare categories."""
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    # Smoothing: blend category mean toward global mean for rare categories
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    train_df[f'{col}_te'] = train_df[col].map(smooth).fillna(global_mean)
    valid_df[f'{col}_te'] = valid_df[col].map(smooth).fillna(global_mean)
    return train_df, valid_df
```

**Use when**: high-cardinality categories (city, zip code, user_id). **Warning**: prone to target leakage — always compute on training fold only.

### Embedding Encoding (for very high cardinality)

```python
from tensorflow.keras.layers import Embedding, Flatten

# For user_id with 1M+ unique values
user_input = tf.keras.Input(shape=(1,))
user_emb = Embedding(input_dim=1_000_000, output_dim=32)(user_input)
user_emb = Flatten()(user_emb)
```

---

## Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = np.array([[1, 1000], [2, 2000], [3, 1500], [100, 1800]])

# StandardScaler: (x - mean) / std → mean=0, std=1
# Use for: linear models, neural networks, PCA, K-Means, SVMs
ss = StandardScaler()
X_standard = ss.fit_transform(X)

# MinMaxScaler: (x - min) / (max - min) → range [0, 1]
# Use for: image pixels, when you need bounded range
mms = MinMaxScaler()
X_minmax = mms.fit_transform(X)

# RobustScaler: (x - median) / IQR → robust to outliers
# Use for: data with many outliers
rs = RobustScaler()
X_robust = rs.fit_transform(X)
```

| Scaler | Formula | Use When |
|--------|---------|---------|
| StandardScaler | (x-μ)/σ | Normally distributed features |
| MinMaxScaler | (x-min)/(max-min) | Bounded, no large outliers |
| RobustScaler | (x-median)/IQR | Outliers present |
| Log transform | log(x+1) | Right-skewed, positive values |

---

## Feature Crosses & Interactions

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features: adds x1², x2², x1*x2
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Manual interaction features
df['age_x_salary'] = df['age'] * df['salary']
df['is_senior_male'] = ((df['age'] > 50) & (df['gender'] == 'M')).astype(int)

# Binning (quantile buckets)
df['age_bucket'] = pd.qcut(df['age'], q=5, labels=['very_young', 'young', 'mid', 'senior', 'old'])
df['salary_bucket'] = pd.cut(df['salary'], bins=[0, 30000, 60000, 100000, np.inf],
                              labels=['low', 'mid', 'high', 'very_high'])
```

---

## Date & Time Features

```python
def extract_datetime_features(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])

    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_hour'] = df[col].dt.hour
    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    df[f'{col}_quarter'] = df[col].dt.quarter
    df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
    df[f'{col}_is_month_end'] = df[col].dt.is_month_end.astype(int)

    # Cyclical encoding (avoids January-December discontinuity)
    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
    df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[f'{col}_hour'] / 24)
    df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[f'{col}_hour'] / 24)

    # Days since a reference event
    df[f'{col}_days_since_epoch'] = (df[col] - pd.Timestamp('2020-01-01')).dt.days

    return df
```

---

## Text Features

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

corpus = ["the cat sat on the mat", "the dog played in the park", "machine learning is great"]

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),       # Unigrams and bigrams
    min_df=2,                 # Ignore terms in <2 documents
    max_df=0.95,              # Ignore terms in >95% of documents
    stop_words='english',
    sublinear_tf=True         # Replace tf with 1 + log(tf) to dampen high freq
)
X_tfidf = tfidf.fit_transform(corpus)

# In a pipeline
pipeline = Pipeline([('tfidf', tfidf), ('clf', LogisticRegression())])
```

### Pre-trained Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(corpus)  # Shape: (n_docs, 384)

# Use embeddings as features for any downstream model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(embeddings)
```

---

## Feature Selection

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# F-test (ANOVA): linear relationship between feature and target
selector_f = SelectKBest(score_func=f_classif, k=20)
X_filtered = selector_f.fit_transform(X, y)

# Mutual Information: captures non-linear relationships
selector_mi = SelectKBest(score_func=mutual_info_classif, k=20)
X_filtered_mi = selector_mi.fit_transform(X, y)

# Correlation filter
corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_uncorrelated = pd.DataFrame(X).drop(columns=to_drop)
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=20)
X_rfe = rfe.fit_transform(X, y)
selected_features = np.array(feature_names)[rfe.support_]

# RFE with cross-validation (finds optimal number of features)
rfecv = RFECV(estimator=RandomForestClassifier(), cv=5, scoring='accuracy')
rfecv.fit(X, y)
print(f"Optimal features: {rfecv.n_features_}")
```

### Embedded Methods (L1/Tree Importance)

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Lasso (L1) for linear models
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)
selector_l1 = SelectFromModel(lasso, prefit=True)
X_lasso = selector_l1.transform(X_scaled)

# Tree feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# SHAP-based selection (most reliable)
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
mean_shap = np.abs(shap_values[1]).mean(axis=0)
top_features = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)
```

---

## Target Leakage

Target leakage means using information in features that would not be available at prediction time, artificially inflating model performance.

```python
# Example: predicting loan default
# LEAKY: 'days_delinquent' is populated AFTER default occurs
# NOT LEAKY: 'credit_score' is available before the loan decision

# Time-based leakage example
# WRONG: scaling entire dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)         # Leaks test stats into training
X_train, X_test = train_test_split(X_scaled)

# CORRECT: fit scaler on training only
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)              # Only transform, never fit

# Target encoding leakage: compute encoding on training fold only
# (see target encoding section above — use k-fold within training)
```

---

## Interview Q&A

**Q1: What is the difference between StandardScaler and MinMaxScaler?**
StandardScaler transforms to mean=0, std=1 (Z-score). It doesn't bound the output — outliers remain extreme but less dominant. MinMaxScaler transforms to [0,1] — it preserves relative distances but is sensitive to outliers (one extreme value squashes everything else). Use StandardScaler for linear models, neural nets, PCA, SVMs. Use MinMaxScaler when you need bounded values (image pixels, when the algorithm requires [0,1]).

**Q2: When would you use target encoding over one-hot encoding?**
Target encoding (mean encoding) is better for high-cardinality categoricals (city with 10,000 unique values, user_id) where one-hot would create too many sparse columns. However, target encoding requires careful application — always compute on the training fold only (within cross-validation) to prevent target leakage. Add smoothing to handle rare categories.

**Q3: What is target leakage and how do you detect it?**
Target leakage is using information in features that wouldn't be available at prediction time. Signs: suspiciously high model performance, features with very high importance that don't make business sense, test performance much lower than validation performance. Prevention: think causally — "would this feature be available before the prediction is needed?"

**Q4: What's the difference between filter, wrapper, and embedded feature selection methods?**
- **Filter**: rank features using a statistical score (correlation, mutual information, ANOVA F-test) independent of the model. Fast but ignores feature interactions.
- **Wrapper**: use model performance to select features (RFE, forward selection). Captures interactions but computationally expensive.
- **Embedded**: feature selection built into the model training (L1/Lasso shrinks to zero, tree importance, SHAP). Best of both worlds in practice.

**Q5: Why is it important to fit scalers on training data only?**
If you fit a scaler on the entire dataset (including test data), the test data's statistics (mean, std, min, max) contaminate the training scaler — a form of data leakage. The model sees information about the test distribution during training, leading to optimistically biased evaluation. Always fit on training set only, then transform both train and test.

**Q6: What are cyclical features and when do you use them?**
Cyclical features encode circular variables (hour of day, month, day of week) using sine and cosine transforms: `sin(2π*x/period)` and `cos(2π*x/period)`. Without this, a model would see December (month=12) and January (month=1) as far apart (distance=11), but they're adjacent. The sin/cos pair correctly captures circularity.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Fitting scaler on all data | Data leakage from test set | Fit only on train, transform test |
| High-cardinality one-hot encoding | Sparse, memory-heavy, many columns | Use target encoding or embeddings |
| Not adding missingness indicators | Lose information about why data is missing | Add `feature_missing` binary column |
| Encoding categories before train/test split | Target leakage in target encoding | Encode within cross-validation folds |
| Ignoring feature interactions | Model may miss non-linear relationships | Add interaction/polynomial features |
| Using MNAR data without investigation | Biased imputation | Investigate why data is missing; model the missingness |
| Feature selection before cross-validation | Optimistic bias (test features influence selection) | Do feature selection inside CV pipeline |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Clustering](./intro_clustering.md) | Feature prep is critical for clustering quality |
| [Dimensionality Reduction](./intro_dimensionality_reduction.md) | DR is a form of feature transformation |
| [Time Series](./intro_time_series.md) | Lag features, rolling statistics for sequential data |
| [Model Explainability](../mlops/intro_model_explainability.md) | SHAP for feature importance in selection |
| [Study Pattern](../docs/study-pattern.md) | Feature Engineering is an Intermediate (🟡) topic |
