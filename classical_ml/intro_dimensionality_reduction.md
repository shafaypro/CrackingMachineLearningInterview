# Dimensionality Reduction (Deep Dive)

Dimensionality reduction reduces the number of features while preserving meaningful structure. It combats the curse of dimensionality, speeds up training, enables visualization, and can improve model generalization by removing noise.

---

## Table of Contents
1. [Why Dimensionality Reduction?](#why-dimensionality-reduction)
2. [PCA – Principal Component Analysis](#pca--principal-component-analysis)
3. [LDA – Linear Discriminant Analysis](#lda--linear-discriminant-analysis)
4. [t-SNE](#t-sne)
5. [UMAP](#umap)
6. [Autoencoders](#autoencoders)
7. [When to Use Each Method](#when-to-use-each-method)
8. [Interview Q&A](#interview-qa)
9. [Common Pitfalls](#common-pitfalls)
10. [Related Topics](#related-topics)

---

## Why Dimensionality Reduction?

| Problem | How DR Helps |
|---------|-------------|
| Curse of dimensionality | High-d spaces are sparse; distances lose meaning |
| Slow training | Fewer features = fewer parameters = faster models |
| Overfitting | Remove noise dimensions |
| Visualization | Project to 2D/3D for exploration |
| Storage/memory | Smaller feature vectors |

---

## PCA – Principal Component Analysis

PCA is a **linear, unsupervised** technique that finds orthogonal directions of maximum variance.

### Step-by-Step

```
1. Center the data: subtract mean from each feature
2. Compute covariance matrix: C = (1/n) * X^T * X
3. Compute eigenvectors and eigenvalues of C
4. Sort eigenvectors by descending eigenvalue
5. Project data onto top-k eigenvectors (principal components)
```

### Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Always standardize before PCA
X_scaled = StandardScaler().fit_transform(X)

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("Explained variance per component:", explained_var[:10])
print("Cumulative explained variance:", cumulative_var[:10])

# Choose k components explaining 95% of variance
k = np.argmax(cumulative_var >= 0.95) + 1
print(f"Components for 95% variance: {k}")

# Project to k dimensions
pca_k = PCA(n_components=k)
X_reduced = pca_k.fit_transform(X_scaled)

# For visualization (2D)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
```

### Scree Plot

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_var[:15]) + 1), explained_var[:15])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_var[:15]) + 1), cumulative_var[:15], 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
```

### PCA Biplot: What Do Components Mean?

```python
# Loadings: how much each original feature contributes to each PC
loadings = pca_k.components_  # shape: (k, n_features)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# Top contributing features for PC1
pc1_loadings = pd.Series(loadings[0], index=feature_names).abs().sort_values(ascending=False)
print("Top features for PC1:", pc1_loadings.head(5))
```

### Incremental PCA (for large datasets)

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in np.array_split(X_large, 100):
    ipca.partial_fit(batch)
X_reduced = ipca.transform(X_large)
```

---

## LDA – Linear Discriminant Analysis

LDA is a **linear, supervised** technique that finds directions maximizing class separability. Unlike PCA (maximizes variance), LDA maximizes the ratio of between-class to within-class scatter.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA requires class labels y
lda = LinearDiscriminantAnalysis(n_components=2)  # max components = n_classes - 1
X_lda = lda.fit_transform(X_scaled, y)

# Explained variance ratio for each discriminant
print("Explained variance ratio:", lda.explained_variance_ratio_)
```

### PCA vs LDA

| | PCA | LDA |
|-|-----|-----|
| **Supervised?** | No | Yes (needs labels) |
| **Objective** | Maximize variance | Maximize class separability |
| **Max components** | min(n, p) | n_classes - 1 |
| **Use case** | Feature compression, visualization | Classification preprocessing |
| **Assumption** | None about classes | Gaussian classes, equal covariance |

---

## t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a **non-linear, unsupervised** technique primarily for **visualization**. It preserves local structure (nearby points stay nearby) but can distort global structure.

### How It Works

```
1. Compute pairwise similarities in high-d space using Gaussian kernel: p(j|i) ∝ exp(-||xi - xj||² / 2σ²)
2. Define pairwise similarities in low-d space using t-distribution: q(ij) ∝ (1 + ||yi - yj||²)^-1
3. Minimize KL divergence between P and Q distributions via gradient descent
```

The t-distribution in low-d space has heavier tails than Gaussian — this allows moderate distances to be well-separated and avoids the "crowding problem."

```python
from sklearn.manifold import TSNE

# t-SNE for visualization only (not for preprocessing)
tsne = TSNE(
    n_components=2,
    perplexity=30,        # Effective number of neighbors (typical: 5-50)
    learning_rate=200,    # 'auto' in newer sklearn
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.title('t-SNE Visualization')
plt.colorbar(label='Class')
```

### t-SNE Hyperparameter Guide

| Parameter | Effect | Typical Range |
|-----------|--------|--------------|
| **perplexity** | Controls local vs global structure tradeoff | 5–50 |
| **learning_rate** | Step size for optimization | 10–1000 (or 'auto') |
| **n_iter** | Number of optimization steps | 500–5000 |
| **early_exaggeration** | Spread between clusters | 4–12 |

**Warning**: t-SNE is **non-deterministic** and **non-parametric** — you cannot transform new data points without re-running the full algorithm. Use for exploration only.

---

## UMAP

UMAP (Uniform Manifold Approximation and Projection) is a **non-linear** technique that is faster than t-SNE and better preserves global structure.

### UMAP vs t-SNE

| | t-SNE | UMAP |
|-|-------|------|
| **Speed** | Slow (O(n²)) | Faster (approximate O(n log n)) |
| **Global structure** | Poor | Better |
| **New data transform** | No | Yes (`.transform()`) |
| **Use case** | Visualization only | Visualization + preprocessing |
| **Deterministic** | No | Yes (with `random_state`) |

```python
import umap

reducer = umap.UMAP(
    n_components=2,      # Can also use 10, 50 for preprocessing
    n_neighbors=15,      # Controls local vs global structure (like perplexity)
    min_dist=0.1,        # Minimum distance between embedded points
    metric='euclidean',  # Or 'cosine' for text embeddings
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

# UMAP supports out-of-sample transformation
X_new_umap = reducer.transform(X_new_scaled)

# UMAP for preprocessing (50 dimensions before clustering)
reducer_50d = umap.UMAP(n_components=50, random_state=42)
X_umap_50d = reducer_50d.fit_transform(X_scaled)
```

---

## Autoencoders

Autoencoders are **non-linear** neural network-based dimensionality reduction.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Encoder: compress to bottleneck
def build_autoencoder(input_dim, encoding_dim=32):
    # Encoder
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(input_dim=X.shape[1], encoding_dim=16)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, validation_split=0.1, verbose=0)

X_encoded = encoder.predict(X_scaled)  # Compressed representations
```

---

## When to Use Each Method

| Method | Use When | Avoid When |
|--------|----------|-----------|
| **PCA** | Linear relationships, need interpretability, preprocessing for models | Non-linear manifold structure |
| **LDA** | Classification task, labeled data, need class-aware projection | No labels; non-Gaussian classes |
| **t-SNE** | Visualization only, exploring cluster structure | New data transforms needed; preprocessing |
| **UMAP** | Visualization + preprocessing, out-of-sample data, speed matters | Need exact reproducibility across libraries |
| **Autoencoder** | Non-linear structure, images/sequences, denoising | Small datasets (risk of overfitting) |

---

## Interview Q&A

**Q1: What is the difference between PCA and t-SNE?**
PCA is linear, deterministic, unsupervised, and maximizes variance. It's interpretable (principal components have loadings) and supports out-of-sample transformation. t-SNE is non-linear, stochastic, and optimized for 2D/3D visualization — it preserves local neighborhood structure but distorts global distances. PCA is used for preprocessing; t-SNE is used only for visualization.

**Q2: What does the perplexity parameter in t-SNE control?**
Perplexity roughly corresponds to the number of effective nearest neighbors considered for each point. Low perplexity (5-10) focuses on very local structure; high perplexity (30-50) considers broader neighborhoods. The optimal value depends on dataset size and density. Always try multiple values and compare.

**Q3: Why must you standardize before PCA?**
PCA finds directions of maximum variance. If one feature has range 0–1000 and another 0–1, the first will dominate the first principal components purely due to scale, not because it's more informative. StandardScaler ensures all features contribute equally.

**Q4: How many principal components should you keep?**
Two common rules:
- Keep components explaining ≥95% of cumulative variance (for compression)
- Use the scree plot: keep components before the "elbow" (where adding more gives diminishing returns)
For downstream ML tasks, treat k as a hyperparameter and tune via cross-validation.

**Q5: What is the difference between PCA and LDA?**
PCA is unsupervised — finds directions of maximum variance regardless of class labels. LDA is supervised — finds directions that best separate classes (maximizes between-class / within-class scatter ratio). LDA is better for classification preprocessing; PCA is better for general compression or when labels are unavailable.

**Q6: Can you use t-SNE for feature extraction (preprocessing for a classifier)?**
No. t-SNE is designed for visualization and has two major limitations: (1) it's non-parametric — you can't transform new test points without re-running from scratch; (2) it's stochastic — different runs give different embeddings. Use PCA or UMAP for preprocessing. Use t-SNE only for visualization.

**Q7: When would you use UMAP over t-SNE?**
UMAP when: you need to transform new data points (`.transform()`), you want better global structure preservation, you have large datasets (UMAP is faster), or you're using it as preprocessing for clustering/classification. t-SNE when: you specifically want local neighborhood visualization and global distortion is acceptable.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Not standardizing before PCA | Scale-dominated components | Always `StandardScaler` before PCA |
| Interpreting t-SNE global distances | Cluster distances are meaningless | Only trust local neighborhoods |
| Using t-SNE for preprocessing | Non-parametric; can't transform new data | Use PCA or UMAP instead |
| Running t-SNE only once | Stochastic; may give different structure | Run multiple times; compare |
| Choosing k=2 for PCA always | Discards most variance | Use explained variance to choose k |
| Not using LDA when you have labels | Leaving signal on the table | LDA often outperforms PCA for classification |
| PCA before distance-based methods without re-scaling | PCA output may still need scaling | Scale after PCA too |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Clustering](./intro_clustering.md) | PCA/UMAP often precedes clustering to reduce noise |
| [Feature Engineering](./intro_feature_engineering.md) | DR is a form of feature transformation |
| [Classic Question Bank](../README.md#classic-question-bank) | PCA, t-SNE, UMAP covered in Q&A 59-62 |
| [Study Pattern](../docs/study-pattern.md) | DR is an Intermediate (🟡) topic |
