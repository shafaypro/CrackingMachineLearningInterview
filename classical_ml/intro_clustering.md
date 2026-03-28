# Clustering Algorithms

Clustering is an unsupervised learning task that groups similar data points together without labeled data. It's used for customer segmentation, anomaly detection, document grouping, image compression, and exploratory data analysis.

---

## Table of Contents
1. [K-Means](#k-means)
2. [K-Means++](#k-means)
3. [DBSCAN](#dbscan)
4. [Hierarchical Clustering](#hierarchical-clustering)
5. [Gaussian Mixture Models](#gaussian-mixture-models)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Choosing the Right Algorithm](#choosing-the-right-algorithm)
8. [Interview Q&A](#interview-qa)
9. [Common Pitfalls](#common-pitfalls)
10. [Related Topics](#related-topics)

---

## K-Means

K-Means partitions n observations into k clusters by minimizing intra-cluster variance (within-cluster sum of squares).

### Algorithm

```
1. Initialize k centroids randomly
2. Assign each point to nearest centroid (Euclidean distance)
3. Recompute centroids as mean of assigned points
4. Repeat 2-3 until convergence (centroids stop moving)
```

### Implementation

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Fit K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Inertia: {inertia:.2f}")
print(f"Cluster sizes: {np.bincount(labels)}")
```

### Choosing k: Elbow Method

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

# Plot: look for the "elbow" — point where adding more k stops reducing inertia significantly
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

### Silhouette Score

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette={score:.3f}")

# Higher silhouette = better separation and cohesion (range: -1 to 1)
best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Best k by silhouette: {best_k}")
```

---

## K-Means++

Standard K-Means initializes centroids randomly, which can lead to poor convergence. K-Means++ uses a smarter initialization:

```
1. Choose first centroid uniformly at random
2. For each subsequent centroid:
   - Compute distance D(x) from each point to nearest chosen centroid
   - Choose next centroid with probability proportional to D(x)²
3. Proceed with standard K-Means
```

This gives O(log k) approximation guarantee. Use `init='k-means++'` in sklearn (default).

---

## DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters as dense regions separated by low-density regions. It can find **arbitrarily shaped clusters** and identifies **outliers as noise**.

### Key Concepts

- **eps (ε)**: neighborhood radius
- **min_samples**: minimum points within ε to form a core point
- **Core point**: has ≥ min_samples within ε
- **Border point**: within ε of a core point, but < min_samples neighbors
- **Noise point**: not within ε of any core point

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# DBSCAN is sensitive to scale — always standardize
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(
    eps=0.5,           # Neighborhood radius
    min_samples=5,     # Minimum points to form a core point
    metric='euclidean'
)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")

# Noise points have label = -1
core_samples = dbscan.core_sample_indices_
```

### Choosing eps with k-distance plot

```python
from sklearn.neighbors import NearestNeighbors

# Fit k-NN with k = min_samples
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
kth_distances = np.sort(distances[:, -1])[::-1]

# Plot: look for "knee" — that's a good eps value
plt.plot(kth_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5th nearest neighbor distance')
plt.title('k-distance plot for eps selection')
```

---

## Hierarchical Clustering

Builds a tree (dendrogram) of clusters without requiring k in advance.

**Agglomerative (bottom-up)**: Start with each point as its own cluster; repeatedly merge the closest pair.

**Divisive (top-down)**: Start with one cluster; repeatedly split.

### Linkage Methods

| Method | Distance Between Clusters | Behavior |
|--------|--------------------------|---------|
| **Single** | Min distance between any two points | Can create elongated "chain" clusters |
| **Complete** | Max distance between any two points | Tends to create compact, similar-size clusters |
| **Average** | Mean distance between all pairs | Balance between single and complete |
| **Ward** | Minimizes total within-cluster variance | Most commonly used; similar to K-Means |

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

# Compute linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.axhline(y=15, color='r', linestyle='--', label='Cut threshold')

# Extract flat clusters by cutting the dendrogram
labels_hier = fcluster(Z, t=4, criterion='maxclust')  # cut to get 4 clusters

# sklearn API
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X)
```

---

## Gaussian Mixture Models

GMM assumes data is generated from a mixture of k Gaussian distributions. Unlike K-Means, it provides **soft cluster assignments** (probability of belonging to each cluster).

**Expectation-Maximization (EM) Algorithm:**
1. **E-step**: Compute responsibility r(i,k) = P(cluster k | x_i) for each point
2. **M-step**: Update Gaussian parameters (mean, covariance, mixing weight) to maximize likelihood
3. Repeat until convergence

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',    # 'full', 'tied', 'diag', 'spherical'
    n_init=5,
    random_state=42
)
gmm.fit(X)

# Hard assignments
labels_gmm = gmm.predict(X)

# Soft assignments (probabilities)
probs = gmm.predict_proba(X)  # shape: (n_samples, n_components)

# Model selection: use BIC or AIC
bic_scores = []
for k in range(1, 10):
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(X)
    bic_scores.append(gm.bic(X))
best_k_gmm = np.argmin(bic_scores) + 1
```

---

## Evaluation Metrics

### When Ground Truth is Available

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Adjusted Rand Index: -1 to 1 (1 = perfect match)
ari = adjusted_rand_score(y_true, labels)

# Normalized Mutual Information: 0 to 1 (1 = perfect)
nmi = normalized_mutual_info_score(y_true, labels)
```

### When Ground Truth is NOT Available (most common)

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Silhouette Score (-1 to 1, higher is better)
# Measures: (inter-cluster distance - intra-cluster distance) / max(both)
sil = silhouette_score(X, labels)

# Davies-Bouldin Index (lower is better)
# Ratio of within-cluster scatter to between-cluster separation
db = davies_bouldin_score(X, labels)

# Calinski-Harabasz Score (higher is better)
# Ratio of between-cluster to within-cluster dispersion
ch = calinski_harabasz_score(X, labels)

print(f"Silhouette: {sil:.3f}, Davies-Bouldin: {db:.3f}, Calinski-Harabasz: {ch:.1f}")
```

### Metric Summary

| Metric | Range | Best Value | Requires Labels |
|--------|-------|-----------|-----------------|
| Inertia (K-Means) | 0 to ∞ | Lower | No |
| Silhouette Score | -1 to 1 | Higher (→1) | No |
| Davies-Bouldin | 0 to ∞ | Lower (→0) | No |
| Calinski-Harabasz | 0 to ∞ | Higher | No |
| Adjusted Rand Index | -1 to 1 | 1 | Yes |
| NMI | 0 to 1 | 1 | Yes |

---

## Choosing the Right Algorithm

| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| Need k clusters, roughly spherical shapes | **K-Means++** | Fast, scalable, simple |
| Arbitrary cluster shapes, outliers present | **DBSCAN** | No need for k, identifies noise |
| Want soft probabilistic assignments | **GMM** | Probabilistic framework |
| Want hierarchical structure, no k needed | **Hierarchical (Ward)** | Dendrogram visualization |
| Very large datasets (millions of points) | **Mini-Batch K-Means** | Faster K-Means approximation |
| High-dimensional sparse data (text) | **K-Means with TF-IDF** | Works well with cosine similarity |

```python
from sklearn.cluster import MiniBatchKMeans

# For large datasets
mbk = MiniBatchKMeans(n_clusters=4, batch_size=1000, random_state=42)
mbk.fit(X_large)
```

---

## Interview Q&A

**Q1: What are the main limitations of K-Means?**
1. Must specify k in advance
2. Assumes spherical, equal-size clusters
3. Sensitive to outliers (centroid pulled toward them)
4. Sensitive to feature scale (always standardize)
5. Non-deterministic (different runs may give different results — use n_init > 1)
6. Poor performance on non-convex cluster shapes

**Q2: How does DBSCAN handle outliers differently from K-Means?**
K-Means assigns every point to a cluster — outliers get forced into the nearest cluster, distorting centroids. DBSCAN explicitly labels low-density points as noise (label = -1), effectively handling outliers without them affecting cluster shapes.

**Q3: What is the difference between K-Means and GMM?**
K-Means makes hard cluster assignments (each point belongs to exactly one cluster) and assumes spherical clusters of equal size. GMM makes soft assignments (probabilities) and can model elliptical clusters of different sizes and orientations via full covariance matrices. GMM is a generalization of K-Means (K-Means ≈ GMM with spherical covariance and hard assignments).

**Q4: When would you use hierarchical clustering over K-Means?**
When: (1) you don't know k upfront and want to explore the tree structure; (2) you need a hierarchy for downstream tasks (taxonomies, phylogenetics); (3) the dataset is small enough for the O(n² log n) cost; (4) you want the dendrogram for visualization and interpretability.

**Q5: How do you evaluate clustering quality without ground truth labels?**
Use internal metrics: silhouette score (measures cohesion vs separation), Davies-Bouldin index (lower = better-separated clusters), Calinski-Harabasz score. Additionally: visualize with t-SNE/UMAP, inspect cluster statistics (size, centroid distance), run qualitative checks on sampled cluster members.

**Q6: What preprocessing steps are essential for clustering?**
1. **Standardize features** (Z-score) — K-Means and DBSCAN use distance metrics; unscaled features dominate
2. **Handle missing values** — impute before clustering
3. **Dimensionality reduction** (PCA, UMAP) — reduces noise and curse of dimensionality
4. **Remove obvious outliers** before K-Means (but not before DBSCAN which handles them naturally)

**Q7: What is the curse of dimensionality in clustering?**
In high dimensions, the distance between the nearest and farthest points converges — all points appear equidistant. This makes distance-based clustering (K-Means, DBSCAN) ineffective. Solutions: PCA before clustering, use cosine similarity instead of Euclidean, or use subspace clustering methods.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Not standardizing features | Scale dominates distance | `StandardScaler` before clustering |
| Choosing k arbitrarily | Suboptimal clustering | Elbow + silhouette analysis |
| Applying K-Means to non-spherical data | Incorrect cluster shapes | Use DBSCAN or GMM |
| Not running K-Means multiple times | Local optima | Use `n_init=10` or more |
| Using silhouette score as the only metric | Single metric can mislead | Use multiple metrics + visual inspection |
| Clustering on raw text without vectorization | Text is not numeric | Use TF-IDF or embeddings first |
| Ignoring cluster sizes | One cluster with 99% of data | Check `np.bincount(labels)` |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Dimensionality Reduction](./intro_dimensionality_reduction.md) | PCA/UMAP often precedes clustering |
| [Feature Engineering](./intro_feature_engineering.md) | Feature prep is critical for clustering quality |
| [Classic Question Bank](../README.md#classic-question-bank) | KMeans vs KNN covered in main Q&A |
| [Study Pattern](../docs/study-pattern.md) | Clustering is an Intermediate (🟡) topic |
