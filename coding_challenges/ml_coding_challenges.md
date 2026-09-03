# ML Coding Challenges — Implement From Scratch

Many ML interviews include a round where you implement an algorithm with NumPy only — no scikit-learn, no PyTorch. The point is not memorization; it is whether you understand the math well enough to translate it into code, and whether you can reason about shapes, numerical stability, and complexity out loud.

Each challenge below gives the problem, a reference solution, and the follow-up questions interviewers actually ask.

---

## Table of Contents
1. [How to Approach These Rounds](#how-to-approach-these-rounds)
2. [Linear Regression with Gradient Descent](#1-linear-regression-with-gradient-descent)
3. [Logistic Regression](#2-logistic-regression)
4. [K-Means](#3-k-means)
5. [K-Nearest Neighbors](#4-k-nearest-neighbors)
6. [Decision Tree Split (Gini and Entropy)](#5-decision-tree-split-gini-and-entropy)
7. [Train/Test Split and K-Fold Cross-Validation](#6-traintest-split-and-k-fold-cross-validation)
8. [Metrics from Scratch](#7-metrics-from-scratch)
9. [PCA](#8-pca)
10. [Softmax and Cross-Entropy (Numerically Stable)](#9-softmax-and-cross-entropy-numerically-stable)
11. [A Two-Layer Neural Network with Backprop](#10-a-two-layer-neural-network-with-backprop)
12. [Scaled Dot-Product Attention](#11-scaled-dot-product-attention)
13. [Cosine Similarity Search](#12-cosine-similarity-search)
14. [Text Chunking with Overlap](#13-text-chunking-with-overlap)
15. [Rate Limiter with Exponential Backoff](#14-rate-limiter-with-exponential-backoff)
16. [Reciprocal Rank Fusion](#15-reciprocal-rank-fusion)
17. [Complexity Reference](#complexity-reference)
18. [Common Pitfalls](#common-pitfalls)
19. [Related Topics](#related-topics)

---

## How to Approach These Rounds

1. **Clarify the contract first.** Input shapes, dtypes, whether a bias term is expected, what happens on empty input. Thirty seconds here prevents a rewrite.
2. **State the math before typing.** "Gradient of MSE with respect to w is `2/n · Xᵀ(Xw - y)`" — this earns credit even if the code has a bug.
3. **Write shapes as comments.** `# (n, d) @ (d,) -> (n,)`. Shape errors are the number one failure in these rounds, and annotating prevents most of them.
4. **Vectorize, but get it correct first.** A working loop beats a broken broadcast. Then say "this loop is `O(n·k)`; here's the vectorized version" and rewrite it.
5. **Mention numerical stability unprompted.** Subtract the max before `exp`, clip before `log`, add epsilon to denominators. Interviewers weight this heavily because it separates people who have shipped models from people who have read about them.
6. **Test on a tiny example out loud.** Two points, two clusters — walk through one iteration.

---

## 1. Linear Regression with Gradient Descent

**Problem**: Implement linear regression trained by batch gradient descent. Support a bias term.

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept

    def _add_bias(self, X):
        return np.c_[np.ones(len(X)), X] if self.fit_intercept else X

    def fit(self, X, y):
        X = self._add_bias(np.asarray(X, dtype=float))   # (n, d+1)
        y = np.asarray(y, dtype=float)                   # (n,)
        n, d = X.shape
        self.w = np.zeros(d)
        self.history = []

        for _ in range(self.n_iters):
            y_pred = X @ self.w                          # (n, d) @ (d,) -> (n,)
            error = y_pred - y                           # (n,)
            grad = (2 / n) * (X.T @ error)               # (d, n) @ (n,) -> (d,)
            self.w -= self.lr * grad
            self.history.append(np.mean(error ** 2))
        return self

    def predict(self, X):
        return self._add_bias(np.asarray(X, dtype=float)) @ self.w
```

**Closed form** (the normal equation) — know both:

```python
def fit_closed_form(X, y, ridge=0.0):
    """w = (XᵀX + λI)⁻¹ Xᵀy — O(d³), exact, no learning rate."""
    X = np.c_[np.ones(len(X)), X]
    d = X.shape[1]
    reg = ridge * np.eye(d)
    reg[0, 0] = 0.0                              # never penalize the intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)   # solve beats inv() — faster and stabler
```

**Follow-ups**
- *When use gradient descent over the closed form?* The normal equation is `O(d³)` and needs `XᵀX` in memory, so it's impractical above a few thousand features; it also requires `XᵀX` to be invertible, which fails with collinear features (ridge fixes that). Gradient descent scales to large `n` and `d`, works out-of-core, and generalizes to non-convex models.
- *Why `np.linalg.solve` instead of `np.linalg.inv`?* Solving the system directly is roughly 2x faster and numerically more stable — explicitly inverting amplifies conditioning problems.
- *What if the loss diverges?* Learning rate too high, or features on wildly different scales. Standardize, or lower the LR.

---

## 2. Logistic Regression

**Problem**: Binary logistic regression with gradient descent and L2 regularization.

```python
class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2

    @staticmethod
    def _sigmoid(z):
        # Stable: avoid exp() overflow on large-magnitude z
        out = np.empty_like(z, dtype=float)
        pos, neg = z >= 0, z < 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), np.asarray(X, dtype=float)]
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)

        for _ in range(self.n_iters):
            p = self._sigmoid(X @ self.w)
            grad = (X.T @ (p - y)) / n            # same form as linear regression!
            grad[1:] += self.l2 * self.w[1:]      # no penalty on the intercept
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.c_[np.ones(len(X)), X] @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

**Follow-ups**
- *Why is the gradient identical in form to linear regression's?* Both are generalized linear models; for a GLM with the canonical link, the gradient of the log-likelihood is always `Xᵀ(prediction - y)`. Nice result to state.
- *Why not squared error for classification?* With a sigmoid, squared error is non-convex in `w` and its gradient vanishes when predictions are confidently wrong (the sigmoid saturates). Log loss is convex and its gradient stays proportional to the error.
- *How do you extend to multiclass?* Softmax regression with cross-entropy, or one-vs-rest.

---

## 3. K-Means

**Problem**: Implement K-Means with k-means++ initialization.

```python
def kmeans(X, k, n_iters=100, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # --- k-means++ init: spread initial centroids by D² sampling ---
    centroids = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum()
        centroids.append(X[rng.choice(n, p=probs)])
    centroids = np.array(centroids)                      # (k, d)

    for _ in range(n_iters):
        # Assignment: squared distance from every point to every centroid
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)   # (n, k)
        labels = dists.argmin(axis=1)                                    # (n,)

        # Update: mean of each cluster; keep old centroid if a cluster empties
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    inertia = ((X - centroids[labels]) ** 2).sum()
    return labels, centroids, inertia
```

**Follow-ups**
- *Memory problem with this distance computation?* `X[:, None, :] - centroids[None, :, :]` materializes an `(n, k, d)` array. For n=1M, k=100, d=128 that's 51 GB. Use the identity `‖x-c‖² = ‖x‖² - 2x·c + ‖c‖²` to compute distances with one `(n,d)@(d,k)` matmul.
- *Why k-means++?* Random initialization frequently converges to a poor local minimum; k-means++ samples each new centroid proportional to squared distance from existing ones, giving an `O(log k)` approximation guarantee in expectation and much more stable results.
- *Convergence?* Both steps monotonically decrease inertia and there are finitely many assignments, so it always converges — to a local optimum, which is why you run it with several seeds (`n_init`).
- *Empty cluster?* Reinitialize it to the point farthest from its centroid, or keep the old centroid (as above).

---

## 4. K-Nearest Neighbors

```python
def knn_predict(X_train, y_train, X_test, k=5, task='classification'):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)

    # ‖a - b‖² = ‖a‖² - 2a·b + ‖b‖²  — avoids the (n_test, n_train, d) tensor
    d2 = (
        (X_test ** 2).sum(axis=1)[:, None]
        - 2 * X_test @ X_train.T
        + (X_train ** 2).sum(axis=1)[None, :]
    )                                                    # (n_test, n_train)

    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]  # O(n) per row, not O(n log n)
    neighbor_labels = np.asarray(y_train)[idx]           # (n_test, k)

    if task == 'regression':
        return neighbor_labels.mean(axis=1)
    # Majority vote
    return np.array([np.bincount(row).argmax() for row in neighbor_labels])
```

**Follow-ups**
- *Why `argpartition` over `argsort`?* You need the k smallest, not a full ordering — `O(n)` vs `O(n log n)` per row.
- *Why does KNN fail in high dimensions?* Distance concentration: as `d` grows, the ratio between nearest and farthest neighbor distances approaches 1, so "nearest" stops being meaningful. Reduce dimensions first (PCA) or use a learned metric.
- *Scaling required?* Yes — KNN is distance-based, so a feature measured in dollars swamps one in fractions. Always standardize.

---

## 5. Decision Tree Split (Gini and Entropy)

**Problem**: Find the best split for a node.

```python
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - (p ** 2).sum()

def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]                       # avoid log(0)
    return -(p * np.log2(p)).sum()

def best_split(X, y, criterion=gini):
    n, d = X.shape
    parent_impurity = criterion(y)
    best = {'gain': -np.inf, 'feature': None, 'threshold': None}

    for feature in range(d):
        # Candidate thresholds: midpoints between consecutive unique values
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2

        for t in thresholds:
            left = X[:, feature] <= t
            n_left = left.sum()
            if n_left == 0 or n_left == n:
                continue
            # Weighted child impurity
            child = (n_left / n) * criterion(y[left]) + ((n - n_left) / n) * criterion(y[~left])
            gain = parent_impurity - child
            if gain > best['gain']:
                best = {'gain': gain, 'feature': feature, 'threshold': t}
    return best
```

**Follow-ups**
- *Gini vs entropy?* They agree on the chosen split the overwhelming majority of the time. Gini is cheaper (no logarithm) and is scikit-learn's default; entropy is grounded in information theory. Not a decision worth agonizing over.
- *Complexity?* This is `O(d · n²)` because of the inner loop over thresholds recomputing impurity. Sorting each feature once and updating class counts incrementally as the threshold sweeps gives `O(d · n log n)`. Say this — it's the optimization interviewers look for.
- *How would you handle continuous vs categorical?* Continuous uses thresholds as above; categorical either uses subset splits (exponential, so usually restricted) or is ordered by mean target value first.

---

## 6. Train/Test Split and K-Fold Cross-Validation

```python
def train_test_split_manual(X, y, test_size=0.2, stratify=False, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)

    if not stratify:
        idx = rng.permutation(n)
        cut = int(n * (1 - test_size))
        train_idx, test_idx = idx[:cut], idx[cut:]
    else:
        train_idx, test_idx = [], []
        for cls in np.unique(y):                      # preserve class ratio per split
            cls_idx = rng.permutation(np.where(y == cls)[0])
            cut = int(len(cls_idx) * (1 - test_size))
            train_idx.extend(cls_idx[:cut])
            test_idx.extend(cls_idx[cut:])
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)                    # handles n not divisible by k
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val
```

**Follow-up**: *What changes for time series?* Nothing about this works — shuffling leaks the future. Use forward chaining: fold `i` trains on `[0..i]` and validates on `[i+1]`, with a gap equal to the label latency.

---

## 7. Metrics from Scratch

```python
def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred, eps=1e-12):
    tp, fp, fn, _ = confusion(y_true, y_pred)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def roc_auc(y_true, y_score):
    """AUC == P(score of a random positive > score of a random negative).
    Computed via the rank-sum (Mann-Whitney U) identity — O(n log n), no curve needed."""
    y_true = np.asarray(y_true)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Average ranks within ties so tied scores don't bias the result
    scores_sorted = np.asarray(y_score)[order]
    i = 0
    while i < len(scores_sorted):
        j = i
        while j + 1 < len(scores_sorted) and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1

    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

**Follow-up**: *Why the rank-sum formula rather than integrating the ROC curve?* It's the same quantity, computed in one sort with no threshold sweep, and it makes the probabilistic interpretation of AUC explicit.

---

## 8. PCA

```python
def pca(X, n_components):
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)                  # centering is mandatory

    # SVD is more numerically stable than eigendecomposing the covariance matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]                   # (n_components, d)
    explained_variance = (S ** 2) / (len(X) - 1)
    explained_ratio = explained_variance / explained_variance.sum()

    X_transformed = X_centered @ components.T        # (n, n_components)
    return X_transformed, components, explained_ratio[:n_components]
```

**Follow-ups**
- *Why SVD over eigendecomposition of `XᵀX`?* Forming `XᵀX` squares the condition number, losing precision; SVD works on `X` directly. It's also cheaper when `d >> n`.
- *Must you standardize as well as center?* Center always. Standardize when features are on different scales — otherwise the component with the largest units dominates the variance and PCA just finds that axis.
- *Choosing `n_components`?* Cumulative explained variance (e.g. 95%), an elbow in the scree plot, or downstream task performance.

---

## 9. Softmax and Cross-Entropy (Numerically Stable)

```python
def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z_shifted = z - np.max(z, axis=axis, keepdims=True)   # prevents exp() overflow
    e = np.exp(z_shifted)
    return e / e.sum(axis=axis, keepdims=True)

def cross_entropy(logits, labels, eps=1e-12):
    """logits: (n, k) raw scores. labels: (n,) integer class indices."""
    probs = softmax(logits, axis=1)
    n = len(labels)
    return -np.log(probs[np.arange(n), labels] + eps).mean()

def cross_entropy_grad(logits, labels):
    """dL/dlogits = (softmax(logits) - onehot(labels)) / n — strikingly simple."""
    probs = softmax(logits, axis=1)
    n = len(labels)
    probs[np.arange(n), labels] -= 1
    return probs / n
```

**Follow-ups**
- *Why subtract the max?* `softmax(z) == softmax(z - c)` for any constant `c`, and subtracting the max caps the largest exponent at `e^0 = 1`, preventing overflow. Costs nothing, prevents NaNs. Interviewers specifically watch for this.
- *Why do frameworks fuse softmax and cross-entropy?* `log(softmax(z))` computed separately loses precision when probabilities are tiny; the fused log-sum-exp form is stable, which is why PyTorch's `CrossEntropyLoss` takes raw logits and applying softmax first is a bug.

---

## 10. A Two-Layer Neural Network with Backprop

```python
class TwoLayerNet:
    """input -> Linear -> ReLU -> Linear -> softmax cross-entropy"""

    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hidden))   # He init for ReLU
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / d_hidden), (d_hidden, d_out))
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1        # (n, h)
        self.a1 = np.maximum(0, self.z1)       # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2  # (n, k) logits
        return self.z2

    def backward(self, labels, lr=0.01):
        n = len(labels)
        # dL/dz2 for softmax + cross-entropy
        dz2 = softmax(self.z2, axis=1)
        dz2[np.arange(n), labels] -= 1
        dz2 /= n                                # (n, k)

        dW2 = self.a1.T @ dz2                   # (h, n) @ (n, k) -> (h, k)
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T                   # (n, k) @ (k, h) -> (n, h)
        dz1 = da1 * (self.z1 > 0)               # ReLU derivative is the mask

        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        for param, grad in ((self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2)):
            param -= lr * grad
```

**Follow-ups**
- *Why is the ReLU backward pass just a mask?* Its derivative is 1 for positive inputs and 0 otherwise, so the incoming gradient is passed through or zeroed. (The derivative at exactly 0 is undefined; any convention works in practice.)
- *Why He initialization here?* ReLU zeroes about half its inputs, halving the output variance per layer; the `2/fan_in` variance compensates so activations neither vanish nor explode with depth.
- *Gradient check?* Compare the analytic gradient with `(L(w+ε) - L(w-ε)) / 2ε` for a few random coordinates; relative error below ~1e-7 means the backward pass is right.

---

## 11. Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (..., n_q, d_k)  K: (..., n_k, d_k)  V: (..., n_k, d_v)"""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)      # (..., n_q, n_k)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)            # causal or padding mask

    weights = softmax(scores, axis=-1)                      # (..., n_q, n_k)
    return weights @ V, weights                             # (..., n_q, d_v)


def causal_mask(n):
    """Position i may attend only to positions <= i."""
    return np.tril(np.ones((n, n), dtype=bool))
```

**Follow-ups**
- *Why divide by `√d_k`?* If `q` and `k` have i.i.d. components with unit variance, their dot product has variance `d_k`. Without scaling, logits grow with dimension, softmax saturates into a near-one-hot distribution, and gradients vanish. The `√d_k` keeps the variance at 1.
- *Complexity?* `O(n² · d)` time and `O(n²)` memory for the score matrix — the reason long context is expensive, and what FlashAttention addresses by never materializing that matrix.
- *Multi-head?* Project to `h` sets of (Q,K,V) of dimension `d/h`, attend in parallel, concatenate, and apply an output projection. The point is letting different heads attend to different relationships.

---

## 12. Cosine Similarity Search

```python
def top_k_similar(query_vec, doc_matrix, k=5):
    """query_vec: (d,)   doc_matrix: (n, d)"""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    docs = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)

    sims = docs @ q                                   # (n, d) @ (d,) -> (n,)
    idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]                 # sort only the k survivors
    return idx, sims[idx]
```

**Follow-ups**
- *Why is the epsilon there?* A zero vector (empty document, failed embedding) would divide by zero and produce NaNs that silently propagate through the ranking.
- *Scaling past a few million vectors?* Exact search is `O(n·d)` per query. Move to an ANN index — HNSW for high recall at moderate memory, IVF-PQ for large corpora with tight memory budgets. Both trade a small recall loss for orders-of-magnitude speedup.

---

## 13. Text Chunking with Overlap

A frequent practical question in AI engineering interviews.

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Character-based chunking with overlap, respecting word boundaries."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Don't cut mid-word: back off to the last space, unless that loses too much
        if end < len(text):
            space = text.rfind(' ', start, end)
            if space > start + chunk_size // 2:
                end = space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap          # step forward, keeping `overlap` characters
    return chunks
```

**Follow-ups**
- *Why overlap at all?* An answer straddling a boundary would otherwise be split across two chunks and match neither query well. 10–20% overlap is the usual range.
- *Why does the `end >= len(text)` break matter?* Without it, when `overlap` is large relative to the final chunk, `start` can fail to advance and the loop never terminates. Infinite-loop edge cases are exactly what interviewers probe.
- *Better than character-based?* Token-based chunking matches what the model actually sees; structural splitting (headers, functions) preserves coherence better than either.

---

## 14. Rate Limiter with Exponential Backoff

```python
import time
import random

def call_with_backoff(fn, max_retries=5, base_delay=1.0, max_delay=60.0):
    """Retry with exponential backoff and full jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (RateLimitError, TransientError) as exc:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(random.uniform(0, delay))      # full jitter, not a fixed delay


class TokenBucket:
    """Allows bursts up to `capacity` while enforcing an average `rate` per second."""

    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity
        self.tokens = float(capacity)
        self.last = time.monotonic()

    def acquire(self, tokens=1):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

**Follow-up**: *Why jitter?* Without it, every client that got rate-limited at the same moment retries at the same moment, producing a synchronized thundering herd that re-triggers the limit. Randomizing the delay spreads the retries out — this is why AWS's guidance is "full jitter", and it's a strong signal to mention it.

---

## 15. Reciprocal Rank Fusion

Combining a keyword ranking and a vector ranking — the standard hybrid-search question.

```python
def reciprocal_rank_fusion(rankings, k=60):
    """rankings: list of ranked doc-id lists, best first. Returns fused ids, best first."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**Follow-ups**
- *Why fuse ranks instead of scores?* BM25 scores and cosine similarities live on incomparable scales, and normalizing them requires per-query calibration that is fragile. Ranks are unitless and directly comparable.
- *What does `k=60` do?* It damps the influence of the very top ranks so a single system can't dominate the fusion. The value is empirical, from the original RRF paper, and is a reasonable default.

---

## Complexity Reference

| Algorithm | Train | Predict (per sample) | Memory |
|---|---|---|---|
| Linear regression (closed form) | `O(nd² + d³)` | `O(d)` | `O(d²)` |
| Linear/logistic regression (GD) | `O(nd)` per iteration | `O(d)` | `O(d)` |
| KNN | `O(1)` (lazy) | `O(nd)` | `O(nd)` |
| K-Means | `O(nkdi)` | `O(kd)` | `O(nd + kd)` |
| Decision tree | `O(dn log n)` (sorted) | `O(depth)` | `O(nodes)` |
| Random Forest | `O(T · dn log n)` | `O(T · depth)` | `O(T · nodes)` |
| PCA (SVD) | `O(min(n²d, nd²))` | `O(dc)` | `O(nd)` |
| Self-attention | `O(n²d)` | — | `O(n²)` |

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| `exp()` without subtracting the max | Overflow → `inf` → NaN | `z - z.max(axis, keepdims=True)` |
| `log(0)` in cross-entropy | `-inf` propagates through the loss | Add epsilon, or use the fused log-sum-exp form |
| Broadcasting to an `(n, k, d)` tensor | Blows up memory on real data | Expand `‖a-b‖²` into matmuls |
| `argsort` when you need top-k | `O(n log n)` instead of `O(n)` | `np.argpartition`, then sort the k survivors |
| Forgetting to center before PCA | The first component points at the mean | Subtract the column means |
| Regularizing the intercept | Biases predictions toward zero | Exclude index 0 from the penalty |
| Not standardizing for distance-based models | Large-scale features dominate | Standardize for KNN, K-Means, SVM; not needed for trees |
| Mutating an input array in place | Caller's data silently changes | Copy, or document it clearly |
| Off-by-one in a chunking loop | Infinite loop or dropped tail | Explicit termination check on the final chunk |
| Skipping shape comments | Shape bugs are the #1 failure mode | Annotate every matmul |

---

## Related Topics

- [Python Coding Challenges](./python_coding_challenges.md)
- [SQL Coding Challenges](./sql_coding_challenges.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Ensemble Methods](../classical_ml/intro_ensemble_methods.md)
- [Clustering](../classical_ml/intro_clustering.md)
- [Dimensionality Reduction](../classical_ml/intro_dimensionality_reduction.md)
- [Neural Network Training](../deep_learning/intro_neural_network_training.md)
- [Transformers](../deep_learning/intro_transformers.md)
