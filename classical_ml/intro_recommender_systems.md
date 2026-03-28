# Recommender Systems

Recommender systems predict what a user might like based on past behavior or item attributes. They power Netflix, Spotify, Amazon, YouTube, and LinkedIn. They're a key topic for ML Engineer and Applied Scientist roles.

---

## Table of Contents
1. [Types of Recommender Systems](#types-of-recommender-systems)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Matrix Factorization](#matrix-factorization)
4. [Content-Based Filtering](#content-based-filtering)
5. [Two-Tower Models (Modern Approach)](#two-tower-models-modern-approach)
6. [Hybrid Systems](#hybrid-systems)
7. [Cold Start Problem](#cold-start-problem)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Production Architecture](#production-architecture)
10. [Interview Q&A](#interview-qa)
11. [Common Pitfalls](#common-pitfalls)
12. [Related Topics](#related-topics)

---

## Types of Recommender Systems

```
                    Recommender Systems
                    ─────────────────────
          ┌─────────────┼──────────────┐
          │             │              │
   Collaborative    Content-Based    Hybrid
    Filtering         Filtering      Systems
    ─────────         ─────────
  User-Based       Item features    CF + CB
  Item-Based       User profiles    Two-Tower
  Matrix Factor.
  Neural CF
```

---

## Collaborative Filtering

Collaborative filtering assumes that users who agreed in the past will agree in the future. It only uses the user-item interaction matrix (ratings, clicks, purchases).

### User-Based CF

"Find users similar to you; recommend what they liked."

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-item rating matrix (rows=users, cols=items)
ratings = np.array([
    [5, 4, 0, 1, 0],  # User 0
    [4, 0, 0, 1, 2],  # User 1
    [0, 0, 5, 4, 3],  # User 2
    [0, 3, 4, 0, 0],  # User 3
    [1, 0, 0, 5, 4],  # User 4
])

# Compute user-user cosine similarity
user_sim = cosine_similarity(ratings)
np.fill_diagonal(user_sim, 0)  # Exclude self-similarity

def user_based_predict(ratings, user_id, item_id, k=2):
    """Predict rating for user_id on item_id using top-k similar users."""
    sim_scores = user_sim[user_id]
    # Get users who rated the item
    rated_mask = ratings[:, item_id] > 0
    rated_mask[user_id] = False

    if not any(rated_mask):
        return 0.0

    sims = sim_scores[rated_mask]
    item_ratings = ratings[rated_mask, item_id]

    # Top-k similar users
    top_k_idx = np.argsort(sims)[-k:]
    top_sims = sims[top_k_idx]
    top_ratings = item_ratings[top_k_idx]

    if top_sims.sum() == 0:
        return 0.0
    return np.dot(top_sims, top_ratings) / top_sims.sum()

# Predict User 0's rating for Item 2
pred = user_based_predict(ratings, user_id=0, item_id=2)
print(f"Predicted rating: {pred:.2f}")
```

### Item-Based CF

"Find items similar to what you liked; recommend those."

```python
# Compute item-item cosine similarity
item_sim = cosine_similarity(ratings.T)  # Transpose: items as rows
np.fill_diagonal(item_sim, 0)

def item_based_predict(ratings, user_id, item_id, k=2):
    """Predict rating using top-k similar items the user already rated."""
    sim_scores = item_sim[item_id]
    user_ratings = ratings[user_id]
    rated_mask = user_ratings > 0

    if not any(rated_mask):
        return 0.0

    sims = sim_scores[rated_mask]
    user_item_ratings = user_ratings[rated_mask]

    top_k_idx = np.argsort(sims)[-k:]
    top_sims = sims[top_k_idx]
    top_ratings = user_item_ratings[top_k_idx]

    return np.dot(top_sims, top_ratings) / (top_sims.sum() + 1e-10)
```

---

## Matrix Factorization

Matrix factorization decomposes the user-item matrix R (m×n) into two lower-rank matrices: R ≈ U × V^T, where U (m×k) represents users and V (n×k) represents items in a latent factor space.

### Alternating Least Squares (ALS)

ALS is commonly used for implicit feedback (clicks, purchases, views) — popular in Spark ML.

```python
from sklearn.decomposition import NMF

# Non-negative Matrix Factorization
nmf = NMF(n_components=10, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(ratings)  # User latent factors (m x k)
H = nmf.components_             # Item latent factors (k x n)

# Reconstruct full ratings matrix
ratings_pred = W @ H
```

### SVD-based Matrix Factorization (Surprise library)

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

# Prepare data in Surprise format
data_df = pd.DataFrame({
    'user_id': [0, 0, 1, 1, 2, 3],
    'item_id': [0, 1, 0, 3, 2, 2],
    'rating': [5, 4, 4, 1, 5, 4]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df, reader)

# SVD (Simon Funk SVD / FunkSVD)
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
print(f"RMSE: {cv_results['test_rmse'].mean():.3f}")

# Predict
trainset, testset = train_test_split(data, test_size=0.25)
algo.fit(trainset)
predictions = algo.test(testset)
print(f"Test RMSE: {accuracy.rmse(predictions):.3f}")
```

---

## Content-Based Filtering

Content-based filtering recommends items similar to what a user has liked, based on item features (genre, keywords, attributes).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie content-based filtering example
movies = pd.DataFrame({
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Iron Man'],
    'description': [
        'dream heist sci-fi thriller complex',
        'space time wormhole sci-fi exploration',
        'batman joker dark crime superhero',
        'marvel superhero team avengers action',
        'iron man tony stark superhero action'
    ]
})

# Build TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Cosine similarity between all movies
item_cosine_sim = cosine_similarity(tfidf_matrix)

def get_content_recommendations(title, n=3):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(item_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

print(get_content_recommendations('Inception'))
```

---

## Two-Tower Models (Modern Approach)

Two-tower (dual encoder) models are the industry standard for large-scale recommendations (YouTube, Pinterest, Spotify). They separately encode users and items into a shared embedding space, then score by dot product.

```
                    User Tower          Item Tower
                    ──────────          ──────────
User features    →  [Dense]         Item features → [Dense]
(age, history,      [Dense]    ←→    (title, genre,  [Dense]
 context)           [Dense]           description)   [Dense]
                        ↓                               ↓
                  User Embedding    ·    Item Embedding
                  (128-dim)              (128-dim)
                              ↓
                        Dot Product = Relevance Score
```

```python
import tensorflow as tf

class TwoTowerModel(tf.keras.Model):
    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim=64):
        super().__init__()
        # User tower
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.user_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')

        # Item tower
        self.item_embedding = tf.keras.layers.Embedding(item_vocab_size, embedding_dim)
        self.item_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def encode_user(self, user_ids):
        x = self.user_embedding(user_ids)
        return tf.nn.l2_normalize(self.user_dense(x), axis=-1)

    def encode_item(self, item_ids):
        x = self.item_embedding(item_ids)
        return tf.nn.l2_normalize(self.item_dense(x), axis=-1)

    def call(self, inputs):
        user_ids, item_ids = inputs
        user_emb = self.encode_user(user_ids)
        item_emb = self.encode_item(item_ids)
        return tf.reduce_sum(user_emb * item_emb, axis=-1)  # Dot product

model = TwoTowerModel(user_vocab_size=10000, item_vocab_size=50000)

# At inference: pre-compute all item embeddings, store in vector DB
# Retrieve top-k candidates using ANN (HNSW, FAISS)
```

---

## Hybrid Systems

Combine collaborative filtering and content-based filtering:

| Strategy | Description | Example |
|----------|-------------|---------|
| **Weighted** | Average CF and CB scores | Score = 0.7*CF + 0.3*CB |
| **Switching** | Use CB for cold start, CF once enough data | New users get CB; returning users get CF |
| **Feature augmentation** | Use CB features as input to CF | Add item genre embedding to MF model |
| **Cascade** | CF generates candidates, CB re-ranks | YouTube: CF retrieval → CB ranking |

---

## Cold Start Problem

| Type | Problem | Solutions |
|------|---------|-----------|
| **New user** | No interaction history | Ask onboarding preferences, use demographics, default to popularity |
| **New item** | No ratings yet | Use content features (CB), promote in exploration slots |
| **New system** | No data at all | Start with content-based; gather data gradually |

```python
def get_recommendations(user_id, interaction_count, cf_model, cb_model, threshold=10):
    """Switch between CB and CF based on interaction count."""
    if interaction_count < threshold:
        # Cold start: use content-based
        return cb_model.recommend(user_id)
    else:
        # Warm user: use collaborative filtering
        return cf_model.recommend(user_id)
```

---

## Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@k** | Relevant in top-k / k | Fraction of recommended items that are relevant |
| **Recall@k** | Relevant in top-k / Total relevant | Fraction of relevant items that were recommended |
| **NDCG@k** | Normalized Discounted Cumulative Gain | Ranking quality — rewards placing relevant items higher |
| **MAP@k** | Mean Average Precision | Average precision across users |
| **Hit Rate@k** | % of users with ≥1 relevant item in top-k | Broad coverage metric |
| **MRR** | Mean Reciprocal Rank | Rank of first relevant result |

```python
def ndcg_at_k(recommended, relevant, k):
    """NDCG@k: higher is better, max=1.0"""
    dcg = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # +2: log2(1)=0, so shift by 1

    # Ideal DCG: all relevant items at top
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

# Example
recommended = ['item1', 'item3', 'item5', 'item2', 'item8']
relevant = {'item1', 'item2', 'item4'}
print(f"NDCG@5: {ndcg_at_k(recommended, relevant, k=5):.3f}")
```

---

## Production Architecture

```
User Request
     │
     ▼
┌──────────────┐
│  Retrieval   │  ← Two-tower ANN (FAISS/HNSW): ~100ms, 100-1000 candidates
│   Stage      │    Pre-computed item embeddings in vector DB
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Ranking     │  ← Gradient boosting or deep ranking model
│   Stage      │    Uses rich features (user context, item stats, recency)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Re-ranking  │  ← Business rules (diversity, freshness, sponsored items)
│   Stage      │    Deduplication, safety filters
└──────┬───────┘
       │
       ▼
Final Top-N Recommendations
```

---

## Interview Q&A

**Q1: What is the difference between collaborative filtering and content-based filtering?**
CF uses the user-item interaction matrix only — it leverages collective user behavior. It can recommend items that are hard to describe but popular with similar users. CB uses item/user features — it can explain recommendations ("because you liked sci-fi") and handles new items well. CF suffers from cold start; CB doesn't generalize beyond item similarity.

**Q2: How do you handle the cold start problem in collaborative filtering?**
New users: collect onboarding preferences (ask them to rate a few items), use demographic-based CF, or fall back to content-based until enough interactions. New items: use content features in a hybrid model, or give new items exploration slots (epsilon-greedy).

**Q3: What are the advantages of two-tower models over traditional matrix factorization?**
Two-tower models: (1) can incorporate rich features beyond IDs (text, images, context); (2) scale to billions of users/items via ANN retrieval; (3) support real-time user context; (4) naturally handle cold start via feature encoding. MF is simpler but limited to ID-based embeddings and doesn't generalize to unseen users/items.

**Q4: How do you evaluate a recommender system offline vs online?**
Offline: precision@k, recall@k, NDCG@k on held-out interactions (use time-based split, not random). Online: A/B test CTR, conversion rate, watch time, user retention. Offline metrics don't always correlate with online performance — the bandit feedback problem means you can only evaluate on items that were actually shown.

**Q5: What is matrix factorization and how does it work?**
MF decomposes the sparse user-item rating matrix R (m×n) into two dense matrices U (m×k) and V (n×k), where k << min(m,n). The k dimensions represent latent factors (like genre preference, price sensitivity). We optimize U and V to minimize reconstruction error: ||R - UV^T||². Gradient descent or ALS updates one matrix while holding the other fixed.

**Q6: How would you add diversity to recommendations?**
Maximum Marginal Relevance (MMR): re-rank to balance relevance and diversity. At each step, select the item that maximizes: λ * relevance_score - (1-λ) * max_similarity_to_already_selected. Or use DPP (Determinantal Point Processes) for principled diversity. Alternatively, apply business rules (no more than 2 items from same category).

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Random train/test split | Simulates future leaking into past | Use time-based split |
| Only evaluating accuracy | Ignores novelty, diversity, coverage | Add NDCG, diversity, coverage metrics |
| Ignoring cold start | New users/items get poor recommendations | Hybrid model or fallback strategy |
| Popularity bias | Always recommend popular items | Add exploration (epsilon-greedy, UCB) |
| Feedback loop / filter bubble | Model trains on its own predictions | Inject exploration; measure diversity over time |
| Evaluating offline only | Offline metrics ≠ online metrics | Always A/B test before full rollout |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Clustering](./intro_clustering.md) | User/item clustering for neighborhood CF |
| [Dimensionality Reduction](./intro_dimensionality_reduction.md) | Latent factor models are a form of DR |
| [Feature Engineering](./intro_feature_engineering.md) | Feature prep for content-based and hybrid models |
| [Vector Databases](../ai_genai/intro_vector_databases.md) | ANN retrieval for two-tower models |
| [Study Pattern](../docs/study-pattern.md) | Recommender Systems is an Advanced (🔴) topic |
