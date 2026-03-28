# Recommendation System Design

A complete ML system design for a large-scale recommendation system (e.g., Netflix, Spotify, Amazon).

---

## Table of Contents

1. [Collaborative vs Content-Based vs Hybrid Filtering](#collaborative-vs-content-based-vs-hybrid-filtering)
2. [Two-Tower Model Architecture](#two-tower-model-architecture)
3. [Candidate Generation → Ranking → Re-Ranking Pipeline](#pipeline)
4. [Handling Cold Start](#handling-cold-start)
5. [Offline vs Online Evaluation](#offline-vs-online-evaluation)
6. [Real-Time Feature Serving](#real-time-feature-serving)
7. [Production Architecture](#production-architecture)

---

## Collaborative vs Content-Based vs Hybrid Filtering

### Collaborative Filtering

Recommends based on the behavior of similar users or similar items.

**User-based CF:** Find users with similar taste, recommend what they liked.
```
Similarity(user_A, user_B) = cosine similarity of their rating vectors
Recommendation for A = items rated highly by users similar to A
```

**Item-based CF:** Find items similar to what the user liked.
```
Similarity(item_i, item_j) = cosine similarity of their rating vectors
Recommendation for user = items similar to user's highly-rated items
```

**Matrix Factorization (MF):** Decompose user-item interaction matrix into latent factors.
```
R ≈ U · Vᵀ
U: (users × k) user embeddings
V: (items × k) item embeddings
Predicted rating: R̂[u,i] = U[u] · V[i]
```

```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np

# User-item interaction matrix (sparse)
interaction_matrix = csr_matrix(user_item_ratings)  # (n_users, n_items)

# Matrix factorization
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T  # (n_items, 50)

# Predict ratings for a user
def recommend_for_user(user_id, user_factors, item_factors, top_k=10):
    scores = user_factors[user_id] @ item_factors.T
    top_items = np.argsort(scores)[::-1][:top_k]
    return top_items
```

**Pros:** Captures complex user preferences, no content features needed.
**Cons:** Cold start for new users/items, popularity bias, scalability.

### Content-Based Filtering

Recommends items similar to what the user has interacted with, based on item features.

```
Item embedding: TF-IDF on description, genre, tags
User profile: Weighted average of item embeddings from history
Score(user, item) = cosine_similarity(user_profile, item_embedding)
```

**Pros:** No cold start for items, explainable, doesn't require other users.
**Cons:** Limited discovery (filter bubble), requires good item features.

### Hybrid Filtering

Combines collaborative and content-based. Most production systems use hybrid.

```
Final Score = α × CF_score + β × CB_score + γ × context_score

Where context includes: time of day, device, location, session behavior
```

---

## Two-Tower Model Architecture

The standard approach for large-scale recommendation candidate retrieval.

```
User Tower                    Item Tower
─────────────                ─────────────
user_id embedding            item_id embedding
age group                    genre embedding
country                      director embedding
watch_history_sequence       avg_rating
time_of_day                  release_year
device_type                  duration
recent_genres (avg)          popularity_score
       ↓                            ↓
 Dense layers               Dense layers
(256 → 128 → 64)           (256 → 128 → 64)
       ↓                            ↓
  User Embedding              Item Embedding
       (d=64)                      (d=64)
            ↘                ↙
         dot product / cosine similarity
                    ↓
             Similarity Score
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, num_genres, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 32)
        self.genre_embed = nn.Embedding(num_genres, 16)

        self.mlp = nn.Sequential(
            nn.Linear(32 + 16 + 8, 256),  # user + genre + context features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, user_id, genre_history, context_features):
        user_emb = self.user_embed(user_id)
        genre_emb = self.genre_embed(genre_history).mean(dim=1)  # Average genre history
        x = torch.cat([user_emb, genre_emb, context_features], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)  # L2 normalize for cosine similarity


class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres, embed_dim=64):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, 32)
        self.genre_embed = nn.Embedding(num_genres, 16)

        self.mlp = nn.Sequential(
            nn.Linear(32 + 16 + 4, 256),  # item + genre + item_features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, item_id, item_genres, item_features):
        item_emb = self.item_embed(item_id)
        genre_emb = self.genre_embed(item_genres).mean(dim=1)
        x = torch.cat([item_emb, genre_emb, item_features], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genres, embed_dim)
        self.item_tower = ItemTower(num_items, num_genres, embed_dim)

    def forward(self, user_inputs, item_inputs):
        user_emb = self.user_tower(*user_inputs)
        item_emb = self.item_tower(*item_inputs)
        return (user_emb * item_emb).sum(dim=-1)  # Dot product

    def compute_user_embedding(self, user_inputs):
        return self.user_tower(*user_inputs)

    def compute_item_embedding(self, item_inputs):
        return self.item_tower(*item_inputs)
```

### Training: In-Batch Negative Sampling

```python
# During training, use all other items in the batch as negatives
def batch_softmax_loss(user_embeddings, item_embeddings, temperature=0.1):
    """
    user_embeddings: (batch_size, embed_dim)
    item_embeddings: (batch_size, embed_dim)
    """
    # Similarity matrix: (batch, batch) — diagonal is positive pairs
    logits = torch.matmul(user_embeddings, item_embeddings.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return F.cross_entropy(logits, labels)
```

---

## Pipeline

### Candidate Generation → Ranking → Re-Ranking

```
10M+ items in catalog
       ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: CANDIDATE GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Narrow to ~1000 candidates in < 20ms

Methods (run in parallel):
├── Two-Tower ANN Search (FAISS/ScaNN): top 200 items by embedding similarity
├── Trending Items: top 200 currently trending in user's region
├── Similar to Last Watched: top 200 items similar to recent watches
├── User's Saved List: all saved/watchlisted items
└── Editorial/Promoted: curated items (promotions, new releases)

Merge & deduplicate: ~1000 candidates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: RANKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Score all ~1000 candidates, narrow to top 100

Model: Wide & Deep / Gradient Boosting / Two-Tower with interaction features
Features: Rich user × item cross features, real-time context
Output: Predicted engagement probability (watch rate, completion rate)
Latency: < 50ms for 1000 candidates (batch inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: RE-RANKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Apply business rules, diversity, and freshness to top 100

Business rules:
├── Diversity: Maximum 2 items from same genre in first 10 results
├── Freshness: Boost new releases < 30 days old
├── Promotions: Boost sponsored/promoted content (with disclosure)
├── Avoid recently watched (last 7 days)
├── Filter age-restricted content based on user age
└── Respect user preferences (disliked genres, blocked creators)

Final output: Top 20-50 recommendations
```

---

## Handling Cold Start

### New User Cold Start

```
Strategies (in order of preference):

1. Onboarding flow:
   - Ask user to rate 5-10 items or select genres
   - Ask usage intent ("I want to discover movies" vs "Catch up on shows")
   - Use preferences immediately for first session

2. Context-based recommendations:
   - Device type (phone → mobile-first content)
   - Time of day (morning → shorter content)
   - Location/language (local content)

3. Popular/trending recommendations:
   - Show trending content with regional filtering
   - Personalized within demographics if available

4. Quick learning (within first session):
   - Contextual bandits: explore-exploit, learn from clicks
   - Update implicit signals: hover time, scroll depth, watch %
   - Update user embedding in real-time using a fast online learning layer
```

### New Item Cold Start

```
1. Content-based embedding:
   - Use item features (genre, description, metadata) for the item tower
   - Even without interactions, the item tower can produce reasonable embeddings

2. Exploration boost:
   - Show new items to a small fraction of users (epsilon-greedy)
   - Use multi-armed bandit (UCB) to maximize learning from early interactions
   - "New Release" boost in re-ranking for first 30 days

3. Similar item mapping:
   - If new item is similar to existing items, bootstrap with similar items' embedding
   - Find k nearest items in content space, initialize with average embedding

4. External signals:
   - Editorial metadata quality score
   - Third-party ratings/reviews (Rotten Tomatoes, IMDb)
   - Social buzz signals (mentions, shares)
```

---

## Offline vs Online Evaluation

### Offline Evaluation

Used during model development to compare models without production traffic.

| Metric | Description | Formula |
|--------|-------------|---------|
| **NDCG@K** | Normalized Discounted Cumulative Gain at K | Rewards relevant items at top positions |
| **MAP@K** | Mean Average Precision at K | Precision averaged at each relevant item |
| **Recall@K** | Coverage of relevant items in top K | Fraction of relevant items in top K |
| **Hit Rate@K** | Did at least one relevant item appear in top K | Binary per user |
| **Coverage** | Fraction of catalog recommended to any user | Diversity measure |

```python
from sklearn.metrics import ndcg_score
import numpy as np

# NDCG@K
def ndcg_at_k(y_true_list, y_score_list, k=10):
    """
    y_true_list: list of actual item IDs user interacted with
    y_score_list: list of (score, item_id) from model
    """
    scores = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        # Top K predictions
        top_k_items = [item for _, item in sorted(y_score, reverse=True)[:k]]

        # Relevance: 1 if item was interacted with, 0 otherwise
        relevance = [1 if item in y_true else 0 for item in top_k_items]

        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

        # Ideal DCG (all relevant items at top)
        ideal_rel = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rel))

        scores.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(scores)
```

### Online Evaluation

A/B testing in production:

```
Control: Existing recommendation model (model v1)
Treatment: New model (model v2)

Primary metrics:
- Click-Through Rate (CTR): % of recommended items clicked
- Watch Rate: % of clicked items watched > 3 minutes
- Session Length: Total time spent in session
- Return Rate: User returns next day

Guardrail metrics (must not regress):
- Catalog coverage: New model shouldn't create filter bubbles
- Long-term satisfaction: Avoid clickbait that drives short-term clicks
- Content diversity: Items from multiple genres

Statistical requirements:
- Minimum detectable effect: 0.5% relative CTR improvement
- Significance level: α = 0.05
- Statistical power: 0.80
- Minimum runtime: 2 weeks (capture weekly seasonality)
```

---

## Real-Time Feature Serving

```python
# Feature categories:
# 1. Precomputed (refreshed daily/hourly)
# 2. Real-time (computed at request time)
# 3. Streaming (updated continuously from Kafka)

from redis import Redis

redis = Redis(host="feature-store.redis", decode_responses=True)

def get_user_features(user_id: str) -> dict:
    """
    Retrieve user features from online feature store.
    Latency target: < 5ms
    """
    # Precomputed features (updated daily)
    user_profile = redis.hgetall(f"user:profile:{user_id}")

    # Streaming features (updated in real-time from Kafka)
    velocity = redis.hgetall(f"user:velocity:{user_id}")

    return {
        # Profile features
        "account_age_days": int(user_profile.get("account_age_days", 0)),
        "preferred_genres": user_profile.get("top_genres", "").split(","),
        "watch_history_count": int(user_profile.get("watch_count_30d", 0)),

        # Real-time velocity features
        "items_watched_session": int(velocity.get("items_watched_1h", 0)),
        "genres_watched_today": velocity.get("genres_today", "").split(","),
        "session_duration_min": float(velocity.get("session_duration", 0)),
    }

def get_item_features_batch(item_ids: list[str]) -> dict:
    """Batch fetch item features — much faster than one at a time."""
    pipeline = redis.pipeline()
    for item_id in item_ids:
        pipeline.hgetall(f"item:features:{item_id}")
    results = pipeline.execute()
    return dict(zip(item_ids, results))
```

---

## Production Architecture

```
USER REQUEST
     │
     ▼
API Gateway (Kong / Envoy)
     │
     ▼
Recommendation Service
     │
     ├──────────────────────────────────────────┐
     │                                          │
     ▼                                          ▼
Feature Fetcher                         ANN Search (FAISS/ScaNN)
└── Redis Online Store                  └── Precomputed item embeddings
    (user profile, velocity)                 from Two-Tower item tower
     │                                          │
     ├──────────────────────────────────────────┘
     │ ~1000 candidates + user features
     ▼
Ranking Service
└── Wide & Deep Model (TensorFlow Serving / Triton)
    Input: user_features × candidate_items (batch)
    Output: engagement probability per item
     │
     │ top 100 ranked items
     ▼
Re-Ranking Service
└── Business rules, diversity, promotions
     │
     │ top 20-50 recommendations
     ▼
Response → User

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Background Pipelines
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Interactions → Kafka → Streaming Pipeline → Update online features (Redis)
                                                → Training data (S3)

Daily Training Pipeline (Airflow):
S3 training data → Two-Tower training (PyTorch)
               → Ranking model training (TensorFlow)
               → Evaluate offline metrics (NDCG, hit rate)
               → Shadow deploy → Canary → Production
               → Precompute item embeddings → FAISS index

Weekly Jobs:
- Update item catalog embeddings (item tower inference on all items)
- Rebuild FAISS/ScaNN index for ANN search
- Update trending item lists by region
- Refresh user long-term profile features
```

### Infrastructure Sizing

```
For 100M daily active users, 50 requests/user/day = 5B requests/day = ~60K RPS

Recommendation Service:
├── 100 API pods × 1000 RPS each = 100K RPS capacity
├── Auto-scaling: 50 → 200 pods based on load
└── Redis cluster: 20 nodes × 50GB = 1TB for user/item features

ANN Search (FAISS):
├── 10M items × 64 dimensions × 4 bytes = 2.5 GB per index
├── 10 ANN servers (replicas for load distribution)
└── Index rebuilt weekly; in-memory for fast lookup

Ranking Model:
├── 50 Triton servers with T4 GPUs
├── Batch 100 candidates per request → very efficient GPU usage
└── Latency: < 30ms for 1000 candidate ranking on GPU
```

---

## Key Interview Points

1. **Why use a two-stage pipeline (retrieval + ranking)?** Running a complex ranking model on 10M+ items is infeasible (too slow). The two-stage pipeline uses a fast approximate method (ANN on embeddings) to narrow to 1000 candidates, then uses a rich model for precise ranking. This balances quality and latency.

2. **How do you address the explore-exploit trade-off?** Use contextual bandits (LinUCB or Thompson Sampling) for exploration — occasionally show items with high uncertainty to gather data. Balance with exploiting known good recommendations. Also use time-varying models to keep recommendations fresh.

3. **How do you prevent filter bubbles?** Enforce diversity constraints in re-ranking (max N items per genre), include content-based exploration items, monitor catalog coverage in A/B tests as a guardrail metric.

4. **How would you scale to 10x traffic?** Horizontal scaling of recommendation service, shard the FAISS index across servers, increase Redis cluster, use pre-computation for popular users, consider pre-generating recommendations at cache warm-up time.

5. **How do you evaluate recommendation quality long-term?** Short-term CTR doesn't capture long-term satisfaction. Use session-level metrics (did user find something to watch?), next-day return rate, and avoid optimizing for clickbait by including long watch-time metrics in training targets.
