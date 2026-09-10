# Search and Ranking System Design

"Design search for our product" is one of the three most common ML system design prompts, alongside recommendations and fraud. It rewards a specific structure — multi-stage retrieval and ranking — and it punishes candidates who jump straight to a model without establishing what relevance means or where the training signal comes from.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [The Multi-Stage Architecture](#the-multi-stage-architecture)
3. [Query Understanding](#query-understanding)
4. [Candidate Retrieval](#candidate-retrieval)
5. [Ranking](#ranking)
6. [Learning to Rank](#learning-to-rank)
7. [Features](#features)
8. [Training Data and Click Bias](#training-data-and-click-bias)
9. [Evaluation](#evaluation)
10. [Serving Architecture](#serving-architecture)
11. [Capacity and Cost](#capacity-and-cost)
12. [Failure Modes](#failure-modes)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Clarify the Problem First

Spend the first three minutes here. Interviewers score this heavily, and the answers change the design.

**What kind of search?** Web, e-commerce, internal documents, code, people. E-commerce optimizes purchases and has inventory constraints; document search optimizes finding one right answer; code search needs exact-symbol matching.

**What defines success?** Click-through rate, conversion, time-to-first-click, task completion, revenue. This determines the label and the objective.

**Scale numbers to ask for:**
- Corpus size: 1M documents is one machine; 10B needs sharding
- QPS: peak and average
- Latency SLO: search is typically 100–300 ms end to end
- Update freshness: how fast must a new item become searchable?

**Constraints:** personalization allowed? Multi-language? Permissions (enterprise search must filter by ACL)? Do results need explanations?

For the rest of this guide, assume a concrete brief: **e-commerce search, 100M products, 5k QPS peak, p99 under 200 ms, new products searchable within 5 minutes, optimizing for conversion.**

---

## The Multi-Stage Architecture

The organizing idea: you cannot run an expensive model over 100M documents in 200 ms, so use progressively more expensive models over progressively smaller candidate sets.

```
Query
  │
  ▼
┌──────────────────────┐
│ Query understanding  │  normalize, spell-correct, segment, classify intent
└──────────┬───────────┘
           ▼
┌──────────────────────┐   100M ──► ~1000
│ Retrieval (recall)   │  BM25 + vector ANN + filters, run in parallel
└──────────┬───────────┘   cheap per doc, must not miss
           ▼
┌──────────────────────┐   1000 ──► ~100
│ Light ranking        │  gradient boosted trees on cheap features
└──────────┬───────────┘
           ▼
┌──────────────────────┐   100 ──► ~10
│ Heavy ranking        │  cross-encoder / deep model, rich features
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Business logic       │  diversity, dedup, boosts, ads, inventory
└──────────┬───────────┘
           ▼
        Results
```

The economics: retrieval costs microseconds per document over millions; heavy ranking costs milliseconds per document over dozens. Total budget stays inside the SLO while the top results get real model capacity.

**Recall is the ceiling.** If retrieval misses the best product, no amount of ranking recovers it. Optimize retrieval for recall@1000 and ranking for precision at the top.

---

## Query Understanding

Cheap and disproportionately valuable — often more so than a better ranker.

| Step | What it does | Example |
|---|---|---|
| **Normalization** | Lowercase, strip punctuation, Unicode fold | `Nike Air-Max` → `nike air max` |
| **Spell correction** | Fix typos against a corpus lexicon | `nkie` → `nike` |
| **Tokenization / segmentation** | Language-aware splitting | Needed for CJK; compound splitting for German |
| **Query expansion** | Add synonyms | `sneakers` → `+ trainers, running shoes` |
| **Entity recognition** | Extract structured attributes | `red nike shoes size 10` → brand, colour, size |
| **Intent classification** | Navigational / informational / transactional | `nike.com` vs `best running shoes` |
| **Query rewriting** | LLM-based reformulation for hard queries | `something for sore feet` → `orthopedic insoles` |

**Head vs tail** is the key structural fact. Search traffic is extremely skewed — a small number of queries make up most of the volume. That justifies a two-track approach: **precompute and cache** results for head queries (huge latency and cost win, plus you can afford expensive offline ranking), and rely on generalization for the long tail, where per-query behavioural data is sparse.

Entity extraction converts free text into filters. `red nike shoes size 10` becomes a structured query with `brand=nike, color=red, size=10`, which does more for precision than any ranking improvement.

---

## Candidate Retrieval

Run **lexical and semantic retrieval in parallel** and fuse. They fail in complementary ways.

### Lexical (BM25)

```
BM25(q, d) = Σ_terms IDF(t) · (f(t,d) · (k+1)) / (f(t,d) + k · (1 - b + b·|d|/avgdl))
```

Strong at exact matches — SKUs, model numbers, brand names, rare terms. An inverted index maps each term to a posting list of documents; scoring is a merge over those lists, extremely fast.

### Semantic (vector ANN)

Embed query and documents into a shared space and use approximate nearest neighbour search. Handles paraphrase and synonymy — "shoes for jogging" matching a running-shoe listing with no shared terms.

The retrieval model is a **two-tower (bi-encoder)**: separate encoders for query and document, so all document embeddings are precomputed offline and only the query is encoded at request time. That's what makes it fast enough for retrieval.

| ANN index | Trade-off |
|---|---|
| **HNSW** | High recall, fast, memory-hungry |
| **IVF-PQ** | Memory-efficient at billion scale, lower recall |
| **ScaNN** | Strong quality/speed balance |

### Fusion

**Reciprocal Rank Fusion** avoids having to calibrate BM25 scores against cosine similarities, which live on incomparable scales:

```python
def rrf(rankings, k=60):
    """Fuse ranked lists by rank, not score. k damps the influence of top ranks."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

### Filters

Hard constraints — in stock, ships to country, price range, **permissions** — must be applied *inside* the retrieval, not after. Post-filtering can return an empty page when the top-1000 are all filtered out, and in enterprise search it leaks the existence of documents the user can't see.

---

## Ranking

### Light ranking

A gradient boosted tree over a few dozen cheap features, scoring ~1000 candidates in a few milliseconds. Purpose: cut to ~100 without losing the good ones.

### Heavy ranking

A **cross-encoder** jointly encodes query and document, so attention can compare them token by token. Far more accurate than the bi-encoder used in retrieval, and far too slow for anything but a short list.

```
Bi-encoder (retrieval):   encode(q) · encode(d)      — d precomputed, ~microseconds
Cross-encoder (ranking):  model([q; d]) → score      — no precompute, ~milliseconds each
```

That contrast is worth stating explicitly in an interview; it's the reason the architecture has stages at all.

---

## Learning to Rank

| Approach | Trains on | Loss | Notes |
|---|---|---|---|
| **Pointwise** | One (q, d) at a time | Regression / classification | Simple; ignores that ranking is relative |
| **Pairwise** | Pairs (d⁺, d⁻) | Which of the two ranks higher | RankNet, LambdaRank; strong practical default |
| **Listwise** | Whole result list | Optimizes a list metric directly | LambdaMART, ListNet; usually best |

**LambdaMART** — gradient boosted trees with gradients weighted by the NDCG change a swap would cause — remains the workhorse for tabular ranking features, and saying so signals practical experience. Neural rankers win when the signal is in the text itself.

The reason pointwise underperforms: ranking only cares about *relative order* within a query. A pointwise model spends capacity on absolute score calibration that the metric never rewards, and treats an easy query and a hard query as equally important.

---

## Features

| Group | Examples |
|---|---|
| **Query** | Length, intent class, rarity, language, is-navigational |
| **Document** | Price, rating, review count, recency, popularity, quality score |
| **Query-document match** | BM25, cosine similarity, exact title match, attribute overlap, cross-encoder score |
| **Behavioural** | Historical CTR for this (query, doc), conversion rate, dwell time |
| **User / context** | Location, device, session history, past purchases |
| **Business** | Margin, stock level, promotion status, seller quality |

**Behavioural features are usually the strongest and the most dangerous.** Historical CTR for a (query, document) pair is enormously predictive — and it's precisely the feature that entrenches whatever was already ranked highly, creating the rich-get-richer loop that suppresses good new items. Mitigate with position-debiasing, exploration, and confidence-adjusted estimates (a document with 2 clicks out of 3 impressions should not outrank one with 500 out of 1000).

---

## Training Data and Click Bias

The central difficulty of search, and where strong candidates separate.

You have abundant click logs but **clicks are not relevance judgments**. They're contaminated by:

| Bias | Mechanism |
|---|---|
| **Position bias** | Position 1 gets far more clicks regardless of quality |
| **Presentation bias** | Only what was shown can be clicked |
| **Selection bias** | The current ranker decides what gets a chance |
| **Trust bias** | Users assume top results are better and click accordingly |

Naively training on clicks teaches the model to reproduce the existing ranker, permanently.

**Mitigations, roughly in order of practicality:**

1. **Position-based propensity weighting (IPW)** — estimate the probability of examination at each position and weight each click by `1/propensity`, so a click at position 8 counts far more than one at position 1.
2. **Randomization for propensity estimation** — occasionally swap adjacent positions or shuffle within the top-k on a small traffic slice. This measures position bias directly rather than assuming a curve, and it's the principled foundation.
3. **Click models** — cascade / dynamic Bayesian network models that separate "examined" from "attractive" from "satisfied".
4. **Exploration** — inject some randomness into ranking so new documents accumulate impressions.
5. **Human relevance judgments** — expensive, but essential as an unbiased evaluation set even if too small for training.

```python
def ipw_weighted_loss(clicks, positions, propensities, model_scores):
    """Weight each observed click by 1/P(examined at this position)."""
    weights = 1.0 / np.clip(propensities[positions], 0.01, 1.0)   # clip: avoid huge weights
    return -(weights * clicks * np.log(sigmoid(model_scores))).mean()
```

**Strong negatives matter as much as positives.** Random negatives are too easy — the model learns to separate "unrelated" from "related", which it could already do. Use in-batch negatives plus *hard* negatives: documents the current system retrieved but the user skipped.

---

## Evaluation

### Offline

| Metric | Use |
|---|---|
| **Recall@k** (retrieval) | The ceiling on everything downstream |
| **NDCG@k** | Primary ranking metric; handles graded relevance and position discount |
| **MRR** | When there's one right answer |
| **MAP** | Multiple relevant items |

NDCG is the default because it captures both that relevance is graded and that position matters:

```
DCG@k = Σ (2^rel_i - 1) / log₂(i+1)      NDCG@k = DCG@k / IDCG@k
```

**Interleaving** deserves specific mention — it's far more sensitive than A/B testing for ranking changes. Instead of splitting users, you interleave results from both rankers into one list and attribute clicks. Because each user sees both rankers, within-user variance is removed and you need roughly an order of magnitude less traffic to detect the same difference. Team-draft interleaving is the standard variant.

### Online

CTR, conversion rate, time to first click, query reformulation rate (a strong *negative* signal — reformulation means the user didn't find it), abandonment rate, revenue per search.

**Guardrails**: latency p99, zero-result rate, and diversity. A ranker that improves CTR by showing near-duplicates is a regression.

---

## Serving Architecture

```
Client → API Gateway → Search Service ─┬─► Query Understanding (cache)
                                        ├─► Inverted Index shards (BM25)
                                        ├─► Vector Index shards (ANN)
                                        ├─► Feature Store (doc + behavioural)
                                        ├─► Ranker (light → heavy)
                                        └─► Business Logic → Response

Indexing:  Product DB → CDC stream → Enrichment → Index writers → Shards
```

**Sharding**: at 100M documents, shard by document ID (each shard holds a slice, every query hits all shards, results merged). Scatter-gather means p99 is governed by the *slowest* shard, so hedged requests — send a duplicate to a replica after a delay and take the first response — are the standard tail-latency fix.

**Caching layers**: query understanding results, full result sets for head queries (short TTL), document features, and embeddings. Head-query caching alone often serves a large share of traffic.

**Index freshness**: a real-time segment holds recent updates and is searched alongside the main index, with periodic merges. This gives 5-minute freshness without rebuilding a 100M-document index.

---

## Capacity and Cost

Worth doing out loud — interviewers look for it.

```
100M docs × 768-dim fp32 embeddings = 100e6 × 768 × 4 B ≈ 307 GB
  → int8 quantization: ~77 GB
  → 384-dim model instead: ~38 GB   → fits in memory across a few nodes

5k QPS, 200 ms p99, ~50 ms average service time
  → concurrency ≈ 5000 × 0.05 = 250 in-flight requests
  → with ~100 concurrent requests per node: ~3 nodes + replication + headroom
```

The two big levers are embedding dimension and quantization, both of which trade a little recall for a lot of memory. Measure recall@1000 at each setting rather than assuming.

---

## Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| **Zero results** | Over-filtering, rare query, strict AND | Relax to OR, spell-correct, semantic fallback, show alternatives |
| **Rich-get-richer** | Behavioural features entrench the incumbent | Exploration slots, position debiasing, new-item boost |
| **Near-duplicate flooding** | No diversity constraint | Dedup by content hash; MMR or per-seller caps |
| **Stale index** | Slow pipeline | Real-time segment; monitor indexing lag |
| **Tail latency spikes** | Slowest shard governs scatter-gather | Hedged requests, timeouts with partial results |
| **Head-query cache poisoning** | A bad result gets cached widely | Short TTL, invalidation on index update |
| **Permission leak** | Post-filtering by ACL | Filter inside retrieval |
| **Query drift** | New products, new vocabulary, seasonality | Continuous retraining; monitor zero-result rate |

---

## Interview Q&A

#### Why a multi-stage architecture instead of one good model?

Because of the cost asymmetry between accuracy and scale. A cross-encoder that jointly attends over query and document is dramatically more accurate than a dot product, but costs milliseconds per document — over 100M documents that's hours per query. A dot product over precomputed embeddings costs microseconds and can be indexed for sublinear search, but it's much less accurate.

Staging lets you spend the accuracy where it pays. Retrieval is cheap per document and optimized for **recall** — the only requirement is not to lose the good ones from the top ~1000. Ranking is expensive per document and optimized for **precision at the top**, over a list short enough to afford it. Total latency stays in budget while the results the user actually sees get real model capacity.

The corollary worth stating: recall at the retrieval stage is a hard ceiling. If the best product isn't in the candidate set, no ranker can recover it — so when search quality is poor, measure recall@1000 before touching the ranker.

#### Clicks are your only labels. What's wrong with training on them directly?

Clicks measure *examination and attractiveness*, not relevance, and they're heavily contaminated by the ranker that produced them. Position bias means position 1 gets far more clicks regardless of quality. Presentation bias means only shown documents can be clicked. Selection bias means the current ranker decides what gets a chance at all.

Train naively and you learn to reproduce the incumbent ranker — the model concludes that whatever ranked first is best, and it's self-confirming forever.

The fixes: **inverse propensity weighting**, weighting each click by `1/P(examined at that position)`, so a click at position 8 counts far more than one at position 1. **Randomization on a small traffic slice** — swapping adjacent results — to *measure* those propensities rather than assume a curve. **Click models** that separately model examination, attractiveness, and satisfaction. **Exploration slots** so new documents accumulate impressions. And a **human-judged evaluation set** which, even if too small to train on, gives you an unbiased read on whether any of it is working.

#### How do you handle the head/tail query distribution?

Query traffic is extremely skewed, and that asymmetry justifies two different strategies.

For **head queries**, precompute. You can afford expensive offline ranking, human curation, and full-result caching, which cuts both latency and cost enormously and lets you hand-fix specific high-value queries. You also have abundant behavioural data per query, so (query, document) CTR features are reliable.

For the **tail**, you have almost no per-query signal, so you must generalize: semantic retrieval matters much more (no behavioural history to lean on), query understanding does the heavy lifting through spell correction and entity extraction, and features must be query-independent or based on generalizable text matching rather than memorized pairs. Confidence-weighting behavioural features is important here — a document with 2 clicks from 3 impressions shouldn't outrank one with 500 from 1000.

#### Why is interleaving better than A/B testing for ranking?

An A/B test splits users, so the comparison is between-user and the metric carries all the variance of user heterogeneity — some users click a lot, some never do, and that noise swamps a small ranking difference.

Interleaving merges results from both rankers into a single list shown to every user, then attributes each click to whichever ranker contributed that result. Every user experiences both systems, so the comparison is within-user and the between-user variance cancels. In practice that's roughly an order of magnitude more sensitive, so you can detect real differences with far less traffic and iterate much faster.

The limits: it measures relative ranking preference, not absolute business metrics like revenue or session length, and it can't evaluate changes to the *set* of results (like a new UI layout). So the usual pattern is interleaving to select among ranker candidates, then a conventional A/B test on the winner to confirm business impact.

#### Your search returns results but users keep reformulating their query. What do you investigate?

Reformulation is one of the strongest negative signals available — it means the user looked and didn't find it — so I'd treat the reformulation rate as a primary metric and segment it.

I'd check, in order: **recall** first, by taking a sample of reformulated queries and verifying whether a relevant document was even in the candidate set — if not, it's retrieval, not ranking. Then **query understanding**: are these queries failing to spell-correct, failing entity extraction, or in a language the tokenizer mishandles? Then **the reformulation pairs themselves**, which are a goldmine — the second query is effectively the user telling you what the first should have meant, and mining those pairs gives synonym and rewriting rules directly.

I'd also look at zero-result and near-zero-result rates, filters that are too aggressive, and whether the failures concentrate in the tail — which would point at generalization rather than a broken component.

#### How do you keep a 100M-document index fresh within 5 minutes?

Not by rebuilding — a full rebuild at that scale takes hours.

The standard design is a **tiered index**: a large, periodically-rebuilt main index plus a small real-time segment holding recent changes. Queries search both and merge results. The real-time segment is small enough to update within seconds, and it's merged into the main index during periodic compaction.

Updates arrive through **change data capture** from the product database into a stream, through an enrichment step (compute embeddings, derive features), then to index writers. Deletes need a tombstone mechanism so a removed product stops appearing immediately even before compaction.

The critical thing to monitor is **indexing lag** — time from source change to searchable — as a first-class SLO with alerting, because a silently stalled pipeline surfaces as gradually worsening relevance that nobody attributes to freshness.

#### How would you add personalization without hurting everyone else?

Carefully, and late in the pipeline. Personalization signals belong in the **ranking** stage as features (past purchases, category affinity, brand affinity, session context), not in retrieval — narrowing the candidate set per user builds a filter bubble and wrecks recall.

I'd start with cheap, robust signals: session context (what they've clicked in this session) is far more valuable and less privacy-sensitive than long-term history, and it handles the cold-start problem naturally. Then blend a personalization score with the general relevance score rather than replacing it, with the blend weight tuned by experiment.

Guardrails matter: measure whether personalization helps *new* users and *tail* queries, not just the aggregate — it usually helps heavy users and hurts everyone else, which an average metric hides. Cap how far personalization can move a result so a bad user model can't destroy an obviously relevant match. And keep a non-personalized control arm permanently, since personalization interacts badly with A/B measurement over time.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Jumping to a model before defining relevance | The whole design rests on the objective | Establish success metric and labels first |
| Optimizing ranking when recall is the bottleneck | Ranking can't recover missing candidates | Measure recall@1000 first |
| Training on raw clicks | Reproduces the incumbent ranker forever | IPW, randomization, click models, exploration |
| Post-filtering by permissions | Empty pages, and leaks document existence | Filter inside retrieval |
| Pure vector search | Loses exact matches on SKUs and model numbers | Hybrid BM25 + vector with RRF |
| Random negatives when training the retriever | Too easy; model learns nothing useful | Hard negatives from current top results |
| No diversity constraint | Near-duplicates flood the page | Dedup, MMR, per-seller caps |
| Ignoring the head/tail split | Wastes head opportunity, fails on tail | Cache and curate head; generalize for tail |
| A/B testing every ranking change | Underpowered; slow iteration | Interleaving for ranker selection |
| Uncalibrated behavioural features | 2/3 clicks outranks 500/1000 | Confidence-adjusted (smoothed) rates |
| No hedging in scatter-gather | p99 is the slowest shard | Hedged requests, timeouts, partial results |
| Personalization in the retrieval stage | Filter bubble, wrecked recall | Personalize at ranking, blended and capped |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Recommendation System Design](./recommendation_system.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Backend AI System Design](./intro_backend_ai_system_design.md)
- [Embeddings](../ai_genai/intro_embeddings.md)
- [Vector Databases — Advanced](../ai_genai/intro_vector_databases_advanced.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Causal Inference and Uplift](../classical_ml/intro_causal_inference.md)
