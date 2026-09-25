# News Feed Ranking System Design

"Design the ranking system for a social media news feed" sounds like a standard recommendation problem, and much of the machinery is the same: candidate generation, a heavy ranker, a re-ranker. What makes feeds different is the objective. There is no single label. A post can be liked, commented on, shared, read for a minute, hidden, or reported, and the system has to turn all of those into one ordering. The feed also sits on a social graph, so ranking decisions change what creators post, what friends see, and how experiments behave. Strong answers spend their time on the value model, the graph, and the long-term effects, and treat the generic recsys pieces as known.

For two-tower retrieval, generic cold start, and a baseline recsys serving stack, see [Recommendation System Design](./recommendation_system.md). This guide focuses on what is specific to feeds.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [Defining the Objective](#defining-the-objective)
3. [The Feed Ranking Funnel](#the-feed-ranking-funnel)
4. [Candidate Generation](#candidate-generation)
5. [Ranking Models](#ranking-models)
6. [Features](#features)
7. [Re-Ranking and Feed Composition](#re-ranking-and-feed-composition)
8. [Bias in Logged Data](#bias-in-logged-data)
9. [Negative Feedback](#negative-feedback)
10. [Evaluation](#evaluation)
11. [Serving Architecture](#serving-architecture)
12. [Capacity and Cost](#capacity-and-cost)
13. [Cold Start](#cold-start)
14. [Feedback Loops, Integrity and Wellbeing](#feedback-loops-integrity-and-wellbeing)
15. [Interview Q&A](#interview-qa)
16. [Common Pitfalls](#common-pitfalls)
17. [Related Topics](#related-topics)

---

## Clarify the Problem First

Feed prompts hide two very different products behind one sentence. Pin down which one before designing.

**Which feed?** A **connected feed** shows posts from accounts the user follows or is friends with. The candidate set is bounded by the graph, and the job is mostly ordering. An **interest-based ("for you") feed** can show any public post. The candidate set is the whole platform, and retrieval becomes the hard part. Most modern apps blend both, so ask what share of the feed is in-network versus out-of-network.

**Graph type?** Symmetric friendships (bounded, a few hundred connections typical) or asymmetric follows (unbounded; some accounts have tens of millions of followers). Asymmetric follows create the celebrity problem in fan-out.

**Content types?** Text, photos, short video, long video, links, reshares, group posts, events, ads. Each has different engagement patterns. A like on a photo and a like on a long video are not equally informative, and video needs watch-time labels.

**What does success mean?** Daily active users, time spent, meaningful interactions, retention, creator growth, reported satisfaction. This decides the labels and the value model weights, so get the interviewer to commit.

**Scale numbers to ask for:**
- Daily active users and feed requests per user per day
- Posts created per day, and the distribution of follower counts
- Latency budget for a feed page (often a few hundred ms end to end, with ranking getting a slice)
- How fast a new post must be eligible to appear (seconds to minutes)

**Constraints:** integrity policy (misinformation, borderline content), age-appropriate experiences for minors, ads load, regulatory requirements such as offering a non-personalized feed option in some jurisdictions.

For the rest of this guide, assume a concrete brief: **a mixed feed (friends, followed pages, and recommended posts), 500M DAU, ~20 feed requests per user per day, p99 of 300 ms for the ranking stack, new posts eligible within a minute.** These are working assumptions for the exercise, not figures from any real platform.

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Ranking latency (p99) | ~300 ms for retrieval + ranking + re-ranking |
| Freshness | New posts rankable within ~1 minute; engagement counters within seconds |
| Availability | Degrade to a simpler ranking (or chronological) rather than an empty feed |
| Consistency | Deleted or blocked content must disappear immediately, even from caches |

---

## Defining the Objective

### The value model

There is no single "relevance" label. The standard approach is to predict the probability of each action separately, then combine them into one score with a weighted sum:

```
score(u, p) =  w_like    · P(like | u, p)
             + w_comment · P(comment | u, p)
             + w_share   · P(share | u, p)
             + w_dwell   · P(dwell > t | u, p)
             + w_click   · P(click | u, p)
             + w_survey  · P(survey says "worth your time" | u, p)
             − w_hide    · P(hide | u, p)
             − w_report  · P(report | u, p)
```

The model predicts the probabilities. The weights are a **product and policy decision**, not something the model learns. Changing a weight changes what the platform rewards, and creators respond to it.

Two consequences:

- **Every head must be calibrated.** A weighted sum of probabilities only makes sense if the probabilities are on the right scale. If P(comment) is inflated 2x after a model change, you have silently doubled the comment weight. Monitor calibration per head, the same way ads monitors pCTR (see [Calibration](./ads_ctr_prediction.md#calibration)).
- **Rare actions need large weights.** Shares and comments are much rarer than likes. If weights are chosen as "how much we value this action," rare, high-value actions matter only when their weight compensates for their low base rate.

### Choosing and tuning weights

| Approach | How it works | Trade-off |
|---|---|---|
| **Business-set values** | Product teams assign relative value per action (e.g. a comment is worth k likes) | Interpretable; easy to argue about; may not reflect what drives retention |
| **Regression on a long-term outcome** | Fit how much each short-term action predicts a long-term metric (retention, future sessions) and use the coefficients as weights | Ties weights to what matters; correlational, so confounded |
| **Online tuning** | Treat weights as hyperparameters and run experiments, often with Bayesian optimization over several arms, optimizing a primary metric under guardrails | Most direct; slow, and each experiment must run long enough to see long-term effects |
| **Constrained optimization** | Maximize one metric subject to others not dropping more than x% | Makes trade-offs explicit; needs a clear metric hierarchy |

Weights also differ by context. The right weight on watch time for a video-heavy surface differs from a text-heavy one, so many systems keep weights per surface or per content type.

### Short-term engagement vs long-term satisfaction

Optimizing clicks and dwell alone reliably finds content that people engage with but regret: outrage bait, clickbait, sensational posts. Engagement goes up for weeks while satisfaction, trust, and eventually retention fall.

Ways to push the objective toward long-term value:

- **Survey labels.** Ask a random sample of users about specific posts they just saw ("Was this worth your time?", "Do you want to see more like this?"). Train a head to predict the answer and include it in the value model. Survey responses are sparse and biased (only some users answer), so reweight respondents and treat the head as one signal among many.
- **Meaningful-interaction heads.** Weight actions between people who know each other (a comment from a friend, a reply thread) more than passive consumption.
- **Negative heads with real weight.** Hides, "see less," unfollows, and reports should reduce a score, not just be ignored.
- **Long-term metrics as the experiment decision criterion.** Ship based on retention and satisfaction measured over weeks, with short-term engagement as a secondary signal.

```python
from dataclasses import dataclass

@dataclass
class ValueWeights:
    like: float = 1.0
    comment: float = 4.0       # illustrative values, set by product policy
    share: float = 6.0
    dwell: float = 0.5
    survey_worth_it: float = 8.0
    hide: float = 10.0
    report: float = 50.0

def value_score(preds: dict, w: ValueWeights) -> float:
    """Combine calibrated per-action probabilities into one ranking score."""
    positive = (w.like * preds["like"] + w.comment * preds["comment"]
                + w.share * preds["share"] + w.dwell * preds["dwell"]
                + w.survey_worth_it * preds["survey_worth_it"])
    negative = w.hide * preds["hide"] + w.report * preds["report"]
    return positive - negative
```

---

## The Feed Ranking Funnel

```
Feed request (user, device, session context)
  │
  ▼
┌──────────────────────────────┐   millions ──► ~2-5k
│ Candidate generation         │  in-network (follow graph inbox)
└──────────────┬───────────────┘  + out-of-network (embeddings, ANN, trending)
               ▼
┌──────────────────────────────┐   ~5k ──► ~500
│ Light ranking                │  two-tower or small MLP, cheap features
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐   ~500 ──► ~100
│ Heavy ranking                │  multi-task model: P(like), P(comment),
└──────────────┬───────────────┘  P(share), P(dwell), P(hide), ... → value score
               ▼
┌──────────────────────────────┐
│ Re-ranking / feed composition│  integrity demotions, diversity, freshness,
└──────────────┬───────────────┘  dedup, creator fairness, ads insertion
               ▼
        Feed page ──► impression / engagement logs ──► training data
```

The shape is the same as search and ads. The feed-specific parts are the in-network candidate source, the multi-head value model, and a re-ranker that has to compose a page (with ads, diversity, and integrity rules) rather than just sort a list.

---

## Candidate Generation

### In-network: the follow graph

For connected content, candidates are recent posts from accounts the user follows, plus posts from groups and pages they joined. The question is how to assemble those posts fast. Two classic strategies:

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Fan-out on write (push)** | When an account posts, write the post ID into every follower's inbox (a per-user list in a key-value store) | Reads are one fetch; feed load is fast | A post from an account with 50M followers means 50M writes; wasted work for inactive followers |
| **Fan-out on read (pull)** | At request time, look up who the user follows and fetch each account's recent posts | No write amplification; always current | Slow reads for users following thousands of accounts; heavy load at peak |
| **Hybrid** | Push for most accounts; pull for high-follower accounts and merge at read time | Bounds both write and read cost | Two code paths; merge logic; threshold tuning |

**The celebrity problem** is why hybrid is the standard answer. Pushing one post from a very large account to every follower creates a write storm and delays delivery for everyone else in the queue. Instead, mark accounts above a follower threshold as "pull" accounts, and at read time fetch their recent posts from a per-author cache (which is hot, since many users read it) and merge with the pushed inbox.

Other refinements: skip fan-out to users who haven't been active for weeks (build their inbox on demand when they return), and cap inbox length since only recent posts matter.

```python
CELEBRITY_THRESHOLD = 100_000   # tuned from write cost vs read cost

def on_new_post(author_id, post_id, graph, inbox_store, author_cache):
    author_cache.push(author_id, post_id)            # always: per-author recent posts
    if graph.follower_count(author_id) >= CELEBRITY_THRESHOLD:
        return                                        # pulled at read time instead
    for follower in graph.active_followers(author_id):
        inbox_store.prepend(follower, post_id, max_len=1000)

def in_network_candidates(user_id, graph, inbox_store, author_cache, k=2000):
    pushed = inbox_store.get(user_id, limit=k)
    pulled = [p for a in graph.followed_celebrities(user_id)
                for p in author_cache.recent(a, limit=20)]
    return dedup_by_id(pushed + pulled)[:k]
```

### Out-of-network: interest-based retrieval

For "for you" content, the candidate set is every eligible public post. Several sources run in parallel and their outputs are merged:

- **Two-tower ANN.** User embedding against post embeddings in an ANN index. The mechanics are covered in [Two-Tower Model Architecture](./recommendation_system.md#two-tower-model-architecture). The feed-specific issue is freshness: posts are short-lived, so new post embeddings must enter the index within minutes, which favors indexes that support incremental inserts.
- **Graph-based.** Posts engaged with by people you follow or by users similar to you ("friends of friends liked this"). Random-walk methods over a user-post interaction graph also work well here.
- **Author-based.** Recent posts from authors similar to authors you engage with (author embeddings are more stable than post embeddings).
- **Topic and trending.** Posts that are gaining engagement quickly within topics or regions the user cares about.

Each source gets a quota so no single source dominates. Quotas are tuned by experiment, and each source should be evaluated on recall of posts the user later engaged with.

---

## Ranking Models

### Light ranking

A two-tower model or small MLP over a few dozen cheap features, trained to predict the same value score (or a distilled version of the heavy model's score). Its job is recall: cut ~5k to ~500 without losing the posts the heavy ranker would put on top. Distilling from the heavy ranker keeps the two stages aligned.

### Heavy ranking: multi-task

One model with a shared representation and one output head per action. Training separate models per action is wasteful (the same features and embeddings everywhere) and gives rare heads like share or report too little data. Multi-task learning lets sparse heads borrow representation from dense ones.

The difficulty is **negative transfer**: tasks that conflict (e.g. click and hide can both be high for clickbait) pull a shared bottom in different directions.

| Architecture | Idea | When it helps |
|---|---|---|
| **Shared bottom** | One shared MLP, per-task towers | Tasks are closely related; simplest baseline |
| **MMoE** | Several shared experts; each task has a softmax gate choosing a mix of experts | Tasks partly conflict; lets them use different experts |
| **PLE** | Task-specific experts plus shared experts, stacked in several extraction layers | Stronger task conflicts; reduces the "seesaw" where improving one task hurts another |

An MMoE implementation is shown in [Model Progression](./ads_ctr_prediction.md#model-progression). For feeds you would have 6 to 12 heads instead of 2.

**Loss.** Sum of per-task binary cross-entropy (or regression for dwell time), each head trained only on examples where its label is defined. For example, P(comment | click) heads only train on clicked examples, so conditional heads need care about sample selection, much like CVR in ads. Per-task loss weights are tuned so rare heads aren't drowned out.

**Sequence features.** Heavy rankers often include the user's recent engagement sequence (last N posts interacted with) processed by attention against the candidate post. This captures short-term intent that aggregate features miss.

---

## Features

| Group | Examples |
|---|---|
| **User** | User embedding, account age, activity level, content-type preferences, recent engagement sequence, device, language |
| **Author** | Author embedding, follower count, historical engagement rate, posting frequency, integrity/quality scores |
| **Post** | Content type, text/image/video embeddings, length or duration, age of post, language, topic classifier outputs, link domain quality |
| **User-author affinity** | How often the user liked, commented on, or messaged this author; profile visits; closeness in the graph; recency of last interaction |
| **Context** | Time of day, session depth (how far into the feed), connection type, surface, time since last session |
| **Real-time engagement** | Likes, comments, shares, hides in the last 5 minutes / hour; engagement velocity (rate of change); engagement from the user's friends |

**User-author affinity** is usually the strongest group for in-network content. People care most about posts from people they actually interact with, and a follow edge alone says little about that. Affinity is typically a set of decayed interaction counters per (user, author) pair, updated in streaming.

**Engagement velocity** separates a post that is taking off from one that had its moment yesterday. Compute it from streaming counters at several windows, and normalize by impressions (engagements per impression), otherwise it mostly measures how widely the post was already distributed.

**Point-in-time correctness** matters more here than in most systems, because real-time counters change every second. A training example must use counter values from the moment of the impression. Logging the served features at impression time is the simplest way to guarantee that.

---

## Re-Ranking and Feed Composition

Sorting by value score gives a list. A feed is a page, and several properties of a good page are about the set, not about individual posts.

| Concern | Typical rule or method |
|---|---|
| **Author diversity** | No more than k posts from the same author in any window of n positions; penalty that grows with repeats |
| **Content-type diversity** | Mix of video, photo, text; avoid long runs of one type |
| **Topic diversity** | MMR-style penalty on similarity to already-placed posts |
| **Freshness** | Time decay on score; boost unseen recent posts from close connections |
| **Dedup** | Collapse reshares and near-duplicate media (perceptual hash) into one item |
| **Seen-post suppression** | Drop or heavily demote posts already shown in earlier sessions |
| **Creator fairness** | Minimum exposure for new or small creators; cap on concentration of impressions among top creators |
| **Integrity demotions** | Multiply score down for borderline content, low-quality domains, likely misinformation pending review |
| **Ads insertion** | Place ads at allowed slots with minimum spacing, trading ad value against predicted organic value lost |

A simple greedy re-ranker handles most of these in one pass: at each position, pick the candidate with the best adjusted score given what is already on the page.

```python
def rerank(candidates, page_size=20, author_cap=2, window=5, sim_penalty=0.3):
    """Greedy slate construction with author caps and a similarity penalty.
    candidates: list of dicts with 'score', 'author', 'embedding' (unit norm),
    'integrity_multiplier' in (0, 1]."""
    pool = sorted(candidates, key=lambda c: c["score"] * c["integrity_multiplier"],
                  reverse=True)
    page = []
    while pool and len(page) < page_size:
        best, best_val = None, float("-inf")
        for c in pool:
            recent = page[-window:]
            if sum(p["author"] == c["author"] for p in recent) >= author_cap:
                continue
            max_sim = max((float(c["embedding"] @ p["embedding"]) for p in page),
                          default=0.0)
            val = c["score"] * c["integrity_multiplier"] - sim_penalty * max_sim
            if val > best_val:
                best, best_val = c, val
        if best is None:          # every remaining post hits a cap; relax it
            best = pool[0]
        page.append(best)
        pool.remove(best)
    return page
```

The penalty mixes score units with similarity units, so `sim_penalty` has to be tuned by experiment. The integrity multiplier is applied before diversity so a demoted post cannot win a slot just because it is different from its neighbors.

**Ads insertion.** Ads are ranked by their own system (see [Ads CTR Prediction](./ads_ctr_prediction.md)) and then placed into the organic feed. A good placement rule compares the ad's expected value with the predicted organic value lost by pushing posts down, plus a user-experience cost, and respects a minimum gap between ads. Ad load is usually tuned with long-term holdouts, because the cost of too many ads shows up as slowly falling engagement.

---

## Bias in Logged Data

Every training example comes from a feed the previous model built. That creates two biases.

**Position bias.** Posts near the top of the feed get more attention regardless of quality, and posts below where the user stopped scrolling were never seen at all. In a feed, "position" means scroll depth, and the right definition of an impression is **the post was in the viewport for some minimum time**, not "the server returned it." Treating returned-but-never-viewed posts as negatives is one of the most common data bugs in feed ranking.

Fixes carry over from search and ads (see [Position Bias](./ads_ctr_prediction.md#position-bias)): position as a training-only feature or bias tower, inverse propensity weighting with propensities measured from randomized swaps on a small traffic slice.

**Exposure (selection) bias.** You only have labels for posts the old ranker chose to show. Posts it never showed have no labels, so the new model learns nothing about them and tends to reproduce the old ranker's preferences. Mitigations:

- **Exploration traffic.** A small, randomized share of impressions (random candidates from retrieval inserted into feeds) gives unbiased labels and makes off-policy evaluation possible.
- **Logging propensities.** If ranking has any stochasticity, log the probability each post had of being shown at its position. Without logged propensities, off-policy correction is guesswork.
- **Unbiased evaluation sets.** Evaluate on exploration traffic, where the old ranker's preferences didn't decide what was shown.

---

## Negative Feedback

Negative signals are rare but carry far more information per event than a like. Treat them as first-class labels.

| Signal | Strength | How to use |
|---|---|---|
| **Scroll past quickly** | Weak, very noisy | Low-dwell label; don't treat as negative without viewport data |
| **"See less of this" / "Not interested"** | Strong, explicit | Negative head; also update user-topic and user-author affinity immediately |
| **Hide post** | Strong | Negative head with a large weight in the value model |
| **Unfollow / mute author** | Very strong, about the author | Remove author from in-network candidates; user-author affinity to zero |
| **Report** | Strongest; also an integrity signal | Negative head; feeds integrity review queues |

Two practical points. First, act on explicit feedback **within the session**. If a user hides a post and the next three posts are from the same author or topic, the feedback feels ignored. A real-time filter layer that reads recent negative actions is cheaper and faster than waiting for the model to learn. Second, negative heads need large weights in the value model, because their base rates are tiny. A well-calibrated P(hide) of 0.5% still needs to outweigh a P(like) of 5% for the posts users most dislike.

---

## Evaluation

### Offline

| Metric | What it tells you |
|---|---|
| **AUC / log loss per head** | Whether each action predictor improved; log loss also catches calibration |
| **Calibration ratio per head** | Required because heads are combined by weighted sum |
| **NDCG / recall@k on the value score** | Ordering quality of the combined score, using the value-weighted actions as graded relevance |
| **Retrieval recall per source** | Whether the posts users engaged with were in the candidate set |
| **Off-policy estimates** | Predicted online value of the new policy from logged data |

An improvement in one head's AUC does not guarantee a better feed, because the value score depends on all heads and on weights. Always evaluate the combined score too.

### Counterfactual (off-policy) evaluation

The question is "what would the reward have been if the new ranker had chosen the feed?" using logs from the old ranker. The basic estimator is inverse propensity scoring:

```
V̂_IPS(π_new) = (1/N) Σᵢ [ π_new(aᵢ | xᵢ) / π_log(aᵢ | xᵢ) ] · rᵢ
```

It is unbiased only if the logging policy gave every action the new policy might take a non-zero probability, which a deterministic ranker does not. That is why exploration traffic with logged propensities matters. In practice:

- **Clip or self-normalize** the importance weights (SNIPS) to control variance.
- **Doubly robust** estimators combine a reward model with IPS, so they stay reasonable when either the propensities or the reward model is off.
- **Slates are hard.** The action space of full feeds is huge, so exact slate propensities are tiny. Practical systems estimate per-position rewards with assumptions (such as additive rewards across positions) or evaluate at the item level.

```python
import numpy as np

def snips(rewards, p_new, p_log, clip=10.0):
    """Self-normalized IPS with weight clipping. p_log must be logged at serve time."""
    w = np.minimum(p_new / np.clip(p_log, 1e-6, None), clip)
    return float((w * rewards).sum() / w.sum())
```

Use OPE to screen candidates and pick which ones deserve an A/B test, not to replace the test.

### Online

| Metric type | Examples |
|---|---|
| **Primary** | Daily active users, sessions per user, meaningful interactions, survey-based satisfaction |
| **Engagement** | Likes, comments, shares, time spent, feed depth |
| **Guardrails** | Hide and report rates, integrity prevalence (share of views on violating content), ad revenue, latency, creator-side metrics (posting rate, reach of new creators) |

**Novelty effects.** A new ranker often shows users different content, and people engage with novelty. The treatment lift is largest in week one and can fade or reverse. Look at the effect over time, not just the average, and run important tests for several weeks.

**Long-term holdouts.** Keep a small group of users on an older ranker (or without a whole class of changes) for months. Comparing them with everyone else measures the cumulative effect of many launches that each looked small, and it catches slow harms like falling satisfaction.

**Network effects and interference.** Feed experiments break the usual assumption that one user's treatment doesn't affect another user. If treatment ranks comments higher, treated users comment more, and their friends in control get more notifications and engage more. Both arms move, and the measured difference is biased. Mitigations:

- **Graph cluster randomization:** partition the social graph into clusters with few cross-cluster edges and randomize whole clusters.
- **Creator-side experiments:** when the change affects distribution (who gets reach), randomize creators and measure creator outcomes, not only viewer outcomes.
- **Measure spillover directly:** compare control users with many treated friends against control users with few.

---

## Serving Architecture

```
                           NEWS FEED RANKING
═══════════════════════════════════════════════════════════════════════════

 Client ──► Feed Service ──┬──► In-network: Inbox store (pushed) ──┐
                           │                + Author cache (pulled)│
                           ├──► Out-of-network: ANN index,         ├─► Merge,
                           │    graph walks, trending               │   filter
                           │                                        │  (blocks,
                           ├──► Real-time filter ◄──────────────────┘   seen,
                           │    (recent hides, blocks, deletes)         deleted)
                           │                  │
                           │                  ▼
                           ├──► Feature Store ─► Light Ranker ─► Heavy Ranker
                           │    (user, author,                   (multi-task,
                           │     post, affinity,                  GPU, batched)
                           │     streaming counters)                  │
                           │                                          ▼
                           ├──► Integrity scores ──────────────► Re-ranker ◄── Ads system
                           │                                          │
                           ▼                                          ▼
                       Response ◄──────────────────────────────── Feed page
                           │
                           ▼
    Viewport impressions, engagements, negative feedback ──► Kafka
                           │                                    │
                           ▼                                    ▼
         Streaming aggregates (velocity,             Training data (logged
         affinity, counters) ──► Feature Store       features + labels)
                                                                │
                                                                ▼
                                           Continuous training ──► Validation ──► Push

═══════════════════════════════════════════════════════════════════════════
```

**Real-time features.** A stream processor consumes engagement events and updates per-post counters (at several time windows), per-(user, author) affinity, and per-user session state. These land in a low-latency key-value store. Keep the same aggregation code for training and serving, or log served values, to avoid training/serving skew.

**Caching.**
- **Post features and embeddings** are shared across many requests; cache them on ranking hosts with short TTLs.
- **Precomputed feeds.** Some systems rank a feed ahead of time (for example when a user is likely to open the app soon) and serve it on request, re-ranking lightly with fresh signals. This cuts request latency but spends compute on sessions that never happen.
- **Pagination.** Rank a larger slate on the first request and serve later pages from it, refreshing when the user pulls to refresh or when the slate is stale.

**Deletes and blocks.** Caches must never serve a deleted post or a post from a blocked author. Apply a final filter at response time against a fast deletion and block set, regardless of what upstream caches return.

**Fallbacks.** If the heavy ranker times out, serve light-ranker order. If candidate generation fails, fall back to a chronological in-network feed. An empty feed is the worst outcome.

---

## Capacity and Cost

Work these through out loud, stating assumptions.

```
Request rate:
  500M DAU × 20 requests/day = 10B requests/day
  10B / 86,400 s ≈ 116k requests/s average; assume ~3x at peak ≈ 350k/s

Heavy ranking load:
  350k req/s × 500 candidates ≈ 175M post scorings/s at peak
  → batch per request on GPU; the light ranker's 5k → 500 cut exists to
    keep this affordable; ranking larger slates less often (pagination)
    cuts it further

Fan-out on write (illustrative):
  If 100M posts/day go through push with an average of 300 active followers
  → 30B inbox writes/day ≈ 350k writes/s average
  → hence skipping inactive followers and pulling for large accounts
```

---

## Cold Start

Generic strategies (onboarding, content features, bandits) are covered in [Handling Cold Start](./recommendation_system.md#handling-cold-start). The feed-specific points:

**New users** have no graph yet, so the connected feed is empty. The feed is interest-based by default at first. Use onboarding topic choices, contacts or friend suggestions (with consent), and popular posts in their region and language. Suggesting accounts to follow is part of solving feed cold start, since each follow adds in-network candidates. Adapt within the session from dwell and early engagement.

**New posts** have no engagement, and engagement velocity is one of the strongest features. For in-network posts, the author and affinity features carry the prediction. For out-of-network distribution, a common pattern is **staged distribution**: show the post to a small audience that is likely to be interested (such as the author's followers or a matched interest cluster), measure engagement per impression, and widen the audience if it performs. This is exploration with a budget, and it is where integrity checks should run before large-scale reach.

**New creators** face both problems. Without exposure they get no engagement, and without engagement they get no exposure. Creator-fairness rules in re-ranking (a minimum exposure budget for new creators) break the loop.

---

## Feedback Loops, Integrity and Wellbeing

The ranker decides what people see, which decides what they engage with, which becomes the next training set. Creators also learn what the ranker rewards and produce more of it.

| Failure | Cause | Mitigation |
|---|---|---|
| **Engagement bait** | Posts ask for likes/comments/shares to game heads | Classifiers for bait patterns; demote; don't count bait-driven actions |
| **Outrage amplification** | Comments and shares are high on divisive content | Weight comments by quality (e.g. between friends); survey heads; integrity demotions |
| **Filter bubbles** | Model narrows to what the user already engaged with | Topic diversity in re-ranking, exploration slots, monitor topic entropy per user |
| **Popularity concentration** | Large creators get more impressions, better-trained embeddings, and more impressions again | Creator fairness constraints, exposure caps, new-creator budgets |
| **Returned ≠ seen labels** | Unviewed posts logged as negatives | Viewport-based impression logging |
| **Creator gaming of weights** | Creators optimize for whichever action is weighted highest | Change weights deliberately, audit what content gains reach after each change |
| **Harmful content going viral** | Engagement velocity boosts it before review | Velocity-triggered review; cap reach until integrity classifiers clear it |
| **Problematic use** | Endless feed optimized for time spent | Wellbeing guardrails, "you're all caught up" markers, time-spent tools, age-appropriate defaults |

**Integrity integration.** Integrity classifiers (see [Content Moderation System Design](./content_moderation_system.md)) produce two kinds of output. Clear violations are removed and never reach ranking. Borderline content (sensational, low-quality, likely-false but unconfirmed) is **demoted**: its score is multiplied down in re-ranking. Prevalence, the share of all feed views that land on violating content, is the standard way to measure whether this works, since it reflects what users actually saw.

**Wellbeing.** Some objectives are hard to express as a per-post prediction, such as how users feel about their time on the platform, or effects on younger users. They show up as survey metrics, guardrails in experiments, and product decisions (defaults, time limits, chronological options) rather than as model heads. In an interview, naming these trade-offs and saying they belong in the decision criteria, not just the model, is what separates a strong answer.

---

## Interview Q&A

#### How do you combine likes, comments, shares, and hides into one ranking score?

Predict each action's probability with a multi-task model, calibrate every head, and combine them with a weighted sum in which negative actions (hide, report, unfollow) get negative weights. The weights are a product decision. They express how much the platform values each action, and they are tuned by online experiments against long-term metrics, with guardrails on negative feedback and integrity.

Calibration matters because the sum only means something if each probability is on the right scale. If a model update inflates P(share) by 50%, the effective share weight rises by 50% without anyone deciding it. So check calibration per head, per surface, and per content type after every model change.

#### Why not just optimize time spent?

Because time spent is easy to increase with content people regret: outrage, clickbait, endless low-effort video. Engagement rises for a while, then satisfaction and retention fall, and the damage shows up only in long-term metrics. Time spent also treats a minute of passive scrolling the same as a minute of conversation with a friend.

A better objective combines several actions, includes negative feedback, and adds heads trained on survey responses about whether posts were worth the user's time. Launch decisions should be based on long-term retention and satisfaction, with time spent as one metric among several.

#### Fan-out on write or fan-out on read?

Hybrid. Fan-out on write makes reads cheap, since each user's inbox is precomputed, which matters because reads far outnumber posts. But for an account with tens of millions of followers, one post means tens of millions of writes, which is slow and wasteful when most of those followers won't open the app soon.

So push for regular accounts, skip inactive followers, and pull for accounts above a follower threshold. At read time, merge the user's pushed inbox with recent posts from the large accounts they follow, fetched from a per-author cache that stays hot because many users read it. The threshold is tuned by comparing write cost with read-time merge cost.

#### Your new ranker wins the A/B test by 2% on engagement in week one. Do you ship?

Not yet. First, check whether the lift holds over time. New rankers often show different content and get a novelty boost that fades. Plot the treatment effect by day and run the test for several weeks.

Second, check guardrails: hide and report rates, integrity prevalence, survey satisfaction, and creator-side metrics. A ranker that raises engagement by surfacing more borderline content is a regression. Third, consider interference. If the change affects how much users post or comment, control users are affected through their friends, so the measured difference may be biased. For changes like that, use graph cluster randomization. Finally, ship with a long-term holdout so the cumulative effect can be measured months later.

#### How do you evaluate a new ranker offline when you only have logs from the old one?

Per-head AUC and log loss tell you whether predictions improved on posts the old ranker chose to show, but not whether the new ranker would choose better posts. For that you need off-policy evaluation: reweight logged rewards by how much more or less likely the new policy is to take the logged action (IPS), with clipping or self-normalization to control variance, or a doubly robust estimator that also uses a reward model.

This requires logged propensities and some exploration, since IPS cannot say anything about actions the logging policy never takes. Full-slate OPE has very high variance, so in practice teams make simplifying assumptions (per-position rewards) and use OPE to decide which candidates are worth an online test.

#### A user hides a post. What should happen?

Several things at different speeds. Immediately, a real-time filter should remove similar posts from the rest of the session: other posts from that author, reshares of the same content, and possibly the same topic. Within minutes, streaming updates should lower the user-author and user-topic affinity features. At the next training run, the hide is a positive label for the hide head.

If hides cluster on one author or post across many users, that's also a signal for integrity and quality systems, and for demoting the post more broadly.

#### How do you make sure small creators get a fair chance?

Without intervention, exposure concentrates. Large creators have more engagement data, better embeddings, and higher predicted scores, so they get more impressions and more data. Mitigations: content features (text, image, and video embeddings) so new posts get sensible predictions without history, staged distribution that tests a new post on a small matched audience before widening, a minimum exposure budget for new creators in re-ranking, and caps on how concentrated impressions can become.

Measure it with creator-side metrics: the distribution of reach across creators, new-creator retention, and the share of impressions going to creators outside the user's network.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Ranking by a single engagement label | Rewards clickbait and outrage | Multi-head value model with negative terms and survey heads |
| Uncalibrated heads in a weighted sum | Model changes silently change effective weights | Calibrate and monitor each head |
| Counting returned posts as impressions | Unseen posts become false negatives | Viewport-based impression logging |
| Pure fan-out on write | Celebrity posts cause write storms | Hybrid push/pull with a follower threshold |
| Ignoring user-author affinity | Friends' posts lose to generic viral content | Decayed per-pair interaction features |
| Deciding launches on week-one engagement | Novelty effects fade; long-term harms missed | Multi-week tests, long-term holdouts |
| User-level A/B for social changes | Spillover through the graph biases results | Graph cluster randomization, spillover measurement |
| No exploration or logged propensities | OPE impossible; ranker reproduces itself | Small randomized slice; log propensities |
| Ignoring explicit negative feedback in-session | Users see more of what they just hid | Real-time filter on recent negative actions |
| Diversity only by author | Same topic or media repeated | Topic and content-type diversity; near-duplicate collapse |
| Integrity as a separate afterthought | Borderline content goes viral before review | Demotions in re-ranking; velocity-triggered review |
| No creator-side metrics | Concentration and creator churn go unnoticed | Track reach distribution and new-creator outcomes |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Recommendation System Design](./recommendation_system.md)
- [Ads CTR Prediction and Ranking](./ads_ctr_prediction.md)
- [Search and Ranking System Design](./search_ranking_system.md)
- [Content Moderation System Design](./content_moderation_system.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Causal Inference and Uplift](../classical_ml/intro_causal_inference.md)
- [Embeddings](../ai_genai/intro_embeddings.md)
- [Vector Databases: Advanced](../ai_genai/intro_vector_databases_advanced.md)
- [Apache Kafka](../data_engineering/intro_apache_kafka.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Feature Store](../mlops/intro_feature_store.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md)
