# Ads CTR Prediction and Ranking System Design

"Design an ads click-through-rate prediction system" looks like a binary classification problem, and candidates who treat it as one tend to do badly. The model's output is not a ranking score. It is a probability that goes into an auction and sets what advertisers pay. That one fact explains most of the design: calibration is mandatory, log loss beats AUC, negative downsampling has to be corrected, and the feedback loops are about money as well as relevance.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [Business Objective and Auction Basics](#business-objective-and-auction-basics)
3. [The Ads Funnel](#the-ads-funnel)
4. [Labels and Delayed Feedback](#labels-and-delayed-feedback)
5. [Position Bias](#position-bias)
6. [Features](#features)
7. [Model Progression](#model-progression)
8. [Calibration](#calibration)
9. [Evaluation](#evaluation)
10. [Training at Scale](#training-at-scale)
11. [Serving Architecture](#serving-architecture)
12. [Capacity and Cost](#capacity-and-cost)
13. [Exploration and Cold Start](#exploration-and-cold-start)
14. [Feedback Loops and Failure Modes](#feedback-loops-and-failure-modes)
15. [Privacy](#privacy)
16. [Interview Q&A](#interview-qa)
17. [Common Pitfalls](#common-pitfalls)
18. [Related Topics](#related-topics)

---

## Clarify the Problem First

Ads prompts are underspecified on purpose. Establish these before drawing any boxes.

**What surface?** Search ads (there is a query, so intent is explicit), feed or social ads (no query, so intent comes from user history), or display ads on third-party sites (often bought through an external exchange). The surface decides the strongest features and whether you run the auction or bid into someone else's.

**How do advertisers pay?** Cost per click (CPC), cost per thousand impressions (CPM), or cost per action/conversion (CPA). This decides which probabilities the system must predict. CPC needs pCTR; CPA needs pCTR × pCVR.

**Who runs the auction?** If you own it, you choose the auction rules and the pricing. If you are a bidder on an exchange, you need a bidding strategy and possibly bid shading.

**Scale numbers to ask for:**
- Ad requests per second, peak and average
- Number of active ads and advertisers (this sizes the candidate pool)
- Latency budget: the whole ad response usually has to fit inside the page's latency, so scoring gets a slice, typically in the 10 to 50 ms range
- How fast a new campaign must start serving, and how fast budget changes take effect

**Constraints:** advertiser targeting rules, policy and brand-safety filters, frequency caps, regulatory limits on sensitive categories, and what user data can be used at all.

For the rest of this guide, assume a concrete brief: **ads in a social feed, 500k ad requests/sec at peak, ~10M active ads, CPC and CPA billing, we run a second-price-style auction, p99 of 30 ms for the full ranking stack, new campaigns eligible within minutes.** These are working assumptions for the exercise, not industry figures.

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Scoring latency (p99) | ~30 ms for retrieval + ranking + auction |
| Throughput | 500k requests/sec peak |
| Availability | Ads failing open (show no ad) is acceptable; wrong charges are not |
| Calibration | Predicted clicks within a few percent of observed, per segment |

---

## Business Objective and Auction Basics

### What are we maximizing?

Three parties have to be served at once:

- **The platform:** revenue, long term, not just today's.
- **Advertisers:** return on ad spend. If ads don't convert, budgets shrink.
- **Users:** relevance and experience. Bad ads drive users away, which cuts future inventory.

The standard ranking score is **expected value per impression**, usually expressed as eCPM (effective cost per mille, i.e. value per 1000 impressions):

```
CPM bid:   eCPM = bid_cpm
CPC bid:   eCPM = bid_cpc × pCTR × 1000
CPA bid:   eCPM = bid_cpa × pCTR × pCVR × 1000
```

This puts advertisers who pay by different events on one comparable scale. Many systems add a quality or user-experience term, for example a penalty for predicted hides or negative feedback:

```
score = eCPM + λ · (predicted user value) − μ · (predicted negative feedback)
```

The weights λ and μ are business decisions, tuned by long-running experiments. They are not learned by the CTR model.

### Auction types

| Auction | Winner pays | Properties |
|---|---|---|
| **First-price** | Their own bid | Simple, transparent; bidders have an incentive to shade bids below true value |
| **Second-price (Vickrey, single slot)** | Second-highest bid | Truthful bidding is a dominant strategy for a single item |
| **Generalized second price (GSP)** | Minimum needed to keep their slot | Multi-slot extension; not strictly truthful, but widely used for search ads |
| **VCG** | Externality imposed on others | Truthful for multiple slots; harder to explain and compute |

In a quality-weighted GSP with CPC bids, ads are sorted by `bid × pCTR`, and ad *i* pays per click just enough to beat the ad below it:

```
price_per_click_i = (bid_{i+1} × pCTR_{i+1}) / pCTR_i
```

**This is why calibration is not optional.** pCTR appears in the ranking *and* in the price. If pCTR is inflated for one advertiser, they win auctions they shouldn't and are charged the wrong amount. A model with great AUC and bad calibration still ranks well within one advertiser's ads, but it misprices the whole marketplace.

Display ad exchanges moved largely to first-price auctions (partly because of header bidding), so bidders there use **bid shading**: learning how far below their value they can bid and still win. For an interview, it is enough to know both exist and that the CTR model's job is the same in both. It supplies a calibrated probability.

---

## The Ads Funnel

The same cost-versus-accuracy logic as search: cheap models over many candidates, expensive models over few.

```
Ad request (user, context, placement)
  │
  ▼
┌──────────────────────────┐   10M ──► ~10k
│ Targeting & retrieval    │  inverted index on targeting rules (geo, age,
└────────────┬─────────────┘  interests) + embedding ANN; budget/policy filters
             ▼
┌──────────────────────────┐   10k ──► ~500
│ Lightweight ranking      │  two-tower or small model, cheap features,
└────────────┬─────────────┘  approximate eCPM
             ▼
┌──────────────────────────┐   500 ──► ~50
│ Heavy ranking            │  deep CTR/CVR model, full feature set,
└────────────┬─────────────┘  calibrated pCTR, pCVR, p(negative feedback)
             ▼
┌──────────────────────────┐
│ Auction                  │  score = bid × pCTR (× pCVR) + quality terms
└────────────┬─────────────┘  reserve prices, pricing rule, slot allocation
             ▼
┌──────────────────────────┐
│ Pacing & budget control  │  throttle or scale bids so budgets last the day
└────────────┬─────────────┘  frequency caps, advertiser diversity
             ▼
        Served ad(s) ──► impression / click / conversion logs
```

**Targeting is a hard filter.** An advertiser who targeted "US, 25 to 34" must never be shown to someone outside that group, however high the pCTR. Apply targeting inside retrieval, not after ranking.

**Budget and pacing interact with ranking.** An advertiser who spends their whole daily budget by 9 a.m. misses the evening audience and is unhappy even with a high ROI. Pacing usually works in one of two ways:

- **Probabilistic throttling:** the ad takes part in only a fraction of eligible auctions, and that fraction is adjusted to track a target spend curve.
- **Bid multipliers:** effective bids are scaled down while spend runs ahead of plan, often by a feedback controller such as PID. This is a separate control loop reading near-real-time spend, not part of the CTR model.

---

## Labels and Delayed Feedback

### Click labels

An impression becomes a positive if a click occurs within an **attribution window**, which for clicks is short (seconds to minutes). The label pipeline joins impression logs to click logs by impression ID. Issues that come up:

- **Invalid traffic:** bot clicks and accidental clicks (for example, clicks within milliseconds of render) should be filtered, or the model learns to predict fraud.
- **Late joins:** a click that arrives after the impression was already written as a negative. The streaming join needs a waiting window, and it must decide what to do with events that arrive after it closes.
- **Impression definition:** "served" is not "viewed." Training on served-but-never-viewed impressions adds noise. Many systems use a viewability threshold.

### Conversion labels and delay

Conversions (purchase, install, signup) can happen hours or days after the click, and attribution windows are often measured in days. That creates a real tension:

- **Wait for the full window:** labels are correct but the model trains on stale data, which hurts in a fast-moving marketplace.
- **Train immediately:** fresh data, but recent clicks that will later convert are labeled negative. The model underpredicts CVR, most of all for new campaigns.

Practical approaches:

| Approach | Idea |
|---|---|
| **Fixed wait window** | Only train on clicks older than *k* hours; accept some label noise beyond *k* |
| **Fake negative + correction** | Ingest as negative immediately, re-ingest as positive when conversion arrives; correct the resulting bias with importance weights |
| **Delay modeling** | Jointly model P(convert) and the delay distribution, so an unconverted recent click counts as "not yet," not "never" |
| **Multi-horizon labels** | Predict conversion within 1h, 1d, 7d as separate heads; short heads train fresh, long heads train late |

Fraud detection has the same structure with chargebacks, but here freshness matters more, so fixed wait windows are rarely enough on their own.

---

## Position Bias

Ads in the top slot get more clicks regardless of quality. If position is ignored, the model learns "ads that were placed high are good." That entrenches whatever the previous model preferred.

The problem has a specific twist for ads: at serving time the model is scoring ads **before** positions are assigned, since the auction assigns them. So you cannot feed the true position as an input at inference.

Common fixes:

1. **Position as a training-only feature.** Include position (and slot type, device) as input during training. At serving, set it to a fixed reference value, such as position 1, for all candidates. The model learns to attribute part of the CTR to position, so the remaining signal is closer to intrinsic attractiveness.
2. **Separate bias tower.** A shallow tower takes only position/context features and outputs a logit that is added to the main model's logit during training, then dropped at serving. This keeps position from interacting with content features in ways that won't be available later.
3. **Inverse propensity weighting.** Estimate examination probability per position, from randomized swaps on a small slice of traffic, and reweight training examples.
4. **Factorization at pricing time.** Model pCTR(ad, position) ≈ pCTR(ad) × P(examine | position). The auction then uses the position-specific estimate for each slot.

```python
import torch
import torch.nn as nn

class CTRWithPositionTower(nn.Module):
    """Main tower learns ad/user relevance; bias tower absorbs position effects."""
    def __init__(self, main_tower: nn.Module, n_positions: int):
        super().__init__()
        self.main = main_tower
        self.position_bias = nn.Embedding(n_positions, 1)

    def forward(self, features, position=None):
        logit = self.main(features)
        if self.training and position is not None:
            logit = logit + self.position_bias(position).squeeze(-1)
        return logit                  # at serving: no position term
```

---

## Features

| Group | Examples |
|---|---|
| **User** | User ID embedding, demographics (where permitted), interest categories, historical CTR, recent engagement sequence |
| **Ad** | Ad ID, campaign ID, advertiser ID embeddings, creative type, text/image embeddings, landing page category, ad age |
| **Context** | Placement, device, OS, time of day, day of week, connection type, page/query content |
| **Cross (user × ad)** | User's historical CTR on this advertiser/category, user interest × ad category, query × ad keyword match |
| **Counters** | Smoothed CTR per ad, per advertiser, per (placement, ad category) over several windows |

### High-cardinality IDs

User, ad, advertiser, and keyword IDs are the most predictive features and the hardest to handle: millions to billions of values, most seen rarely.

**Hashing trick.** Map each (feature name, value) pair to one of *M* buckets. There's no vocabulary to maintain, new IDs work immediately, and memory is fixed. The cost is collisions, which are usually fine when *M* is large compared with the number of active values.

```python
import mmh3   # MurmurHash3

def hashed_index(field: str, value: str, n_buckets: int = 2**24) -> int:
    """Stable hash of field:value into a fixed-size index space."""
    return mmh3.hash(f"{field}={value}", signed=False) % n_buckets

def cross(field_a, val_a, field_b, val_b, n_buckets=2**24):
    """Hashed feature cross, e.g. user_country x ad_category."""
    return hashed_index(f"{field_a}_x_{field_b}", f"{val_a}|{val_b}", n_buckets)
```

**Embeddings.** In deep models each hashed ID indexes a row in a learned embedding table. The embedding tables are usually most of the model's parameters and memory, far more than the dense layers.

**Smoothed counters.** Raw CTR for a new ad with 1 click in 2 impressions is 50%, which is meaningless. Use a Bayesian-smoothed rate: `(clicks + α) / (impressions + α + β)`, where the prior comes from the ad's category or advertiser. Compute counters **point in time**, so a training example only sees counts from before its impression. Otherwise the label leaks into the feature.

---

## Model Progression

Walking through the history shows you understand *why* each step was taken. That's more useful in an interview than naming the newest architecture.

| Model | What it adds | Limitation |
|---|---|---|
| **Logistic regression + hashed crosses** | Fast, calibrates well, scales to billions of sparse features; online-learnable (e.g. FTRL) | Every useful interaction must be hand-crafted as a cross |
| **GBDT + LR** | Trees learn nonlinear feature combinations; each tree's leaf index becomes a sparse feature for LR | Trees are retrained in batch; poor with very high-cardinality IDs |
| **FM (factorization machines)** | Learns every pairwise interaction via dot products of latent vectors; generalizes to unseen pairs | Only pairwise; one latent vector per feature for all interactions |
| **FFM (field-aware FM)** | Separate latent vector per (feature, other field) | Parameter count grows with number of fields |
| **Wide & Deep** | Wide linear part memorizes crosses; deep MLP over embeddings generalizes | Wide part still needs manual crosses |
| **DeepFM** | Replaces the wide part with an FM sharing embeddings with the deep part | Interactions beyond pairwise left to the MLP |
| **DCN / DCN-v2** | Explicit cross layers learn bounded-degree interactions efficiently | Tuning cross depth and rank |
| **Multi-task (MMoE, shared-bottom, ESMM)** | Predicts CTR, CVR, negative feedback jointly; shares representation, handles sparse CVR labels | Task conflicts; more complex calibration per head |

### Factorization machines

```
ŷ = w₀ + Σᵢ wᵢ xᵢ + Σᵢ Σ_{j>i} ⟨vᵢ, vⱼ⟩ xᵢ xⱼ
```

The pairwise term can be computed in O(k·n) rather than O(k·n²) using the identity `Σ_{i<j} ⟨vᵢ,vⱼ⟩xᵢxⱼ = ½ Σ_f [(Σᵢ v_{i,f} xᵢ)² − Σᵢ v_{i,f}² xᵢ²]`. Because the interaction weight is a dot product of learned vectors, the model can score a (user segment, ad category) pair it has never seen together, provided it has seen each one with other partners.

### DCN cross layer

```
x_{l+1} = x₀ ⊙ (W_l x_l + b_l) + x_l          (DCN-v2 form)
```

Each layer raises the polynomial degree of interactions by one, and the residual connection keeps lower-order terms. Stacking a few cross layers next to (or before) an MLP is a strong default for heavy ranking.

### Multi-task: CTR + CVR

CVR has two problems. Its labels are rare, and it is only observed on clicked impressions, while at serving it has to be predicted for all impressions (sample selection bias). Two common patterns:

- **ESMM-style:** train pCTR on all impressions and pCTCVR = pCTR × pCVR on all impressions, with pCVR as an intermediate that is never directly supervised on the biased click-only set.
- **MMoE:** several shared expert networks, and a per-task softmax gate over experts. Tasks that disagree can use different experts instead of fighting over one shared bottom.

```python
class MMoE(nn.Module):
    def __init__(self, input_dim, n_experts=4, expert_dim=128, tasks=("ctr", "cvr")):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU())
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleDict({t: nn.Linear(input_dim, n_experts) for t in tasks})
        self.towers = nn.ModuleDict({t: nn.Linear(expert_dim, 1) for t in tasks})

    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)   # (B, E, D)
        logits = {}
        for task, gate in self.gates.items():
            w = torch.softmax(gate(x), dim=-1).unsqueeze(-1)            # (B, E, 1)
            logits[task] = self.towers[task]((w * expert_out).sum(1)).squeeze(-1)
        return logits   # per-task logits; each needs its own calibration
```

---

## Calibration

### Why it's mandatory

In most classification problems a monotone transform of the score changes nothing, because only the ranking matters. In ads, the absolute value of pCTR is multiplied by bids from *different advertisers* and used to set *prices*. If the model is systematically 20% high for mobile video ads, those ads win auctions they should lose and pay prices computed from the wrong probability. The auction stops allocating slots to the ads that are actually most valuable.

The calibration check:

```
calibration ratio = Σ predicted pCTR / Σ observed clicks      (target ≈ 1.0)
```

Check it overall **and per segment**: by advertiser, placement, device, ad age, and pCTR bucket. A model can be perfectly calibrated overall while being 30% off for new ads.

### Negative downsampling correction

Click-through rates are low, so negatives are usually downsampled to cut training cost. If negatives are kept at rate *w* (e.g. *w* = 0.1 keeps 10%), the model's predicted odds are inflated by a factor of 1/*w*. Undo it at serving:

```
p = p' / (p' + (1 − p') / w)
```

where p' is the model's raw prediction and p is the corrected probability. For example, p' = 0.10 with w = 0.1 gives p = 0.1 / (0.1 + 9) ≈ 0.011.

```python
def correct_downsampling(p_prime: float, w: float) -> float:
    """Undo negative downsampling at rate w (fraction of negatives kept)."""
    return p_prime / (p_prime + (1.0 - p_prime) / w)
```

Forgetting this correction is a classic bug: offline AUC is unchanged, while the eCPM of every CPC- or CPA-bid ad is inflated by roughly 1/*w* for low-CTR ads. Those ads then beat CPM-bid ads and clear reserve prices they shouldn't.

### Post-hoc calibration

Even without downsampling, deep models trained with regularization, early stopping, or multi-task losses drift from calibration. Fit a calibrator on recent held-out data:

| Method | How | When |
|---|---|---|
| **Platt scaling** | Fit `σ(a·logit + b)` | Few parameters, stable, good when miscalibration is a smooth shift |
| **Isotonic regression** | Monotone piecewise-constant map | Flexible; needs more data; can overfit in sparse regions |
| **Per-segment calibration** | Separate Platt/isotonic per placement or device | When miscalibration differs by segment |
| **Calibration layer in model** | Learned per-segment bias on the logit, trained continuously | When the model is updated online |

Refresh calibrators as often as the model, since calibration drifts faster than ranking quality.

---

## Evaluation

### Offline

| Metric | What it tells you |
|---|---|
| **Log loss** | Primary. Penalizes both poor ranking and poor calibration |
| **Normalized entropy (NE)** | Log loss divided by the entropy of the background CTR. Below 1.0 means better than predicting the average; comparable across datasets with different base rates |
| **AUC** | Ranking quality only; insensitive to calibration. Useful as a secondary check |
| **Calibration ratio** | Σ predicted / Σ observed, overall and per segment |
| **Reliability diagram** | Predicted vs observed CTR by pCTR bucket |

```python
import numpy as np

def normalized_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    ctr = y.mean()
    background = -(ctr * np.log(ctr) + (1 - ctr) * np.log(1 - ctr))
    return log_loss / background
```

Evaluate on a **time-based split**: train on days 1 to N, test on day N+1. Random splits leak future counters and trends into training and overstate gains.

### Online

Offline log loss does not guarantee online gains. The auction, pacing, and advertiser behavior all react to the new model. Always A/B test.

| Metric type | Examples |
|---|---|
| **Platform** | Revenue per thousand requests, auction clearing price, fill rate |
| **Advertiser** | Cost per conversion, conversion rate, ROI, advertiser retention and spend growth |
| **User (guardrails)** | Ad hide/report rate, session length, long-term engagement, ad load tolerance |
| **Model health** | Online calibration ratio, pCTR distribution, latency |

**Marketplace experiments are tricky.** If treatment and control share advertiser budgets, a model that makes treatment spend faster drains the budget that control would have used, and treatment looks better than it is. Mitigations: budget-split experiments (each arm gets its own slice of each advertiser's budget) or advertiser-level randomization. Also watch long-term user metrics. Showing more clickable but lower-quality ads can raise revenue for weeks while slowly eroding engagement.

---

## Training at Scale

### Data

```
Impression logs ─┐
Click logs ──────┼──► Streaming join (impression_id, wait window) ──► Labeled examples
Conversion logs ─┘                                                        │
                                                   Feature snapshot at impression time
                                                                          ▼
                                     Downsample negatives (rate w, logged per example)
                                                                          ▼
                                       Continuous trainer ──► Model checkpoints
```

Log the **features as served** at impression time, rather than recomputing them later from tables. That guarantees training and serving see the same values and removes a big source of training/serving skew.

### Freshness and continuous learning

Ad inventory changes constantly: new campaigns, new creatives, seasonal events, sudden trends. A model trained once a week falls noticeably behind. The usual pattern:

- **Periodic full retrain** (e.g. daily or weekly) on a long window, to reset and allow architecture changes.
- **Incremental / online updates** from the full model, on streaming data every few minutes to hours, pushed to serving as new checkpoints.
- **Embedding tables updated most often**, since new ad and user IDs appear continuously. Dense layers can be updated less often.

Risks with continuous training: a bad data batch (a logging bug, a bot attack) can poison the live model within minutes. Guard with automated validation before each push (log loss on a fresh holdout, calibration ratio, prediction distribution checks) and keep fast rollback to the previous checkpoint.

### Embedding tables

With hashed IDs, the tables can reach hundreds of GB. Common techniques:

- Shard embedding tables across parameter servers or across GPUs (model parallel), while dense layers are data-parallel.
- Use lower dimensions for low-frequency features, and mixed or reduced precision for storage.
- Evict or merge rows for IDs that haven't appeared recently.

---

## Serving Architecture

```
                         ADS CTR PREDICTION AND RANKING
═══════════════════════════════════════════════════════════════════════════

 Client ──► Ad Server ──┬──► Targeting / Retrieval ──► Candidate ads (~10k)
                        │        (inverted index, ANN, budget & policy filter)
                        │                     │
                        │                     ▼
                        │          Lightweight Ranker (~500)
                        │                     │
                        │                     ▼
                        ├──► Feature Store ─► Heavy Ranker (GPU/CPU, batched)
                        │    (user, ad,        pCTR, pCVR, p(neg)  ──► Calibrator
                        │     counters)                                    │
                        │                                                  ▼
                        ├──► Pacing Service ──────────────────────►  Auction
                        │    (bid multipliers,                    (score, price,
                        │     throttle rates)                      slot allocation)
                        │                                                  │
                        ▼                                                  ▼
                    Response ◄───────────────────────────────── Winning ad(s)
                        │
                        ▼
        Impression / click / conversion events ──► Kafka ──► Label join
                                                                 │
                                                                 ▼
                                      Continuous training ──► Validation ──► Model push
                                                                 │
                                                        Budget / spend aggregator
                                                        (feeds Pacing Service)

═══════════════════════════════════════════════════════════════════════════
```

**Feature store.** User features are fetched once per request. Ad features are fetched once per candidate, but ads are shared across many requests, so they can be cached locally on ranking servers with short TTLs. Real-time counters (last-hour CTR, spend so far) come from a streaming aggregator backed by a low-latency key-value store.

**Model latency.** Batch all candidates for a request into one inference call. Precompute what doesn't depend on the user: ad-side tower outputs in a two-tower light ranker can be cached per ad. Use quantization or distillation to fit the heavy ranker in budget.

**Caching.** User embeddings can be cached for seconds to minutes within a session. Full-score caching is rarely safe, since pCTR depends on context and bids change constantly.

**Fallbacks.** If the heavy ranker times out, fall back to light-ranker scores with a conservative calibration, or show no ad. Never charge advertisers based on a stale or defaulted pCTR without logging it.

---

## Capacity and Cost

Say these estimates out loud; interviewers look for this.

```
Heavy-ranker inference:
  500k req/s × 500 candidates = 250M ad scorings/s at peak
  → Too many for per-candidate calls; batch per request, then
    batch requests on GPU. The light ranker's cut from 10k to 500
    exists precisely to keep this number affordable.

Embedding memory (hashed, illustrative):
  User table:  1B rows × 64 dims × 4 B  ≈ 256 GB
  Ad table:    100M rows × 64 dims × 4 B ≈ 26 GB
  → Shard across parameter servers; fp16 halves it; frequency
    thresholds and eviction cut it further.

Training data:
  500k req/s × ~1 ad each × 86,400 s ≈ 43B impressions/day
  → Downsample negatives to w = 0.05-0.1 → a few billion examples/day
```

---

## Exploration and Cold Start

A new ad has no click history, so its ID embedding is untrained and its counters are empty. If the model is conservative it scores low, gets no impressions, and never gathers the data it needs to prove itself.

| Technique | How it helps |
|---|---|
| **Content features** | Creative text/image embeddings, advertiser history, landing page category give a meaningful prior from the start |
| **Hierarchical priors** | Back off from ad → campaign → advertiser → category for counters and embeddings |
| **Exploration budget** | Reserve a small share of impressions for under-explored ads |
| **Bandit-style boosts** | Add an uncertainty bonus (UCB) or sample pCTR from a posterior (Thompson sampling) for ads with few impressions |
| **New-ad calibration** | Track calibration specifically for ads < N impressions; new ads are often systematically mispredicted |

```python
import numpy as np

def thompson_pctr(clicks, impressions, prior_ctr, prior_strength=100):
    """Sample pCTR from a Beta posterior centered on a category prior."""
    alpha = prior_ctr * prior_strength + clicks
    beta = (1 - prior_ctr) * prior_strength + (impressions - clicks)
    return np.random.beta(alpha, beta)
```

Exploration costs revenue in the short term, and in a paid auction there is a fairness question: who pays for the exploration? A common answer is to boost *ranking* for exploration without inflating the *price* charged. That means computing the price from an unboosted pCTR.

---

## Feedback Loops and Failure Modes

The model decides what gets shown, and what gets shown becomes the training data. That loop creates most of the long-term failure modes.

| Failure | Cause | Mitigation |
|---|---|---|
| **Rich-get-richer** | Ads with more impressions have better-trained embeddings and win more | Exploration, uncertainty-aware scoring, content features |
| **Position entrenchment** | Top-slot ads look better because they are in the top slot | Position debiasing; randomized swaps on a small slice |
| **Calibration drift** | Seasonality, new placements, downsampling rate change | Per-segment calibration monitoring, frequent calibrator refresh |
| **Clickbait creatives** | High CTR but low conversion and poor user experience | Optimize eCPM with CVR and negative-feedback terms, not CTR alone |
| **Advertiser gaming** | Advertisers adapt creatives and bids to the model | Policy review, quality scores, monitoring sudden score shifts |
| **Bot / invalid clicks** | Fraudulent clicks in training labels | Invalid traffic filtering before label join; refund invalid clicks |
| **Poisoned online update** | Logging bug or attack enters continuous training | Pre-push validation, holdout checks, instant rollback |
| **Budget-exhaustion bias** | Budget-capped ads disappear from logs late in the day | Account for pacing when analyzing data; log eligibility, not only wins |
| **Training/serving skew** | Features recomputed differently offline | Log served features; train on logged snapshots |

---

## Privacy

Ads systems depend heavily on cross-context user data, and that data is getting harder to use. At a general level:

- **Third-party cookies** are blocked by default in some browsers, and support in others has been subject to changing plans. Cross-site tracking for targeting and conversion measurement is less reliable than it used to be.
- **Mobile platforms** increasingly require user opt-in before an app can track across other companies' apps and sites, which reduces device-level conversion data.
- **Regulation** (for example GDPR in the EU and state privacy laws in the US) restricts processing of personal data, requires consent in many cases, and limits use of sensitive categories for targeting.

Design consequences:

| Change | Response |
|---|---|
| Fewer user-level conversion signals | Aggregated or delayed conversion reporting; models trained on coarser, noisier labels; conversion modeling to fill gaps |
| Less cross-site user history | Rely more on first-party and contextual signals (page content, placement, query) |
| Aggregated, noised measurement APIs | Calibration and evaluation must tolerate noise and aggregation; avoid per-user metrics |
| Consent varies by user | Features must handle missing values explicitly; do not let consent status leak into targeting in discriminatory ways |

Privacy-preserving techniques that come up: on-device inference or learning, differential privacy noise on aggregated reports, k-anonymity thresholds on audience size, and data retention limits. In an interview it is enough to show that you know the signal is shrinking and that the model and measurement must be designed to handle it.

---

## Interview Q&A

#### Why does calibration matter so much more here than in other classifiers?

Because pCTR is not just used to sort. It is multiplied by bids from different advertisers and used to compute prices. If the model ranks perfectly within each advertiser's ads but is 20% high for one advertiser, that advertiser wins auctions it should lose and the prices are wrong. Advertisers lose money, competitors lose inventory, and the platform's revenue estimate is off.

That is why log loss (which penalizes miscalibration) is the primary offline metric and AUC is secondary, and why the calibration ratio is monitored per segment in production. It is also why negative downsampling has to be corrected with `p = p' / (p' + (1 − p') / w)` before scores reach the auction.

#### You downsampled negatives to 10%. What changes?

Training is about ten times cheaper and positives get more weight. But the model's predicted odds are now inflated by a factor of 1/w = 10. Ranking within a request is unaffected, since the transform is monotone. Everything that uses absolute probability is broken: the auction, pricing, pacing, and budget forecasts.

Fix it by applying the correction formula at serving, or equivalently by adding log(w) to the logit. Log the sampling rate with each training example so the correction stays right if the rate changes. Then check calibration ratio on unsampled held-out traffic to confirm it is close to 1.0.

#### Why log loss or normalized entropy rather than AUC as the main metric?

AUC measures only whether positives score above negatives. It does not change if you double every prediction. Log loss penalizes both mis-ranking and miscalibration, which matches how the scores are used.

Normalized entropy divides log loss by the entropy of a model that always predicts the average CTR. That makes numbers comparable across datasets or time periods with different base CTRs, which raw log loss is not. A 1% NE improvement means the same thing on a 0.5% CTR placement and a 3% CTR placement.

#### How do you handle position bias when the position isn't known at scoring time?

The model scores before the auction assigns positions, so the true position can't be an input at serving. Instead, use position as a training-only feature, usually through a separate shallow tower whose logit is added to the main model's during training and dropped at serving. The main model learns relevance with the position effect factored out.

If the auction needs per-slot estimates, factor pCTR(ad, slot) into pCTR(ad) × P(examine | slot), where the examination curve is measured from a small slice of randomized-position traffic. That also gives an unbiased way to validate the correction.

#### How would you predict conversions when they arrive days after the click?

Separate the problem into pCTR (fast labels) and pCVR (slow labels), ideally in a multi-task model. For pCVR, pick between freshness and label accuracy. Options: wait a fixed window before training, ingest clicks as negatives immediately and then correct with importance weights when conversions arrive, or model the delay distribution explicitly so recent unconverted clicks are treated as uncertain.

CVR labels also exist only for clicked impressions, while serving predicts over all impressions. ESMM-style training (supervising pCTR × pCVR on the full impression space) or a shared multi-task model addresses that sample selection bias.

#### A new advertiser launches. How do they get impressions?

Their ad has no history, so ID-based features are empty and the model will tend to underpredict. First, give the model content features that exist from the start: creative embeddings, landing page category, advertiser vertical. Next, back off counters hierarchically to the campaign, advertiser, and category level. Finally, add explicit exploration, either a reserved impression budget or an uncertainty bonus via UCB or Thompson sampling, until enough impressions accumulate.

Apply the exploration bonus to ranking, not pricing, so the advertiser isn't charged for the platform's exploration. Monitor calibration specifically for low-impression ads, since that's where errors concentrate.

#### Offline log loss improved, but the online A/B test shows flat revenue. What do you check?

First, **calibration online**, per segment. A model can improve log loss mostly on segments that don't matter for revenue while getting slightly worse on high-bid segments. Second, **the experiment itself**: if arms share advertiser budgets, budget-constrained advertisers cap how much revenue can move, and spillover between arms can mask or fake an effect. Third, **auction dynamics**: better predictions can lower clearing prices by separating ads more clearly, which is good for advertisers even if short-term revenue is flat. Check advertiser metrics like cost per conversion.

Finally, check **training/serving skew**: features that differ between the logged training snapshot and live serving, or a calibrator that wasn't refreshed with the new model.

#### How often would you retrain, and how do you keep continuous training safe?

Ads data shifts quickly, so a combination works best. A periodic full retrain on a long window, plus incremental updates every few minutes to hours on streaming data, with embedding tables updated most frequently.

For safety, every checkpoint must pass automated gates before it is pushed: log loss and NE on a very recent holdout, calibration ratio overall and per major segment, and prediction distribution checks against the current model. Keep the last known-good checkpoint ready for instant rollback. Alert on sudden shifts in average pCTR or spend rate, which are often the first sign of a logging bug or a click-fraud attack in the training stream.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Optimizing AUC only | Ignores calibration, which sets auction prices | Log loss / NE as primary; calibration ratio per segment |
| Forgetting downsampling correction | Every absolute probability is inflated by ~1/w | `p = p' / (p' + (1 − p')/w)` at serving; log w per example |
| Ranking by pCTR instead of eCPM | Ignores bids and conversion value | Rank by bid × pCTR (× pCVR) plus quality terms |
| Ignoring position bias | Model reproduces the previous ranker | Position as training-only feature; randomized calibration slice |
| Random train/test split | Leaks future counters and trends | Time-based split: train on the past, test on the next day |
| Non-point-in-time counters | Label leaks into features | Compute counters from data strictly before each impression |
| Training on CVR only for clicked impressions | Sample selection bias at serving | ESMM-style or multi-task over all impressions |
| Treating clicks as the only objective | Rewards clickbait; hurts conversions and users | Multi-objective score with CVR and negative-feedback terms |
| No exploration for new ads | Cold-start ads never get impressions | Content features, hierarchical priors, bandit boosts |
| Shared-budget A/B tests | Arms interfere through budgets; biased results | Budget-split or advertiser-level randomization |
| Unguarded continuous training | Bad data reaches production in minutes | Pre-push validation gates, instant rollback |
| Applying targeting after ranking | Wasted compute; risk of violating targeting | Enforce targeting and policy in retrieval |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Search and Ranking System Design](./search_ranking_system.md)
- [Recommendation System Design](./recommendation_system.md)
- [Fraud Detection System Design](./fraud_detection.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Causal Inference and Uplift](../classical_ml/intro_causal_inference.md)
- [Embeddings](../ai_genai/intro_embeddings.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Feature Store](../mlops/intro_feature_store.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md)
