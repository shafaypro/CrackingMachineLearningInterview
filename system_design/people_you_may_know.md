# People You May Know System Design

"Design a People You May Know system" is a recommendation problem where the items are other users and the data is a graph. That changes almost everything that makes it interesting. The candidate set is huge but highly structured (most future connections are two hops away), the label is a two-sided decision (one person sends, the other accepts), and a bad suggestion is not just irrelevant: it can reveal something private, push an unwanted invite, or connect a minor to a stranger. The model also changes the graph it learns from, so every experiment and every training set is affected by earlier versions of the system. Strong answers spend their time on candidate generation over the graph, label construction, leakage, experiment interference, and privacy, and treat the generic ranking stack as known.

For two-tower retrieval and a baseline recsys serving stack, see [Recommendation System Design](./recommendation_system.md). For GNN mechanics, see [Graph Neural Networks](../deep_learning/intro_graph_neural_networks.md). This guide focuses on what is specific to connection recommendation.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [Defining the Objective](#defining-the-objective)
3. [The PYMK Funnel](#the-pymk-funnel)
4. [Candidate Generation](#candidate-generation)
5. [Link-Prediction Features](#link-prediction-features)
6. [Ranking Model and Labels](#ranking-model-and-labels)
7. [Leakage in Link Prediction](#leakage-in-link-prediction)
8. [Evaluation](#evaluation)
9. [Serving Architecture](#serving-architecture)
10. [Cold Start](#cold-start)
11. [Privacy and Safety](#privacy-and-safety)
12. [Feedback Loops](#feedback-loops)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Clarify the Problem First

**What kind of edge?** A **symmetric friendship** (both sides must agree, as in a request and accept flow) or an **asymmetric follow** (one side acts, no approval). Symmetric edges make the label two-sided and make unwanted invites a cost to the recipient. Follow graphs look more like creator recommendation and have a much heavier degree tail.

**What kind of network?** A personal social network, a professional network, or a messaging app with a contact graph. This decides which side signals exist (schools, employers, groups, phone contacts) and which relationships are sensitive.

**Where do suggestions surface?** A dedicated PYMK page, a carousel inside the feed, post-signup onboarding, notifications or emails ("You may know X"), and profile pages ("People also viewed"). Each surface has its own position effects and its own tolerance for mistakes. A push notification about a person is far more intrusive than a row in a carousel.

**How fresh must suggestions be?** After a user accepts a request or joins a group, should new suggestions appear within minutes, or is a daily refresh enough? New users are the case where freshness matters most, since their graph changes fastest in the first sessions.

**Scale numbers to ask for:**
- Number of users and edges, and the degree distribution (median degree, size of the largest hubs)
- Daily active users and PYMK impressions per user per day
- Latency budget for the PYMK module (usually a slice of a page load, so tens of ms)
- New edges created per day and new signups per day

For the rest of this guide, assume a concrete brief: **a symmetric friendship network, 1B users, a few hundred friends at the median with a long tail of hubs, 300M DAU, suggestions on a PYMK page, a feed carousel and onboarding, a p99 of ~50 ms for serving, and suggestions reflecting new friendships within minutes for new users and within a day for everyone else.** These are working assumptions for the exercise, not figures from any real platform.

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Serving latency (p99) | ~50 ms, which rules out heavy graph traversal at request time |
| Freshness | Minutes for new users and after big graph events; daily batch otherwise |
| Safety | Blocks, removals, and "don't suggest" choices take effect immediately, even from caches |
| Availability | Fall back to cached suggestions or hide the module; never show unfiltered candidates |

---

## Defining the Objective

### What a good suggestion is

The obvious label is "a request was sent." That is the wrong target on its own, because the easiest way to get more requests sent is to suggest people the viewer is curious about but who do not want to hear from them. A better chain of outcomes:

```
impression ──► request sent ──► request accepted ──► connection is used
   (viewer)       (viewer)        (recipient)        (both: messages,
                                                     comments, views)
```

A reasonable value score for a candidate pair (u, v):

```
score(u, v) =  P(send | u, v, shown) · P(accept | u, v, sent) · value(u, v, accepted)
             - w_ignore · P(send | ...) · (1 - P(accept | ...))
             - w_report · P(report or "don't know" | u, v, shown)
```

The first term is the expected accepted connection, weighted by how much the new edge will be used. The second term charges for requests that go unanswered, since those are a cost to the recipient. The third term charges for suggestions that feel wrong. As in feed ranking, the weights are a product and policy decision, and each probability must be calibrated for the sum to mean anything.

### Asymmetric costs

The cost of mistakes is not symmetric:

| Error | Cost |
|---|---|
| Missing a good suggestion | The user connects later or not at all; a small, recoverable loss |
| Suggesting a stranger | A wasted slot; mild annoyance |
| Suggesting someone who will ignore the request | A pending request the recipient did not want; spam-like if repeated |
| Suggesting a "creepy" match | Reveals that the system knows something private (a therapist, an ex, a secret contact); damages trust in the whole product |
| Suggesting an adult stranger to a minor | A safety failure, not a relevance failure |

Because of this, the system optimizes for precision near the top and applies hard filters for sensitive cases before any score is computed. A model that raises accepted connections by 3% while doubling "I don't know this person" reports is a regression.

### Downstream value and network health

Accepted requests are still a proxy. Connections that never produce any interaction add little. Longer-horizon targets:

- **Engagement on the new edge** in the next 28 days (messages, comments, profile views).
- **Retention of the new user**, since early connections are one of the strongest predictors of whether a new account stays.
- **Recipient experience**: incoming requests per user, and the share accepted. A user flooded with low-quality requests is being harmed even if the sender is happy.

---

## The PYMK Funnel

```
PYMK request (viewer, surface)
  │
  ▼
┌──────────────────────────────┐   ~1B users ──► ~1-5k
│ Candidate generation         │  friends-of-friends, PPR / random walks,
└──────────────┬───────────────┘  embedding ANN, contacts, co-membership
               ▼
┌──────────────────────────────┐
│ Hard filters                 │  existing friends, pending requests, blocks,
└──────────────┬───────────────┘  "don't suggest", age rules, privacy settings
               ▼
┌──────────────────────────────┐   ~5k ──► ~200
│ Light ranking                │  GBDT or small model on cheap graph features
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐   ~200 ──► ~50
│ Heavy ranking                │  multi-task: P(send), P(accept), P(report)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Re-ranking                   │  diversity of source, request caps per
└──────────────┬───────────────┘  recipient, dedup across surfaces
               ▼
     PYMK module ──► impressions, sends, accepts, dismissals ──► training data
```

Most of the funnel is precomputed in batch (candidate generation, graph features, often the ranking itself), with a light online layer that applies filters and fresh signals. See [Serving Architecture](#serving-architecture).

---

## Candidate Generation

Most new friendships close a triangle: two people with a mutual friend become friends. That makes the two-hop neighbourhood the main candidate source. The other sources cover what triangles miss.

### Friends of friends (FoF)

For viewer u, every friend-of-friend v is a candidate, scored by the number of common neighbours. The computation is a walk over the two-hop neighbourhood:

```
for each friend f of u:
    for each friend v of f:
        if v != u and v not in friends(u):
            count[v] += 1
```

The cost is the sum of the friends' degrees. For a user with 500 friends of average degree 500, that is 250k edge reads, fine in batch and too slow per request.

**The hub (celebrity) problem.** If one of u's friends has 100k friends, that single friend contributes 100k candidates, each with a common-neighbour count of 1, and dominates the work. Hubs are also weak evidence: being connected through a person who knows everyone says little about whether u knows v. Mitigations:

- **Skip or sample hubs.** Cap the number of neighbours read per intermediate node (for example, sample 1,000 of a hub's neighbours) or skip intermediate nodes above a degree threshold.
- **Down-weight hubs in the score.** Adamic-Adar does this directly by weighting each common neighbour by 1 / log(degree).
- **Cap per-viewer work.** Stop after a budget of edge reads and keep the top candidates by partial count.

At full-graph scale, FoF counting is a batch job. It can be written as a sparse matrix product (A · A gives common-neighbour counts), but A² is far denser than A, so real pipelines compute it per viewer in a distributed join with hub sampling and keep only the top candidates per viewer.

### Random walks and personalized PageRank

Personalized PageRank (PPR) from u estimates how often a random walk that restarts at u visits each node. It generalizes FoF: it reaches three hops and beyond, rewards many short paths, and naturally down-weights hubs because their probability mass is spread over many neighbours. Monte Carlo estimation (run many short walks with restart probability around 0.15 to 0.3 and count visits) is easy to distribute and to bound in cost, and it makes the hub problem a sampling problem.

### Embedding ANN

Learn a vector per user so that connected users are close, then retrieve nearest neighbours with an ANN index.

| Method | How it learns | Notes |
|---|---|---|
| **node2vec / DeepWalk** | Skip-gram over random-walk sequences | Transductive: new users have no vector until retraining |
| **GNN (GraphSAGE-style)** | Aggregates neighbour features over 2-3 hops, trained on link prediction | Inductive: can embed a new user from profile features and a few edges |
| **Two-tower on side features** | User towers over profile, school, employer, location, interests | Works for users with no edges; weaker on graph structure |

Embedding retrieval catches candidates with few or no common friends who share context (same employer, same city and school years). Its weakness is that proximity in embedding space does not guarantee the pair actually knows each other, so these candidates need strong ranking features to avoid "similar stranger" suggestions. See [Scaling to Large Graphs](../deep_learning/intro_graph_neural_networks.md#scaling-to-large-graphs) for why GNN embeddings are usually precomputed in batch.

### Contact book and co-membership

- **Uploaded contacts.** If u uploaded their phone contacts and v's number is in them, that is strong evidence. Only use contacts the uploader consented to share for this purpose, respect the other person's settings on discoverability by phone or email, and never use one person's uploaded contacts to reveal information to a third party (see [Privacy and Safety](#privacy-and-safety)).
- **Co-membership.** Same school and graduation year, same employer and team, same group, same event. Large groups (a 200k-member group) are weak evidence, so weight membership by group size, the same way Adamic-Adar weights hubs.
- **Interaction without an edge.** Profile views, tags in the same photo, replies in the same thread. These are strong but some are sensitive (profile views in particular), so they should be used as ranking features, never exposed as explanations.

Each source gets a quota, and each is evaluated on recall of connections formed later. A pair found by several sources is usually a better candidate, so the set of sources that produced a candidate is itself a ranking feature.

---

## Link-Prediction Features

| Group | Features |
|---|---|
| **Neighbourhood overlap** | Common neighbours, Adamic-Adar, Jaccard, resource allocation, common neighbours among close friends only |
| **Paths** | Shortest-path distance (capped at 3 or 4), number of 2-hop and 3-hop paths, PPR score from u to v and v to u |
| **Degree** | Degree of u and v, ratio of degrees, v's recent incoming request volume |
| **Shared context** | Same school or employer (with overlap years), same city, same groups weighted by group size, contact-book match, same language |
| **Interaction** | u viewed v's profile, v viewed u's profile, co-tagged in photos, comments on the same posts, messages in shared group chats |
| **Embedding** | Cosine similarity of graph embeddings, GNN pair score |
| **Viewer and candidate** | Account age, activity level, acceptance rate of requests v receives, how many requests u has sent recently |
| **Source** | Which candidate generators produced the pair, and rank within each |

A small pure-Python version of the core pair features:

```python
import math
from collections import deque

def pair_features(adj, u, v, max_dist=4):
    """Link-prediction features for (u, v) from an undirected adjacency dict
    {node: set(neighbours)}. Caller must pass a graph that excludes the edge
    being predicted and any edges created after the prediction time."""
    nu, nv = adj.get(u, set()), adj.get(v, set())
    common = nu & nv
    union = nu | nv
    adamic_adar = sum(1.0 / math.log(len(adj[w])) for w in common if len(adj[w]) > 1)
    resource_alloc = sum(1.0 / len(adj[w]) for w in common)
    return {
        "common_neighbours": len(common),
        "jaccard": len(common) / len(union) if union else 0.0,
        "adamic_adar": adamic_adar,
        "resource_allocation": resource_alloc,
        "pref_attachment": len(nu) * len(nv),
        "distance": shortest_path(adj, u, v, max_dist),
    }

def shortest_path(adj, src, dst, max_dist):
    """BFS distance, capped: returns max_dist + 1 if farther or unreachable."""
    if src == dst:
        return 0
    seen, frontier = {src}, deque([(src, 0)])
    while frontier:
        node, d = frontier.popleft()
        if d == max_dist:
            continue
        for nb in adj.get(node, ()):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, d + 1))
    return max_dist + 1

def fof_candidates(adj, u, hub_cap=1000, top_k=100):
    """Friends-of-friends by common-neighbour count, skipping large hubs."""
    counts = {}
    for f in adj.get(u, ()):
        if len(adj[f]) > hub_cap:
            continue                      # or sample hub_cap neighbours instead
        for v in adj[f]:
            if v != u and v not in adj[u]:
                counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

if __name__ == "__main__":
    edges = [("ana", "ben"), ("ana", "cai"), ("ben", "cai"), ("ben", "dee"),
             ("cai", "dee"), ("dee", "eli"), ("hub", "ana"), ("hub", "dee"),
             ("hub", "eli"), ("hub", "fay")]
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    print(fof_candidates(adj, "ana"))      # [('dee', 3), ('eli', 1), ('fay', 1)]
    print(pair_features(adj, "ana", "dee"))
```

Two notes. Adamic-Adar and resource allocation beat raw common neighbours mainly because they discount hubs. And these features are only honest if `adj` is the graph as it existed at prediction time, which is the subject of [Leakage in Link Prediction](#leakage-in-link-prediction).

---

## Ranking Model and Labels

### Label construction

There are three natural labels, each defined on a different population:

| Label | Population | What it measures | Pitfall |
|---|---|---|---|
| **Send** | Impressions (viewport, not just returned) | Viewer interest | Rewards curiosity about people who will not accept |
| **Accept** | Sent requests | Recipient agreement | Only defined on sends, so it has selection bias |
| **Used edge** | Accepted connections | Downstream value | Sparse, delayed by weeks |

The standard design is a multi-task model with separate heads trained on their own populations, combined as P(send) · P(accept | send), like CTR · CVR in ads. Training P(accept) only on sent requests means it never sees pairs the viewer would not send to, so it is biased on the full candidate set. Options are to train it with inverse propensity weights from the send head, or to train an end-to-end P(send and accept | impression) head alongside it (the ESMM idea from ads).

**Negatives.** Use impressions with no send, not random pairs. Random pairs are trivially unconnected and teach the model to rank by degree. Label windows matter: a request can be accepted days later, so wait a fixed window (such as 14 days) before labelling a send as "not accepted," or treat recent examples as censored.

**Explicit negatives.** "Remove" or "I don't know this person" on a suggestion, ignoring or declining a request, and reports are strong negative labels and should drive a separate head with a large negative weight.

### Model choice

- **Light ranker:** gradient-boosted trees on the overlap, path, and context features. Trees handle count features with heavy tails well and are cheap enough to score thousands of pairs per viewer in batch.
- **Heavy ranker:** a multi-task neural network over the same features plus user and pair embeddings (from node2vec or a GNN), with heads for send, accept, and report. A GNN can also be the heavy ranker directly, scoring pairs from message-passed embeddings, but it is usually cheaper to precompute GNN embeddings and feed them to a standard ranker.

**Position and surface.** A suggestion in slot 1 of the feed carousel gets far more sends than the same suggestion in slot 8 of the PYMK page. Include surface and position as training features and set them to a fixed value at serving time, or model them with a separate bias tower.

---

## Leakage in Link Prediction

Link prediction has a leakage trap that ordinary tabular ranking does not: the features are computed from the graph, and the graph contains the answer.

**Remove the edge being predicted.** If u and v are friends in the snapshot used to compute features, then v is in u's neighbourhood, common-neighbour counts are shifted, the shortest-path distance is 1, and a GNN aggregates v directly into u's embedding. Any model trained this way learns "distance = 1 means they will connect."

**Remove all future edges, not just the target.** Suppose u and v connected on day 10, and u also connected to w (a friend of v) on day 9. If features are computed from a day-12 snapshot, the u-w edge inflates common neighbours for (u, v), and it was often caused by the same event (u joined a new school). Point-in-time correctness means every feature for an example at time t uses only edges created before t.

**Use temporal splits.** Train on connections formed before a cutoff and evaluate on connections formed after it, with features computed from the graph at the cutoff. Random edge splits let the model use later edges to predict earlier ones and overstate offline quality.

**Embeddings leak too.** node2vec or GNN embeddings trained on today's graph encode edges that did not exist at the time of older training examples. Train embeddings on the snapshot at the training cutoff, or log the embedding versions used at serving time. See [How do you split data for link prediction without leaking?](../deep_learning/intro_graph_neural_networks.md#how-do-you-split-data-for-link-prediction-without-leaking).

```
            graph snapshot G(t0)                     edges formed in (t0, t1]
 ─────────────────────────────────────────┬─────────────────────────────────────
  features, embeddings, candidates         │  labels: which candidate pairs
  computed from edges created before t0    │  became connections
 ─────────────────────────────────────────┴─────────────────────────────────────
                                          t0                                   t1
```

The cleanest guarantee is to log the features served at impression time and train on those, which also removes training/serving skew.

---

## Evaluation

### Offline

| Metric | What it tells you |
|---|---|
| **Recall@k of candidate generation on future edges** | Share of connections formed in (t0, t1] that were in each source's top k at t0 |
| **Recall@k / NDCG@k of the ranker** | Ordering quality against future accepted connections |
| **AUC and log loss per head** | Send, accept, report heads separately |
| **Calibration per head** | Needed because heads are multiplied and summed |
| **Precision@k on "don't know" reports** | How often top suggestions are strangers |
| **Sliced metrics** | New users, low-degree users, minors, users with contacts uploaded or not |

Recall on future edges is biased toward the old system: many future edges were formed because the old PYMK suggested them. A new model that finds different good candidates is penalized. Exploration traffic (a small share of randomized suggestions from the candidate pool) gives a less biased evaluation set.

### Online

| Metric type | Examples |
|---|---|
| **Primary** | Accepted connections per viewer, connections that produce interaction within 28 days, new-user retention |
| **Funnel** | Impressions, send rate, acceptance rate |
| **Guardrails** | "Don't know" and remove rates, reports, pending-request backlog per recipient, incoming requests per user, requests to minors |

**Network growth effects.** Every accepted connection changes the graph: both users get new friends of friends, their feeds change, and they may attract more requests. A good PYMK change shows effects well beyond the PYMK surface (more feed engagement, more messages), and those effects compound over weeks. Run tests long enough to see them.

### Interference in A/B tests on graphs

User-level randomization assumes one user's treatment does not affect another's outcome. PYMK breaks this directly. When a treated user sends more requests, control users receive them and accept them, so control acceptance and connection counts also rise. The treatment effect on "connections formed" is double-counted across arms, and on the recipient side is attributed to the wrong arm.

| Approach | How it works | Trade-off |
|---|---|---|
| **Graph cluster randomization** | Partition the graph into clusters with few cross-cluster edges (for example with a balanced graph partitioner) and randomize whole clusters | Fewer, larger units, so less power; clusters are never perfectly isolated |
| **Ego-network randomization** | Treat a user and their neighbourhood as a unit, measure outcomes only on the ego | Neighbourhoods overlap, so needs careful design |
| **Measure recipient-side metrics by arm** | Attribute incoming requests and acceptances to the sender's arm | Captures who caused what, not total spillover |
| **Spillover analysis** | Compare control users with many treated neighbours against those with few | Estimates the size of the bias |

```python
import hashlib

def cluster_arm(user_id, cluster_of, experiment="pymk_v7", treat_share=0.5):
    """Assign treatment by graph cluster, not by user. cluster_of is precomputed
    by a graph partitioner on a snapshot taken before the experiment starts."""
    key = f"{experiment}:{cluster_of[user_id]}".encode()
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 10_000
    return "treatment" if bucket < treat_share * 10_000 else "control"
```

The partition must come from a snapshot taken before the experiment, otherwise the treatment itself changes the clusters. For the causal reasoning behind this, see [Causal Inference and Uplift](../classical_ml/intro_causal_inference.md).

---

## Serving Architecture

```
                         PEOPLE YOU MAY KNOW
═══════════════════════════════════════════════════════════════════════════

  Graph store (edges, timestamps) ──► Daily batch jobs
                                         │  FoF with hub sampling
                                         │  PPR / random walks
                                         │  node2vec / GNN embeddings ──► ANN index
                                         │  co-membership joins
                                         ▼
                                   Candidate set per user (top ~1-5k)
                                         │
                                         ▼
                                   Batch ranking (light + heavy)
                                         │
                                         ▼
                                   PYMK store: user ──► ranked list (~100-500)

  Graph events (accept, join group, ──► Stream processor ──► Incremental
  signup, contact upload, block)         │                    refresh for the
                                         │                    affected users
                                         ▼
                                   Real-time filter set
                                   (blocks, removals, pending requests,
                                    new friendships, "don't suggest")

  Client ──► PYMK Service ──► PYMK store ──► filter ──► light re-score with
                                                        fresh features ──►
                                                        re-rank ──► response
                                                                     │
                                                                     ▼
            impressions, sends, accepts, dismissals, reports ──► Kafka ──► training

═══════════════════════════════════════════════════════════════════════════
```

**Batch first.** Candidate generation over billions of edges does not fit a 50 ms budget. Run it daily per user, rank in batch, and store a list per user. The online path fetches the list, applies filters, and optionally re-scores with a few fresh features.

**Refresh on graph events.** When a user accepts a request, their friend's friends become new candidates. For most users, waiting until the next batch run is fine. For new users and users who just joined a school or company group, trigger an incremental job: recompute FoF for that user alone (cheap for a low-degree user), add candidates from the new group, and score them. A new edge also makes its two endpoints poor candidates for each other, so the filter set must be updated in seconds.

**Filtering at read time.** Precomputed lists go stale in ways that matter for safety. A block, a new friendship, a pending request, or a "remove suggestion" must take effect immediately. Apply these checks against a fast key-value set at response time, whatever the cache says.

**Deduplication across surfaces.** The same person should not be suggested on the PYMK page, the feed carousel, and in an email on the same day. Keep a per-viewer impression history and apply frequency caps, with a cooldown after a dismissal (often permanent for "don't know this person").

**Recipient-side caps.** A popular user can be suggested to millions of viewers and receive a flood of requests. Cap how often a user is shown as a candidate, or penalize candidates whose incoming pending requests are already high.

**Fallbacks.** If the store is unavailable, hide the module. Serving an unfiltered or stale list is worse than serving nothing.

---

## Cold Start

**New users** have no edges, so FoF, PPR, and graph embeddings produce nothing. This is also the moment PYMK matters most, since a user who makes no connections in the first days is likely to leave. Sources that work without edges:

- **Contact upload** (with explicit consent), matched against users who allow being found by phone or email.
- **Profile context** from onboarding: school, employer, city. Co-membership candidates and an inductive model (a GNN or two-tower model over profile features) can score them.
- **Reverse signals:** existing users who have the new user's phone number or email in their contacts. This is powerful but sensitive; see [Privacy and Safety](#privacy-and-safety).
- **The inviter.** If the user joined through an invite, the inviter and the inviter's close friends are strong candidates.

After the first accepted connection, FoF starts working, so trigger an incremental refresh on every early accept. The generic playbook is in [Handling Cold Start](./recommendation_system.md#handling-cold-start).

**New edges of old users.** When an existing user joins a new company or moves city, the relevant candidates are in a part of the graph where they have no edges yet. Co-membership and event-triggered refresh handle this.

**Low-degree users.** Users with a handful of friends get few FoF candidates and noisy features. Evaluate them as a separate slice, since averages are dominated by well-connected users.

---

## Privacy and Safety

A PYMK suggestion is a statement: "the system thinks you know this person." It can reveal information neither user chose to share.

| Risk | Example | Mitigation |
|---|---|---|
| **Revealing sensitive relationships** | Suggesting a patient to their therapist, or two people who share a support group | Exclude sensitive group memberships and professional contacts as candidate sources; never explain a suggestion with a sensitive signal |
| **Contact-book leakage** | A user is suggested to someone whose number they never shared, because a third party uploaded both | Only match on contact data the owner consented to use; respect "don't let people find me by phone" settings |
| **Location inference** | Suggesting people who were in the same place at the same time | Do not use precise location or co-location as a signal |
| **Profile-view leakage** | "People who viewed your profile" leaking through suggestions | Use views as a feature at most; never as a visible explanation |
| **Blocked users** | Suggesting someone the viewer blocked, or someone who blocked the viewer | Symmetric hard filter at read time; also exclude them from explanations ("12 mutual friends" should not list a blocked user) |
| **Minors** | Suggesting unconnected adults to minors | Hard rules: no adult-to-minor suggestions without strong existing ties, stricter defaults for minors' discoverability, age-aware filters applied before ranking |
| **Harassment** | Suggesting an ex or someone a user has tried to avoid | "Don't suggest" and block controls that act immediately; downrank pairs with recent unfriend or decline history |
| **Sensitive attribute inference** | Clusters that reveal religion, sexuality, health, or politics | Avoid features derived from sensitive groups; audit suggestions for these patterns |

**Explanations.** "12 mutual friends" or "Works at X" helps acceptance and lets users understand why they see a suggestion. Explanations must only use signals the viewer can already see. "You have this person's number in your contacts" is fine only if it is the viewer's own upload. "This person viewed your profile" is never fine.

**Hard filters before the model.** Safety rules are not features the model can trade off against engagement. They are filters applied before scoring and again at read time. Keep them in a separately owned, audited layer. See [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md) for how to structure audits.

---

## Feedback Loops

PYMK does not just predict the graph; it builds it. A large share of new edges on a platform with a PYMK surface can come from suggestions, and those edges become the next training data and the next FoF candidates.

| Loop | Cause | Mitigation |
|---|---|---|
| **Rich get richer** | Popular users are suggested more, gain more connections, and become even more central (preferential attachment) | Recipient-side exposure caps; degree-normalized features like Adamic-Adar; monitor concentration of incoming requests |
| **Homophily** | Suggestions follow existing clusters (same school, same group), so the graph becomes more segregated | Some exploration outside the dominant cluster; monitor cross-cluster edge share; weigh this against the fact that most real ties are local |
| **Self-confirming evaluation** | Recall on future edges rewards the old model, whose suggestions caused those edges | Evaluate on exploration traffic; long-term holdouts |
| **Triadic closure bias** | FoF dominance means users with few friends get few suggestions, stay small | Non-graph sources and quotas for low-degree users |
| **Request spam** | A model trained on sends learns to suggest people who get lots of requests | Acceptance and recipient-experience terms in the objective |

**Long-term holdouts.** Keep a small group of users on an older PYMK model, or with the module removed entirely, for months. This measures how much PYMK changes graph growth, retention, and feed engagement, and it is the only reliable estimate of the causal effect of the system as a whole.

---

## Interview Q&A

#### How would you generate candidates for PYMK on a billion-user graph?

Several sources in parallel, mostly precomputed in batch. The main one is friends of friends, because most new connections close a triangle. Count common neighbours per two-hop candidate, sample or skip hub nodes so one friend with 100k connections does not dominate the work, and keep the top few thousand per user. Add personalized PageRank or random walks to reach three hops and down-weight hubs naturally, embedding ANN (node2vec or a GNN) for users with shared context but few mutual friends, and co-membership and consented contact matches for new users.

Each source has a quota and is measured by recall of connections formed after the snapshot. Run it daily, with incremental refresh for new users and after big graph events.

#### Why is Adamic-Adar often better than common neighbours?

Common neighbours treats all mutual friends equally. A mutual friend with 5,000 connections is weak evidence that two people know each other, while a mutual friend with 30 connections is strong evidence. Adamic-Adar weights each common neighbour by 1 / log(degree), so low-degree mutual friends count more. Resource allocation uses 1 / degree, a stronger discount. In practice you include all of them as features and let the ranker learn the weighting.

#### What label do you train on: send, accept, or something else?

Several, in a multi-task model. P(send | impression) measures viewer interest but rewards curiosity about people who will not accept. P(accept | send) measures whether the recipient agrees but is only observed on sent requests, so it needs correction for selection bias. The product P(send) · P(accept | send) gives the expected accepted connection. Add a head for "don't know this person" or reports with a large negative weight, and, where possible, a longer-term signal such as interaction on the new edge within 28 days. Negatives come from viewed impressions, not random pairs.

#### How can link-prediction features leak?

Features come from the graph, and the graph contains the label. If the target edge is in the snapshot, the distance is 1 and common neighbours shift. Subtler: other edges formed after the prediction time, often caused by the same event, inflate overlap features. The fix is point-in-time features from a snapshot at the cutoff (or features logged at serving time), temporal train/test splits, and embeddings trained only on the graph as of the training cutoff.

#### Your PYMK A/B test shows +5% connections in treatment. What could be wrong?

Interference. Treated users send more requests, and many recipients are in control. Control users accept those requests, so control connections also rise, and connection counts are shared between arms. The measured difference can be biased in either direction, and totals across the platform are not what the test estimated.

Use graph cluster randomization, attribute recipient-side outcomes to the sender's arm, and measure spillover by comparing control users with many versus few treated neighbours. Also check guardrails: a lift in connections that comes with more "don't know" reports or more ignored requests is not a win.

#### How do you handle a brand-new user with no friends?

Use signals that do not need edges: consented contact upload, onboarding profile (school, employer, city) scored by an inductive model, and the person who invited them. Once the first request is accepted, refresh their candidates right away, since FoF now has something to walk from. New users are the highest-value slice for PYMK, so measure it separately, and be careful with reverse contact matches, which are the easiest place to leak information.

#### What makes a suggestion "creepy," and how do you prevent it?

A suggestion is creepy when it shows the system knows something the user did not share: a therapist, someone met at a clinic, an ex, a person whose number was uploaded by a third party. Prevent it by excluding sensitive sources from candidate generation (sensitive groups, precise location, professional-client contacts), respecting contact discoverability settings, never using profile views in explanations, and applying hard filters for blocks and minors before ranking. Track "don't know" and report rates by source, since a spike from one source usually points to a signal that should not be used.

#### How do you avoid rich-get-richer?

Popular users gain connections through PYMK, which makes them more central and more likely to be suggested. Use degree-discounted features, cap how often a person is shown as a suggestion, penalize candidates with large pending-request backlogs, and monitor the concentration of incoming requests. Long-term holdouts measure how much the system shapes the graph over months.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Training only on "request sent" | Rewards suggestions that will be ignored or feel invasive | Multi-task: send, accept, report; downstream value |
| Target edge left in the feature graph | Model learns distance = 1; offline scores meaningless | Point-in-time snapshots; log served features |
| Random edge splits | Future edges predict past ones | Temporal splits at a cutoff |
| Random pairs as negatives | Trivial; model learns degree | Viewed impressions with no send as negatives |
| Full neighbour expansion through hubs | Work explodes; weak evidence dominates | Hub sampling or skipping; Adamic-Adar |
| Request-time graph traversal | Blows the latency budget | Batch candidates and ranking; online filter only |
| Filters only in batch | Blocked or new friends still suggested | Read-time filter set updated in seconds |
| User-level A/B tests | Requests cross arms; biased effects | Graph cluster randomization; sender attribution |
| Evaluating only on future edges from logs | Rewards the old model's own suggestions | Exploration traffic; long-term holdouts |
| Sensitive signals as sources or explanations | Reveals private relationships | Exclude sensitive sources; explanations from viewer-visible data only |
| No minors policy | Safety failure | Age-aware hard filters before ranking |
| No recipient-side view | Popular users flooded with requests | Exposure caps; pending-request penalties |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Recommendation System Design](./recommendation_system.md)
- [News Feed Ranking System Design](./news_feed_ranking.md)
- [Ads CTR Prediction and Ranking](./ads_ctr_prediction.md)
- [Fraud Detection System Design](./fraud_detection.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Graph Neural Networks](../deep_learning/intro_graph_neural_networks.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Causal Inference and Uplift](../classical_ml/intro_causal_inference.md)
- [Embeddings](../ai_genai/intro_embeddings.md)
- [Vector Databases: Advanced](../ai_genai/intro_vector_databases_advanced.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Feature Store](../mlops/intro_feature_store.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md)
