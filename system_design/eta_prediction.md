# ETA Prediction System Design

"Design an ETA prediction system for a ride-hailing or food-delivery app" sounds like a plain regression problem: predict minutes, minimize error. Candidates who stop there miss most of the design. The ETA is shown to users, used to price trips, and fed into the dispatcher that decides which driver or courier gets which job. A routing engine already produces a decent physics-based estimate from the road graph, so the ML model's job is usually to correct it, not to replace it. And the cost of an error depends on its sign: telling someone "5 minutes" when it is really 12 does more damage than the reverse. Those facts drive the choices below.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [Why ETA Matters](#why-eta-matters)
3. [Baseline: The Routing Engine](#baseline-the-routing-engine)
4. [Decomposing a Delivery ETA](#decomposing-a-delivery-eta)
5. [Features](#features)
6. [Labels and Data Cleaning](#labels-and-data-cleaning)
7. [Model Progression](#model-progression)
8. [Loss Functions and Uncertainty](#loss-functions-and-uncertainty)
9. [Evaluation](#evaluation)
10. [Serving Architecture](#serving-architecture)
11. [Re-Prediction During the Trip](#re-prediction-during-the-trip)
12. [Feedback Loops](#feedback-loops)
13. [Cold Start in New Cities](#cold-start-in-new-cities)
14. [Monitoring and Drift](#monitoring-and-drift)
15. [Interview Q&A](#interview-qa)
16. [Common Pitfalls](#common-pitfalls)
17. [Related Topics](#related-topics)

---

## Clarify the Problem First

"ETA" means several different things on one platform. Pin down which one before designing anything.

**Which ETA?**
- **Pickup ETA (ride-hailing):** time for a driver to reach the rider. Shown before the rider requests, and used by the dispatcher to choose a driver.
- **Trip ETA (ride-hailing):** time from pickup to dropoff. Shown in the fare quote and during the trip.
- **Delivery ETA (food):** time from order placement to food at the door. It includes restaurant prep time, courier assignment, travel to the restaurant, waiting, and travel to the customer. Much of it is not driving at all.

**When is it shown and who consumes it?** Before the order (a quote on the home screen or checkout page, shown for many candidate restaurants or products at once), at order time (a promise), and continuously during the trip (a live countdown). Internal consumers matter as much as users: dispatch, pricing, and batching all call the same service, often at much higher volume than the UI does.

**Point estimate or range?** Food apps commonly show a range ("25 to 35 min"). A range needs a calibrated distribution, not a single number.

**Scale numbers to ask for:**
- Requests per second, peak and average. Dispatch may score every (driver, request) pair in a neighbourhood, so internal calls can be an order of magnitude above user-facing ones.
- Latency budget: ETA sits inside the dispatch loop and the page render, so the model usually gets a slice in the range of single-digit to tens of milliseconds.
- Number of cities and how different they are (road networks, traffic patterns, two-wheelers versus cars).

For the rest of this guide, assume: **a food-delivery and ride-hailing platform in a few hundred cities, ~200k ETA requests/sec at peak including internal dispatch calls, p99 of 30 ms for the ETA service, shown as a range for delivery and a point estimate for rides.** These are working assumptions for the exercise, not industry figures.

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Latency (p99) | ~30 ms per call, lower for batched dispatch calls |
| Throughput | ~200k requests/sec peak |
| Availability | Must always return something; a routing-only ETA is an acceptable fallback |
| Accuracy | Low absolute error and low bias, per city and per time band |
| Interval quality | An "80% interval" should contain the actual time about 80% of the time |

---

## Why ETA Matters

ETA is an input to three separate decisions, which is why small improvements are worth a lot:

- **Dispatch and matching.** The dispatcher assigns drivers or couriers by minimizing something like total pickup time. A biased pickup ETA leads to worse assignments across the whole fleet, not just one bad trip.
- **Pricing.** Upfront fares are usually a function of predicted distance and duration. Underestimate duration and the platform or driver is underpaid; overestimate and riders are overcharged or leave.
- **User trust and conversion.** The ETA shown on the menu or quote screen affects whether people order at all. A late order against a promised time produces refunds, support tickets, and churn.

### Asymmetric cost

Errors do not cost the same in both directions.

| Error | Effect |
|---|---|
| **Underestimate** (arrives later than promised) | Broken promise, angry customer, cold food, refunds, cancellations, worse ratings |
| **Overestimate** (arrives earlier than promised) | Lower conversion at quote time (the app looks slow), idle couriers, food waiting at the door; mild if small |

So the target is usually not the mean. For the displayed promise, many teams aim for a higher quantile, or add a buffer, so that "late" is rare. For dispatch, where ETAs are compared against each other, an unbiased mean estimate is what you want. **The same model can serve both if it predicts a distribution**, and each consumer reads the quantile it needs.

---

## Baseline: The Routing Engine

Before any ML, a routing engine computes an ETA from the road network.

**Road graph.** Nodes are intersections, edges are road segments. Each edge carries a travel time, derived from length divided by an expected speed, plus turn costs and penalties for traffic lights.

**Shortest path.**
- **Dijkstra** finds the minimum-time path but explores too many nodes on a continental graph to meet a millisecond budget.
- **A\*** adds a heuristic (straight-line distance divided by maximum speed) to guide the search toward the goal. Faster, but still not enough at scale.
- **Contraction hierarchies (CH)** preprocess the graph: nodes are contracted in order of importance and shortcut edges are added so that queries only move "upward" through the hierarchy in a bidirectional search. Queries become very fast. The catch is that preprocessing assumes fixed weights.
- **Customizable approaches** (for example customizable route planning or customizable CH) split preprocessing into a metric-independent part and a fast "customization" step, so edge weights can be refreshed with live traffic every few minutes without redoing everything.

**Segment speeds.** Edge weights come from historical speed profiles (by segment, time of day, day of week) blended with real-time speeds computed from recent GPS traces on that segment.

```
routing_eta = Σ over segments in route  (length_s / speed_s(t))  +  Σ turn and signal penalties
```

### Why ML on top: residual correction

The routing ETA is systematically wrong in predictable ways. It ignores time spent finding parking, entering a gated complex, walking to the door, waiting at pickup, driver-specific behaviour, and local quirks the speed profile does not capture. Rather than learning travel time from scratch, the standard design trains a model to predict the **residual**:

```
final_eta = routing_eta + f(features)          # additive residual
final_eta = routing_eta × g(features)          # or multiplicative correction
```

Reasons this works well:
- The routing engine already encodes route geometry and traffic; the model does not have to rediscover physics.
- The residual has a smaller range and simpler structure than raw duration, so it is easier to learn.
- If the model fails, `routing_eta` is a sensible fallback.
- Routing improvements (better maps, fresher speeds) help automatically; the model only needs retraining to recalibrate.

The routing engine also exposes useful intermediate outputs: route distance, number of turns, share of time on highways, and the list of segments. Those become model features.

---

## Decomposing a Delivery ETA

A food-delivery ETA is a sum of stages with very different drivers. Modelling them separately makes each easier to learn, debug, and update as the order progresses.

```
Order placed
   │
   ├── Restaurant accepts ─────────────── (seconds to minutes)
   │
   ├── Food preparation ───────────────── prep model: restaurant, items, current load
   │
   ├── Courier assignment ─────────────── dispatch latency, courier supply nearby
   │
   ├── Courier travels to restaurant ──── routing ETA + residual
   │
   ├── Wait at restaurant ─────────────── max(0, food_ready − courier_arrival) + handoff
   │
   ├── Travel to customer ─────────────── routing ETA + residual
   │
   └── Last mile (parking, building, walk) ── location-specific model
Delivered
```

The stages are not simply additive. Pickup time is roughly `max(food_ready_time, courier_arrival_time) + handoff`, which means the total depends on which of two uncertain times is later. Two ways to handle it:

| Approach | Idea | Trade-off |
|---|---|---|
| **Stage models + composition** | Predict each stage (ideally as a distribution) and combine with the max and sums, possibly by Monte Carlo sampling | Interpretable; each stage re-predicts as it completes; composition errors add up |
| **End-to-end model with stage features** | One model predicts total time, using stage predictions as inputs | Learns correlations between stages; harder to debug; must still update as stages complete |

A common compromise is stage models for the operational parts (prep, travel) and an end-to-end model on top that uses them as features and is trained on the total.

```python
import numpy as np

def delivery_eta_samples(prep_samples, courier_to_rest_samples,
                         handoff_samples, rest_to_cust_samples, last_mile_samples):
    """Compose stage distributions into a total delivery-time distribution.
    Each argument is an array of samples (minutes) from that stage's model."""
    pickup = np.maximum(prep_samples, courier_to_rest_samples) + handoff_samples
    return pickup + rest_to_cust_samples + last_mile_samples

# total = delivery_eta_samples(...)
# low, high = np.percentile(total, [10, 90])   # range to display
```

Note that sampling stages independently ignores correlations (a busy Friday evening slows the kitchen and the roads at once). Either sample stages jointly from shared context, or let the end-to-end model correct for it.

---

## Features

| Group | Examples |
|---|---|
| **Route** | Routing ETA, route distance, segment IDs along the route, road classes, number of turns and traffic lights, share of highway, bridges or tolls |
| **Traffic** | Historical speed per segment and time band, real-time speed from recent GPS pings, ratio of current to typical speed, incidents and closures |
| **Time** | Hour of day, day of week, holiday flags, minutes since rush hour start; encode cyclically or as embeddings |
| **Geography** | Origin and destination H3 cells at a few resolutions, city ID, zone type (dense downtown, suburb, airport) |
| **Weather** | Rain, snow, temperature; ideally current and forecast for the trip duration |
| **Events** | Concerts, sports games, large public events near the route |
| **Driver / courier** | Vehicle type (car, bike, scooter, walking), driver's historical speed relative to routing ETA, experience, current state |
| **Restaurant** | Historical prep time by hour, prep time for similar item counts, current open orders (kitchen load), recent handoff delays |
| **Destination** | Historical last-mile time at this building or cell (apartments, campuses, gated areas take longer) |
| **Supply / demand** | Available couriers nearby, open orders in the zone, recent assignment latency |

### Geospatial cells

Raw latitude and longitude are poor features for most models. Discretize with a hierarchical grid such as **H3**, which tiles the earth with hexagons at multiple resolutions. Hexagons have uniform neighbour distances, which makes smoothing and neighbour lookups simple.

- Use coarse cells (city-district scale) for stable aggregates and fine cells (block scale) for last-mile and pickup-point effects.
- Aggregate targets per cell with hierarchical backoff: fine cell → coarse cell → city, so sparse cells borrow strength from their parents.
- Origin-destination cell pairs are useful for caching and for aggregates like "typical residual between these two areas at this hour."

```python
import h3   # h3-py v4 API

def geo_features(lat, lng, resolutions=(7, 9)):
    """Hex cell IDs at multiple resolutions, used as categorical keys
    for embeddings and for looking up aggregated statistics."""
    return {f"h3_r{r}": h3.latlng_to_cell(lat, lng, r) for r in resolutions}
```

### Point-in-time correctness

Traffic and kitchen-load features change minute by minute. The training example must use the value **as it was at prediction time**, not a later snapshot. The safest approach is to log the feature vector at serving time and train on those logs. Recomputing features from tables after the fact leaks information (for example, a "current speed" that already reflects the congestion the trip ran into).

---

## Labels and Data Cleaning

The label is simple in principle: actual arrival time minus the time of prediction. In practice it is noisy.

**GPS noise.** Pings drift in urban canyons, jump between parallel roads, and drop out in tunnels. Map-match traces to the road graph (typically with an HMM-style matcher) before computing segment speeds or trip durations.

**Event timestamp noise.** "Arrived" and "delivered" are often button presses. Couriers mark delivered early or late, drivers start trips before the rider is in the car. Where possible, derive events from geofences and GPS rather than taps, or use taps only when they agree with GPS within a tolerance.

**Trip cleaning.** Drop or separately label:
- Cancelled or reassigned trips (the label is censored, not a valid duration)
- Trips with long unexplained stops, detours far off the planned route, or implausible speeds
- Batched deliveries where the courier carried several orders; label each leg separately or include batch features
- Test orders and internal traffic

**Outliers.** Some trips take much longer (accident, wrong address). Do not blindly delete them: users experience those too, and removing them makes the model overconfident. Instead cap extreme values for the mean model, keep them for the tail quantile models, and use robust losses.

**Censoring.** A trip cancelled after 20 minutes of waiting tells you the true time was *more than* 20 minutes. Dropping these biases the data toward fast trips. Survival-style losses or at least tracking the share of censored trips per segment keeps this visible.

**Which ETA is the label for?** Pickup ETA labels exist only for the driver actually dispatched, not for the alternatives the dispatcher considered. That is a selection effect that matters for the feedback loop discussed below.

---

## Model Progression

| Model | What it adds | Limitation |
|---|---|---|
| **Routing ETA only** | Physics and traffic; no training needed | Misses parking, pickup waits, local quirks; systematic bias by area |
| **Routing ETA + per-cell/hour bias table** | Cheap correction for known biases | No interactions; sparse cells are noisy |
| **GBDT on aggregates (residual)** | Nonlinear interactions between route summary, time, weather, city; strong on tabular data; fast on CPU | Sees the route only through summary statistics |
| **Deep model over features (transformer / MLP)** | Embeddings for high-cardinality categoricals (cells, segments, restaurants); scales with data | Needs more data and care to serve fast |
| **Sequence / graph models over route segments** | Models the route as a sequence or subgraph; captures spatial propagation of congestion | Heavier to train and serve; latency pressure |

### GBDT on aggregates

The strong first model. Inputs are the routing ETA, route summaries (distance, highway share, turn count), time and weather features, and aggregated statistics such as the median residual for this origin cell at this hour. Target is the residual or the log-ratio `log(actual / routing_eta)`. It trains quickly, handles missing values natively, and serves in well under a millisecond.

```python
import lightgbm as lgb
import numpy as np

# Target: log ratio of actual to routing ETA; errors become relative
y = np.log(df["actual_min"] / df["routing_eta_min"])

model = lgb.LGBMRegressor(
    objective="huber", alpha=0.3,       # robust to label outliers (alpha is the Huber delta)
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    min_child_samples=200,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])

eta = df_serve["routing_eta_min"] * np.exp(model.predict(X_serve))
```

### Transformer over features

One published industry approach embeds every feature (continuous features after discretization into buckets, categorical features such as cells, time buckets, vehicle type) as a token, then uses self-attention across those tokens to learn feature interactions. The output head predicts the residual on top of the routing ETA. The ideas that matter generically:

- **Bucketize continuous features** and embed the buckets. This lets the model learn nonlinear effects without hand-tuned transforms.
- **Attention across feature tokens** learns interactions (for example, rain × downtown × rush hour) instead of relying on manual crosses.
- **Keep it shallow and use efficient attention** so it fits the latency budget; the feature count is small, so this is cheaper than it sounds.
- **Separate output calibration per request type** (pickup, dropoff, delivery) with a small bias term or head, since each has a different error profile.

### Sequence and graph models over the route

A trip is a sequence of road segments. Summary statistics lose where the slow part is and how congestion on one segment affects its neighbours.

- **Sequence model:** encode each segment (ID embedding, length, road class, current and typical speed) and run an RNN, 1D convolution, or transformer over the sequence, then pool to predict total time.
- **Graph neural network on the road graph:** nodes are segments (or groups of adjacent segments), edges connect segments that feed into each other. Message passing propagates congestion: a jam downstream slows segments upstream in the next few minutes. Published work on large-scale map ETAs has used GNNs over groups of connected segments for exactly this reason. Per-segment travel times from the GNN can be summed along the route or used to refresh routing weights.

```python
import torch
import torch.nn as nn

class RouteSequenceETA(nn.Module):
    """Encode the segments along a route and predict a correction to routing ETA."""
    def __init__(self, n_segments, n_dense, d=64):
        super().__init__()
        self.seg_emb = nn.Embedding(n_segments, d)
        self.dense_proj = nn.Linear(n_dense, d)      # length, speeds, road class...
        layer = nn.TransformerEncoderLayer(d, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, 1)

    def forward(self, seg_ids, seg_dense, pad_mask):
        x = self.seg_emb(seg_ids) + self.dense_proj(seg_dense)   # (B, L, d)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0).sum(1) / (~pad_mask).sum(1, keepdim=True)
        return self.head(h).squeeze(-1)    # log-ratio correction
```

**Practical choice:** start with GBDT on the residual. Move to a deep model when the data volume justifies it and you need embeddings for high-cardinality entities. Use sequence or graph models where congestion dynamics dominate the error (dense cities, long trips), and weigh their latency against the gain.

---

## Loss Functions and Uncertainty

The loss decides what "best" means, so tie it to how the ETA is used.

| Loss | Optimal prediction | Notes |
|---|---|---|
| **MSE** | Conditional mean | Dominated by outlier trips; rarely the right choice with noisy labels |
| **MAE** | Conditional median | Robust; a good default for a point estimate |
| **Huber** | Between mean and median | Robust to outliers while smooth near zero |
| **MAPE** | Biased low (penalizes overestimates more, relative to actual) | Explodes on very short trips; tends to push predictions down, which is the wrong direction for ETA |
| **MAE on log duration** | Median of duration; errors are relative | Good compromise when relative error matters across short and long trips |
| **Quantile (pinball)** | Chosen quantile *q* | Directly encodes asymmetric cost: *q* > 0.5 penalizes underestimates more |
| **Distributional (e.g. Gaussian/log-normal NLL, mixture)** | Full distribution | Gives intervals and any quantile from one model |

Pinball loss for quantile *q*, with residual `r = y − ŷ`:

```
L_q(r) = q · r          if r ≥ 0     (underestimate: actual later than predicted)
         (q − 1) · r    if r < 0     (overestimate)
```

With *q* = 0.8, an underestimate costs four times as much per minute as an overestimate, and the optimal prediction is the 80th percentile.

```python
import torch

def pinball_loss(pred, target, quantiles=(0.1, 0.5, 0.9)):
    """pred: (B, Q) one column per quantile; target: (B,)."""
    q = torch.tensor(quantiles, device=pred.device)
    r = target.unsqueeze(1) - pred
    return torch.maximum(q * r, (q - 1) * r).mean()
```

### Predicting intervals

Train several quantile heads (e.g. P10, P50, P90) together, or a parametric distribution. Practical details:

- **Quantile crossing.** Independent heads can produce P90 < P50. Predict P50 and non-negative offsets (via softplus) instead of raw quantiles.
- **Conformal calibration.** On a recent holdout, measure how often actuals fall inside the predicted interval and widen or shrink it until coverage matches the target, per segment. This is cheap and robust.
- **Display policy is separate from the model.** The product decides whether to show P50-P90, round to five minutes, or pad for new restaurants. Keep that logic outside the model so it can change without retraining.

---

## Evaluation

### Offline

| Metric | What it tells you |
|---|---|
| **MAE** | Average absolute error in minutes; easy to explain |
| **MAPE** or **MAE on log** | Relative error; compares short and long trips fairly (watch MAPE on very short trips) |
| **P90 / P95 absolute error** | The tail; the bad experiences that drive complaints |
| **Mean signed error (bias)** | Systematic early or late predictions; a model can have good MAE and be biased late in one city |
| **Late rate** | Share of trips arriving after the promised time (or after the upper end of the range) |
| **Interval coverage** | Share of actuals inside the P10-P90 interval; should be close to 80% |
| **Interval width** | Narrow intervals are only useful if coverage holds |

**Slice everything.** Report each metric by city, hour band, trip-length bucket, vehicle type, stage (pickup vs dropoff), and new versus established restaurants. Aggregate MAE hides the segments where the model is badly biased, and those are usually where complaints come from.

**Compare against the routing baseline** in every slice. A model that improves overall MAE but gets worse than plain routing on airport trips has a problem to explain.

Evaluate on **time-based splits** (train on weeks 1 to N, test on week N+1), and include holiday weeks in the test set at least once a year, since those are where drift shows.

```python
import numpy as np
import pandas as pd

def eta_report(df, by="city"):
    """df has columns: actual, pred, p10, p90, and the slicing column."""
    err = df["pred"] - df["actual"]
    out = df.assign(abs_err=err.abs(), signed_err=err,
                    late=df["actual"] > df["pred"],
                    covered=(df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"]))
    return out.groupby(by).agg(
        n=("abs_err", "size"), mae=("abs_err", "mean"),
        p90_abs=("abs_err", lambda s: np.percentile(s, 90)),
        bias=("signed_err", "mean"), late_rate=("late", "mean"),
        coverage_80=("covered", "mean"))
```

### Online

Offline accuracy does not guarantee better outcomes, because the ETA changes behaviour: dispatch picks different drivers, users order from different restaurants, drivers react to the times they see. A/B test.

| Metric type | Examples |
|---|---|
| **Accuracy (live)** | MAE, bias, late rate, coverage measured on completed trips in each arm |
| **User** | Conversion from quote to order, cancellations after ordering, support contacts about lateness, ratings, retention |
| **Marketplace** | Average pickup time, courier utilization, idle time at restaurants, completed trips per hour |
| **Guardrails** | Latency, fallback rate, fare accuracy |

**Marketplace interference.** Treatment and control share the same drivers. If the new ETA makes dispatch pull drivers toward treatment requests, control gets worse and the difference is overstated. Use city- or region-level switchback experiments (alternate the model across time windows within a city) when the change affects dispatch.

---

## Serving Architecture

```
                              ETA PREDICTION SYSTEM
═══════════════════════════════════════════════════════════════════════════════

 Rider / Eater app ──┐
 Dispatch service ───┼──► ETA API ──┬──► Cache (H3 origin-dest pair, time bucket)
 Pricing service ────┘              │        │ hit ─────────────────────────┐
                                    │        ▼ miss                         │
                                    ├──► Routing Engine ──► route, segments,│
                                    │    (CH / customizable graph,          │
                                    │     live segment weights)             │
                                    │                                       │
                                    ├──► Feature Store ───► restaurant load,│
                                    │    (online KV)         courier supply,│
                                    │                        cell aggregates│
                                    │                                       ▼
                                    └──► Model Server ──► residual / quantiles
                                         (GBDT or small NN, batched)        │
                                                                            ▼
                                                            Interval calibration
                                                            + display policy
                                                                            │
                                                                            ▼
                                                                 ETA response
                                                                  (logged with
                                                                   features)

 GPS pings ──► Kafka ──► Map matching ──► Segment speed aggregator ──► Routing weights
                               │                   (1-5 min windows)   + feature store
                               ▼
                   Trip events (pickup, dropoff) ──► Label join ──► Training data
                                                                        │
                                            Retrain ──► Offline eval ──► Shadow ──► Rollout

═══════════════════════════════════════════════════════════════════════════════
```

**Latency budget.** The routing query is often the most expensive step. The model itself, if GBDT or a small MLP, is cheap. Batch dispatch calls: the dispatcher asks for ETAs from many drivers to one pickup at once, which maps to a one-to-many routing query that is much cheaper than many separate queries.

**Precomputation and caching.** Many requests are near-duplicates. The menu screen asks for delivery ETAs to one user location from dozens of restaurants; nearby users ask for nearly the same things.
- Cache routing results and even full ETAs keyed on `(origin H3 cell, destination H3 cell, time bucket)` at a resolution fine enough that the error introduced is small relative to model error. Short TTLs (a minute or two) keep traffic fresh.
- Precompute one-to-many travel time tables for high-demand zones (for example, restaurant clusters to surrounding cells) on a schedule.
- Use the cache for browsing screens where approximate is fine, and compute fresh ETAs for order confirmation and dispatch.

**Feature freshness.** Real-time speeds and kitchen load come from streaming aggregates over the last few minutes, written to an online key-value store. Each feature carries a timestamp, and the model receives its age as a feature, or falls back to the historical value if it is too stale.

**Fallbacks**, in order:
1. Model unavailable or timed out → routing ETA × a per-city/hour correction factor from a lookup table.
2. Routing engine unavailable → historical travel time for the H3 cell pair and time bucket.
3. No history → straight-line distance / typical city speed, with a wide interval.

Log which fallback level produced each response, so accuracy dashboards can separate model quality from infrastructure problems.

---

## Re-Prediction During the Trip

Once a trip starts, the ETA should update as new information arrives: the driver's current position, whether food is ready, current speed.

- **Re-predict on events and on a timer.** Recompute when a stage completes (food ready, courier picked up) and every 30 to 60 seconds in between, or when the driver deviates from the planned route.
- **Condition on elapsed progress.** The remaining-time model uses features such as distance remaining, time already spent, and how the trip has gone so far relative to the original ETA. A trip running slow often continues to run slow.
- **Smooth what the user sees.** Raw re-predictions jitter. Users notice when the countdown jumps up and down. Apply damping, allow the displayed time to decrease freely but increase only when the change exceeds a threshold, and never show a remaining time that is negative.
- **Training data for re-prediction** is multiple examples per trip, one per re-prediction point. Weight or subsample so long trips do not dominate, and split train/test by trip, not by row, so examples from one trip do not leak across the split.

```python
def displayed_eta(prev_display, new_pred, alpha=0.3, jump_threshold=2.0):
    """Exponential smoothing with a threshold for upward revisions (minutes remaining)."""
    smoothed = alpha * new_pred + (1 - alpha) * prev_display
    if new_pred > prev_display and new_pred - prev_display < jump_threshold:
        return prev_display                    # ignore small upward noise
    return max(0.0, smoothed)
```

---

## Feedback Loops

ETA is not a passive forecast. It changes the outcomes it predicts.

| Loop | Mechanism | Mitigation |
|---|---|---|
| **Dispatch selection** | The dispatcher picks the driver with the lowest predicted ETA. Underestimates are selected more often, so the observed errors on chosen drivers are biased late (a winner's-curse effect) | Evaluate on all candidates with small randomized exploration; correct with propensity weighting; monitor bias on dispatched vs. all |
| **Promise shapes behaviour** | Couriers and restaurants pace themselves to the displayed time; a padded ETA gets used up | Evaluate on hard labels (GPS events), not taps; A/B test padding changes |
| **Demand shifts** | Short ETAs attract more orders to a restaurant or area, overloading it and making ETAs longer | Include live load features; re-predict frequently; coordinate with demand shaping |
| **Routing and traffic** | Drivers follow the suggested route; if many do, they create congestion on it | Real-time speeds pick it up; mainly a concern at very high fleet share |
| **Training on own predictions** | Features derived from past ETAs (e.g. "promised time") leak model behaviour into the labels | Avoid feeding the model's own previous outputs as features without care |

The dispatch selection effect is the one interviewers most often probe. When you choose the minimum over noisy estimates, the chosen estimate is optimistic on average. Labels only exist for the chosen option, so the model never sees the counterfactual, and it can drift toward underestimating exactly the cases dispatch prefers.

---

## Cold Start in New Cities

A new city has a road graph (from map data) but no trip history, no segment speed profiles from your fleet, and no restaurant prep history.

| Technique | How it helps |
|---|---|
| **Routing-only ETA first** | Map data and posted speed limits give a physics baseline from day one |
| **Transfer from similar cities** | Train a global model with city features (density, road mix, climate, vehicle mix); a new city borrows from similar ones |
| **Hierarchical backoff** | Restaurant → cuisine type and city → global prep-time priors; fine cell → coarse cell → city for residuals |
| **Third-party traffic data** | Licensed historical and real-time speeds fill the gap until your own GPS volume builds up |
| **Wider intervals and conservative padding** | Uncertainty is higher; widen displayed ranges and aim for a higher quantile until coverage is verified |
| **Fast feedback** | Retrain or recalibrate city bias terms frequently in the first weeks |

The same logic applies to a new restaurant in an established city: use cuisine, menu size, and neighbourhood priors for prep time, then shrink toward the restaurant's own history as orders arrive, e.g. `(n × own_mean + k × prior_mean) / (n + k)`.

---

## Monitoring and Drift

ETA models drift constantly, because the world they model changes: roads close, new buildings open, seasons change traffic, and the fleet mix shifts.

**What to monitor, per city and hour:**
- Live MAE, bias, late rate, and interval coverage on completed trips (labels arrive within an hour, so this is fast compared with fraud or ads conversions)
- Distribution of routing ETA versus final ETA; a sudden shift in the residual suggests routing or map changes
- Feature freshness: age of real-time speed features, share of requests falling back to historical values
- Fallback rate and latency

**Known sources of drift and responses:**

| Source | Response |
|---|---|
| **Holidays and special days** | Holiday features, training on prior years' holidays, or a separate calibration for known dates |
| **Road closures and construction** | Map updates feed the routing engine; real-time speed features catch unannounced closures; residual per segment spikes are an alert |
| **Weather events** | Weather features; real-time speeds adapt within minutes |
| **Large events** | Event calendar features; zone-level real-time demand and speeds |
| **Seasonal and long-term change** | Regular retraining on a rolling window |
| **Fleet or product changes** | New vehicle types or batching logic change the label distribution; segment metrics by vehicle and batch type |

A useful safety net is an **online bias corrector**: a per-(city, hour) additive term updated continuously from the last hour of residuals. It catches sudden systematic shifts much faster than a full retrain, and it is easy to reason about. Cap how far it can move so a data bug cannot push ETAs to absurd values.

---

## Interview Q&A

#### Why not just use the routing engine's ETA?

It is a strong baseline and should always be the fallback, but it has systematic errors. It does not model parking, building access, waiting at pickup, driver-specific speed, restaurant prep, or local effects that speed profiles miss. Those errors are predictable from features like location, time, and vehicle type, so a model trained on the residual between actual and routing ETA removes much of the bias. Keeping the routing ETA as an input also means the model inherits map and traffic improvements for free.

#### Why predict a residual instead of total duration?

The routing ETA already captures most of the variance from distance and traffic, so the residual has a smaller range and simpler structure. The model spends its capacity on what routing misses. Predicting `log(actual / routing_eta)` makes errors relative, so short and long trips contribute comparably. It also gives a natural fallback: if the model fails, the correction is zero.

#### Which loss would you use?

It depends on the consumer. For dispatch, which compares ETAs across drivers, an unbiased median or mean estimate matters, so MAE or Huber on the log ratio. For the promise shown to customers, underestimates cost more than overestimates, so a quantile loss at *q* above 0.5, or better, predict several quantiles and let the product choose. I would avoid MAPE as the training loss: it is unstable on short trips and pulls predictions low, which is the wrong direction when lateness is the expensive error.

#### How do you show a range like "25 to 35 minutes" and make sure it's honest?

Predict quantiles (e.g. P10, P50, P90) with pinball loss, using offset parameterization so they don't cross. Then calibrate on recent data: measure empirical coverage per city and time band and adjust interval width with a conformal-style correction until an 80% interval contains about 80% of actuals. Monitor coverage live. Rounding and display rules live in a separate policy layer.

#### How would you structure a food-delivery ETA?

Decompose it into stages: restaurant acceptance, prep, courier assignment, travel to restaurant, handoff, travel to customer, last mile. Pickup time is the max of food-ready time and courier arrival, plus handoff, so it is not a simple sum. I would build stage models (especially prep time and travel), then a total-time model that uses them as features. As each stage completes, its uncertainty drops to zero and the total is re-predicted.

#### Dispatch picks the driver with the lowest ETA. What problem does that create for the model?

Selection bias. Choosing the minimum over several noisy estimates favours drivers whose ETA was underestimated, so the realized pickup times for chosen drivers run later than predicted even if the model is unbiased across all candidates. Labels only exist for chosen drivers, so retraining on them does not fix it. Mitigations: measure bias separately for dispatched trips, use small amounts of randomized assignment to get unbiased labels, reweight by selection propensity, and include driver-level features that reduce the noise that selection exploits.

#### How do you serve this at 200k requests per second within 30 ms?

Most cost is in routing, not the model. Use a precomputed hierarchy (contraction hierarchies or a customizable variant that accepts live weights), one-to-many queries for dispatch, and a cache keyed on origin and destination H3 cells plus a time bucket for browsing screens. Features come from an online store fed by streaming aggregates. The model is GBDT or a small network, batched per request. A tiered fallback (lookup correction, historical cell-pair time, distance heuristic) guarantees a response.

#### ETA accuracy in one city got worse overnight. How do you debug it?

First check whether the routing ETA or the residual changed: a jump in routing ETA points to a map or traffic-feed problem, a jump in the residual points to the model or its features. Check feature freshness (stale real-time speeds are a common cause) and fallback rate. Slice by zone and hour to see if it is local (a road closure, an event) or city-wide (weather, holiday, a logging change in trip events). Check whether a product change such as new batching rules changed what the label means. The online bias corrector buys time while the root cause is fixed.

#### How do you launch in a new city?

Start from routing with posted speeds, plus a global model that uses city-level features so the new city borrows from similar cities. Use third-party traffic data if available. Show wider intervals and aim for a higher quantile at first. Recalibrate per-city bias terms daily as trips accumulate, and switch to the regular model once coverage and bias are verified on local data.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Ignoring the routing baseline | Model relearns physics poorly; no fallback | Predict a residual on top of routing ETA |
| Training with MSE on raw minutes | Dominated by outlier trips; mean is not what users need | MAE/Huber on log ratio; quantile loss for promises |
| Using MAPE as the loss | Unstable on short trips; biases predictions low | Log-space loss or quantile loss |
| Treating over- and under-estimates equally | Late deliveries cost far more than early ones | Asymmetric or quantile loss; separate display policy |
| Reporting only overall MAE | Hides bias in specific cities, hours, trip types | Slice MAE, bias, P90 error, coverage by segment |
| Recomputing features after the fact | Leaks congestion the trip experienced into features | Log served features; train on logs |
| Using tap events as labels | Couriers tap early or late; noisy labels | Derive events from GPS geofences; clean trips |
| Dropping cancelled trips | Biases data toward fast trips | Treat as censored; track censoring rate |
| Summing stage estimates naively | Pickup is a max, not a sum; stages are correlated | Compose distributions; end-to-end model on top |
| Random train/test split | Leaks traffic conditions across the split | Time-based split; split re-predictions by trip |
| Ignoring dispatch selection bias | Model looks unbiased offline, runs late in production | Monitor dispatched-trip bias; exploration; reweighting |
| Jittery live countdown | Users lose trust when the ETA jumps | Smoothing and thresholds on upward revisions |
| User-level A/B test for dispatch changes | Arms share drivers; results biased | Switchback or region-level experiments |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Ads CTR Prediction System Design](./ads_ctr_prediction.md)
- [Fraud Detection System Design](./fraud_detection.md)
- [Search and Ranking System Design](./search_ranking_system.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Geospatial Data Engineering](../data_engineering/intro_geospatial.md)
- [Apache Kafka](../data_engineering/intro_apache_kafka.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md)
- [Ensemble Methods](../classical_ml/intro_ensemble_methods.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Graph Neural Networks](../deep_learning/intro_graph_neural_networks.md)
- [Sequence Models](../deep_learning/intro_sequence_models.md)
- [Transformers](../deep_learning/intro_transformers.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Feature Store](../mlops/intro_feature_store.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
