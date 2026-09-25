# Dynamic Pricing System Design

"Design a dynamic pricing system" covers ride-hailing surge, hotel and airline revenue management, and e-commerce price updates. Candidates often answer with a demand forecasting model and stop. The hard part is different: the system has to know what happens to demand **when it changes the price**, and historical data answers that question badly, because past prices were set in response to demand. Pricing is a causal problem wearing a forecasting costume. On top of that, prices are visible to customers, regulators, and competitors, so guardrails, fairness, and explainability are part of the design, not an afterthought.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [Business Objective and Constraints](#business-objective-and-constraints)
3. [System Overview](#system-overview)
4. [Demand Forecasting vs Price Elasticity](#demand-forecasting-vs-price-elasticity)
5. [Estimating Causal Elasticity](#estimating-causal-elasticity)
6. [Optimal Price Under Constant Elasticity](#optimal-price-under-constant-elasticity)
7. [Two-Sided Marketplace Surge](#two-sided-marketplace-surge)
8. [Price Exploration with Bandits](#price-exploration-with-bandits)
9. [Competitor Prices and Market Signals](#competitor-prices-and-market-signals)
10. [Segmentation and Personalised Pricing](#segmentation-and-personalised-pricing)
11. [Guardrails](#guardrails)
12. [Evaluation](#evaluation)
13. [Serving Architecture](#serving-architecture)
14. [Monitoring and Feedback Loops](#monitoring-and-feedback-loops)
15. [Explaining Prices to Users](#explaining-prices-to-users)
16. [Interview Q&A](#interview-qa)
17. [Common Pitfalls](#common-pitfalls)
18. [Related Topics](#related-topics)

---

## Clarify the Problem First

The three common settings look similar but have different mechanics. Ask which one the interviewer means.

| Setting | What is priced | Key feature |
|---|---|---|
| **Ride-hailing surge** | A trip in a zone and time window | Two-sided: price moves riders and drivers; minutes matter |
| **Hotel / airline** | A room-night or seat for a future date | Fixed perishable inventory; price over a booking horizon |
| **E-commerce** | A product (SKU) | Large catalog, competitor prices visible, slower changes |

**Questions to ask:**
- **Who sets the price today?** A rules engine, analysts, or nothing (fixed prices)? The existing policy is the baseline and also the source of confounding in the data.
- **How often can prices change?** Every minute (surge), several times a day (airline fare buckets), daily or weekly (retail).
- **What is the unit of pricing?** Per zone and minute, per SKU, per SKU and region, per customer? Per-customer pricing raises legal and trust issues covered below.
- **What levers exist besides price?** Driver incentives, discounts, fees, shipping cost, bundling. Price is often not the only control.
- **Scale.** Number of SKUs or zones, quote requests per second, latency budget for a price quote.

For the rest of this guide, assume: **a ride-hailing platform in a few hundred cities, prices recomputed per zone every minute, ~50k price quotes/sec at peak, p99 quote latency of 50 ms; plus notes on the retail case where it differs.** These are working assumptions for the exercise, not industry figures.

### Non-Functional Requirements

| Requirement | Target |
|---|---|
| Quote latency (p99) | ~50 ms, since the price is shown before the user requests |
| Price freshness | Zone multipliers updated every 1 to 2 minutes |
| Availability | Always return a price; a base fare with multiplier 1.0 is the fallback |
| Consistency | A quoted price is honoured for a short window (e.g. a few minutes) |
| Auditability | Every quoted price is logged with inputs, model version, and guardrail decisions |

---

## Business Objective and Constraints

"Maximize revenue" is rarely the right answer. Pick the objective explicitly, because it changes the optimal price.

| Objective | Optimizes | Risk if used alone |
|---|---|---|
| **Revenue** | Price × quantity | Ignores cost; can sell at a loss or price too high for volume |
| **Margin / profit** | (Price - cost) × quantity | Can shrink volume and market share |
| **Marketplace balance** | Match rate, wait times, completed trips | Cheap for riders now, drivers leave later |
| **Long-term value** | Retention, lifetime value, trust | Hard to measure; slow feedback |

A practical formulation is to maximize short-term margin subject to constraints on the rest, or to optimize a weighted objective with a long-term penalty term that is estimated from experiments (for example, the retention cost of a price increase).

**Hard constraints:**
- **Floors and ceilings.** Never below cost or a legal minimum wage-related fare; never above a surge cap or a regulated maximum.
- **Regulation.** Price-gouging laws during declared emergencies, taxi fare caps in some cities, consumer protection rules on drip pricing and hidden fees, and anti-discrimination law.
- **Fairness.** Price must not vary by protected attributes or close proxies for them (neighbourhood can be one).
- **Consistency.** Rapid price flicker between refreshes damages trust even if each price is "optimal."
- **Contracts.** Airlines and hotels have fare classes and channel parity agreements; retailers have minimum advertised price agreements with suppliers.

---

## System Overview

```
                            DYNAMIC PRICING SYSTEM
═══════════════════════════════════════════════════════════════════════════════

  Offline / slow loop (hours to days)
  ───────────────────────────────────
  Transactions, quotes, ──► Demand model ──────────► baseline demand q0(x, t)
  conversions, experiments    (forecast)
                          ──► Elasticity model ────► e(x): causal price response
                              (experiments, DML, IV)
  Competitor feeds ───────► Market signal features
                                        │
                                        ▼
  Near-real-time loop (minutes)     Price optimizer
  ─────────────────────────────     argmax objective(p) s.t. constraints
  Open requests, available  ──►         │
  drivers, ETAs per zone               ▼
                                   Guardrails (floors, caps, max step,
                                   fairness rules, overrides, kill switch)
                                        │
                                        ▼
                                   Price table per (zone, time bucket)
                                        │
  Request path (ms)                     ▼
  ─────────────────             Quote API ──► user sees price ──► accept / reject
                                        │                              │
                                        └──── log quote + features ◄───┘
                                                       │
                                                       ▼
                                  Training data, monitoring, experiment analysis

═══════════════════════════════════════════════════════════════════════════════
```

The key design choice is the split between **what demand would be at a reference price** (forecasting) and **how demand responds to changing the price** (elasticity). The optimizer combines them.

---

## Demand Forecasting vs Price Elasticity

These are different questions and need different data.

| | Demand forecasting | Price elasticity |
|---|---|---|
| **Question** | How many requests or sales will there be? | How much does demand change if we change price? |
| **Type** | Prediction | Causal (intervention) |
| **Good data** | Any historical data | Data where price varied for reasons unrelated to demand |
| **Model** | Time series, GBDT on calendar, weather, events | Experiments, IV, double ML, structural models |
| **Failure mode** | Poor accuracy (visible in backtests) | Biased slope (invisible in backtests) |

Elasticity is defined as `e = -(dq/q) / (dp/p) = -d log q / d log p`. With the minus sign, `e = 2` means a 1% price increase reduces quantity by about 2%. Demand is **elastic** when `e > 1` and **inelastic** when `e < 1`.

### Why observational price-demand data is confounded

Under any existing pricing policy, price is set **because of** demand. Surge goes up when many people request rides. Hotels raise rates for dates that are filling up. So in historical data, high prices co-occur with high demand.

```
        Demand shock (event, rain, holiday)
            │                      │
            ▼                      ▼
          Price  ───────────►  Quantity
                (causal effect,
                 what we want)
```

Regress quantity on price and you mix the causal effect (negative) with the demand shock (positive). The estimated slope is biased toward zero and can even come out positive, which would imply "raise prices to sell more." Adding features that capture the demand shock helps only to the extent they capture everything the old policy reacted to, and the old policy often used real-time signals that were never logged.

A second problem: if the old policy was a deterministic function of logged features, price has **no independent variation** once you condition on those features. No amount of modelling recovers elasticity from data with no variation (the positivity assumption fails).

---

## Estimating Causal Elasticity

| Method | Source of price variation | Strength | Weakness |
|---|---|---|---|
| **Randomized price experiments** | Random price perturbations (e.g. ±5%) per zone-time, SKU-day, or session | Unbiased; the gold standard | Costs revenue; customers may notice; interference |
| **Natural experiments** | Price changes caused by things unrelated to demand: rounding, cost shocks, tax changes, system outages, rule thresholds | Free | Rare; may not generalize |
| **Instrumental variables** | A variable that moves price but affects demand only through price (supplier cost changes, surge-rule thresholds, random experiment assignment) | Handles unobserved confounders | Exclusion restriction is untestable; weak instruments |
| **Double ML** | Residual price variation after controlling for rich covariates | Flexible nonlinear controls; valid confidence intervals | Needs unconfoundedness given X and some residual variation |

**Regression discontinuity** also appears in practice: a surge rule that switches from 1.0x to 1.2x when the demand-supply ratio crosses a threshold gives nearly identical zones on either side of the cutoff with different prices.

**The practical recipe:** build small, permanent randomized price jitter into the policy (a few percent, within guardrails). It creates the independent variation every other method depends on, and it can serve directly as an instrument for the realized price. Log the random component separately.

### Double ML for elasticity

Double (debiased) ML, as in the partially linear model `log q = θ·log p + g(X) + ε`, `log p = m(X) + v`:
1. Predict `log q` from X and `log p` from X with any flexible model, using cross-fitting (predictions on held-out folds).
2. Take residuals of both.
3. Regress the quantity residual on the price residual. The slope is `θ`, and elasticity is `-θ`.

It only works if X contains the confounders and there is residual price variation (the `v` term), which is why logging the random jitter matters.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(0)
n = 5000
X = rng.normal(size=(n, 3))                  # demand drivers: hour, weather, events
demand_shock = 1.0 * X[:, 0] + 0.5 * np.sin(X[:, 1])
log_p = 0.6 * demand_shock + 0.1 * rng.normal(size=n)       # old policy + jitter
true_e = 2.0
log_q = 3.0 - true_e * log_p + 1.5 * demand_shock + 0.3 * rng.normal(size=n)

# Naive regression: confounded, slope biased toward zero or even positive
naive_slope = np.polyfit(log_p, log_q, 1)[0]

# Double ML with cross-fitting
model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
q_res = log_q - cross_val_predict(model, X, log_q, cv=5)
p_res = log_p - cross_val_predict(model, X, log_p, cv=5)
theta = (p_res @ q_res) / (p_res @ p_res)

print(f"naive elasticity: {-naive_slope:.2f}")   # wrong sign here
print(f"DML elasticity:   {-theta:.2f}")          # near the true 2.0; naive is far off
```

In practice, use a library such as EconML or DoubleML for standard errors and heterogeneous effects `e(X)`, and check the result against an experiment before trusting it.

### Heterogeneous elasticity

Elasticity varies by context: airport trips are less price-sensitive than short city hops, business travel less than leisure, branded goods less than commodities. Estimate `e(X)` with causal forests or DML with an interaction model, then **shrink segment estimates toward a pooled value**, because per-segment experiments are small and noisy. A noisy elasticity estimate feeds straight into the price formula below, and the formula is very sensitive when `e` is near 1.

---

## Optimal Price Under Constant Elasticity

Assume constant-elasticity demand `q(p) = A · p^(-e)` and unit cost `c`. Profit is:

```
π(p) = (p - c) · A · p^(-e)

dπ/dp = A · p^(-e) - e · (p - c) · A · p^(-e-1) = 0
     => p = e · (p - c)
     => p* = c · e / (e - 1)          valid for e > 1
```

Equivalently, the **Lerner condition**: `(p* - c) / p* = 1 / e`. The markup over cost is set by elasticity alone; the scale `A` (the demand forecast) drops out.

| Elasticity e | p* with c = 10 | Markup over cost |
|---|---|---|
| 1.5 | 30.00 | 200% |
| 2 | 20.00 | 100% |
| 3 | 15.00 | 50% |
| 5 | 12.50 | 25% |

Three points to raise in an interview:
- **If e ≤ 1** the formula has no finite solution: under this model profit keeps rising with price. That means the constant-elasticity assumption has broken down over that range (real demand becomes more elastic at high prices), so the answer is capped by the price ceiling or a better demand curve.
- **Revenue maximization** (c = 0) under constant elasticity has no interior optimum either. With linear demand `q = a - b·p`, revenue is maximized at `p = a / (2b)`, where elasticity equals exactly 1. The functional form matters.
- **Sensitivity.** Moving from e = 1.5 to e = 1.2 moves p* from 30 to 60. Near e = 1, small estimation errors cause huge price errors, which is why guardrails and shrinkage are needed.

```python
import numpy as np

def optimal_price_constant_elasticity(c, e):
    if e <= 1:
        raise ValueError("no finite optimum for e <= 1 under constant elasticity")
    return c * e / (e - 1)

def profit(p, c, e, A=1000.0):
    return (p - c) * A * p ** (-e)

c = 10.0
for e in [1.5, 2.0, 3.0, 5.0]:
    grid = np.linspace(c + 0.01, 10 * c, 200_000)
    p_grid = grid[np.argmax(profit(grid, c, e))]
    p_star = optimal_price_constant_elasticity(c, e)
    print(f"e={e}: closed form {p_star:.2f}, grid search {p_grid:.2f}, "
          f"Lerner (p-c)/p={(p_star - c) / p_star:.3f} vs 1/e={1 / e:.3f}")
```

With capacity limits (seats, rooms, drivers), the unconstrained optimum is not enough. The airline version is a constrained problem: maximize `Σ (p_t - c) · q_t(p_t)` over the booking horizon subject to `Σ q_t ≤ capacity`. The Lagrange multiplier on capacity acts as an opportunity cost that raises the effective `c`, which is why prices climb as a flight fills. Classic revenue management solves this with bid prices or dynamic programming over remaining inventory and time.

---

## Two-Sided Marketplace Surge

Ride-hailing surge is not just "charge more when demand is high." The price has two jobs:

1. **Ration demand in the short term.** Some riders wait, walk, or take transit, so the remaining requests can be served with acceptable wait times.
2. **Attract supply over the next minutes and hours.** Drivers move toward high-multiplier zones and log on when earnings are higher.

```
   Demand > supply in zone Z
            │
            ▼
   Multiplier ↑ ──► fewer requests (rider elasticity, minutes)
            │
            └────► drivers reposition / log on (supply elasticity, 10-30 min)
                         │
                         ▼
               supply ↑, wait times ↓ ──► multiplier ↓
```

**Objective.** Rather than margin alone, the marketplace target is usually a service level: keep pickup ETA and the unfulfilled request rate below thresholds while not overpricing. A common formulation picks the smallest multiplier such that expected demand at that price can be served by expected supply within the target wait time.

**Supply response has a lag.** Riders react in seconds; drivers take 10 to 30 minutes to arrive. If the multiplier reacts only to the current imbalance, drivers flood in after the peak has passed, the multiplier drops, and drivers leave: an oscillation. Mitigations:
- **Predict the imbalance** a few intervals ahead using the demand forecast, not only the current queue.
- **Smooth** the multiplier (exponential smoothing, rate limits per interval).
- **Spatial smoothing** across neighbouring hexagons, so a single cell does not flicker and drivers do not chase noise at cell boundaries.
- **Separate the rider price from driver pay** where allowed (upfront pricing and targeted driver incentives), so each side can be tuned with its own elasticity.

```python
import numpy as np

def surge_multiplier(requests, drivers, prev_mult, target_ratio=1.0,
                     sensitivity=0.8, alpha=0.3, max_step=0.2,
                     floor=1.0, cap=3.0):
    """Zone multiplier from a demand/supply ratio with smoothing and a step limit."""
    ratio = requests / max(drivers, 1)
    raw = 1.0 + sensitivity * max(0.0, ratio - target_ratio)
    smoothed = alpha * raw + (1 - alpha) * prev_mult
    step = np.clip(smoothed - prev_mult, -max_step, max_step)
    return float(np.clip(prev_mult + step, floor, cap))

m = 1.0
for req, drv in [(40, 40), (90, 40), (120, 45), (100, 70), (60, 80)]:
    m = surge_multiplier(req, drv, m)
    print(req, drv, round(m, 2))
```

A learned policy can replace the rule, but keep the structure: forecast, target service level, smoothing, and caps. Reinforcement learning for surge is discussed in the literature; in production, simulation-tested rules with learned inputs are more common because they are easier to reason about and to audit.

---

## Price Exploration with Bandits

When elasticity is uncertain (a new product, a new city, a changed market), the system must try prices to learn. A **contextual bandit** frames each pricing decision as choosing an arm (a price level or multiplier) given context (segment, time, inventory), and observing reward (margin from the sale, or zero).

**Thompson sampling** keeps a posterior over each arm's conversion rate, samples from it, and picks the arm with the highest sampled expected reward. Arms that are uncertain but plausibly good get tried; arms that are clearly bad stop being tried.

```python
import numpy as np

rng = np.random.default_rng(1)
prices = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
cost = 6.0
true_conv = 0.9 * (prices / 8.0) ** -2.5          # unknown to the algorithm
alpha, beta = np.ones(len(prices)), np.ones(len(prices))
true_reward = (prices - cost) * true_conv
regret = 0.0

for t in range(20_000):
    sampled_conv = rng.beta(alpha, beta)
    arm = np.argmax((prices - cost) * sampled_conv)
    sold = rng.random() < true_conv[arm]
    alpha[arm] += sold
    beta[arm] += 1 - sold
    regret += true_reward.max() - true_reward[arm]

print("best price:", prices[np.argmax(true_reward)])
print("pulls per price:", (alpha + beta - 2).astype(int))
print(f"cumulative regret: {regret:.0f}")
```

**The cost of exploration** is regret: margin lost by charging non-optimal prices while learning. Ways to reduce it:
- **Parametric sharing.** Instead of independent arms, fit one demand curve (e.g. log-linear in price) with a Bayesian prior. A sale at 12 then tells you something about 14. This learns much faster than independent arms.
- **Restrict the arm set** to a narrow band around the current price. Wild prices are both expensive and visible.
- **Priors from similar products or cities** so learning starts near the answer.
- **Stop exploring** when the posterior on the best price is tight, but keep a small jitter to detect drift.

Bandit pitfalls specific to pricing: customers who see different prices for the same thing at the same time will notice and complain; delayed rewards (a hotel booking months out) slow learning; and a bandit that optimizes immediate conversion ignores long-term effects like churn from a bad price experience.

---

## Competitor Prices and Market Signals

In retail and travel, competitor prices are among the strongest features. In ride-hailing, the rival app's surge affects which app riders choose.

| Signal | Use |
|---|---|
| **Competitor price for the same item** | Feature in demand model (price gap or ratio); input to rules like "stay within 5% of the lowest major competitor" |
| **Competitor stock-outs** | Demand shifts to you; elasticity drops temporarily |
| **Search and view volume** | Leading indicator of demand before transactions happen |
| **Events, weather, holidays** | Demand forecast features; surge anticipation |
| **Input costs** | Fuel, supplier cost; also a candidate instrument for price |

**Data quality problems:** scraped prices are delayed, can be wrong (a different variant, a member-only price, a regional page), and competitors can serve different prices to scrapers. Keep a freshness timestamp and a match-confidence score on each competitor price, and fall back when they are stale or low confidence.

**Strategic caution:** a rule that matches or undercuts the lowest competitor can start a price war with another algorithm that does the same. And pricing rules that simply follow competitors upward raise legal questions about algorithmic collusion, even without any communication between firms. Have legal review of any rule that references competitor prices directly.

---

## Segmentation and Personalised Pricing

There is a spectrum from uniform prices to fully individual prices:

| Level | Example | Risk |
|---|---|---|
| **Uniform** | Same price for everyone at a given time | Leaves value on the table; lowest risk |
| **Context-based** | By time, location, lead time, inventory | Widely accepted (surge, off-peak, early-bird) |
| **Segment-based, transparent** | Student, senior, member discounts | Accepted if clearly disclosed and opt-in |
| **Inferred-segment** | Price by device type, browsing history, inferred income | Trust damage when discovered; possible discrimination |
| **Individual** | Price from a model of each person's willingness to pay | Highest legal and reputational exposure |

**Why personalised pricing is risky:**
- **Fairness and discrimination.** Features like location, device, or name can proxy for race, age, or income. A model can learn to charge protected groups more without any protected attribute as input. Audit price outcomes by group, not just the features used.
- **Legal exposure.** Anti-discrimination and consumer protection law, data protection rules on profiling and automated decisions, and sector rules (insurance, credit, housing) apply. Requirements differ by jurisdiction, so involve legal early.
- **Customer trust.** People compare prices with friends. Discovering that someone else paid less for the same thing at the same moment feels unfair, even when it is legal. Public backlash has followed reports of this kind of pricing in several industries.
- **Strategic behaviour.** Customers learn to clear cookies, switch devices, or wait for discounts, which erodes the model's signal.

A defensible default: vary prices by **context** (time, place, inventory, lead time) and offer **discounts** through transparent, opt-in programs, rather than raising prices for individuals predicted to be willing to pay more.

---

## Guardrails

The optimizer proposes a price; guardrails decide what ships. They sit between the model and the customer and are owned jointly with product, operations, and legal.

| Guardrail | Purpose |
|---|---|
| **Floor and ceiling** | Cost floor, regulatory caps, surge cap |
| **Max change per interval** | Limit how far price moves per refresh (e.g. ±10% per hour for retail) to stop flicker and runaway loops |
| **Max change vs reference** | Cap deviation from a reference price (list price, 30-day median) |
| **Emergency mode** | Freeze or cap prices during declared emergencies and disasters |
| **Fairness checks** | Block price rules or segments that fail outcome audits |
| **Human override** | Operations can pin a price for a zone, SKU, or event |
| **Kill switch** | Revert everything to a static rule or last-known-good table within seconds |
| **Sanity checks** | Reject prices that are non-numeric, zero, or orders of magnitude off the reference |

```python
import numpy as np

def apply_guardrails(proposed, prev, reference, floor, ceiling,
                     max_step_pct=0.10, max_dev_pct=0.50,
                     override=None, kill_switch=False):
    """Return the price to publish and the reason it differs from the proposal."""
    if kill_switch:
        return reference, "kill_switch"
    if override is not None:
        return override, "override"
    if not np.isfinite(proposed) or proposed <= 0:
        return prev, "invalid_proposal"
    p = np.clip(proposed, prev * (1 - max_step_pct), prev * (1 + max_step_pct))
    p = np.clip(p, reference * (1 - max_dev_pct), reference * (1 + max_dev_pct))
    p = float(np.clip(p, floor, ceiling))
    return p, "ok" if np.isclose(p, proposed) else "clipped"

print(apply_guardrails(proposed=30.0, prev=20.0, reference=20.0,
                       floor=12.0, ceiling=40.0))   # (22.0, 'clipped')
```

Log every guardrail decision. A rising clip rate is an early sign that the model has drifted or its elasticity estimate is off.

---

## Evaluation

### Offline

Offline evaluation of a pricing policy is hard because you only observe demand at the prices that were charged.
- **Demand model accuracy**: forecast error by segment on a time-based split, as for any forecast.
- **Elasticity validation**: compare estimated elasticities against held-out randomized experiments; this is the only check of the causal part.
- **Off-policy evaluation**: with logged propensities from randomized jitter, inverse propensity scoring estimates the reward of a new policy on old data. Without randomization, it is not possible for prices outside the logged range.
- **Simulation**: a marketplace simulator (riders, drivers, movement) calibrated on real data, used to stress-test surge rules for oscillation and extreme events.

### Online experiments and interference

User-level A/B tests are biased in marketplaces. If treatment riders see lower prices, they request more rides and consume drivers that control riders would have gotten. Control looks worse, and the treatment effect is overstated. The same happens in retail with limited inventory. This is a violation of SUTVA (no interference between units).

| Design | How | When |
|---|---|---|
| **User-level A/B** | Randomize users | No shared supply or inventory; price display tests |
| **Switchback** | Alternate the whole zone or city between policies in time blocks (e.g. 30 to 60 min) | Marketplace pricing; the standard choice for surge |
| **Geo / cluster randomization** | Randomize cities or regions | Long-horizon effects; fewer units, lower power |
| **Synthetic control** | Compare treated market to a weighted combination of untreated ones | Single-market launches |

**Switchback details:** block length must exceed the system's memory (drivers repositioning, queued requests), or effects carry over between blocks. Drop a burn-in period at the start of each block, randomize the block order, and analyse with block-level variance (or randomization inference), not per-trip standard errors.

### Metrics

| Type | Examples |
|---|---|
| **Primary** | Margin or contribution per period, completed trips, gross bookings |
| **Marketplace** | Pickup ETA, unfulfilled request rate, driver earnings per hour, utilization |
| **Customer** | Conversion from quote to request, cancellations after seeing price, complaints mentioning price |
| **Long-term** | Retention of riders exposed to high prices, repeat purchase rate, driver churn |
| **Guardrails** | Clip rate, price volatility, fairness outcome gaps, latency |

**Short-term wins can be long-term losses.** Higher prices lift margin in a two-week test while retention damage shows up months later. Use holdout groups that stay on the old policy for months, and track cohort retention of users who were exposed to high prices.

---

## Serving Architecture

```
  Every 1-2 min per zone (streaming job)
  ──────────────────────────────────────
  Requests, driver pings ──► Kafka ──► Zone aggregator ──► imbalance, forecasts
                                                   │
                                                   ▼
                                         Pricing optimizer + guardrails
                                                   │
                                                   ▼
                              Price table: (zone hex, time bucket, product) → multiplier
                                  in online KV store, versioned, TTL
                                                   │
  Quote request ──► Quote API ──► KV lookup ──► base fare(distance, ETA) × multiplier
                        │                              │
                        │                              ▼
                        │                   quote id + expiry, logged
                        └── fallback: last table → static multiplier 1.0
```

- **Precompute rather than compute per request.** The multiplier depends on the zone and time bucket, not the individual rider, so a table lookup keeps quote latency low. Only the base fare (distance, predicted duration) is computed per request.
- **Caching per region and time bucket.** Key by H3 cell and a one-minute bucket; the table is small and fits in memory on each quote server.
- **Quote locking.** Return a quote id with an expiry; the price the user accepts is the price charged, even if the table updates.
- **Fallbacks, in order:** latest table → previous table if the latest is stale → static rule (multiplier 1.0 or a per-city time-of-day schedule). Log which level served each quote.
- **Retail differences:** prices change hourly or daily, so a batch job writes the price file, a review queue catches large changes, and the storefront reads from a price service with a cache. Latency matters less; correctness and audit matter more.

---

## Monitoring and Feedback Loops

**Price anomalies.** Alert on prices at floor or ceiling for many zones at once, sudden jumps versus the same hour last week, zero or extreme prices, and a spike in guardrail clips. Many public pricing incidents come from a bug (a unit mix-up, a missing competitor price read as zero) rather than a bad model; sanity checks and a kill switch catch them.

**Input drift.** Monitor freshness of competitor feeds and supply signals, forecast error, and whether elasticities from ongoing jitter experiments are moving.

**Feedback loops:**

| Loop | Mechanism | Mitigation |
|---|---|---|
| **Own-policy confounding** | Model trained on prices it set learns its own policy, not demand | Keep randomized jitter; train elasticity on the random component |
| **Competitor algorithms** | Two repricers reacting to each other can spiral down (price war) or up | Step limits, floors, dampening; don't key price mechanically off one competitor |
| **Driver chasing** | Drivers move to high multipliers, which then drop | Forecast-based surge, smoothing, spatial smoothing |
| **Customer adaptation** | Users learn to wait out surge or buy on discount days | Track timing shifts; long-term metrics |
| **Stock-out loop** | Low price sells out, demand data is censored at zero inventory | Treat stock-outs as censored demand, not zero demand |

---

## Explaining Prices to Users

A price the user cannot understand feels arbitrary, and arbitrary feels unfair.

- **Show the reason in plain terms**: "Fares are higher because demand is high in this area," with the multiplier or the extra amount visible before the user commits.
- **Show the alternative**: "Prices are expected to drop in about 10 minutes," or a cheaper product option. Only show a forecast if it is reliable.
- **Upfront, all-in prices.** Hidden fees added at checkout damage trust and are restricted by consumer protection rules in many places.
- **Consistent explanations.** The explanation must come from the actual pricing inputs (logged per quote), not a generic message; otherwise support agents and regulators will find mismatches.
- **Internal explainability.** Operations and legal need per-quote breakdowns: base fare, multiplier, which guardrails fired, and model version. This is also required to answer complaints and audits.

---

## Interview Q&A

#### Why can't you estimate price elasticity from historical sales data?

Because the old pricing policy set prices in response to demand. High demand periods had high prices, so price and quantity move together for reasons that have nothing to do with the causal effect of price. A regression of quantity on price picks up both and is biased toward zero, sometimes even positive. If the old policy was a deterministic function of logged features, there is no independent price variation left after conditioning, so no model can recover the effect. You need price variation that is unrelated to demand: randomized experiments, natural experiments, or an instrument.

#### What is the optimal price under constant elasticity?

With demand `q = A·p^(-e)` and unit cost `c`, profit `(p - c)·A·p^(-e)` is maximized at `p* = c·e/(e - 1)` for `e > 1`, which is the Lerner condition `(p - c)/p = 1/e`. With `c = 10` and `e = 2`, `p* = 20`. The demand scale drops out, so only elasticity and cost matter. For `e ≤ 1` there is no finite optimum under this model, which signals the model is wrong over that range and the ceiling binds.

#### How would you design surge pricing for a ride-hailing app?

Compute per-zone imbalance every minute from open requests, available drivers, and a short-term demand forecast. Pick a multiplier that brings expected demand in line with supply at a target pickup time, using rider elasticity for the demand side and driver supply response for the medium term. Smooth over time and across neighbouring hexagons, limit the change per interval, and cap the multiplier. Precompute a (zone, time bucket) table so quotes are a lookup. Evaluate with switchback experiments on pickup time, completion rate, driver earnings, and rider retention.

#### Why do surge prices oscillate, and how do you stop it?

Riders react instantly but drivers take 10 to 30 minutes to arrive. A multiplier driven by the current imbalance attracts drivers who arrive after the peak, the multiplier then drops, drivers leave, and the imbalance returns. Use forecasted imbalance instead of current, smooth the multiplier, rate-limit changes, smooth spatially, and test the rule in a simulator for stability before launch.

#### How would you run an experiment on a new pricing algorithm in a marketplace?

Not with user-level randomization, because treatment and control share drivers or inventory and the difference is biased. Use switchback experiments that alternate the whole zone between policies in time blocks longer than the system's carryover, with burn-in periods dropped and variance computed at the block level. For long-term effects, add geo-level holdouts that stay on the old policy for months.

#### When would you use a bandit for pricing, and what does it cost?

When elasticity is uncertain and conditions allow repeated decisions with fast feedback: new products, new markets, or detecting drift. Thompson sampling over a small set of price levels, or better a Bayesian parametric demand curve, balances learning and earning. The cost is regret, margin lost while trying worse prices, plus the reputational cost of customers seeing different prices. Keep arms within a narrow band, share information across arms and segments, and use priors from similar products.

#### Should prices be personalised?

Be careful. Context-based variation (time, place, inventory) and transparent opt-in discounts are widely accepted. Pricing individuals by predicted willingness to pay risks discrimination through proxy features, legal exposure under consumer protection and data protection law, and loss of trust when customers compare prices. If segments are used, audit price outcomes by group, and prefer discounts to targeted increases.

#### A competitor's repricing bot and yours are both running. What can go wrong?

If both undercut the other, prices spiral down to the floor. If both follow each other upward, prices drift up in a way that can look like collusion even without communication. Use step limits and floors, avoid rules that mechanically match one competitor, dampen responses to competitor changes, and monitor for sustained drift and correlated price moves.

#### What guardrails would you put around the pricing model?

Floors and ceilings (cost, regulation, caps), a maximum change per interval, a maximum deviation from a reference price, emergency-mode caps, fairness audits, human overrides per zone or SKU, a kill switch that reverts to a static rule, and sanity checks against obviously wrong prices. Log every guardrail action; a rising clip rate signals model drift.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Treating pricing as demand forecasting | Forecast accuracy says nothing about the effect of changing price | Separate forecasting from causal elasticity |
| Estimating elasticity by regressing sales on price | Confounded by demand; slope biased toward zero or wrong sign | Randomized jitter, IV, natural experiments, DML with checks |
| No independent price variation in logs | Elasticity is not identifiable | Build small randomized perturbations into the policy |
| Plugging noisy elasticity into p* = c·e/(e-1) | Near e = 1, small errors produce huge prices | Shrink estimates, bound e, apply ceilings |
| Maximizing short-term revenue only | Hurts retention, driver supply, and trust | Constrain on marketplace and long-term metrics |
| Surge reacting only to current imbalance | Oscillation from delayed supply response | Forecast-based surge, smoothing, step limits |
| User-level A/B tests in a marketplace | Interference biases the result | Switchback or geo experiments |
| Trusting scraped competitor prices blindly | Stale or mismatched prices drive bad decisions | Freshness and match-confidence checks, fallbacks |
| Personalised prices from inferred traits | Discrimination via proxies, legal risk, backlash | Context-based pricing, transparent discounts, outcome audits |
| No kill switch or override | A bug publishes absurd prices at scale | Guardrail layer with sanity checks and fast revert |
| Treating stock-outs as zero demand | Underestimates demand and distorts elasticity | Model censored demand |
| Unexplained price changes | Users perceive unfairness | Show reasons, upfront all-in prices, consistent explanations |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [ETA Prediction System Design](./eta_prediction.md)
- [Ads CTR Prediction System Design](./ads_ctr_prediction.md)
- [Fraud Detection System Design](./fraud_detection.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Causal Inference and Uplift Modeling](../classical_ml/intro_causal_inference.md)
- [Time Series](../classical_ml/intro_time_series.md)
- [Statistics and Probability](../classical_ml/intro_statistics_probability.md)
- [Anomaly Detection](../classical_ml/intro_anomaly_detection.md)
- [Reinforcement Learning](../deep_learning/intro_reinforcement_learning.md)
- [Geospatial Data Engineering](../data_engineering/intro_geospatial.md)
- [Apache Kafka](../data_engineering/intro_apache_kafka.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Responsible AI and Fairness](../mlops/intro_responsible_ai_fairness.md)
- [Model Explainability](../mlops/intro_model_explainability.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
