# Causal Inference and Uplift Modeling

Prediction answers "what will happen?"; causal inference answers "what happens **if I intervene?**" Those are different questions with different mathematics, and conflating them is the single most expensive mistake in applied data science: it produces models that predict beautifully and recommend disastrously.

This is core Data Scientist interview material, increasingly asked of ML Engineers, and the foundation under every "should we ship this?" decision.

---

## Table of Contents
1. [Why Prediction Is Not Enough](#why-prediction-is-not-enough)
2. [Potential Outcomes](#potential-outcomes)
3. [Confounding and DAGs](#confounding-and-dags)
4. [Randomized Experiments](#randomized-experiments)
5. [Propensity Scores and Matching](#propensity-scores-and-matching)
6. [Difference-in-Differences](#difference-in-differences)
7. [Instrumental Variables](#instrumental-variables)
8. [Regression Discontinuity](#regression-discontinuity)
9. [Synthetic Control](#synthetic-control)
10. [Uplift Modeling](#uplift-modeling)
11. [Evaluating Uplift Models](#evaluating-uplift-models)
12. [Method Selection](#method-selection)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Why Prediction Is Not Enough

The canonical example, and a great one to have ready:

> A churn model identifies customers likely to leave. Marketing sends them a discount. Churn among the targeted group is high anyway. Did the discount fail?

Unanswerable as posed. The model selected people who were *already* going to churn; you have no idea what they would have done without the discount. Worse, the discount may have actively *caused* churn in some segment by reminding dormant users the subscription exists: the "sleeping dogs" effect, which is real and measurable in retention campaigns.

The predictive model answers "who is likely to churn?". The business question is "**for whom does the discount change the outcome?**": a causal quantity, and often uncorrelated with the predictive score.

| Question | Type | Tool |
|---|---|---|
| Who will churn? | Prediction | Any classifier |
| Why do people churn? | Explanation | Causal inference |
| Who should we send an offer to? | **Intervention** | Uplift modeling |
| What would have happened without the campaign? | **Counterfactual** | Causal inference |

---

## Potential Outcomes

The Neyman-Rubin framework. For each unit `i`, define two potential outcomes:

- `Y_i(1)`: outcome if treated
- `Y_i(0)`: outcome if not treated

The **individual treatment effect** is `τ_i = Y_i(1) - Y_i(0)`.

**The fundamental problem of causal inference**: you only ever observe one of them. The other is counterfactual, permanently unobservable. So individual effects are not identifiable, and we estimate averages instead:

| Estimand | Meaning | When it's what you want |
|---|---|---|
| **ATE** | `E[Y(1) - Y(0)]` over everyone | Effect of treating the whole population |
| **ATT** | Effect among the *treated* | Evaluating a campaign that already ran |
| **CATE** | `E[Y(1) - Y(0) \| X]`: effect conditional on features | **Targeting**: this is what uplift models estimate |
| **LATE** | Effect among *compliers* | What an instrumental variable identifies |

Naming the right estimand is half of a good interview answer. "Did the campaign work?" is ATT. "Should we roll it out to everyone?" is ATE. "Who should we target?" is CATE.

**Three assumptions** underpin everything:

1. **SUTVA**: one unit's treatment doesn't affect another's outcome, and there's a single version of the treatment. Broken by network effects (a social feature spreading between friends) and by marketplace spillovers (a discount to some buyers changes prices for all).
2. **Unconfoundedness / ignorability**: conditional on observed covariates, treatment is as good as random. Untestable, and the assumption everything hinges on.
3. **Positivity / overlap**: every unit has a non-zero chance of either treatment. Violated when some group is *always* treated, and no method can extrapolate into a region with no comparison units.

---

## Confounding and DAGs

A **confounder** causes both treatment and outcome, creating spurious association.

```
        Confounder (income)
         /            \
        v              v
  Treatment  ---->  Outcome
  (premium plan)    (retention)
```

Higher-income users both buy premium *and* retain better. Naive comparison attributes the income effect to the plan.

Drawing the DAG is the discipline that tells you what to adjust for: and, just as important, what **not** to adjust for. Three structures every candidate should be able to distinguish:

| Structure | Shape | Adjust for the middle? |
|---|---|---|
| **Confounder (fork)** | `T ← C → Y` | **Yes**: this is the bias to remove |
| **Mediator (chain)** | `T → M → Y` | **No**: it's part of the effect you're measuring |
| **Collider** | `T → C ← Y` | **No**: conditioning *creates* bias |

**Controlling for a mediator** answers a different question. If a discount increases retention *by* increasing usage, adjusting for usage measures only the direct effect and reports a much smaller number: technically correct for a different estimand, but usually not what was asked.

**Collider bias** is the counter-intuitive one and a favourite interview probe. Condition on a common *effect* of two variables and you induce association between them where none existed. The classic illustration: among hospitalized patients, two unrelated diseases appear negatively correlated, because being admitted is a collider: having one disease means you needed less of the other to get admitted. In product analytics this appears constantly as selection bias: analyzing only users who completed signup conditions on a collider.

The rule is **not** "control for everything you have." That advice actively introduces bias.

---

## Randomized Experiments

Randomization is the gold standard because it breaks every arrow into treatment, making the groups exchangeable on *all* covariates: observed and unobserved. That last part is what no observational method can replicate.

```python
import numpy as np
from scipy import stats

def analyze_ab_test(control, treatment):
    """Difference in means with a CI: the whole estimator when randomization holds."""
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    t_stat = diff / se
    df = len(control) + len(treatment) - 2
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return {"lift": diff, "ci": ci, "p_value": p}
```

**Randomization can still fail in practice**: check before trusting results:
- **SRM (sample ratio mismatch)**: assignment counts deviate from the intended split. A chi-square test failing here invalidates the experiment; it usually means a bug in bucketing or differential data loss.
- **Interference**: SUTVA violations in social products and marketplaces. Use cluster randomization (by region, by social graph community) or switchback designs.
- **Non-compliance**: assigned to treatment but didn't receive it. Analyze by intention-to-treat, and use assignment as an instrument to recover the effect on compliers.
- **Attrition**: differential dropout between arms is a collider: conditioning on "still present at the end" biases the comparison.

**CUPED** is the standard variance-reduction technique and worth naming: use a pre-experiment covariate `X` (typically the same metric measured before the test) to reduce variance without touching the expectation.

```python
def cuped_adjust(y, x_pre):
    """Y_adj = Y - θ(X_pre - E[X_pre]) with θ = Cov(Y, X_pre)/Var(X_pre)."""
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean())
```

Typical variance reductions of 30–50% translate directly into shorter experiments: the single most effective improvement to most experimentation platforms.

---

## Propensity Scores and Matching

When randomization isn't possible, the propensity score `e(X) = P(T=1 | X)` reduces multi-dimensional adjustment to a single number.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. Model treatment assignment from covariates
ps_model = LogisticRegression(max_iter=1000).fit(X, treatment)
propensity = ps_model.predict_proba(X)[:, 1]

# 2. Check overlap BEFORE estimating anything
#    If treated and control propensity distributions barely overlap, stop:
#    there is no comparable control group and no method fixes that.

# 3. IPW: reweight to a pseudo-population where treatment is independent of X
weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
weights = np.clip(weights, 0, np.percentile(weights, 99))   # trim extreme weights
ate = np.average(y[treatment == 1], weights=weights[treatment == 1]) - \
      np.average(y[treatment == 0], weights=weights[treatment == 0])
```

Approaches on top of the propensity score: **matching** (pair each treated unit with a similar control), **stratification** (bucket by propensity and average within-bucket effects), **IPW** (reweighting, above), and **doubly robust** estimators (AIPW), which combine an outcome model with a propensity model and stay consistent if *either* is correctly specified: the reason they're the modern default.

**Always report covariate balance after adjustment.** Standardized mean differences below 0.1 are the usual bar. An unbalanced covariate means the adjustment failed, and the estimate is not trustworthy no matter how tight the confidence interval looks.

The limitation to state plainly: propensity methods adjust for **observed** confounders only. An unmeasured confounder biases the result with no warning sign in the diagnostics.

---

## Difference-in-Differences

When a treatment rolls out to some group at a known time, compare the *change* in treated versus the *change* in control, differencing away both fixed group differences and common time trends.

```
Effect = (Treated_after - Treated_before) - (Control_after - Control_before)
```

```python
import statsmodels.formula.api as smf

# The interaction term is the DiD estimate
model = smf.ols("outcome ~ treated + post + treated:post", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["unit_id"]}   # cluster SEs, see below
)
print(model.params["treated:post"])
```

The identifying assumption is **parallel trends**: absent treatment, the two groups' outcomes would have moved in parallel. It's untestable for the post period, but you support it by plotting pre-treatment trends over several periods and running an event-study specification with leads and lags: a significant "effect" *before* treatment falsifies the design.

Two technical points interviewers probe: **cluster standard errors** at the treatment-assignment level (serial correlation makes naive SEs far too small, which is the classic DiD error), and be careful with **staggered rollouts**: the two-way fixed effects estimator is biased when units are treated at different times with heterogeneous effects, which is why Callaway–Sant'Anna and similar estimators exist.

---

## Instrumental Variables

When unmeasured confounding is unavoidable, an instrument `Z` can rescue identification. It must satisfy three conditions:

1. **Relevance**: `Z` affects treatment (testable; a weak first stage is a serious problem).
2. **Exclusion**: `Z` affects the outcome *only* through treatment (untestable, and the assumption everyone argues about).
3. **Independence**: `Z` is as good as randomly assigned.

```
Z ──> T ──> Y
      ^
      │
   Unobserved confounder (doesn't touch Z)
```

The most defensible instrument in a tech context is **randomized encouragement**: randomly nudge some users toward a feature. Assignment is random by construction (independence), it moves adoption (relevance), and the nudge plausibly affects the outcome only via adoption (exclusion). Two-stage least squares then recovers the **LATE**: the effect among compliers, i.e. people who adopt *because* of the nudge. That's a narrower population than ATE, and saying so is the mark of someone who understands the method rather than reciting it.

Check the first-stage F-statistic; below ~10 the instrument is weak and the IV estimate can be more biased than plain OLS.

---

## Regression Discontinuity

When treatment is assigned by a threshold on a continuous running variable, units just above and just below the cutoff are effectively randomized.

Examples: a loyalty tier at 1,000 points, a scholarship at a test score, free shipping above a basket value.

```python
# Local linear regression on each side of the cutoff, within a bandwidth
band = df[(df.running_var > cutoff - h) & (df.running_var < cutoff + h)].copy()
band["above"] = (band.running_var >= cutoff).astype(int)
band["centered"] = band.running_var - cutoff
rd = smf.ols("outcome ~ above + centered + above:centered", data=band).fit()
# `above` coefficient = the local treatment effect at the cutoff
```

Validity checks that a strong answer mentions: **no manipulation** of the running variable (a McCrary density test, a spike just above the threshold means people are gaming it, destroying the design), **covariate continuity** at the cutoff, and **bandwidth sensitivity** (results shouldn't flip with a reasonable change in `h`).

The estimate is **local**: it's the effect at the cutoff, which may not generalize to units far from it.

---

## Synthetic Control

For a single treated unit (one city, one market, one large customer), build a weighted combination of untreated units that reproduces the treated unit's pre-treatment trajectory, then use it as the counterfactual afterwards.

The weights are chosen to minimize pre-period discrepancy. Credibility rests on close pre-period fit over a long window; inference uses placebo tests: apply the method to each untreated unit and check whether the real treated unit's post-period gap is extreme relative to that placebo distribution.

This is the standard tool for geo-experiments and market-level rollouts where you can't randomize at the user level.

---

## Uplift Modeling

Uplift (heterogeneous treatment effect, CATE) modeling estimates **`τ(x) = E[Y|T=1, X=x] - E[Y|T=0, X=x]`**, who to treat, not who will convert.

### The four-quadrant framing

The mental model interviewers look for:

| Segment | Behavior | Action |
|---|---|---|
| **Persuadables** | Convert only if treated | **Target these**: the entire ROI |
| **Sure things** | Convert either way | Don't waste budget |
| **Lost causes** | Never convert | Don't waste budget |
| **Sleeping dogs** | Convert only if **not** treated | **Actively avoid**: treatment hurts |

A response model targets sure things (they have the highest conversion probability) and can't distinguish sleeping dogs at all. That's why a campaign can post great conversion rates among the targeted and produce no incremental revenue.

### Meta-learners

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# --- T-learner: separate model per arm; simple, but errors compound ---
m1 = GradientBoostingRegressor().fit(X[T == 1], y[T == 1])
m0 = GradientBoostingRegressor().fit(X[T == 0], y[T == 0])
uplift_t = m1.predict(X_new) - m0.predict(X_new)

# --- S-learner: one model with treatment as a feature ---
Xs = np.column_stack([X, T])
ms = GradientBoostingRegressor().fit(Xs, y)
uplift_s = (ms.predict(np.column_stack([X_new, np.ones(len(X_new))]))
            - ms.predict(np.column_stack([X_new, np.zeros(len(X_new))])))
```

| Learner | How | Best when |
|---|---|---|
| **S-learner** | One model, treatment as a feature | Small effects; risks the model ignoring `T` entirely |
| **T-learner** | One model per arm | Ample data in both arms; errors from the two models compound |
| **X-learner** | Impute counterfactuals, model the imputed effects, blend by propensity | **Imbalanced arms**: the usual real-world case |
| **R-learner** | Residualize both `Y` and `T`, then regress | Strong theory, needs careful cross-fitting |
| **Causal Forest** | Trees that split to maximize effect *heterogeneity* | Nonlinear effects; gives valid confidence intervals |

**Causal forests** deserve a sentence in any answer: unlike a standard tree that splits to reduce outcome error, they split to maximize the *difference in treatment effect* between children, with honest sample splitting (one subsample chooses the structure, another estimates the effects) so the confidence intervals are valid.

Critically, **uplift models need experimental data to train on**. Training on observational data means the treatment/control comparison is confounded, and the model learns selection rather than effect.

---

## Evaluating Uplift Models

Standard metrics don't apply: the individual effect is never observed, so there's no per-row ground truth to score against. Evaluation is population-level.

**Qini and uplift curves**: rank the held-out experimental population by predicted uplift, then plot cumulative incremental conversions against the fraction targeted. A good model concentrates incremental gains in the top deciles.

```python
def qini_curve(y, treatment, uplift_scores, bins=10):
    """Cumulative incremental conversions when targeting the top-scored fraction."""
    order = np.argsort(-uplift_scores)
    y, treatment = np.asarray(y)[order], np.asarray(treatment)[order]

    points = []
    for i in range(1, bins + 1):
        k = int(len(y) * i / bins)
        yt, yc = y[:k][treatment[:k] == 1], y[:k][treatment[:k] == 0]
        nt, nc = len(yt), len(yc)
        if nt == 0 or nc == 0:
            points.append((i / bins, np.nan)); continue
        # Incremental conversions attributable to treatment within the top-k
        points.append((i / bins, yt.sum() - yc.sum() * nt / nc))
    return points
```

**Uplift by decile** is the most convincing artifact for stakeholders: bucket by predicted uplift and show actual measured lift per bucket. A working model shows monotonically decreasing lift across deciles, and often a *negative* bottom decile, which is the sleeping-dogs group you should now exclude.

The decisive validation is a **follow-up experiment**: randomize between targeting by the uplift model and targeting by the incumbent rule, and compare incremental outcomes. Offline Qini is a proxy; the second experiment is proof.

---

## Method Selection

| Situation | Method | Key assumption |
|---|---|---|
| You can randomize | **A/B test** | Randomization worked (check SRM) |
| Can't randomize; confounders measured | **Propensity / doubly robust** | Unconfoundedness + overlap |
| Rollout to some units at a known time | **Difference-in-differences** | Parallel trends |
| Unmeasured confounding, valid instrument | **Instrumental variables** | Exclusion restriction |
| Threshold-based assignment | **Regression discontinuity** | No manipulation at the cutoff |
| One treated market/region | **Synthetic control** | Good pre-period fit |
| Deciding *whom* to treat | **Uplift / causal forest** | Trained on experimental data |
| Want variance reduction in a test | **CUPED** | Pre-period covariate available |

---

## Interview Q&A

#### What's the difference between a predictive model and a causal model, and when does it matter?

A predictive model estimates `P(Y|X)` from observed data; a causal model estimates `P(Y|do(T))`: the distribution under an intervention. They coincide only when treatment is randomized.

It matters whenever the model's output drives an action. The standard example: a churn model plus a retention discount. The model ranks by churn probability, but the right target is people whose behaviour the discount *changes*. Those overlap poorly: high-risk users may be leaving for reasons a discount can't fix, and some users churn *because* the offer reminded them they're paying. Optimizing prediction here can produce zero or negative incremental revenue while every predictive metric looks excellent.

The tell that you need causal machinery: the question contains "should we do X" rather than "what will happen".

#### Explain confounding, and why "control for everything" is wrong advice.

A confounder causes both treatment and outcome, so the naive association mixes the treatment effect with the confounder's effect. Adjusting for it removes that bias.

But not every variable should be adjusted for. A **mediator** lies on the causal path (`T → M → Y`); controlling for it removes part of the very effect you're measuring and answers a different question: the direct effect rather than the total. A **collider** is a common effect of two variables (`T → C ← Y`); conditioning on it *creates* association where none existed. The hospitalization example is the classic: among admitted patients, unrelated diseases appear negatively correlated purely because admission is a collider.

So the right procedure is to draw the DAG and apply the backdoor criterion: adjust for the set that blocks all backdoor paths, and no more. "Throw every feature in" reliably introduces bias in the collider and mediator cases.

#### What is uplift modeling and how does it differ from a response model?

A response model predicts `P(convert | treated)` and targets the highest scores. An uplift model estimates `P(convert | treated) - P(convert | not treated)` (the *incremental* effect), and targets the largest differences.

The distinction is the four quadrants: persuadables (convert only if treated), sure things (convert regardless), lost causes (never convert), and sleeping dogs (treatment makes them *worse*). A response model spends the budget on sure things because they have the highest conversion probability, and cannot detect sleeping dogs at all. That's how a campaign posts strong conversion among the targeted group with zero incremental revenue.

The requirement worth stating: uplift models must be trained on **experimental** data. Fit on observational data, the treated/untreated comparison is confounded and you learn selection, not causation.

#### How do you evaluate an uplift model when you never observe individual treatment effects?

You can't score row-by-row, so evaluation is population-level on a held-out **experimental** sample. Rank by predicted uplift and build a Qini or uplift curve: cumulative incremental conversions versus fraction targeted, where incremental means treated conversions minus rescaled control conversions within the top-k. Qini coefficient summarizes the area over random targeting.

The most persuasive artifact for stakeholders is **uplift by decile**: bucket by predicted uplift and plot the actually-measured lift per bucket. A working model is monotonically decreasing, and the bottom decile is often negative, which identifies sleeping dogs you should exclude.

Ultimately the proof is a second experiment: randomize between the uplift-based targeting policy and the incumbent, and compare incremental outcomes directly.

#### Explain difference-in-differences and its key assumption.

DiD compares the change over time in a treated group against the change in an untreated group, differencing out both time-invariant group differences and shared time trends. The estimate is the interaction coefficient in `Y ~ treated + post + treated:post`.

The identifying assumption is **parallel trends**: absent treatment, both groups would have moved together. It's untestable in the post period, so you support it with several pre-treatment periods plotted, and an event-study specification with leads: a significant "effect" before treatment falsifies the design.

Two technical points that separate answers: standard errors must be **clustered at the assignment level**, because serial correlation otherwise makes them far too small; and with **staggered rollouts**, two-way fixed effects is biased under heterogeneous effects, which is why estimators like Callaway–Sant'Anna exist.

#### You can't run an experiment. How do you estimate the effect?

Ask why not first: it's occasionally a policy constraint that can be relaxed, or a randomized *encouragement* design is acceptable even when forced treatment isn't.

Failing that, the design follows the data structure. If I believe I've measured the confounders, propensity-score methods or doubly robust estimation, with overlap checks and post-adjustment covariate balance reported. If treatment turned on at a known time for some units, difference-in-differences with pre-trend validation. If assignment follows a threshold, regression discontinuity. If unmeasured confounding is likely but I have a plausible instrument, IV, recognizing it identifies LATE among compliers, not ATE.

Whatever the choice, I'd state the identifying assumption explicitly, run the diagnostics the design permits, and do a sensitivity analysis: how strong would an unmeasured confounder need to be to overturn the result? That framing is far more honest than a point estimate with a tight confidence interval that ignores the assumption doing all the work.

#### What is CUPED and why does every experimentation platform use it?

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance by adjusting the outcome with a pre-experiment covariate, usually the same metric measured before the test: `Y_adj = Y - θ(X_pre - E[X_pre])` with `θ = Cov(Y, X_pre)/Var(X_pre)`.

Because `X_pre` is measured before randomization, it can't be affected by treatment, so the adjustment leaves the expected treatment effect unbiased while removing the component of outcome variance that the pre-period predicts. Typical reductions are 30–50%, which either shortens the experiment substantially or lets you detect smaller effects at the same duration.

That's why it's ubiquitous: it's essentially free statistical power from data you already have.

#### What is SUTVA and when is it violated in a tech product?

Stable Unit Treatment Value Assumption: one unit's treatment doesn't affect another unit's outcome, and there's only one version of the treatment.

It breaks constantly in practice. **Social products**: a treated user shares the new feature with untreated friends, contaminating control and biasing the effect toward zero. **Marketplaces**: showing some buyers lower prices changes inventory and prices for everyone: control is affected by treatment. **Shared resources**: a treatment that consumes more compute slows the control arm.

Fixes: randomize at the level where interference stops: cluster randomization by geography or by social-graph community; switchback designs alternating the whole marketplace over time; or ego-cluster designs in social graphs. The diagnostic is to run the same test at two randomization granularities and see whether the estimate changes; a large gap indicates interference.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Using a predictive model for a targeting decision | Optimizes conversion, not incremental effect | Uplift / CATE model on experimental data |
| Controlling for a mediator | Removes part of the effect being measured | Draw the DAG; adjust only backdoor paths |
| Controlling for a collider | *Creates* bias that wasn't there | Same: DAG plus backdoor criterion |
| Ignoring overlap before propensity adjustment | Extrapolating where no comparable units exist | Plot propensity distributions; trim non-overlap |
| No covariate balance check after matching | Adjustment may have failed silently | Report standardized mean differences (< 0.1) |
| Unclustered standard errors in DiD | Serial correlation makes SEs far too small | Cluster at the treatment-assignment level |
| No pre-trend check in DiD | Parallel trends may be plainly false | Event study with leads and lags |
| Weak instrument in IV | Can be more biased than OLS | Check first-stage F (≳ 10) |
| Training uplift models on observational data | Learns selection, not causal effect | Requires randomized training data |
| Ignoring sleeping dogs | Campaign actively harms a segment | Inspect the bottom uplift decile; exclude it |
| Reporting ATE when the question is ATT or CATE | Answers a question nobody asked | State the estimand before estimating |
| Assuming SUTVA in a social or marketplace product | Interference biases the estimate | Cluster / switchback randomization |

---

## Related Topics

- [A/B Testing & Experimentation](../mlops/intro_ab_testing.md)
- [Statistics & Probability](./intro_statistics_probability.md)
- [Model Evaluation and Metrics](./intro_model_evaluation.md)
- [Ensemble Methods](./intro_ensemble_methods.md)
- [Recommender Systems](./intro_recommender_systems.md)
- [Model Explainability](../mlops/intro_model_explainability.md)
- [Take-Home Projects](../docs/take-home-projects.md)
- [Classical ML Overview](./README.md)
