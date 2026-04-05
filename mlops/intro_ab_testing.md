# A/B Testing & Experimentation for ML Systems

## Why A/B Testing Matters for ML

Offline metrics (AUC, RMSE, NDCG) don't always predict online impact. A model with better offline metrics can hurt revenue due to:
- Training-serving skew
- Novelty effects
- Interaction with the full system
- Metrics that don't perfectly capture business goals

A/B testing is the gold standard for measuring real user impact.

---

## Statistical Framework

### Null and Alternative Hypotheses

```
H₀: μ_treatment = μ_control      (no difference)
H₁: μ_treatment ≠ μ_control      (two-tailed — use when you don't know direction)
H₁: μ_treatment > μ_control      (one-tailed — when direction is expected)
```

Use two-tailed unless you have strong a priori reason for directionality. Two-tailed is more conservative and prevents p-hacking.

### Key Metrics

**Primary metric:** The one metric that determines launch decision (e.g., revenue per user, CTR)

**Guardrail metrics:** Metrics that must not decrease (e.g., latency, error rate, retention) — these are non-negotiable

**Secondary metrics:** Informational, not part of decision criteria

---

## Sample Size Calculation

This is the most common interview question. You must calculate sample size **before** running the experiment.

```python
import numpy as np
from scipy import stats

def calculate_sample_size(
    baseline_rate: float,    # current conversion rate e.g., 0.05
    min_detectable_effect: float,  # relative change e.g., 0.10 = 10% lift
    alpha: float = 0.05,     # Type I error rate (significance level)
    power: float = 0.80      # 1 - Type II error rate
) -> int:
    """
    Calculate required sample size per variant for a proportion test.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)

    # Pooled proportion under H₀
    p_pooled = (p1 + p2) / 2

    # Z-scores for alpha and beta
    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-tailed
    z_beta  = stats.norm.ppf(power)

    # Sample size formula for proportions
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    denominator = (p2 - p1) ** 2

    n = numerator / denominator
    return int(np.ceil(n))

# Example: baseline CTR = 5%, want to detect 10% relative lift (to 5.5%)
n = calculate_sample_size(0.05, 0.10)
print(f"Required n per variant: {n:,}")
# Required n per variant: ~30,000

# Total experiment duration = n_required / daily_traffic_per_variant
daily_traffic = 10_000  # users per variant per day
duration_days = n / daily_traffic
print(f"Experiment duration: {duration_days:.1f} days")
```

**Rule of thumb:** Smaller effect size → exponentially larger sample needed. A 1% lift takes ~100× more users than a 10% lift.

---

## Running the Test

### Randomization Unit

The unit you randomize on should match the unit you measure:

| Scenario | Randomize on | Why |
|----------|-------------|-----|
| UI changes | User ID | Consistent experience per user |
| Pricing experiments | User ID | Avoid same user seeing different prices |
| Email campaigns | User ID or email | Independence between users |
| Algorithm tests (search) | Query ID | Tests relevance per query |
| Infrastructure changes | Request | Fine-grained, but no personalization data |

**User-level** randomization is most common. Never randomize on time (biased by weekly patterns).

### Hash-Based Assignment

```python
import hashlib

def assign_variant(user_id: str, experiment_id: str, 
                   n_variants: int = 2) -> int:
    """
    Deterministic assignment: same user always gets same variant.
    """
    hash_input = f"{experiment_id}:{user_id}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    return hash_value % n_variants

# Usage
variant = assign_variant("user_12345", "exp_rec_model_v2")
# 0 → control, 1 → treatment
```

**Why hash-based:** Deterministic (reproducible), uniform distribution, no need to store assignments.

### Pre-Experiment Checks

```python
import pandas as pd
from scipy import stats

# 1. A/A Test: run experiment with identical variants
# Both variants should show no significant difference
# If they do, your randomization is broken

# 2. Check balance: verify variant sizes are as expected
def check_balance(df, variant_col='variant', expected_split=0.5):
    counts = df[variant_col].value_counts()
    total = len(df)
    for variant, count in counts.items():
        actual_split = count / total
        print(f"Variant {variant}: {count:,} ({actual_split:.1%}, expected {expected_split:.1%})")

    # Chi-square test for balance
    chi2, p = stats.chisquare(counts.values)
    print(f"Balance test: χ²={chi2:.3f}, p={p:.3f}")
    if p < 0.05:
        print("WARNING: Variants are imbalanced!")

# 3. Check pre-experiment equivalence (SRM - Sample Ratio Mismatch)
# If users are significantly different before experiment, results are invalid
```

---

## Statistical Analysis

### Z-Test for Proportions (e.g., CTR)

```python
import numpy as np
from scipy import stats

def ab_test_proportions(
    n_control: int, n_treatment: int,
    conversions_control: int, conversions_treatment: int,
    alpha: float = 0.05
) -> dict:
    """
    Two-sample z-test for proportion difference.
    """
    p_control   = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment

    # Pooled proportion (under H₀: p_c = p_t)
    p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)

    # Standard error of difference
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    # Z-statistic
    z = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed

    # Confidence interval for the difference
    se_diff = np.sqrt(p_control*(1-p_control)/n_control +
                      p_treatment*(1-p_treatment)/n_treatment)
    ci_lower = (p_treatment - p_control) - 1.96 * se_diff
    ci_upper = (p_treatment - p_control) + 1.96 * se_diff

    relative_lift = (p_treatment - p_control) / p_control

    return {
        'p_control': p_control,
        'p_treatment': p_treatment,
        'relative_lift': relative_lift,
        'z_statistic': z,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_95': (ci_lower, ci_upper),
    }

# Example
result = ab_test_proportions(
    n_control=50_000, n_treatment=50_000,
    conversions_control=2_400, conversions_treatment=2_650
)
print(f"Control CTR: {result['p_control']:.3%}")
print(f"Treatment CTR: {result['p_treatment']:.3%}")
print(f"Relative lift: {result['relative_lift']:.1%}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"95% CI for difference: ({result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f})")
```

### T-Test for Continuous Metrics (e.g., Revenue)

```python
from scipy import stats

def ab_test_continuous(control_values, treatment_values, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
    
    control_mean   = np.mean(control_values)
    treatment_mean = np.mean(treatment_values)
    
    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'relative_lift': (treatment_mean - control_mean) / control_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
    }
```

---

## Common Pitfalls

### 1. Peeking Problem (Optional Stopping)

If you look at results repeatedly and stop when p < 0.05, your actual Type I error rate is much higher than 5%.

```python
# Simulation showing peeking inflates false positives
import numpy as np

def simulate_peeking(n_simulations=10_000, max_n=1_000, check_every=50):
    false_positives = 0
    for _ in range(n_simulations):
        control   = np.random.normal(0, 1, max_n)
        treatment = np.random.normal(0, 1, max_n)  # same distribution = no effect
        for n in range(check_every, max_n+1, check_every):
            _, p = stats.ttest_ind(control[:n], treatment[:n])
            if p < 0.05:
                false_positives += 1
                break
    return false_positives / n_simulations

# Fixed sample: ~5% false positive rate
# Peeking every 50 samples up to 1000: ~20%+ false positive rate!
```

**Solutions:**
- **Pre-register** sample size and analysis plan before starting
- **Sequential testing** (alpha spending functions) if you need early stopping
- **Bayesian testing** — update beliefs continuously without inflating error rates

### 2. Multiple Testing Problem

Testing 20 metrics at α=0.05 expects 1 false positive.

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.20, 0.07, 0.15, 0.02]  # 7 metrics

# Benjamini-Hochberg FDR correction (less conservative than Bonferroni)
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

for i, (raw, corrected, sig) in enumerate(zip(p_values, p_corrected, reject)):
    print(f"Metric {i}: raw p={raw:.3f}, corrected p={corrected:.3f}, significant={sig}")
```

**Hierarchy rule:** Define primary metric in advance. Only apply correction to secondary metrics. This prevents HARKing (Hypothesizing After Results are Known).

### 3. Network Effects / SUTVA Violations

SUTVA (Stable Unit Treatment Value Assumption): treatment of one unit doesn't affect others.

Violated in:
- Social networks: user in treatment group affects their friends in control
- Marketplace: changing prices for some sellers affects overall market
- Email: viral content spread from treatment to control users

**Solutions:**
- **Cluster-based randomization:** randomize entire social clusters together
- **Geo-based experiments:** randomize by geography (city, country)
- **Time-based experiments:** alternate treatment by time period (riskier)

### 4. Simpson's Paradox

A trend can reverse when data is aggregated vs segmented.

```python
import pandas as pd

# Example: New ML model looks better overall but worse in every segment
data = {
    'segment': ['Mobile', 'Mobile', 'Desktop', 'Desktop'],
    'variant':  ['Control', 'Treatment', 'Control', 'Treatment'],
    'users':    [1000, 9000, 9000, 1000],
    'conversions': [50, 360, 900, 90]
}
df = pd.DataFrame(data)
df['rate'] = df['conversions'] / df['users']

# Segment view: Treatment worse in both segments!
print(df[['segment', 'variant', 'rate']])
# Mobile: Control=5%, Treatment=4%
# Desktop: Control=10%, Treatment=9%

# But aggregate: Treatment better overall (because treatment had more mobile users!)
# This is Simpson's Paradox — always analyze by segment
```

**Prevention:** Always segment analysis by major user groups. Check if treatment groups are balanced across key dimensions.

---

## Advanced Techniques

### Variance Reduction with CUPED

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance by subtracting out pre-experiment variation:

```python
def cuped_metric(
    post_metric: np.ndarray,
    pre_metric: np.ndarray
) -> np.ndarray:
    """
    Adjust post-experiment metric using pre-experiment covariate.
    Y_cuped = Y - θ × (X - E[X])
    θ = Cov(Y, X) / Var(X)  — OLS estimate
    """
    theta = np.cov(post_metric, pre_metric)[0, 1] / np.var(pre_metric)
    return post_metric - theta * (pre_metric - np.mean(pre_metric))

# Usage: if pre_metric is correlated with post_metric,
# CUPED significantly reduces variance → smaller required sample size
# Variance reduction of 50-80% is common in practice
```

**Why it works:** Pre-experiment user behavior is correlated with post-experiment behavior. By removing this systematic variation, you see the treatment effect more clearly.

### Multi-Armed Bandits

When you want to minimize regret (opportunity cost) during experimentation rather than pure exploration:

```python
import numpy as np

class ThompsonSampling:
    """
    Bayesian bandit: model CTR as Beta distribution, sample to decide arm.
    """
    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures + 1

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm: int, reward: int):
        self.alpha[arm] += reward
        self.beta[arm]  += 1 - reward

# Compare to fixed A/B: bandits continuously shift traffic to better arms
# Trade-off: MAB minimizes regret, A/B gives cleaner statistical inference
```

**A/B Test vs Bandit trade-off:**
- **A/B Test:** Clean inference, fixed allocation, high regret if one variant is much better
- **Bandit:** Lower regret, adaptively allocates traffic, but inference is more complex
- **Use A/B** for major feature decisions that need clear p-values for stakeholders
- **Use Bandits** for ongoing optimization (content ranking, ad serving, recommendation)

---

## ML Model A/B Testing Specifics

### Shadow Mode Testing

Run new model in parallel without serving its results to users:

```python
# Pseudo-code for shadow mode
def handle_request(user_request):
    # Production: serve from current model
    production_response = current_model.predict(user_request)
    
    # Shadow: run new model, log output but don't serve
    shadow_response = new_model.predict(user_request)
    logger.log({
        'request': user_request,
        'production': production_response,
        'shadow': shadow_response,
    })
    
    return production_response  # only serve production result
```

Use shadow mode to validate correctness and measure latency before any user exposure.

### Interleaving for Ranking Systems

For ranking models (search, recommendations), interleaving is more sensitive than side-by-side A/B:

- Take top-N results from Model A and Model B
- Interleave them in a balanced way
- Show one combined list to the user
- Model that gets more clicks "wins"

Requires ~10-100× fewer users to detect the same effect size vs A/B testing.

### Causal Impact for Non-Randomized Experiments

When you can't randomize (e.g., a feature rolled out to all users at once):

```python
# CausalImpact uses Bayesian structural time series to estimate
# counterfactual (what would have happened without the intervention)

# pip install causalimpact
from causalimpact import CausalImpact
import pandas as pd

# data: time series of metric, with pre/post periods marked
pre_period  = ['2024-01-01', '2024-03-31']
post_period = ['2024-04-01', '2024-06-30']

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
ci.plot()
```

---

## Decision Framework

```
1. Pre-experiment
   ├── Define primary metric (only 1)
   ├── Define guardrail metrics
   ├── Calculate required sample size
   ├── Pre-register analysis plan
   └── Check for data quality issues

2. During experiment
   ├── Do NOT peek at results (or use sequential testing)
   ├── Monitor for sample ratio mismatch (SRM)
   └── Check for data pipeline issues

3. Post-experiment analysis
   ├── Check SRM: was split as expected?
   ├── Primary metric test (z-test or t-test)
   ├── Segment analysis (mobile vs desktop, new vs returning)
   ├── Guardrail metric check
   └── Multiple testing correction if needed

4. Decision
   ├── Ship if: primary metric ↑, no guardrail violations
   ├── Investigate if: mixed signals by segment
   └── Hold if: guardrail metric ↓ (even if primary ↑)
```

---

## Common Interview Questions

**Q: How do you determine how long to run an A/B test?**
Calculate required sample size before running based on baseline conversion rate, minimum detectable effect (MDE), desired power (80-95%), and significance level (5%). Run until reaching that sample size — not based on time. Also run for full weekly cycles (multiples of 7 days) to avoid day-of-week bias.

**Q: The test shows p=0.03 but you checked yesterday and p=0.06. What happened?**
This is the peeking problem. By checking early and stopping when p crossed the threshold, you inflated your Type I error rate. The true false positive rate with repeated peeking is much higher than 5%. You should have pre-committed to a sample size and only looked once at the end.

**Q: How would you design an A/B test for a recommendation model?**
Define primary metric (e.g., CTR or session time), guardrails (latency p99, error rate). Randomize on user ID for personalization. Calculate sample size for the expected effect. Use interleaving for faster signal if testing ranking only. Run shadow mode first to check model correctness. Monitor for novelty effects in first 24-48 hours. Segment analysis by user type (new vs returning, mobile vs desktop).

**Q: What is a Sample Ratio Mismatch (SRM) and why is it critical?**
SRM occurs when the actual split between variants differs significantly from the planned split. It indicates a bug in the randomization/logging/data pipeline. If 50/50 split gives you 45K control and 55K treatment, something is wrong — you cannot trust any of the experiment's results. Always check SRM first before looking at any metrics.

**Q: When would you use CUPED?**
CUPED (pre-experiment covariate adjustment) is useful when users have highly variable baseline behavior. If pre-experiment purchase rate is highly correlated with post-experiment purchase rate, CUPED can reduce variance by 50-80%, allowing you to detect smaller effects or reach significance faster with fewer users. It's essentially regression adjustment — standard in large-scale experimentation.
