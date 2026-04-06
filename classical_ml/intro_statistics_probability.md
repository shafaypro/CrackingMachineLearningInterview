# Statistics & Probability for ML Interviews

## Probability Foundations

### Key Definitions

**Conditional Probability:**
```
P(A|B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

In ML terms: P(label|data) = P(data|label) × P(label) / P(data)
- P(label|data) = posterior
- P(data|label) = likelihood
- P(label) = prior
- P(data) = evidence (normalizing constant)

**Independence:** P(A ∩ B) = P(A) × P(B)

**Law of Total Probability:**
```
P(B) = Σ_i P(B|A_i) × P(A_i)
```

### Common Distributions

| Distribution | Use Case | Parameters | Mean | Variance |
|-------------|----------|-----------|------|----------|
| Bernoulli | Binary outcomes | p | p | p(1-p) |
| Binomial | k successes in n trials | n, p | np | np(1-p) |
| Poisson | Count events in interval | λ | λ | λ |
| Normal | Continuous measurements | μ, σ² | μ | σ² |
| Exponential | Time between events | λ | 1/λ | 1/λ² |
| Uniform | Equal probability range | a, b | (a+b)/2 | (b-a)²/12 |
| Beta | Probability of probability | α, β | α/(α+β) | — |

**When to use each in ML:**
- **Bernoulli/Binomial:** Binary classification outputs, A/B testing outcomes
- **Poisson:** Click counts, fraud event counts, document word counts (NLP)
- **Normal:** Modeling noise, weight initialization, Gaussian processes
- **Exponential:** Time-to-event models, survival analysis
- **Beta:** Prior on click-through rates, Bayesian A/B testing

---

## Descriptive Statistics

### Measures of Central Tendency

```python
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]

mean   = np.mean(data)    # 5.0  — sensitive to outliers
median = np.median(data)  # 4.5  — robust to outliers
mode   = 4                # most frequent — categorical data

# When to use which:
# Symmetric distributions: mean ≈ median ≈ mode
# Right-skewed (income): median preferred
# Categorical data: mode
```

### Measures of Spread

```python
variance = np.var(data, ddof=1)   # sample variance (ddof=1 for unbiased)
std_dev  = np.std(data, ddof=1)
iqr      = np.percentile(data, 75) - np.percentile(data, 25)  # robust to outliers

# Coefficient of Variation: compare spread across different scales
cv = std_dev / mean
```

### Skewness and Kurtosis

- **Skewness > 0**: right tail is longer (income, housing prices)
- **Skewness < 0**: left tail is longer
- **Kurtosis > 3** (excess > 0): heavy tails, more outliers (leptokurtic)
- **Kurtosis < 3** (excess < 0): light tails (platykurtic)

```python
from scipy import stats
skew = stats.skew(data)
kurt = stats.kurtosis(data)  # excess kurtosis (normal = 0)
```

---

## Hypothesis Testing

### Framework

1. **State null hypothesis (H₀)** and alternative (H₁)
2. **Choose significance level α** (typically 0.05 or 0.01)
3. **Compute test statistic**
4. **Find p-value**: P(observing result this extreme | H₀ is true)
5. **Decision**: if p < α, reject H₀

**Errors:**
- **Type I (α):** Reject H₀ when it's true (false positive) — controlled by α
- **Type II (β):** Fail to reject H₀ when it's false (false negative)
- **Power = 1 - β:** Probability of correctly detecting an effect

### t-test

Tests whether the mean of a sample is equal to a hypothesized value, or whether two samples have equal means.

```python
from scipy import stats

# One-sample t-test: is mean different from 5.0?
t_stat, p_value = stats.ttest_1samp(data, popmean=5.0)

# Two-sample independent t-test
control   = [82, 85, 88, 90, 91]
treatment = [88, 92, 95, 97, 99]
t_stat, p_value = stats.ttest_ind(control, treatment)

# Paired t-test (before/after same subjects)
before = [80, 85, 90, 88, 92]
after  = [85, 88, 95, 90, 97]
t_stat, p_value = stats.ttest_rel(before, after)

print(f"t={t_stat:.3f}, p={p_value:.3f}")
# p < 0.05 → statistically significant difference
```

**Assumptions:** Normality (or n > 30 by CLT), independence, equal variance (independent t-test).

### Chi-Square Test

Tests independence between two categorical variables.

```python
import numpy as np
from scipy.stats import chi2_contingency

# Contingency table: Button color vs Click
#              Clicked  Not Clicked
# Blue:          200       800
# Red:           250       750
observed = np.array([[200, 800], [250, 750]])
chi2, p_value, dof, expected = chi2_contingency(observed)
print(f"χ²={chi2:.3f}, p={p_value:.3f}, df={dof}")
# p < 0.05 → button color and click are dependent (not independent)
```

Use case in ML: feature selection (test if a feature is independent of the target).

### ANOVA (Analysis of Variance)

Tests if means of 3+ groups are equal.

```python
group_a = [85, 88, 90, 87]
group_b = [78, 80, 82, 79]
group_c = [92, 94, 91, 95]

f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
# Significant → at least one group differs; use post-hoc tests (Tukey) to find which
```

### Multiple Comparisons Problem

Running 20 tests at α=0.05 → expect ~1 false positive by chance.

**Bonferroni correction:** adjust α → α/n_tests
```python
n_tests = 20
alpha_bonferroni = 0.05 / n_tests  # 0.0025

# Benjamini-Hochberg (FDR control) — less conservative
from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

---

## Statistical Inference for ML

### Confidence Intervals

A 95% CI means: if we repeated sampling many times, 95% of intervals would contain the true parameter.

```python
import scipy.stats as stats
import numpy as np

data = np.array([2.1, 2.5, 3.0, 2.8, 2.3, 2.7])
n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # standard error = std / sqrt(n)

ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
print(f"Mean: {mean:.2f}, 95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")

# Bootstrap CI (model-agnostic, no distributional assumptions)
def bootstrap_ci(data, statistic=np.mean, n_boot=1000, ci=0.95):
    boot_stats = [statistic(np.random.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha*100, (1-alpha)*100])

ci_boot = bootstrap_ci(data)
```

### Central Limit Theorem (CLT)

The distribution of the **sample mean** approaches Normal as n → ∞, regardless of the population distribution.

```
X̄ ~ N(μ, σ²/n) as n → ∞
```

Why it matters for ML:
- Justifies using normal-based tests on large datasets
- Foundation for confidence intervals
- Why many ML metrics (averaged over samples) are approximately normal

**Rule of thumb:** n ≥ 30 is usually sufficient for CLT to hold for means.

---

## Correlation and Covariance

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

cov = np.cov(x, y)[0, 1]  # covariance
corr = np.corrcoef(x, y)[0, 1]  # Pearson correlation [-1, 1]

# Pearson r = Cov(X,Y) / (σ_X × σ_Y)
```

**Spearman vs Pearson:**
- **Pearson:** linear relationships, sensitive to outliers, assumes normality
- **Spearman:** monotonic relationships, rank-based, robust to outliers
- Use Spearman when data has outliers or you only care about ordinal relationship

**Correlation ≠ Causation:** Always. A classic ML interview trap. Two variables can be correlated due to:
- Direct causation (X → Y)
- Reverse causation (Y → X)
- Common cause (Z → X and Z → Y, confounding)
- Coincidence (spurious correlation)

---

## Bayesian Statistics

### Bayesian vs Frequentist

| | Frequentist | Bayesian |
|--|-------------|----------|
| Parameters | Fixed, unknown | Random variables with distributions |
| Probability | Long-run frequency | Degree of belief |
| Output | Point estimate + CI | Posterior distribution |
| Prior knowledge | Not used | Encoded as prior |
| Computation | Usually simpler | Can be expensive |

### Bayesian Updating

```python
# Example: Estimating click-through rate with Beta distribution
# Prior: Beta(α=1, β=1) = Uniform (no prior knowledge)
# Likelihood: Binomial (k clicks in n impressions)
# Posterior: Beta(α + k, β + n - k)

from scipy.stats import beta
import numpy as np

# Prior
alpha_prior, beta_prior = 1, 1

# Observed data: 30 clicks out of 100 impressions
n, k = 100, 30

# Posterior
alpha_post = alpha_prior + k        # 31
beta_post  = beta_prior + n - k     # 71

# Posterior mean = α/(α+β) = 31/102 ≈ 0.304
posterior_mean = alpha_post / (alpha_post + beta_post)

# 95% credible interval
ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"Posterior mean: {posterior_mean:.3f}")
print(f"95% Credible Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
```

### Maximum Likelihood Estimation (MLE)

Find parameters θ that maximize P(data|θ):

```python
# MLE for Gaussian: sample mean and variance are the MLE estimates
# Logistic regression uses MLE to find weights

# In practice, we minimize negative log-likelihood (equivalent, numerically stable)
# NLL = -Σ log P(y_i | x_i, θ)

# For classification: cross-entropy loss IS the negative log-likelihood
import torch.nn as nn
criterion = nn.CrossEntropyLoss()  # maximizes likelihood of correct class
```

### Maximum A Posteriori (MAP)

Extends MLE by incorporating a prior:
```
MAP: θ* = argmax P(θ|data) = argmax P(data|θ) × P(θ)
```

**L2 regularization = MAP with Gaussian prior:**
```
min NLL + λ||θ||²  ↔  MAP with prior θ ~ N(0, 1/2λ)
```

**L1 regularization = MAP with Laplace prior:**
```
min NLL + λ||θ||₁  ↔  MAP with prior θ ~ Laplace(0, 1/λ)
```

This is why L1 produces sparse solutions — the Laplace prior has a sharp peak at 0.

---

## Expected Value, Variance, and Key Identities

```
E[X] = Σ x · P(X=x)           (discrete)
E[X] = ∫ x · f(x) dx           (continuous)
E[aX + b] = aE[X] + b          (linearity)
E[X + Y] = E[X] + E[Y]         (always, even if correlated)
E[XY] = E[X]E[Y]               (only if X, Y independent)

Var(X) = E[X²] - (E[X])²
Var(aX + b) = a²Var(X)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
Var(X + Y) = Var(X) + Var(Y)   (if independent)
```

---

## Common Interview Questions

**Q: What is p-value and how do you interpret it?**
The p-value is the probability of observing a result as extreme as (or more extreme than) what was observed, *assuming the null hypothesis is true*. It is NOT the probability that H₀ is true. p = 0.03 means: if H₀ were true, we'd see this result only 3% of the time by chance. Since this is below our threshold α=0.05, we reject H₀.

**Q: A/B test shows p=0.04. Is the new feature better?**
Not necessarily. Check: (1) Was the test pre-registered with a fixed sample size? (2) Did you correct for multiple comparisons if you tested multiple metrics? (3) Is the effect size practically significant? (4) Did you check for novelty effects? (5) Were there any confounders (day-of-week effects, etc.)? Statistical significance ≠ practical significance.

**Q: Explain the bias-variance tradeoff using statistics.**
Test error = Bias² + Variance + Irreducible Noise. Bias: systematic error from wrong assumptions (high in underfitting). Variance: sensitivity to training data fluctuations (high in overfitting). The tradeoff: reducing bias (more complex model) increases variance and vice versa. Optimal model minimizes their sum.

**Q: How does regularization connect to Bayesian priors?**
L2 regularization (Ridge) is equivalent to MAP estimation with a Gaussian prior on weights. L1 (Lasso) is MAP with a Laplace prior. The regularization strength λ corresponds to the inverse variance of the prior. This Bayesian view explains why L1 induces sparsity: the Laplace prior has infinite density at zero, strongly pulling weights toward zero.

**Q: What is the difference between standard error and standard deviation?**
Standard deviation (σ) measures variability in the *population*. Standard error (SE = σ/√n) measures variability of the *sample mean* — how much the mean estimate varies across different samples. SE decreases with more data (√n in denominator), showing that larger samples give more precise estimates of the true mean.

**Q: When would you use non-parametric tests?**
When data doesn't meet parametric assumptions: non-normality with small n, ordinal data, heavy outliers, or when you can't assume a specific distribution. Examples: Mann-Whitney U (vs t-test), Kruskal-Wallis (vs ANOVA), Spearman (vs Pearson). The trade-off is lower statistical power when assumptions actually hold.
