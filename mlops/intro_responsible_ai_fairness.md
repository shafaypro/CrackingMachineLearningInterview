# Responsible AI: Fairness, Bias, Privacy, and Governance

Any model that decides who gets a loan, a job interview, a medical follow-up, or a content takedown will eventually be asked "is this fair, and can you prove it?" Interviewers use Responsible AI questions to check whether you can move past slogans: name the fairness metric you would use and why, explain why you cannot satisfy all of them at once, say where bias entered the pipeline, and describe how you would protect training data. It comes up in ML system design rounds (especially lending, hiring, healthcare, ads, and moderation), in LLM roles, and increasingly in MLOps roles where audits and documentation are part of the release process.

---

## Table of Contents

1. [What Responsible AI Covers](#what-responsible-ai-covers)
2. [Sources of Bias Across the Lifecycle](#sources-of-bias-across-the-lifecycle)
3. [Protected Attributes and Proxies](#protected-attributes-and-proxies)
4. [Group Fairness Metrics](#group-fairness-metrics)
5. [Computing Fairness Metrics in NumPy](#computing-fairness-metrics-in-numpy)
6. [Impossibility Results](#impossibility-results)
7. [Individual and Counterfactual Fairness](#individual-and-counterfactual-fairness)
8. [Bias Mitigation](#bias-mitigation)
9. [Fairness for LLMs](#fairness-for-llms)
10. [Privacy](#privacy)
11. [Documentation: Model Cards and Datasheets](#documentation-model-cards-and-datasheets)
12. [Governance and Regulation](#governance-and-regulation)
13. [Practical Fairness Audit Checklist](#practical-fairness-audit-checklist)
14. [Interview Q&A](#interview-qa)
15. [Common Pitfalls](#common-pitfalls)
16. [Related Topics](#related-topics)

---

## What Responsible AI Covers

Responsible AI is an umbrella term. In interviews it usually breaks down into five concerns:

| Concern | Core question | Typical tools |
|---|---|---|
| Fairness | Does the system treat groups and individuals equitably? | Disaggregated metrics, mitigation methods, audits |
| Privacy | Can the system leak or misuse personal data? | PII handling, differential privacy, federated learning |
| Transparency | Can people understand what the system does and why? | Model cards, datasheets, explanations |
| Safety and robustness | Does it fail gracefully and resist misuse? | Red-teaming, guardrails, monitoring |
| Accountability | Who owns decisions, and can they be contested? | Governance reviews, logging, appeal processes |

Fairness and privacy get the most technical depth in interviews, so most of this guide focuses there. Explainability has its own guide (see Related Topics).

---

## Sources of Bias Across the Lifecycle

"The model is biased" is not a diagnosis. Bias enters at specific stages, and the fix depends on the stage. A useful taxonomy:

| Type | Where it enters | Example | Typical fix |
|---|---|---|---|
| **Historical** | The world itself | Past hiring decisions favored one group, so "hired" labels encode that preference | Question the label; reconsider the target; constraints; human review |
| **Representation** | Sampling | Face dataset is mostly light-skinned adults; the model underperforms on others | Targeted data collection, stratified sampling, reweighting |
| **Measurement** | Features and labels | "Arrests" used as a proxy for "crime" when policing intensity differs by neighborhood | Choose a better-grounded label; audit proxy validity |
| **Aggregation** | Modeling | One model for a population where the feature-label relationship differs by subgroup (e.g., a clinical marker with different normal ranges) | Subgroup-aware features, interaction terms, or separate models |
| **Evaluation** | Benchmarks and metrics | Test set mirrors the skewed training set, so aggregate accuracy hides subgroup failures | Disaggregated evaluation; representative test sets |
| **Deployment** | Use in context | A risk score built for triage is used for punitive decisions | Clear intended-use documentation; access controls |
| **Feedback loops** | After launch | Predictive policing sends patrols where arrests were high, generating more arrests there, which retrains the model to send more patrols | Log exploration, counterfactual evaluation, monitor outcome drift by group |

Two points worth saying in an interview:

- **Label bias is the hardest.** If the label itself reflects past discrimination, a perfectly accurate model reproduces it. No amount of balancing the features fixes a bad target.
- **Feedback loops only show up after launch.** You see only outcomes for people the model approved (selective labels), so the data you retrain on is shaped by the model's own decisions.

---

## Protected Attributes and Proxies

**Protected attributes** are characteristics that anti-discrimination law or policy says should not drive certain decisions. The exact list depends on jurisdiction and domain, but it commonly includes race, color, sex or gender, religion, national origin, age, disability, and in many places sexual orientation, pregnancy, and marital status.

**Fairness through unawareness** (just drop the protected column) does not work, for two reasons:

1. **Proxies.** Other features correlate with the protected attribute. ZIP code, first name, school attended, shopping patterns, and even writing style can encode race or gender. The model rebuilds the attribute from its proxies.
2. **You lose the ability to measure.** Without the attribute, you cannot compute any group fairness metric. Many teams keep protected attributes out of the model's inputs but store them separately, under access controls, for auditing only.

**Detecting proxies:** train a model to predict the protected attribute from the other features. If it does much better than chance, the information is there and the main model can use it. Per-feature mutual information with the protected attribute helps you find the worst offenders, but dropping them rarely removes the leakage entirely because it is spread across many weak features.

**Disparate treatment vs. disparate impact** (US legal framing): disparate treatment is using the attribute directly; disparate impact is a facially neutral policy that produces substantially different outcomes across groups. A model without the protected attribute can still produce disparate impact.

---

## Group Fairness Metrics

Notation: `Y` is the true label, `Ŷ` the predicted label, `S` a score, `A` the group attribute. Most group metrics compare a conditional rate across values of `A`.

| Metric | Condition | Plain meaning | When it fits |
|---|---|---|---|
| **Demographic (statistical) parity** | `P(Ŷ=1 \| A=a) = P(Ŷ=1 \| A=b)` | Same selection rate per group | When base-rate differences are themselves considered an artifact of injustice, or for exposure/allocation |
| **Equal opportunity** | `P(Ŷ=1 \| Y=1, A=a) = P(Ŷ=1 \| Y=1, A=b)` | Same true positive rate | When missing a qualified person is the main harm (loans, hiring shortlists) |
| **Equalized odds** | Equal TPR **and** equal FPR across groups | Same error rates on both classes | When both false positives and false negatives cause harm (risk scores, fraud flags) |
| **Predictive parity** | `P(Y=1 \| Ŷ=1, A=a) = P(Y=1 \| Ŷ=1, A=b)` | Same precision (PPV) | When a positive prediction must mean the same thing for everyone |
| **Calibration within groups** | `P(Y=1 \| S=s, A=a) = s` for every group and score | A score of 0.7 means 70% risk regardless of group | When scores are shown to humans who act on them |

Ratios versus differences: you can report `rate_a - rate_b` or `rate_a / rate_b`. The **disparate impact ratio** (selection rate of the least-selected group divided by that of the most-selected group) is common in hiring contexts. The US EEOC "four-fifths rule" treats a ratio below 0.8 as evidence of possible adverse impact. It is a rule of thumb for screening, not a legal safe harbor.

Choosing a metric is a value judgment, not a technical one. The honest interview answer is: pick based on which errors harm whom, involve domain and legal stakeholders, and document the choice.

---

## Computing Fairness Metrics in NumPy

```python
import numpy as np

def group_rates(y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    tp = np.sum((yp == 1) & (yt == 1)); fp = np.sum((yp == 1) & (yt == 0))
    fn = np.sum((yp == 0) & (yt == 1)); tn = np.sum((yp == 0) & (yt == 0))
    return {
        "selection_rate": yp.mean(),
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "ppv": tp / max(tp + fp, 1),
        "base_rate": yt.mean(),
    }

def fairness_report(y_true, y_pred, group):
    rates = {g: group_rates(y_true, y_pred, group == g) for g in np.unique(group)}
    sel = [r["selection_rate"] for r in rates.values()]
    tpr = [r["tpr"] for r in rates.values()]
    fpr = [r["fpr"] for r in rates.values()]
    ppv = [r["ppv"] for r in rates.values()]
    summary = {
        "demographic_parity_diff": max(sel) - min(sel),
        "disparate_impact_ratio": min(sel) / max(max(sel), 1e-12),
        "equal_opportunity_diff": max(tpr) - min(tpr),
        "equalized_odds_diff": max(max(tpr) - min(tpr), max(fpr) - min(fpr)),
        "predictive_parity_diff": max(ppv) - min(ppv),
    }
    return rates, summary

def calibration_by_group(y_true, scores, group, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for g in np.unique(group):
        s, y = scores[group == g], y_true[group == g]
        idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
        out[g] = [(s[idx == b].mean(), y[idx == b].mean())  # (mean score, observed rate)
                  for b in range(bins) if np.any(idx == b)]
    return out

rng = np.random.default_rng(0)
n = 10_000
group = rng.choice(["a", "b"], size=n, p=[0.7, 0.3])
y_true = rng.binomial(1, np.where(group == "a", 0.30, 0.20))
scores = np.clip(0.5 * y_true + rng.normal(0.25, 0.2, n) - 0.05 * (group == "b"), 0, 1)
y_pred = (scores >= 0.5).astype(int)

rates, summary = fairness_report(y_true, y_pred, group)
for g, r in rates.items():
    print(g, {k: round(float(v), 3) for k, v in r.items()})
print({k: round(float(v), 3) for k, v in summary.items()})
```

In practice, libraries such as Fairlearn (`MetricFrame`) and AIF360 compute the same quantities with confidence handling and many more metrics. Knowing how to compute them by hand matters because interviewers often ask you to derive them from a confusion matrix.

Always report **group sizes and confidence intervals** next to these numbers. A 5-point TPR gap on a group of 40 positives is mostly noise.

---

## Impossibility Results

The most important theoretical fact in this area: **when base rates differ across groups, several natural fairness criteria cannot all hold at once** except in degenerate cases.

Results by Kleinberg, Mullainathan and Raghavan (2016) and by Chouldechova (2017) show that, if `P(Y=1 | A=a) ≠ P(Y=1 | A=b)`:

- **Calibration within groups** and **equalized odds** (balance of error rates) cannot both hold unless the classifier is perfect.
- **Predictive parity** and **equal FPR and FNR** cannot all hold at once.

Why, intuitively: for a binary classifier in one group,

```
FPR = p / (1 - p) · (1 - PPV) / PPV · TPR
```

where `p` is that group's base rate. If two groups share the same PPV and the same TPR but have different `p`, their FPRs must differ. Fix any two of the quantities and the base rate forces the third apart.

The widely discussed COMPAS recidivism debate is the standard example: one analysis showed unequal false positive rates across racial groups, while the vendor argued the scores were calibrated across groups. With different base rates, both statements can be true at once.

Other tensions worth knowing:

- **Demographic parity vs. calibration:** with different base rates, equal selection rates require a selection threshold that treats the same score differently by group.
- **Fairness vs. accuracy:** constraints usually cost some aggregate accuracy, though often less than people expect, and sometimes none when the unconstrained model was overfitting to a majority group.

The practical takeaway: you choose which criterion matters for the use case and state the tradeoff openly.

---

## Individual and Counterfactual Fairness

Group metrics can be satisfied while individuals are treated arbitrarily (for example, a random subset of one group is approved to hit a quota). Two alternative framings:

**Individual fairness** (Dwork et al., "fairness through awareness"): similar individuals should receive similar outcomes. Formally, a Lipschitz condition `D(f(x), f(x')) ≤ L · d(x, x')`. The hard part is the similarity metric `d`: deciding what "similar with respect to this task" means is exactly the contested question, so the definition moves the problem rather than solving it.

**Counterfactual fairness** (Kusner et al.): a prediction is fair for an individual if it would be the same in the counterfactual world where their protected attribute were different, holding everything not causally downstream of it fixed. It requires a causal graph, and the answer depends on which paths you treat as legitimate (a path through "years of experience" may be acceptable, a path through "name" is not).

A cheap practical approximation is a **counterfactual flip test**: change the protected attribute or an obvious proxy (a name, a pronoun) in the input and measure how much the prediction changes. It is not full counterfactual fairness, because it ignores downstream effects, but it catches direct dependence and is very useful for text models.

---

## Bias Mitigation

Mitigation methods are grouped by where they intervene.

| Stage | Method | How it works | Pros | Cons |
|---|---|---|---|---|
| Pre-processing | **Reweighing** | Weight each `(group, label)` cell so group and label look independent | Model-agnostic, simple | Only fixes the joint distribution, not label bias |
| Pre-processing | Resampling / targeted collection | Over-sample or collect more data for underrepresented groups | Often the best real fix | Collection is slow and costly |
| Pre-processing | Fair representations | Learn features that predict `Y` but not `A` | Reusable downstream | Hard to verify; can lose signal |
| In-processing | **Fairness constraints** | Optimize loss subject to, e.g., TPR gap ≤ ε (reductions approach, Lagrangian penalties) | Directly targets the chosen metric | Needs `A` at training time; tuning |
| In-processing | **Adversarial debiasing** | Predictor tries to predict `Y` while an adversary tries to recover `A` from its output; predictor is penalized when the adversary succeeds | Works with neural nets | Unstable training, like GANs |
| Post-processing | **Group-specific thresholds** | Pick a threshold per group to equalize TPR/FPR or selection rates | Cheap, no retraining | Needs `A` at inference; explicit group treatment may be legally restricted in some domains |
| Post-processing | Reject-option classification | Route uncertain cases near the threshold to humans or adjust them | Targets borderline cases | Needs a human process |

### Reweighing Example

Weights are `w(a, y) = P(A=a) · P(Y=y) / P(A=a, Y=y)`, so under-represented cells (such as positives in a low-base-rate group) get upweighted.

```python
import numpy as np

def reweighing_weights(group, y):
    w = np.empty(len(y), dtype=float)
    for g in np.unique(group):
        for label in np.unique(y):
            cell = (group == g) & (y == label)
            if cell.any():
                w[cell] = (group == g).mean() * (y == label).mean() / cell.mean()
    return w

# model.fit(X, y, sample_weight=reweighing_weights(group, y))
```

### Group Threshold Example

```python
def threshold_for_tpr(scores, y, target_tpr):
    pos_scores = np.sort(scores[y == 1])
    k = int(np.floor((1 - target_tpr) * len(pos_scores)))
    return pos_scores[min(k, len(pos_scores) - 1)]

thresholds = {g: threshold_for_tpr(scores[group == g], y_true[group == g], 0.80)
              for g in np.unique(group)}
```

Equalizing both TPR and FPR exactly with thresholds alone may be impossible, which is why the Hardt, Price and Srebro post-processing method allows randomizing between two thresholds for a group.

### Tradeoffs to Discuss

- **Fairness vs. accuracy:** plot the frontier (accuracy against fairness gap) and let stakeholders choose a point, rather than picking one silently.
- **One metric vs. another:** fixing equal opportunity can worsen predictive parity (see the impossibility results).
- **Legal constraints on the fix:** using the protected attribute to set thresholds can itself be prohibited in some domains, even when the goal is fairness. Get legal review.
- **Fixing the metric vs. fixing the cause:** post-processing can close a gap on paper while the model still performs poorly on the minority group. Better data is usually the durable fix.

---

## Fairness for LLMs

Generative models shift the focus from allocation decisions to **representational harms**: stereotyping, demeaning or erasing groups, uneven toxicity, and quality-of-service gaps (the model works worse in some dialects or languages).

| Harm | Example | How to measure |
|---|---|---|
| Stereotyping | Completing "The nurse said that..." with "she" far more than "he" | Template probes, co-reference benchmarks |
| Differential toxicity or sentiment | More negative continuations for prompts mentioning certain groups | Toxicity/sentiment classifiers over prompt sets |
| Quality-of-service gaps | Worse answers or higher refusal rates for some dialects or languages | Per-language and per-dialect evaluation |
| Allocation via LLM | LLM screens resumes and scores identical CVs differently by name | Counterfactual name-swap tests |
| Erasure | Groups missing from generated content (e.g., images of "a doctor") | Distribution of generated attributes |

**Bias benchmarks** you can name: WinoBias and WinoGender (gender bias in co-reference), BBQ (bias in question answering, including ambiguous contexts), StereoSet and CrowS-Pairs (stereotype preference), BOLD (open-ended generation), and RealToxicityPrompts (toxic continuations). Benchmarks have known validity problems, including noisy or culturally narrow examples, and they are mostly English and US-centric. Treat them as smoke tests, not certificates.

**Practical LLM fairness evaluation:**

1. Build **counterfactual prompt pairs** that differ only in a group signal (name, pronoun, dialect) and compare outputs with an automatic metric or an LLM judge.
2. Measure **refusal rate and answer quality by group and language**, not only toxicity.
3. **Red-team** with diverse testers, including people from affected communities, looking for stereotypes, slurs, and jailbreaks that elicit biased content.
4. Mitigate with data curation, preference tuning (RLHF/DPO with fairness-aware guidelines), system prompts, and output classifiers, then re-run the same evaluation as a regression suite.

LLM-as-judge evaluators can carry their own biases, so spot-check judge outputs against human ratings on a sample.

---

## Privacy

### PII Handling

- **Minimize:** collect only what the task needs; drop or truncate fields you do not use.
- **Detect and redact:** run PII detectors (regex plus NER models) on training data, logs, and prompts. For LLMs, also scrub outputs, since models can regurgitate training data.
- **Pseudonymize vs. anonymize:** replacing names with IDs is pseudonymization. Under GDPR, pseudonymized data is still personal data.
- **Access control and retention:** separate sensitive columns, log access, and set deletion schedules. Remember deletion requests may apply to derived artifacts too, which is hard for trained models (machine unlearning is an open research area).

### Why k-Anonymity Is Not Enough

k-anonymity requires every record to be indistinguishable from at least `k-1` others on quasi-identifiers (ZIP, birth date, sex). Its limits:

- **Homogeneity attack:** if all `k` records in a group share the same sensitive value, you learn it anyway. l-diversity and t-closeness were proposed to address this.
- **Background-knowledge and linkage attacks:** auxiliary datasets re-identify people. Latanya Sweeney showed that ZIP code, birth date, and sex alone uniquely identify a large majority of the US population, and the Netflix Prize data was de-anonymized by linking it with public IMDb ratings.
- **High-dimensional data** (browsing histories, embeddings) cannot be generalized enough without destroying utility.

### Differential Privacy

A randomized mechanism `M` is **(ε, δ)-differentially private** if for any two datasets `D, D'` differing in one person's record and any set of outputs `S`:

```
P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ
```

In words, the output distribution barely changes whether or not any one person is in the data. Smaller **ε** means stronger privacy and more noise. `δ` is a small failure probability, typically chosen well below `1/n`.

Key properties:

- **Composition:** privacy loss adds up across queries or training steps; you manage a **privacy budget**.
- **Post-processing immunity:** anything computed from a DP output is still DP.
- **Protects against any attacker**, regardless of auxiliary data, unlike k-anonymity.

The **Laplace mechanism** for a counting query (sensitivity 1):

```python
import numpy as np

def dp_count(values, predicate, epsilon, rng=np.random.default_rng()):
    true_count = sum(predicate(v) for v in values)
    sensitivity = 1.0  # adding/removing one person changes a count by at most 1
    return true_count + rng.laplace(0.0, sensitivity / epsilon)

ages = [34, 45, 29, 61, 52, 38]
print(dp_count(ages, lambda a: a > 40, epsilon=1.0))
```

**DP-SGD** makes neural network training differentially private:

1. Compute **per-example gradients**.
2. **Clip** each to a maximum L2 norm `C`, bounding any one person's influence.
3. Add **Gaussian noise** with standard deviation proportional to `σ · C` to the summed gradient.
4. Track cumulative privacy loss with an **accountant** (moments accountant or Rényi DP), which also accounts for the privacy amplification from sampling mini-batches.

Libraries: Opacus (PyTorch) and TensorFlow Privacy. The costs are slower training (per-example gradients), an accuracy drop that is usually worse for small or underrepresented groups, and extra hyperparameters. That last point matters for fairness: DP noise tends to hurt the tails of the distribution most.

### Federated Learning

Training happens on devices or silos; only model updates are sent to a server, which aggregates them (e.g., FedAvg). It keeps raw data local, but **it is not a privacy guarantee on its own**: gradients can leak information about the training examples. Production systems combine it with **secure aggregation** (the server sees only the sum of updates) and often **DP** on the updates. Challenges include non-IID data across clients, stragglers, and communication cost.

### Membership Inference and Related Attacks

| Attack | Goal | Signal exploited |
|---|---|---|
| Membership inference | Was this record in the training set? | Lower loss or higher confidence on training examples (overfitting); shadow models |
| Model inversion / attribute inference | Recover sensitive attributes of training records | Model outputs correlated with the attribute |
| Training data extraction | Get verbatim training text out of an LLM | Memorization of repeated or rare sequences |

Defenses: regularization and early stopping (reduce overfitting), deduplicating training data, DP training (the principled defense), limiting output precision (return labels not full probabilities), and rate-limiting queries.

---

## Documentation: Model Cards and Datasheets

**Model cards** (Mitchell et al.) document a trained model: intended use and out-of-scope uses, training and evaluation data, metrics **disaggregated by group and intersection**, fairness analysis, known limitations, ethical considerations, and caveats.

**Datasheets for datasets** (Gebru et al.) document a dataset: motivation, composition, collection process, preprocessing and labeling, recommended uses, distribution, and maintenance. They force questions such as "who is missing from this data?" and "did the people in it consent to this use?"

| Section | Model card | Datasheet |
|---|---|---|
| Purpose | Intended use, users, out-of-scope uses | Why the dataset was created, who funded it |
| Content | Architecture, training setup | Instances, labels, missing data, sensitive fields |
| Evaluation | Metrics per subgroup, fairness metrics | Known biases, splits |
| Risks | Limitations, failure modes | Consent, privacy, potential misuse |
| Lifecycle | Version, owner, update cadence | Maintenance, deprecation plan |

Treat both as living artifacts versioned with the model in your registry, not a one-off PDF. A card generated by the training pipeline (metrics auto-filled) stays accurate far longer than a hand-written one.

---

## Governance and Regulation

Regulation changes quickly and details depend on jurisdiction, so in interviews speak at the level of principles and say you would involve legal counsel. The high-level picture:

### EU AI Act

A risk-based regulation that entered into force in 2024, with obligations phasing in over several years.

| Tier | Examples | Obligations (summary) |
|---|---|---|
| Unacceptable risk | Social scoring by public authorities, manipulative techniques that cause harm, certain biometric uses | Prohibited |
| High risk | AI in hiring, credit scoring, education, critical infrastructure, law enforcement, some medical uses | Risk management, data governance, technical documentation, logging, human oversight, accuracy and robustness, conformity assessment |
| Limited risk | Chatbots, AI-generated or manipulated content | Transparency: tell people they are interacting with AI; label synthetic content |
| Minimal risk | Spam filters, game AI | No specific obligations |

It also adds obligations for providers of general-purpose AI models, such as documentation and additional duties for models considered to pose systemic risk.

### GDPR

- Applies to processing personal data of people in the EU. Relevant principles: lawful basis, purpose limitation, data minimization, storage limitation.
- **Article 22** restricts decisions based **solely** on automated processing that produce legal or similarly significant effects, with exceptions, and requires safeguards such as the right to obtain human intervention and to contest the decision.
- Articles 13 to 15 require "meaningful information about the logic involved" in such automated decisions. Whether this amounts to a full **"right to explanation"** of individual decisions is debated among legal scholars; the safe engineering position is to be able to produce a clear, per-decision explanation anyway.
- Data subject rights (access, erasure) apply to training data and can reach into ML pipelines.

### NYC Local Law 144 (Automated Employment Decision Tools)

Employers and employment agencies in New York City that use automated tools to substantially assist hiring or promotion decisions must have an **independent bias audit** within the past year, publish a summary of the results, and **notify candidates** that such a tool is used. The audit reports selection rates and **impact ratios** across sex and race/ethnicity categories, which is essentially the disparate impact ratio from the metrics section.

### Other Frameworks to Name

- **US sector law:** fair lending (ECOA), fair housing, and employment discrimination law (Title VII) already apply to algorithmic decisions; credit denials require adverse-action reasons.
- **NIST AI Risk Management Framework:** voluntary US framework organized around Govern, Map, Measure, and Manage.
- **ISO/IEC 42001:** management system standard for AI governance.

### Internal Governance

Laws set a floor. A mature organization also has: a risk classification for each ML use case, a review board for high-risk launches, required model cards, pre-launch fairness and privacy sign-off, incident response for AI harms, and an appeal path for affected users.

---

## Practical Fairness Audit Checklist

1. **Scope the harm.** Who is affected, what decision is made, and what does a false positive vs. false negative cost each group?
2. **Choose groups and metrics with stakeholders.** Include intersections (e.g., older women), and write down why the chosen metric fits.
3. **Get group labels responsibly.** Self-reported where possible; if inferred (e.g., BISG for race in lending), acknowledge the error it adds. Store under access control.
4. **Check the data.** Group representation, label quality and label bias, missingness by group, proxies for the protected attribute.
5. **Evaluate disaggregated.** Accuracy, TPR, FPR, PPV, selection rate, and calibration per group, with sample sizes and confidence intervals.
6. **Run counterfactual tests.** Flip names, pronouns, or the attribute; measure output change.
7. **Mitigate and compare.** Try at least one pre-, in-, or post-processing method; show the accuracy-fairness frontier.
8. **Check privacy exposure.** PII in features and logs, membership-inference risk, need for DP.
9. **Document.** Model card and datasheet, including known gaps and the rejected alternatives.
10. **Get sign-off.** Legal, domain experts, and the accountable owner.
11. **Monitor in production.** Track fairness metrics by group over time, watch for feedback loops, and re-audit on retrain or population shift.
12. **Provide recourse.** A way for affected people to understand, contest, and correct decisions.

---

## Interview Q&A

#### Why doesn't removing the protected attribute make a model fair?

Because the information is usually still present through proxies. ZIP code, name, school, purchase history, and many weak features together can reconstruct race or gender quite accurately, and a flexible model will find that signal if it helps predict the label. Removing the attribute also removes your ability to measure fairness. A better approach is to keep the attribute out of the model inputs (if the domain requires that) but retain it for auditing, test how well the other features predict it, and measure outcomes by group directly.

#### Demographic parity vs. equalized odds: when would you use each?

Demographic parity requires equal selection rates across groups regardless of true labels. It suits cases where you believe differences in base rates are themselves produced by unfairness, or where the goal is equal exposure (e.g., showing job ads). Its downside is that with genuinely different base rates it forces accepting less-qualified or rejecting more-qualified people in some group. Equalized odds requires equal TPR and FPR, so errors are distributed evenly among people with the same true outcome. It suits cases where the labels are trustworthy and both error types cause harm, such as fraud or risk scoring. If the labels themselves are biased, equalized odds just equalizes errors with respect to a biased target.

#### Explain the impossibility theorem in plain terms.

If two groups have different base rates of the outcome, you cannot have a score that is calibrated within each group and also has equal false positive and false negative rates across groups, unless it predicts perfectly. The confusion-matrix identity `FPR = p/(1-p) · (1-PPV)/PPV · TPR` makes this concrete: hold PPV and TPR equal and the base rate `p` forces FPR apart. The practical consequence is that "is this model fair?" has no single answer; you must pick the criterion that matches the harm and say what you are giving up.

#### Your credit model has a 10-point TPR gap between two groups. What do you do?

First, check it is real: group sizes, confidence intervals, and whether it holds across time slices. Then diagnose the cause: fewer training examples for the group, noisier features (thin credit files), label bias, or a threshold that falls in a different part of each group's score distribution. Fixes in rough order of durability: better data for the underserved group, features that capture creditworthiness better for thin-file applicants, reweighing or a constrained training objective, and finally group-aware thresholds if legally allowed. I would show the accuracy vs. TPR-gap tradeoff to risk, legal, and business owners, pick a point together, document it in the model card, and monitor the gap in production.

#### What is the difference between individual and group fairness?

Group fairness compares aggregate statistics across groups, such as selection rate or TPR. It is easy to measure but can be satisfied while individuals are treated inconsistently. Individual fairness says similar people should get similar outcomes, formalized as a Lipschitz condition on the model with respect to a task-specific similarity metric. It protects individuals but depends entirely on choosing that metric, which is the contested part. Counterfactual fairness is a causal version: the prediction should not change if the person's protected attribute had been different, holding non-descendant factors fixed.

#### How does adversarial debiasing work?

You train two networks. The predictor maps inputs to the task output. The adversary takes the predictor's output (or its internal representation, and for equalized odds also the true label) and tries to predict the protected attribute. The predictor's loss is its task loss minus a weighted adversary loss, so it is rewarded for making the attribute unrecoverable. If the adversary sees only the prediction, this pushes toward demographic parity; if it also sees the true label, it pushes toward equalized odds. It is flexible but can be unstable and needs careful tuning of the adversary weight.

#### How would you evaluate an LLM for bias?

Combine several methods because no single benchmark is sufficient. Run established benchmarks such as BBQ or WinoBias as smoke tests. Build counterfactual prompt pairs specific to the product (identical requests differing only in a name, pronoun, or dialect) and compare quality, sentiment, refusal rate, and any scores the model produces. Measure performance per language and dialect. Red-team with diverse testers for stereotypes and harmful content. Validate any LLM-as-judge scoring against human ratings. Then turn the failures into a regression suite that runs on every model or prompt change.

#### What is differential privacy, and what does epsilon mean?

A mechanism is ε-differentially private if its output distribution changes by at most a factor of `e^ε` when any one person's data is added or removed. So an attacker who sees the output cannot tell with much confidence whether a given person was in the data, no matter what else they know. Smaller ε means stronger privacy and more noise. Privacy loss composes: running many queries or training steps spends more of the budget. In practice, single-digit ε values are common for deployed ML, and the meaning of a particular ε depends on the unit of privacy (record vs. user) and on δ, so reporting those details matters as much as the number.

#### How does DP-SGD work, and what does it cost?

It clips each example's gradient to a fixed norm `C`, sums the clipped gradients, adds Gaussian noise scaled to `C`, and takes a step. Clipping bounds any one person's influence; noise hides what remains. A privacy accountant tracks cumulative ε across steps, benefiting from the amplification of sampling random mini-batches. The costs are compute (per-example gradients), accuracy loss, and extra hyperparameters (clip norm, noise multiplier, batch size). The accuracy loss is often concentrated on rare or underrepresented examples, so DP can widen fairness gaps and should be evaluated by group.

#### Does federated learning guarantee privacy?

No. It keeps raw data on the device, which reduces exposure, but model updates can still leak information: gradient inversion attacks can reconstruct inputs from updates in some settings, and the final model can memorize data like any other. Real deployments add secure aggregation so the server only sees summed updates, and differential privacy on client updates to bound what any single user contributes. Federated learning is a data-minimization architecture, not a formal privacy guarantee.

#### What is a membership inference attack and how do you defend against it?

The attacker wants to know whether a specific record was in the training set. The basic signal is that models tend to have lower loss and higher confidence on training examples than on unseen ones, so a threshold on loss or confidence, or a classifier trained on shadow models, can distinguish members. It matters when membership is itself sensitive, for example being in a dataset of patients with a particular condition. Defenses: reduce overfitting (regularization, early stopping, more data), deduplicate, train with DP for a formal bound, return labels or rounded scores instead of full probabilities, and rate-limit queries.

#### What goes in a model card for a high-risk model, and who reads it?

Intended use and explicitly out-of-scope uses; training and evaluation data sources with known gaps; metrics disaggregated by group and relevant intersections with sample sizes; the fairness criterion chosen and why; mitigation applied and the tradeoffs; privacy measures; limitations and failure modes; human oversight requirements; version, owner, and review date. Readers include downstream engineers deciding whether to reuse the model, auditors and regulators, product owners, and sometimes affected users. Generating the metrics sections automatically from the training pipeline keeps it accurate.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Dropping the protected attribute and calling it fair | Proxies reconstruct it; you can no longer measure bias | Keep it for auditing under access control; test proxy leakage |
| Reporting only aggregate accuracy | Subgroup failures are hidden by the majority | Disaggregate every metric by group and intersection |
| Tiny subgroups with no confidence intervals | Gaps are noise; false alarms or false comfort | Report counts and bootstrap CIs; pool time windows |
| Picking a fairness metric by default | Different metrics encode different values and conflict | Choose with stakeholders based on harm; document why |
| Trying to satisfy every fairness metric | Impossible with differing base rates | Pick the one that matches the harm; state the tradeoff |
| Treating labels as ground truth | Historical bias in labels is reproduced perfectly | Audit how labels were generated; consider alternative targets |
| Post-processing to close a gap without investigating | Metric looks fair while the model still fails the group | Diagnose root cause; improve data and features |
| Ignoring feedback loops | Model's own decisions shape future training data | Log exploration; counterfactual evaluation; monitor drift by group |
| Assuming anonymization protects privacy | Linkage attacks re-identify records | Use differential privacy for strong guarantees |
| Assuming federated learning is private | Updates and models leak information | Add secure aggregation and DP |
| Applying DP without checking subgroup accuracy | Noise hurts minority groups most | Evaluate DP models per group; tune budget |
| One-time fairness audit at launch | Population and model drift change outcomes | Monitor fairness metrics continuously; re-audit on retrain |
| Relying only on public bias benchmarks for LLMs | Narrow, English-centric, may not match your use case | Build product-specific counterfactual tests and red-team |

---

## Related Topics

- [Model Explainability](intro_model_explainability.md) — SHAP, LIME, and explanation requirements under GDPR and the EU AI Act
- [Model Monitoring](intro_model_monitoring.md) — Tracking drift and per-segment metrics in production
- [Data Quality](intro_data_quality.md) — Validation and data checks that catch representation problems early
- [LLM Evaluation](intro_llm_evaluation.md) — Evaluation methods and LLM-as-judge for generative models
- [Evaluation and Guardrails](intro_evaluation_guardrails.md) — Output filters and safety checks for LLM systems
- [A/B Testing](intro_ab_testing.md) — Measuring impact, including by segment, after launch
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md) — Confusion matrices, calibration, and slicing metrics
- [Causal Inference](../classical_ml/intro_causal_inference.md) — Background for counterfactual fairness
- [LLM Security](../ai_genai/intro_llm_security.md) — Red-teaming, jailbreaks, and data leakage in LLMs
- [Fraud Detection System Design](../system_design/fraud_detection.md) — A high-stakes system where error-rate balance matters
