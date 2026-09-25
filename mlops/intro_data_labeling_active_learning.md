# Data Labeling, Active Learning & Weak Supervision

Most production ML teams spend more time and money getting labels than choosing architectures. Interviewers know this, so "you have 10 million unlabeled examples and a $50k budget — what do you do?" is a common system design and MLOps question. A good answer shows you can write a labeling spec, measure whether annotators agree, find bad labels, spend the budget where the model is weakest (active learning), stretch it with programmatic and model-generated labels (weak supervision, pseudo-labeling, LLM annotators, synthetic data), and keep the whole thing versioned and reproducible.

---

## Table of Contents
1. [Data-Centric AI Framing](#data-centric-ai-framing)
2. [Labeling Guidelines and Ontology Design](#labeling-guidelines-and-ontology-design)
3. [Annotation Workflows](#annotation-workflows)
4. [Inter-Annotator Agreement](#inter-annotator-agreement)
5. [Gold Questions and Adjudication](#gold-questions-and-adjudication)
6. [Label Noise and How to Find It](#label-noise-and-how-to-find-it)
7. [Active Learning](#active-learning)
8. [Weak Supervision](#weak-supervision)
9. [Semi-Supervised Learning](#semi-supervised-learning)
10. [LLMs as Annotators](#llms-as-annotators)
11. [Synthetic Data](#synthetic-data)
12. [Class Imbalance at Labeling Time](#class-imbalance-at-labeling-time)
13. [Data Versioning and Lineage](#data-versioning-and-lineage)
14. [Cost and Budget Estimation](#cost-and-budget-estimation)
15. [Interview Q&A](#interview-qa)
16. [Common Pitfalls](#common-pitfalls)
17. [Related Topics](#related-topics)

---

## Data-Centric AI Framing

Model-centric work holds the data fixed and iterates on the model. Data-centric work holds the model fixed (usually a strong off-the-shelf baseline) and iterates on the data: fixing labels, clarifying the definition of each class, adding examples where the model fails.

More or better data usually beats a better model when:

| Signal | What it tells you | Data action |
|---|---|---|
| Learning curve still rising at full dataset size | Model is data-starved | Label more, ideally targeted |
| Train and val error both high, several architectures plateau at the same point | Ceiling is set by label noise or ambiguous classes | Audit labels, tighten guidelines |
| Errors cluster in specific slices (a language, a device, a rare class) | Coverage gap | Collect or label that slice |
| Human agreement is low on the task | Task definition is fuzzy | Fix the ontology before anything else |
| Model beats humans on val but fails in production | Val set labels or distribution are wrong | Rebuild the eval set from production traffic |

A better model is usually the right move when the learning curve has flattened, labels are clean (agreement is high and audits find few errors), and the error analysis points to a capability gap such as long context, missing features, or wrong inductive bias.

A useful rule of thumb: **label noise sets a ceiling on measured accuracy**. If 8% of your test labels are wrong, a perfect model scores about 92% and a model that learned the same mistakes can score higher. Clean the test set before comparing models.

```python
# Learning curve: is more data likely to help?
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

X, y = make_classification(n_samples=5000, n_features=30, n_informative=10, random_state=0)
sizes, _, val_scores = learning_curve(
    LogisticRegression(max_iter=1000), X, y,
    train_sizes=np.linspace(0.05, 1.0, 8), cv=5, scoring="accuracy",
)
for n, s in zip(sizes, val_scores.mean(axis=1)):
    print(f"{n:5d} examples -> val acc {s:.3f}")
# If the last few points are still climbing, more labels will likely pay off.
```

---

## Labeling Guidelines and Ontology Design

The ontology is the set of labels and their structure. The guidelines are the document annotators follow. Both are product decisions, and most label quality problems trace back to them rather than to careless annotators.

**Ontology design rules**

- **Mutually exclusive and collectively exhaustive** for single-label tasks. Add an explicit `other` and an `unsure / cannot tell` option so annotators don't force a guess.
- **Match the granularity to the decision the model drives.** If the product only acts on "spam vs not spam," a 14-way spam taxonomy costs money and lowers agreement.
- **Hierarchical labels** (`vehicle > car > sedan`) let you label coarse first and refine later, and let you back off to the parent when the child is ambiguous.
- **Multi-label vs multi-class** must be decided up front. Switching later means relabeling.
- **Version the ontology.** When a class is split or merged, old labels need a mapping or a relabel.

**What a good guideline document contains**

| Section | Content |
|---|---|
| Task purpose | One paragraph on what the model does with these labels |
| Label definitions | Precise definition per class, with the decision rule |
| Positive and negative examples | 3-5 each, including near misses |
| Edge cases | Explicit rulings: "sarcastic praise counts as negative" |
| Tie-breakers | Priority order when two labels seem to apply |
| When to escalate | What to mark `unsure` vs guess |
| Changelog | Every rule change, dated, so old labels can be interpreted |

Write the first draft, have 3-5 people label the same 100-200 items, measure agreement, read every disagreement, update the guidelines, repeat. Two or three rounds of this pilot usually fix most ambiguity before you spend real money.

---

## Annotation Workflows

| Option | Best for | Strengths | Weaknesses |
|---|---|---|---|
| **In-house experts** | Medical, legal, policy, security; small high-stakes sets; building the gold set | Deep domain knowledge, tight feedback loop, confidentiality | Expensive, slow to scale, expert time is scarce |
| **Managed vendor** | Large volume with moderate complexity; ongoing production labeling | Trained workforce, project management, QA built in, SLAs | Ramp-up time, less direct control, data must leave your org |
| **Crowdsourcing** | Simple, objective tasks at scale (bounding boxes, sentiment, relevance) | Cheap, fast, elastic | Noisy, needs redundancy and heavy QA, unsuitable for sensitive data |
| **Programmatic / model** | Bootstrapping, high volume, stable patterns | Near-zero marginal cost | Systematic errors; needs human-labeled data to validate |

Common production pattern: experts write guidelines and label the gold set; a vendor or crowd labels volume with gold questions mixed in; a model pre-labels to speed humans up (humans correct instead of labeling from scratch); disagreements and low-confidence items go to experts.

**Pre-labeling caveat:** showing annotators a model's suggestion speeds them up but anchors them. They accept wrong suggestions more often than they would make the same mistake unaided. Measure this by sending a small random sample through without pre-labels and comparing.

---

## Inter-Annotator Agreement

Raw percent agreement is misleading because some agreement happens by chance, especially with imbalanced classes. If 95% of items are "not toxic," two annotators who both always say "not toxic" agree 95% of the time while learning nothing.

| Metric | Raters | Handles missing ratings | Label types | Notes |
|---|---|---|---|---|
| **Cohen's kappa** | Exactly 2 | No | Nominal (weighted variant for ordinal) | κ = (p_o − p_e) / (1 − p_e) |
| **Fleiss' kappa** | Fixed number per item, can be different people per item | No | Nominal | Generalizes Scott's pi to many raters |
| **Krippendorff's alpha** | Any number | Yes | Nominal, ordinal, interval, ratio | α = 1 − D_observed / D_expected; most flexible |

Here `p_o` is observed agreement and `p_e` is the agreement expected if both raters labeled independently at their own marginal rates.

A commonly quoted interpretation scale (Landis and Koch) calls 0.41-0.60 moderate, 0.61-0.80 substantial, and above 0.80 almost perfect. Treat these as rough conventions, not thresholds with statistical meaning. Kappa also depends on class prevalence: with very skewed classes you can get low kappa despite high raw agreement, so report both.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

a = np.array([1, 0, 0, 1, 2, 2, 0, 1, 1, 0, 2, 0])
b = np.array([1, 0, 1, 1, 2, 1, 0, 1, 0, 0, 2, 0])

def cohens_kappa(r1, r2):
    labels = np.union1d(r1, r2)
    p_o = np.mean(r1 == r2)
    p_e = sum(np.mean(r1 == k) * np.mean(r2 == k) for k in labels)
    return (p_o - p_e) / (1 - p_e)

print(f"percent agreement: {np.mean(a == b):.3f}")
print(f"manual kappa:      {cohens_kappa(a, b):.3f}")
print(f"sklearn kappa:     {cohen_kappa_score(a, b):.3f}")
# For ordinal labels (e.g. 1-5 ratings) penalize distant disagreements more:
# cohen_kappa_score(a, b, weights="quadratic")
```

For Fleiss' kappa, `statsmodels.stats.inter_rater.fleiss_kappa` takes an items × categories count table. For Krippendorff's alpha, the `krippendorff` package on PyPI handles missing ratings.

**Low agreement is information.** Before blaming annotators, check whether disagreements concentrate on particular classes or item types. That usually points to a guideline gap or a genuinely ambiguous task, in which case soft labels (the distribution of annotator votes) may be a better training target than a forced majority label.

---

## Gold Questions and Adjudication

**Gold questions** (also called honeypots or test questions) are items with known, expert-verified answers mixed invisibly into the annotation queue.

- Use them to qualify annotators before they start and to monitor them continuously.
- Keep 5-10% of the queue as gold early on; reduce once quality is stable.
- Rotate the gold set. Annotators share answers and memorize repeated items.
- Make gold items representative, including hard cases. A gold set of only easy items tells you who is paying attention, not who understands the guidelines.

**Adjudication** resolves disagreements when multiple people label the same item.

| Strategy | How it works | Use when |
|---|---|---|
| Majority vote | Most common label wins | Crowd tasks with 3-5 labels per item |
| Weighted vote | Weight annotators by gold-question accuracy | Annotator quality varies a lot |
| Probabilistic (Dawid-Skene style) | EM jointly estimates true labels and per-annotator confusion matrices | Many annotators, no reliable gold |
| Expert adjudicator | A senior reviewer decides disagreements | High-stakes labels, small volume |
| Keep the distribution | Train on soft labels | Genuinely subjective tasks |

A cost-saving pattern is **dynamic redundancy**: label once, get a second label only if the first annotator has low accuracy or the model disagrees with them, and escalate to a third or an expert only if the first two disagree.

---

## Label Noise and How to Find It

Real datasets contain wrong labels. Estimates of a few percent label errors in widely used benchmark test sets have been published, and internal datasets built under deadline pressure are usually worse.

**Types of noise**

- **Uniform (symmetric):** label flipped to a random class. Mostly lowers the ceiling; models are fairly robust to moderate amounts.
- **Class-conditional:** specific confusions, such as "wolf" labeled as "husky". More damaging because it is systematic and the model learns it.
- **Instance-dependent:** hard or ambiguous items get mislabeled more often. The most realistic and hardest case.

**Detection approaches**

1. **Loss-based filtering.** Examples a model cannot fit, or fits only late in training, are disproportionately mislabeled. Rank by per-example loss and review the top.
2. **Confident learning.** Get out-of-sample predicted probabilities via cross-validation. For each class compute a threshold equal to the model's average confidence on examples labeled as that class. An example is flagged if the model is confidently (above threshold) predicting a different class than its given label. This estimates the joint distribution of given vs true labels and flags likely errors without needing clean data. The open-source `cleanlab` library implements this (`cleanlab.filter.find_label_issues(labels, pred_probs)`).
3. **Annotator disagreement.** Items where annotators split, or where one annotator disagrees with the consensus, are candidates.
4. **Embedding neighbors.** An item whose label differs from most of its nearest neighbors is suspicious.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

X, y_true = make_classification(n_samples=2000, n_features=20, n_informative=8, n_classes=3,
                                n_clusters_per_class=1, class_sep=2.0, random_state=0)
rng = np.random.default_rng(0)
y = y_true.copy()
flip = rng.choice(len(y), size=100, replace=False)       # inject 5% noise
y[flip] = (y[flip] + rng.integers(1, 3, size=100)) % 3

# Out-of-sample probabilities: never score an example with a model trained on it
probs = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")

# Simplified confident-learning rule: per-class thresholds = mean self-confidence
thresholds = np.array([probs[y == k, k].mean() for k in range(3)])
pred = probs.argmax(axis=1)
confident_other = (pred != y) & (probs[np.arange(len(y)), pred] >= thresholds[pred])
flagged = np.where(confident_other)[0]

precision = np.isin(flagged, flip).mean()
recall = np.isin(flip, flagged).mean()
print(f"flagged {len(flagged)}: precision {precision:.0%}, recall {recall:.0%}")
# Roughly 85% precision / 75% recall here; much lower on a harder task where the
# model itself is weak, which is why flagged items go to review, not deletion.
```

**What to do with flagged examples:** send them to review rather than deleting them automatically. Automatic removal tends to discard hard but correct examples and the minority class. If you must act automatically, prefer removing from training data over editing labels, and never auto-clean the test set without human review.

---

## Active Learning

Active learning chooses which unlabeled examples to send to annotators so that each label improves the model as much as possible. The standard setting is **pool-based**: a large unlabeled pool, a small labeled seed, and a loop of train → score pool → query → label → retrain.

### Query Strategies

| Strategy | Score (higher = query first) | Intuition |
|---|---|---|
| **Least confidence** | `1 − max_k p(k\|x)` | Model's top guess is weak |
| **Margin** | `−(p_top1 − p_top2)` | Top two classes are close; good for multi-class |
| **Entropy** | `−Σ p_k log p_k` | Uncertainty spread over all classes |
| **Query-by-committee** | Disagreement among an ensemble (vote entropy or KL to the mean) | Committee members disagree where the hypothesis space is unconstrained |
| **Bayesian (e.g. BALD)** | Mutual information between prediction and model parameters, often via MC dropout | Separates model uncertainty from inherent noise |
| **Diversity / core-set** | Pick points that cover embedding space, e.g. k-center greedy | Avoid redundant queries; works without calibrated probabilities |
| **Hybrid** | Cluster the uncertain candidates, pick one per cluster | Uncertain and diverse; common default in practice |

### Practical Issues

- **Batch mode.** Retraining after every label is impractical. When you pick the top-B uncertain points at once, they tend to be near-duplicates from the same region. Fix by pre-filtering to the top few × B uncertain points and then choosing a diverse subset (clustering, k-center, or methods like BADGE that combine gradient magnitude with diversity).
- **Cold start.** With very few labels the model's uncertainty is unreliable. Seed with a random or stratified sample, or a diversity-based selection over pretrained embeddings, before switching to uncertainty.
- **Outliers.** Pure uncertainty sampling loves junk: corrupted images, off-language text, spam. Filter the pool or weight by density.
- **Sampling bias.** The labeled set is no longer a random sample of the data. Keep a separate, randomly sampled test set for evaluation and never evaluate on actively selected data.
- **Transfer.** Data chosen to be informative for one model is not guaranteed to be informative for a different architecture trained later.
- **Stopping.** Stop when validation performance plateaus, when the maximum uncertainty in the pool drops, or when the budget runs out.

### Pool-Based Loop in scikit-learn

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=20, n_informative=6,
                           n_classes=3, n_clusters_per_class=2, random_state=0)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(axis=1)

def run(strategy, seed_size=30, batch=20, rounds=15):
    rng = np.random.default_rng(0)      # same random seed set for both strategies
    labeled = list(rng.choice(len(X_pool), size=seed_size, replace=False))  # cold start: random
    accs = []
    for _ in range(rounds):
        model = LogisticRegression(max_iter=1000).fit(X_pool[labeled], y_pool[labeled])
        accs.append(model.score(X_test, y_test))
        unlabeled = np.setdiff1d(np.arange(len(X_pool)), labeled)
        if strategy == "entropy":
            scores = entropy(model.predict_proba(X_pool[unlabeled]))
            pick = unlabeled[np.argsort(-scores)[:batch]]
        else:
            pick = rng.choice(unlabeled, size=batch, replace=False)
        labeled.extend(pick)            # "annotator" reveals y_pool[pick]
    return accs

for s in ["random", "entropy"]:
    accs = run(s)
    print(f"{s:8s} start {accs[0]:.3f} -> after {30 + 20 * 14} labels {accs[-1]:.3f}")
```

To swap in margin sampling, sort the probabilities and use `-(p[:, -1] - p[:, -2])` as the score. For query-by-committee, train several models on bootstrap samples of the labeled set and score by vote entropy. Libraries such as `modAL` and `small-text` wrap these patterns.

---

## Weak Supervision

Weak supervision replaces hand labels with many cheap, noisy sources and then combines them. The Snorkel framework popularized the approach.

1. **Labeling functions (LFs):** small programs that vote on a label or abstain. Sources include keyword rules, regexes, heuristics, existing knowledge bases (distant supervision), older models, and LLM prompts.
2. **Label model:** learns how accurate each LF is and how LFs correlate, using only their agreements and disagreements on unlabeled data (no ground truth needed). It outputs a probabilistic label per example.
3. **End model:** a normal discriminative model trained on the probabilistic labels. It generalizes beyond the LFs because it learns from features, not from the rules themselves, so it can cover examples where every LF abstained.

```python
import numpy as np
ABSTAIN, HAM, SPAM = -1, 0, 1

def lf_has_link(t):     return SPAM if "http" in t else ABSTAIN
def lf_free_money(t):   return SPAM if "free" in t and "$" in t else ABSTAIN
def lf_short_reply(t):  return HAM if len(t.split()) < 5 else ABSTAIN
def lf_mentions_you(t): return HAM if "you mentioned" in t else ABSTAIN

lfs = [lf_has_link, lf_free_money, lf_short_reply, lf_mentions_you]
texts = ["free $ at http://x.co", "thanks!", "as you mentioned, see http://docs",
         "claim your free $ prize before midnight", "meeting moved to 3pm today please confirm"]

L = np.array([[lf(t) for lf in lfs] for t in texts])   # label matrix: n_items x n_lfs
coverage = (L != ABSTAIN).mean(axis=0)
print("LF coverage:", {f.__name__: round(float(c), 2) for f, c in zip(lfs, coverage)})

def majority_vote(row):
    votes = row[row != ABSTAIN]
    return ABSTAIN if len(votes) == 0 else np.bincount(votes).argmax()

print("labels:", [int(majority_vote(r)) for r in L])  # ties go to the lower class id
# A real label model (e.g. snorkel.labeling.model.LabelModel) weights LFs by
# estimated accuracy instead of counting votes equally.
```

**LF diagnostics to track:** coverage (fraction of items it labels), overlap and conflict with other LFs, and empirical accuracy on a small hand-labeled dev set. You still need a few hundred human labels to develop LFs and to evaluate the end model.

**When it works well:** domain experts can express knowledge as rules, labels are needed in bulk, and the task definition changes often (edit an LF and relabel everything in minutes). **When it struggles:** tasks with no expressible heuristics (fine-grained visual categories) or where the LFs share one blind spot, which the label model cannot detect.

---

## Semi-Supervised Learning

Semi-supervised learning trains on labeled and unlabeled data together.

| Method | Idea | Watch out for |
|---|---|---|
| **Pseudo-labeling / self-training** | Train, predict on unlabeled data, add high-confidence predictions as labels, retrain | Confirmation bias: the model reinforces its own mistakes. Use high thresholds and per-class thresholds |
| **Consistency regularization** | Penalize different predictions for two augmentations of the same unlabeled input (Pi-model, Mean Teacher, UDA) | Needs meaningful augmentations for the modality |
| **FixMatch-style** | Pseudo-label from a weakly augmented view if confidence exceeds a threshold, train the strongly augmented view to match | Threshold tuning; class imbalance in pseudo-labels |
| **Self-supervised pretraining + fine-tune** | Learn representations from unlabeled data first, then fine-tune on labels | Often the strongest option today; many teams start from a pretrained foundation model instead |

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = make_classification(n_samples=3000, n_features=20, random_state=1)
y_partial = y.copy()
rng = np.random.default_rng(1)
y_partial[rng.random(len(y)) < 0.97] = -1       # only ~3% labeled; -1 means unlabeled

base = LogisticRegression(max_iter=1000)
self_train = SelfTrainingClassifier(base, threshold=0.9).fit(X, y_partial)
supervised = LogisticRegression(max_iter=1000).fit(X[y_partial != -1], y[y_partial != -1])
print(f"supervised only: {supervised.score(X, y):.3f}  self-training: {self_train.score(X, y):.3f}")
```

Semi-supervised learning helps most when unlabeled data comes from the same distribution as the labeled data and the classes are reasonably separated. It can hurt when the unlabeled pool contains classes that are absent from the labeled set.

---

## LLMs as Annotators

Large language models can label text (and, with multimodal models, images) at a fraction of human cost and far faster. Treat an LLM as one more annotator whose quality you measure, not as ground truth.

**Workflow**

1. Write the prompt from the human labeling guidelines, including definitions, edge-case rulings, and a few examples.
2. Build a human-labeled gold set of a few hundred items, stratified to include rare classes and hard cases.
3. Run the LLM on gold. Report accuracy, per-class precision and recall, confusion matrix, and Cohen's kappa against humans. Compare to human-human agreement, which is the realistic ceiling.
4. Calibrate: check whether the model's stated confidence, the agreement across several samples, or token log-probabilities actually predict correctness. Route low-confidence items to humans.
5. Keep auditing a random sample of LLM labels in production. Re-run the gold evaluation whenever you change the prompt or the model version.

**Cost math (illustrative numbers; plug in current prices)**

```python
n_items = 200_000
tokens_per_item = 400 + 50            # prompt incl. guidelines + output
price_per_m_tokens = 1.00             # blended $ per million tokens (placeholder)
llm_cost = n_items * tokens_per_item / 1e6 * price_per_m_tokens

human_cost_per_label = 0.08           # $ per label (placeholder)
redundancy = 3                        # humans typically need multiple labels per item
human_cost = n_items * human_cost_per_label * redundancy

review_rate = 0.15                    # share routed to humans because LLM is unsure
hybrid_cost = llm_cost + n_items * review_rate * human_cost_per_label * redundancy
print(f"LLM ${llm_cost:,.0f} | human ${human_cost:,.0f} | hybrid ${hybrid_cost:,.0f}")
```

Prompt caching of the shared guideline prefix and batch APIs can reduce LLM cost further. The cost that is easy to forget is the human gold set and ongoing audits, which you need either way.

**Risks**

- **Systematic bias:** LLM errors are correlated across items, unlike independent human mistakes, so they do not average out with more volume. Known tendencies include favoring certain positions or answer orderings, verbosity, and cultural or dialect bias.
- **Self-preference:** using the same model family to label training data and to judge outputs can inflate scores.
- **Distillation ceiling:** a model trained on LLM labels will learn the LLM's mistakes, and will rarely exceed the labeler on the task.
- **Terms and privacy:** check provider terms on using outputs to train models, and whether the data can be sent to an external API at all.
- **Drift:** a provider model update can silently change label distributions. Pin model versions and log them with every label.

---

## Synthetic Data

| Type | Examples | Good for | Main risk |
|---|---|---|---|
| **Augmentation** | Flips, crops, color jitter, mixup, back-translation, synonym swap, SpecAugment | Invariances you know hold; small datasets | Transform changes the label (flipping a "6", negating a sentence) |
| **Simulation / rendering** | Game engines for driving and robotics, rendered product images, simulated sensor data | Rare or dangerous scenarios, perfect labels for free | Sim-to-real gap; mitigate with domain randomization and real fine-tuning data |
| **Generative models** | Diffusion images, LLM-written examples, instruction data, tabular generators (CTGAN-style) | Rare classes, privacy-preserving stand-ins, bootstrapping a new task | Low diversity, factual errors, leaks of training data |
| **Rule-based / templated** | Fill-in templates, programmatic fraud patterns | Structured formats, unit tests for models | Model learns the template, not the task |

**Using LLM-generated data well:** vary personas, topics, lengths, and difficulty explicitly in the prompt; deduplicate (exact and near-duplicate); filter with a verifier or classifier; check that synthetic examples do not overlap your test set; and always evaluate on real data.

**Model collapse.** When models are trained repeatedly on data produced by earlier models, the tails of the distribution get underrepresented at each generation and the output distribution narrows, eventually losing rare modes. Research published in 2024 (Shumailov et al., Nature) demonstrated this for recursively trained generative models. Mitigations: keep a stable pool of real data in every training mix rather than replacing it, track provenance so you know which examples are synthetic, and monitor diversity metrics on generated data.

---

## Class Imbalance at Labeling Time

If positives are 0.1% of traffic, labeling a random sample of 10,000 items yields about 10 positives. Fix this at sampling time, not only at training time.

- **Pre-filter with a cheap signal:** keyword rules, an existing model's score, user reports. Label the enriched pool.
- **Stratified sampling by model score:** sample from score buckets so you get positives from the top and hard negatives from the middle.
- **Active learning** naturally oversamples the uncertain boundary region, which often contains the minority class.
- **Record the sampling weights.** When you enrich, the labeled set's class ratio no longer matches production. Keep a separate random sample to estimate true prevalence and to compute unbiased precision and recall, or apply importance weights.
- **Don't forget hard negatives.** An enriched set that is all obvious negatives plus positives teaches the model the wrong boundary.

---

## Data Versioning and Lineage

Labels change: guidelines get revised, noise gets fixed, vendors relabel. Without versioning you cannot reproduce a model or explain why metrics moved.

What to record for every label:

| Field | Why |
|---|---|
| Item ID and content hash | Detect when the underlying item changed |
| Label and label schema / ontology version | Interpret old labels after class splits |
| Guideline version | Know which rules the annotator followed |
| Source: annotator ID, vendor, LF name, model name and version, prompt version | Audit and re-weight sources; remove a bad annotator's labels |
| Timestamp and confidence | Filter stale or low-confidence labels |
| Adjudication history | Show how disagreements were resolved |

Tools: DVC or lakeFS for versioning datasets alongside code, Delta Lake or Iceberg tables with time travel for labels stored in the warehouse, and your experiment tracker (for example MLflow) to log the dataset version each model was trained on. Freeze the test set per evaluation cycle and change it deliberately, with a version bump, so model comparisons stay valid.

---

## Cost and Budget Estimation

A back-of-the-envelope estimate interviewers expect you to be able to produce:

```python
def labeling_budget(n_items, seconds_per_item, hourly_rate, redundancy=1.0,
                    gold_rate=0.07, review_rate=0.10, overhead=0.25):
    """Rough cost of a human labeling project."""
    effective_items = n_items * redundancy * (1 + gold_rate)   # extra labels on gold items
    hours = effective_items * seconds_per_item / 3600
    hours *= (1 + review_rate)                                # QA / expert review
    cost = hours * hourly_rate * (1 + overhead)               # PM, tooling, vendor margin
    return round(hours), round(cost)

# Example: 50k text items, 20 s each, 2x redundancy, $15/hr workforce
hours, cost = labeling_budget(50_000, 20, 15, redundancy=2)
print(f"~{hours:,} annotator-hours, ~${cost:,}")
```

Things that dominate real budgets:

- **Time per item** varies by orders of magnitude: a binary sentiment label takes seconds, a polygon segmentation mask can take many minutes, a medical read can need a specialist. Time a pilot rather than guessing.
- **Redundancy** multiplies everything. Use dynamic redundancy instead of a flat 3x.
- **Pilot and guideline iteration** is a fixed cost that saves rework later.
- **Expert time** for gold sets and adjudication is expensive per hour but small in volume.
- **Relabeling** after an ontology change. Design the ontology carefully to avoid paying twice.

Then ask which lever buys the most accuracy per dollar: more random labels, actively selected labels, cleaning existing labels, or a weak or LLM labeler plus human review. A small experiment on each, measured on the same clean test set, answers this better than intuition.

---

## Interview Q&A

#### You have 1M unlabeled examples and budget for 10k labels. How do you spend it?

First, spend a small fraction on a pilot: write guidelines, have several annotators label the same few hundred items, measure agreement, and fix the guidelines. Second, set aside a random, carefully labeled test set (often 1-2k items, more if classes are rare) that is never touched by selection. Third, get a strong baseline cheaply: a pretrained model, zero-shot or few-shot, or an LLM labeler checked against the gold set. Then spend the remaining budget in active learning rounds using uncertainty plus diversity sampling, seeded with a random or stratified sample to avoid cold-start problems. Consider pseudo-labeling or weak supervision on the rest of the pool. Track the learning curve per round so you can show where extra labels stop paying off.

#### Why use Cohen's kappa instead of percent agreement?

Percent agreement ignores agreement that happens by chance. Kappa subtracts the agreement expected if both raters labeled independently at their own class rates: κ = (p_o − p_e)/(1 − p_e). With a 95/5 class split, two raters can agree 90%+ while kappa is near zero, which exposes that they don't agree on the rare class, usually the one you care about. Report both, because kappa is sensitive to prevalence and can look low even when agreement is genuinely good on skewed data.

#### When would you use Fleiss' kappa or Krippendorff's alpha instead?

Cohen's kappa is for exactly two raters. Fleiss' kappa handles more raters when each item gets the same number of ratings, even if different people rate different items, which fits crowd setups. Krippendorff's alpha is the most general: any number of raters, missing ratings, and different distance functions for nominal, ordinal, interval, or ratio labels. For a crowd project where items get a variable number of labels, alpha is usually the right choice.

#### Annotator agreement is low. What do you do?

Read the disagreements before changing anything. If they cluster in specific classes or item types, the guidelines are ambiguous there: add definitions, edge-case rulings, and examples, then re-pilot. If one or two annotators account for most disagreements, retrain or remove them using gold-question accuracy. If the task is inherently subjective (toxicity, humor, relevance grades), accept that, collect multiple labels per item, and consider training on soft labels. Low agreement also caps the model's achievable score, which matters when setting targets.

#### How do you find mislabeled examples in a large dataset?

Get out-of-sample predicted probabilities using cross-validation, then flag examples where the model confidently predicts a different class than the given label. Confident learning makes this principled with per-class confidence thresholds and an estimate of the noise joint distribution; cleanlab implements it. Complementary signals are high per-example training loss, annotator disagreement, and label disagreement with nearest neighbors in embedding space. Send flagged items to review rather than auto-deleting, because hard-but-correct and minority-class examples look similar to noise.

#### Explain uncertainty sampling and its failure modes.

Train on the labeled set, score each unlabeled example by how unsure the model is (least confidence, margin between the top two classes, or entropy), and label the most uncertain. Failure modes: it is unreliable at cold start when the model is poor; it picks redundant near-duplicates in batch mode; it is drawn to outliers and junk; poorly calibrated probabilities (common in deep nets) make the ranking noisy; and the resulting labeled set is biased, so it cannot be used as a test set. Fixes are random seeding, diversity-aware batch selection, outlier filtering, and a separate random test set.

#### What is query-by-committee?

Train several models (a committee) that are all consistent with the labeled data, for example by bootstrapping or using different architectures, and query the examples where they disagree most, measured by vote entropy or average KL divergence from the consensus. Disagreement marks regions where the labeled data hasn't yet pinned down the decision boundary. It is more robust than a single model's uncertainty but costs several times more training per round.

#### How does weak supervision differ from just using heuristic rules?

A heuristic used directly is only as good as its own coverage and accuracy. Weak supervision combines many overlapping, conflicting labeling functions and uses a label model to estimate each one's accuracy and correlations from their agreement patterns, without ground truth, producing probabilistic labels. An end model trained on those labels learns from features, so it generalizes to examples no rule covered and often outperforms the rules. It still needs a small hand-labeled dev set to develop LFs and a test set to evaluate.

#### What is the risk in pseudo-labeling and how do you control it?

Confirmation bias: the model's wrong predictions become training labels, and it gets more confident in its mistakes. It is worse for minority classes, which get fewer and less accurate pseudo-labels. Controls include a high confidence threshold (or per-class thresholds), using a teacher model that is stronger or an ensemble, consistency with augmented views (FixMatch), limiting the pseudo-label share, and validating every round on a clean human-labeled set.

#### Can you replace human annotators with an LLM?

Partly, and only after measuring. Treat the LLM as an annotator: build a human gold set, measure its accuracy and kappa against humans, and compare with human-human agreement. Use it where it matches humans, route low-confidence or disagreeing items to people, and keep auditing a random sample. Human labels are still needed for the gold set, for evaluation, and for subjective or high-stakes classes. The main risk is correlated systematic errors that don't average out, plus drift when the provider updates the model, so pin versions and log them.

#### When does synthetic data help and when does it hurt?

It helps when real data is rare, expensive, dangerous, or private: rare failure modes, edge cases in driving, fraud patterns, or bootstrapping a new task. It hurts when it doesn't match the real distribution (sim-to-real gap), lacks diversity, contains errors, or leaks into evaluation. Recursive training on model outputs can cause model collapse, where distribution tails disappear. Always evaluate on real data, mix in real data, deduplicate, and track which examples are synthetic.

#### How would you estimate the cost of a labeling project?

Items × labels per item × time per label ÷ 3600 × hourly rate, then add gold-question overhead, QA review, project management and tooling, and a pilot. Time per label must come from a timed pilot, since it varies by orders of magnitude across tasks. Then compare against alternatives: fewer labels with active learning, an LLM labeler with human review, or weak supervision, and pick based on accuracy per dollar measured on a clean test set.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Starting to label before guidelines are piloted | Inconsistent labels; expensive relabeling | Pilot with several annotators, measure agreement, iterate |
| Reporting only percent agreement | Chance agreement hides disagreement on rare classes | Report kappa or alpha alongside raw agreement |
| Evaluating on actively selected data | Biased sample; metrics don't reflect production | Keep a separate random test set |
| Pure uncertainty sampling in batch mode | Redundant near-duplicate queries waste budget | Add diversity (clustering, k-center) to batch selection |
| Using uncertainty sampling at cold start | Untrained model's uncertainty is meaningless | Seed with random or diversity-based samples |
| Auto-deleting examples flagged as noisy | Removes hard, correct, and minority-class examples | Human review of flagged items |
| Trusting LLM labels without a gold set | Correlated errors silently corrupt the dataset | Measure against human gold; audit continuously |
| Not pinning the LLM or prompt version | Label distribution shifts without anyone noticing | Log model and prompt version with every label |
| Enriched sampling without recording weights | Prevalence and precision estimates are wrong | Keep a random sample; store sampling weights |
| Pre-labels shown to annotators without checks | Anchoring makes annotators accept model errors | Compare against a pre-label-free sample |
| Replacing real data with synthetic data | Model collapse; sim-to-real gap | Keep real data in every mix; evaluate on real data |
| Labels stored without guideline or source version | Can't reproduce models or trace bad labels | Version labels, ontology, guidelines, and sources |

---

## Related Topics

- [Data Quality & Validation](./intro_data_quality.md)
- [Model Monitoring](./intro_model_monitoring.md)
- [MLflow](./intro_mlflow.md)
- [LLM Evaluation](./intro_llm_evaluation.md)
- [Model Evaluation](../classical_ml/intro_model_evaluation.md)
- [Anomaly Detection](../classical_ml/intro_anomaly_detection.md)
- [Fine-Tuning](../deep_learning/intro_fine_tuning.md)
- [Generative Models](../deep_learning/intro_generative_models.md)
- [Data Engineering for AI](../data_engineering/intro_data_engineering_for_ai.md)
- [Delta Lake](../data_engineering/intro_delta_lake.md)
- [ML System Design Patterns](../system_design/ml_system_design_patterns.md)
- [MLOps Overview](./README.md)
