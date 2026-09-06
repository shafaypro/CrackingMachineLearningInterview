# ML Take-Home Projects and Case Studies

The take-home is the highest-variance stage of an ML loop. Candidates with strong fundamentals routinely score poorly on it, not because the modeling was wrong but because they optimized the wrong thing — chasing an extra point of AUC while shipping a notebook nobody can run, no baseline to compare against, and no statement of what they would do next.

This guide covers what reviewers actually score, how to budget your time, the structure that reads well, and the presentation round that usually follows.

---

## Table of Contents
1. [What Reviewers Actually Score](#what-reviewers-actually-score)
2. [Time Budget](#time-budget)
3. [Repository Structure](#repository-structure)
4. [The README Is the Deliverable](#the-readme-is-the-deliverable)
5. [Modeling Approach](#modeling-approach)
6. [Common Take-Home Types](#common-take-home-types)
7. [The Follow-Up Presentation](#the-follow-up-presentation)
8. [Scope and Boundaries](#scope-and-boundaries)
9. [Self-Review Checklist](#self-review-checklist)
10. [Common Pitfalls](#common-pitfalls)
11. [Related Topics](#related-topics)

---

## What Reviewers Actually Score

Most rubrics weight roughly like this. Note where accuracy sits.

| Dimension | Weight | What earns the marks |
|---|---|---|
| **Problem framing** | High | Restated the business problem, chose a metric and justified it, defined success |
| **Data understanding** | High | Found the leakage, the imbalance, the duplicates, the temporal structure |
| **Methodology soundness** | High | Valid splits, a real baseline, honest evaluation, no leakage |
| **Communication** | High | README a stranger can follow; conclusions stated plainly |
| **Reproducibility** | Medium-high | Runs end to end from a clean checkout |
| **Code quality** | Medium | Readable, structured, tested where it matters |
| **Model performance** | **Low-medium** | Reasonable; the gap between 0.85 and 0.87 rarely decides anything |
| **Production thinking** | Medium | How it would deploy, monitor, fail, and be retrained |

The single most common mis-calibration is spending 80% of the time on the model and 20% on everything else. Reviewers can see modeling ability from a clean pipeline with a well-chosen baseline; they cannot recover framing, honesty, or communication from a high score.

**What gets a submission rejected outright**, regardless of accuracy: data leakage that inflates results, a random split on temporal data, code that does not run, no baseline, and a metric that does not match the stated problem.

---

## Time Budget

If the brief says "about 4 hours," treat that as a real constraint. Reviewers compare submissions against the stated budget, and a 30-hour submission signals poor prioritization as much as it signals effort — it also disadvantages candidates who respected the limit, which reviewers notice.

For a nominal 4–6 hours:

| Phase | Share | Output |
|---|---|---|
| Understand the problem and data | 20% | Framing, metric choice, data quirks found |
| Baseline end to end | 15% | A trivial model, scored, that the pipeline runs through |
| Feature work and modeling | 30% | One or two real iterations with a rationale |
| Evaluation and error analysis | 15% | Slices, confusion, where it fails and why |
| Write-up and cleanup | 20% | README, structure, reproducibility |

**Build the baseline before the good model.** A majority-class predictor, a simple heuristic, or a logistic regression gives you an end-to-end pipeline within the first hour and a number every later result is measured against. Candidates who skip it often discover at hour five that their pipeline has a bug, with nothing to compare against.

If you run out of time, say so explicitly in the README with what you would have done. That reads as prioritization. Silence reads as not knowing.

---

## Repository Structure

Structure signals engineering maturity before a reviewer reads a line of logic.

```
churn-prediction/
├── README.md                  ← the deliverable; write it first, refine last
├── requirements.txt           ← pinned versions
├── Makefile                   ← make setup / make train / make evaluate
├── data/
│   ├── raw/                   ← immutable, gitignored
│   └── processed/             ← generated, gitignored
├── notebooks/
│   └── 01_exploration.ipynb   ← EDA only, clearly labeled as exploratory
├── src/
│   ├── data.py                ← loading and splitting
│   ├── features.py            ← transformations (importable, testable)
│   ├── train.py               ← trains and saves a model
│   └── evaluate.py            ← metrics, slices, plots
├── tests/
│   └── test_features.py       ← a few tests on the transformations that matter
└── reports/
    └── figures/
```

Two things reviewers check immediately and that cost you nothing:

- **Does it run?** Clone into a fresh virtualenv, follow your own README literally, and run it. Absolute paths, an undeclared dependency, or a missing data step are the most common reasons a submission scores badly for reasons unrelated to skill.
- **Is logic in importable modules?** A single 900-cell notebook cannot be tested, reviewed, or reused. Keep exploration in a notebook and put the pipeline in `src/`.

---

## The README Is the Deliverable

Most reviewers spend 10–15 minutes on a submission, and much of it in the README. Write it for someone who will not run your code.

A structure that works:

```markdown
# <Problem> — Approach and Results

## Problem framing
What I understood the task to be, the metric I optimized, and why that metric
fits the stated business context.

## Data
Size, shape, target balance. What I found: leakage risks, duplicates, missingness
patterns, temporal structure. What I did about each.

## Approach
Baseline first, then what I tried and why. Alternatives I considered and rejected,
with reasons.

## Results
| Model | PR-AUC | Recall @ 80% precision | Notes |
|---|---|---|---|
| Majority baseline | 0.031 | 0.00 | Prevalence floor |
| Logistic regression | 0.284 | 0.19 | Interpretable reference |
| Gradient boosting | 0.412 | 0.34 | Final |

## Error analysis
Where it fails, on which segments, and my hypothesis for why.

## Assumptions and limitations
Stated plainly. This section builds more trust than the results table.

## What I'd do with more time
Ranked by expected value, not by what is interesting.

## How to run
Exact commands, verified from a clean checkout.
```

Two sections carry disproportionate weight. **Assumptions and limitations** is where reviewers look for intellectual honesty — a candidate who names the weaknesses of their own work is one who can be trusted with an ambiguous problem. **What I'd do next**, ranked by expected value, demonstrates prioritization, which is most of senior ML work.

---

## Modeling Approach

**Framing before fitting.** Write down the prediction target, the unit of prediction, when the prediction would be made in production, and what information is available at that moment. That last question is what prevents leakage, and stating it explicitly is worth marks by itself.

**Choose the metric from the cost structure.** Accuracy on an imbalanced problem is an immediate red flag. Ask what a false positive costs versus a false negative, then pick accordingly — PR-AUC and recall at a fixed precision for rare-event problems, calibration if scores drive decisions, MAE versus RMSE depending on outlier treatment. Say why in one sentence.

**Split before you look.** Fit every transformation — scalers, imputers, target encoders — inside a pipeline on the training fold only. If the data is temporal, split by time; if entities repeat (users, sessions, patients), split by group. A random split on temporal or grouped data is the most common disqualifying error in take-homes, and it produces suspiciously good numbers that reviewers recognize on sight.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Everything stateful lives inside the pipeline, so CV can never leak
pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])),
    ("model", GradientBoostingClassifier(random_state=42)),
])
```

**Error analysis beats one more model.** Twenty minutes examining where the model fails — which segments, which feature ranges, what the false positives have in common — produces better material than another hour of hyperparameter search. It is also the part that transfers directly into the presentation round.

**Seed everything and state the variance.** Report `mean ± std` across folds rather than a single number. A candidate who reports 0.412 ± 0.03 is more credible than one who reports 0.4118.

---

## Common Take-Home Types

| Type | Real test | Where candidates lose marks |
|---|---|---|
| **Tabular prediction** | Framing, leakage detection, metric choice | Random split on temporal data; accuracy on imbalanced target |
| **Messy data cleaning** | Judgment under ambiguity | Silently dropping rows; not documenting decisions |
| **NLP classification** | Sensible baseline before transformers | Fine-tuning BERT without a TF-IDF baseline to justify it |
| **Recommendation** | Evaluation design | Random split instead of temporal; ignoring cold start |
| **Time series forecasting** | Validation strategy | Any shuffling; no seasonal-naive baseline |
| **Build a small service** | Engineering practice | No tests, no error handling, no input validation |
| **RAG / LLM feature** | Evaluation of a non-deterministic system | No eval set; "looks good" as the evidence |
| **Open-ended analysis** | Prioritization and communication | Exhaustive EDA with no conclusion |

Two type-specific notes worth knowing. For **NLP**, a TF-IDF plus linear model baseline takes ten minutes and makes any subsequent transformer result meaningful; skipping it is a common and avoidable loss. For **RAG or LLM take-homes**, the whole test is usually whether you build an evaluation harness at all — 20 or 30 labeled question-answer pairs with retrieval recall and answer accuracy measured separately outweighs any amount of prompt polish.

---

## The Follow-Up Presentation

Most take-homes are followed by a 30–45 minute discussion. It is a real round with its own score, and it is where a mid-tier submission can be rescued or a strong one undermined.

**Structure a 10-minute walkthrough**: problem framing and metric choice (2 min), data findings (2 min), approach and baseline comparison (3 min), results and error analysis (2 min), limitations and next steps (1 min). Lead with framing, not with the model.

**Questions to expect:**
- "Why that metric?" — have the cost reasoning ready.
- "Why that model, and what else did you consider?"
- "What was your baseline, and how much did the model actually add over it?"
- "Where does it fail?" — never answer "I'm not sure"; you did the error analysis.
- "How would you deploy this?" — features at serving time, latency, monitoring, retraining.
- "What would break at 100x the data?"
- "What did you not have time to do?"
- "What would you do differently?"

**Defend and concede appropriately.** If a reviewer challenges a choice you thought through, explain the reasoning and the alternative you rejected. If they identify a genuine mistake, acknowledge it directly and say what you would change — reviewers frequently probe a known weakness specifically to see whether you will defend the indefensible. Conceding a real error scores better than defending it.

**Know your numbers.** Dataset size, class balance, baseline score, final score, and runtime. Fumbling your own results undermines everything else.

---

## Scope and Boundaries

A few boundary questions come up often enough to answer here.

**Clarifying questions are usually welcome.** One concise email asking about an ambiguous target definition or an unclear evaluation criterion signals engagement. Three rounds of questions signal an inability to act under ambiguity. If nobody responds, state your assumption in the README and proceed — that is the correct behavior, and reviewers score it well.

**Going over the stated time** is a real negative, not the neutral it feels like. If you spent longer, do not claim otherwise; say what you did within the budget and mark clearly anything added afterward.

**On using AI assistance**: follow whatever the brief says. If it is silent, use it as you would at work and be prepared to explain every line — the presentation round exposes code you cannot account for very quickly. Where a brief asks you to disclose usage, disclose it; being caught misrepresenting it ends the process regardless of the work's quality.

**On external data and pretrained models**: allowed unless the brief forbids it, but justify the choice and account for the added complexity. A pretrained model that adds two points over a simple baseline while tripling inference cost is a tradeoff worth discussing rather than an automatic win.

---

## Self-Review Checklist

Run through this before submitting.

**Reproducibility**
- [ ] Cloned to a fresh directory, new virtualenv, followed the README literally — it runs
- [ ] Dependencies pinned; no absolute paths; seeds set
- [ ] Large data and artifacts gitignored, with instructions for obtaining them

**Methodology**
- [ ] A baseline exists and is reported alongside the final model
- [ ] Split respects time and entity structure
- [ ] Every transformation is fit inside the pipeline on training data only
- [ ] Metric matches the stated business problem, with the reason given
- [ ] Results reported as mean ± std, not a single decimal-heavy number

**Communication**
- [ ] README answers: what problem, what approach, what result, what next
- [ ] Assumptions and limitations stated explicitly
- [ ] Next steps ranked by expected value
- [ ] A reader who never runs the code understands the conclusion

**Code**
- [ ] Pipeline logic in importable modules, not only in notebooks
- [ ] Notebooks are clean and labeled as exploratory
- [ ] A few tests on the transformations that matter
- [ ] No commented-out dead code, no leftover debug prints

---

## Common Pitfalls

| Pitfall | Why it costs you | Fix |
|---|---|---|
| No baseline | Results are uninterpretable; reviewers cannot judge the gain | Trivial baseline first, reported alongside |
| Random split on temporal data | Leaks the future; disqualifying | Time-based split, with a gap for label latency |
| Preprocessing fit before the split | Leaks test statistics into training | Fit everything inside a `Pipeline` |
| Accuracy on an imbalanced target | Signals a fundamental gap | PR-AUC, recall at fixed precision, and say why |
| All logic in one notebook | Cannot be tested, reviewed, or reused | Modules in `src/`, notebook for EDA only |
| README written last, in five minutes | It is the most-read artifact | Draft it first, refine at the end |
| Chasing accuracy over everything else | Weighted lowest in most rubrics | Spend the time on framing, evaluation, write-up |
| Massively exceeding the stated time | Signals poor prioritization | Respect the budget; list what you cut |
| No error analysis | Loses the strongest presentation material | Twenty minutes on where and why it fails |
| Unstated assumptions | Reads as not noticing the ambiguity | An explicit assumptions section |
| Submission never run from a clean checkout | The most common avoidable failure | Fresh clone, fresh venv, follow your own README |

---

## Related Topics

- [Behavioral and Project Deep-Dive Guide](./behavioral-interview-guide.md)
- [2026 Interview Roadmap](./2026-interview-roadmap.md)
- [Study Pattern](./study-pattern.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Ensemble Methods](../classical_ml/intro_ensemble_methods.md)
- [ML Coding Challenges](../coding_challenges/ml_coding_challenges.md)
- [ML/AI Project Structure](../project_setup/intro_project_structure.md)
- [GitHub Project Setup](../project_setup/intro_github_project_setup.md)
- [ML System Design Framework](../system_design/README.md)
