# ML Debugging Scenarios: "Your Model Is Broken — What Do You Do?"

Most ML engineer onsites include at least one troubleshooting round. The interviewer describes a symptom — a suspicious metric, a production regression, a latency spike — and watches how you work through it. There is rarely one correct answer. What gets graded is whether your process finds the cause quickly and cheaply, and whether you protect users while you look.

This guide covers a repeatable answer structure and 25 scenarios drawn from real loops, grouped by where the failure shows up.

---

## Table of Contents
1. [How to Answer Scenario Questions](#how-to-answer-scenario-questions)
2. [Offline Training Problems](#offline-training-problems)
3. [Offline-Good, Online-Bad Gaps](#offline-good-online-bad-gaps)
4. [Production Drift and Monitoring](#production-drift-and-monitoring)
5. [Data and Labels](#data-and-labels)
6. [LLM, RAG, and Agent Systems](#llm-rag-and-agent-systems)
7. [Recommendation and Ranking](#recommendation-and-ranking)
8. [Infrastructure and Latency](#infrastructure-and-latency)
9. [Common Answer Mistakes](#common-answer-mistakes)
10. [Related Topics](#related-topics)

---

## How to Answer Scenario Questions

Use the same five steps every time. Say them out loud so the interviewer can follow your structure.

| Step | What you do | What it sounds like |
|---|---|---|
| **1. Clarify** | Pin down the symptom: which metric, how much, since when, all traffic or a segment, anything deployed or changed | "Did this start suddenly or gradually? Was there a deploy, a data change, or a traffic change around then?" |
| **2. Hypothesise** | List 3–5 causes, ranked by *likelihood × cost-to-check* | "Most likely is leakage, and it's cheap to check, so that's first." |
| **3. Check cheapest first** | Run the fast, high-information checks before the slow ones | "Before retraining anything, I'd look at feature importances and single-feature AUCs." |
| **4. Fix** | Mitigate first if users are affected (rollback, fallback, kill switch), then fix the root cause | "I'd roll back to the previous model now, then debug without time pressure." |
| **5. Prevent recurrence** | Add the test, monitor, or process change that catches this class of failure next time | "I'd add a schema check on that upstream table and alert on null rate." |

Three habits make the difference between a mid-level and a senior answer:

- **Rank, don't list.** Reciting twelve possible causes shows memory. Saying which one you'd check first, and why, shows judgment.
- **Mitigate before you diagnose.** In production, the first action is usually to stop the damage. Debugging a live regression while it keeps costing money is a red flag.
- **End with a system change.** "I fixed the bug" is a mid-level ending. "I added a check so nobody on the team can ship this bug again" is a senior one.

Aim to talk for 5–8 minutes per scenario, then let the interviewer steer. They usually have a specific root cause in mind and will give you hints if you ask good clarifying questions.

---

## Offline Training Problems

### 1. "Validation AUC is 0.99 on the first try"

**Likely causes**
1. Target leakage: a feature that is computed after the outcome, or is a proxy for the label (`refund_issued` in a fraud model, `discharge_code` in a readmission model).
2. Split leakage: random split on temporal data, or the same user/patient/session in both train and validation.
3. Duplicate rows across splits.
4. Preprocessing (scaler, target encoder, imputer) fit on the full dataset before splitting.
5. The problem is genuinely easy — possible, but last on the list until proven.

**How to investigate**
- Look at feature importances. One feature carrying most of the importance is the classic leakage signature.
- Compute single-feature AUC for every numeric feature. Anything above ~0.9 alone deserves an explanation.
- Check entity and row overlap between splits.

```python
from sklearn.metrics import roc_auc_score
import pandas as pd

for col in X_val.select_dtypes("number").columns:
    auc = roc_auc_score(y_val, X_val[col].fillna(X_val[col].median()))
    if max(auc, 1 - auc) > 0.9:
        print(f"suspicious: {col} auc={auc:.3f}")

dupes = set(pd.util.hash_pandas_object(X_train, index=False)) & \
        set(pd.util.hash_pandas_object(X_val, index=False))
print("rows in both splits:", len(dupes))
print("shared users:", len(set(train.user_id) & set(val.user_id)))
```

- For each top feature, ask: "At the moment we would make this prediction in production, would this value be known?"

**Fix:** Remove or rebuild leaky features with point-in-time (as-of) logic, switch to a time-based or group-based split, and move all fitted preprocessing inside a pipeline. Expect the metric to drop; report the lower number as the real one.

**Prevent it next time:** Write down the prediction timestamp and the feature availability rule before building features. Add a CI check that fails if train and validation share entity IDs, and a sanity check that flags any single feature with AUC above a threshold.

**What the interviewer is listening for:** That you treat a too-good result as a bug until proven otherwise, and that you know the specific mechanisms of leakage rather than just the word.

### 2. "Training loss goes NaN after 3 hours"

**Likely causes**
1. Learning rate too high once warmup ends, causing gradients to explode.
2. Mixed-precision overflow in fp16 (large activations, attention logits, or a loss scaler that isn't working).
3. A bad batch: `inf` or `NaN` in the input, an empty sequence, or a label outside the valid range.
4. Numerically unstable custom code: `log(0)`, division by a zero norm, `sqrt` of a negative number.

**How to investigate**
- The timing is a clue. If 3 hours coincides with the end of warmup, suspect the LR schedule. If it's random across runs, suspect a data batch.
- Plot gradient norm and loss scale over time. A rising gradient norm before the NaN points to instability; a sudden jump at one step points to one batch.
- Resume from the last good checkpoint and replay the same batches with anomaly detection on.

```python
torch.autograd.set_detect_anomaly(True)          # slow; debugging only
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
    torch.save({"step": step, "batch": batch}, f"bad_batch_{step}.pt")
    raise RuntimeError(f"non-finite loss/grad at step {step}")
```

**Fix:** Add gradient clipping, lower the peak LR or lengthen warmup, switch from fp16 to bf16 where the hardware supports it, and replace hand-written `log(softmax(x))` with `log_softmax` or the fused loss. If it's a bad batch, fix the data and add validation.

**Prevent it next time:** Log gradient norm, loss scale, and per-layer activation stats as standard. Validate inputs in the data loader (finite values, label range). Checkpoint often enough that a NaN costs minutes, not hours.

**What the interviewer is listening for:** Using *when* it failed as evidence, and knowing that clipping, LR, precision, and data are distinct causes with distinct fixes.

### 3. "The loss isn't going down at all"

**Likely causes**
1. A pipeline bug: labels misaligned with inputs after a shuffle or join, or all labels identical.
2. Wrong loss wiring: softmax applied before `CrossEntropyLoss` (which expects logits), or targets in the wrong format.
3. Optimizer problems: `zero_grad()` missing, parameters frozen, LR far too low or too high.
4. Inputs not normalised, or the model is in eval mode.

**How to investigate**
- Overfit a single small batch. A working model and pipeline should drive training loss close to zero on 8–32 examples within a few hundred steps. If it can't, the bug is in code, not data or capacity.
- Check the initial loss. For k-class cross-entropy with a reasonable init it should be close to `ln(k)`; far off means the output layer or loss is wired wrong.
- Print a few `(input, label)` pairs after the full data pipeline and check them by eye.
- Confirm `sum(p.numel() for p in model.parameters() if p.requires_grad)` is what you expect.

**Fix:** Correct the specific bug. Most commonly it's label alignment or a double softmax.

**Prevent it next time:** Keep the single-batch overfit test as a unit test in the training repo, and add assertions on label distribution after every join.

**What the interviewer is listening for:** The overfit-one-batch test. It's the single most efficient debugging step for a network that won't learn, and interviewers expect you to reach for it.

### 4. "Training accuracy is 98%, validation is 70%"

**Likely causes**
1. Overfitting: model capacity too high for the data size, no regularisation.
2. Train and validation come from different distributions (different time period, different source, different label process).
3. Leakage *within* train (duplicates) inflating training accuracy.
4. Train-time augmentation or preprocessing differs from validation preprocessing.

**How to investigate**
- Plot learning curves: if validation loss falls then rises while training loss keeps falling, it's overfitting. If validation is flat and bad from the start, suspect distribution mismatch.
- Run adversarial validation: train a classifier to distinguish train rows from validation rows. AUC near 0.5 means same distribution; high AUC means they differ, and its top features tell you how.
- Plot validation performance against training set size. If it's still climbing, more data will help.

**Fix:** For overfitting: stronger regularisation, early stopping, fewer or simpler features, more data or augmentation. For distribution mismatch: fix the split so validation reflects production, or reweight training data.

**Prevent it next time:** Build the validation set to mirror production first, then tune. Always log train and validation metrics side by side.

**What the interviewer is listening for:** Separating "overfitting" from "validation is a different distribution," because the fixes are opposite.

---

## Offline-Good, Online-Bad Gaps

### 5. "Offline AUC improved 3 points, but the A/B test shows no lift"

**Likely causes**
1. The offline gain is in a region of scores that doesn't change any decision (e.g. better ranking among items that are never shown, or below the action threshold).
2. Training-serving skew: the model in production isn't seeing the features it was evaluated on.
3. The experiment is underpowered, or broken (sample ratio mismatch, assignment leak).
4. The offline metric isn't aligned with the business metric.
5. Added latency cancelled the gain.

**How to investigate**
- Check the experiment first; it's cheapest. Run a sample ratio mismatch test: if you expected a 50/50 split and got 50.8/49.2 on a million users, the assignment is broken.

```python
from scipy.stats import chisquare
stat, p = chisquare([n_control, n_treatment])  # expected equal by default
print("SRM!" if p < 0.001 else "split looks fine", p)
```

- Compute the minimum detectable effect for the sample size you had. "No lift" may mean "couldn't detect the lift we should have expected."
- Log the treatment model's live scores and recompute offline AUC on the live traffic with delayed labels. If it's much lower than the offline number, it's skew.
- Evaluate the metric at the operating point: precision at the threshold, or NDCG on the positions actually displayed.

**Fix:** Depends on the cause: fix skew, re-run with adequate power, or redefine the offline metric to match the decision being made.

**Prevent it next time:** Choose offline metrics that are measured at the operating point, track their historical correlation with online results, and run shadow scoring before any A/B test to catch skew.

**What the interviewer is listening for:** Checking the experiment's validity before blaming the model, and the idea that an offline gain only matters if it changes a decision.

### 6. "The model does much worse in production than on the holdout set from day one"

**Likely causes**
1. Training-serving skew: the offline feature pipeline (SQL/Spark) and the online path (service code) compute a feature differently.
2. Point-in-time errors in training data: offline features used values from after the prediction time, so the offline number was inflated.
3. Stale or missing online features: feature store not refreshed, defaults (0 vs NaN) differ between paths.
4. Preprocessing version mismatch: a different tokenizer, vocabulary, or encoder at serving time.

**How to investigate**
- Log the exact feature vector used for each online prediction. Recompute the same features offline for the same entity and timestamp, join, and compare feature by feature.

```python
merged = online_log.merge(offline_features, on=["entity_id", "ts"], suffixes=("_on", "_off"))
for f in feature_names:
    mismatch = (merged[f"{f}_on"] - merged[f"{f}_off"]).abs() > 1e-6
    print(f, f"{mismatch.mean():.1%} mismatch")
```

- Check null and default rates per feature online versus in training.
- Score the same logged requests with the offline model artifact and compare predictions.

**Fix:** Make one feature definition serve both paths (a feature store with shared transformations, or the same library called from both), and fix point-in-time joins in the training set.

**Prevent it next time:** Keep logging served features, and run an automated skew check that compares a daily sample against offline recomputation and alerts on mismatch rate.

**What the interviewer is listening for:** "Log served features and diff them against offline" as the concrete technique, not just naming "training-serving skew."

### 7. "The model was great in the A/B test, but revenue dropped a month later"

**Likely causes**
1. Proxy metric misalignment: the test optimised clicks or conversions, while revenue per user, returns, or retention moved the other way.
2. Novelty effect: users engaged with something new, and the effect decayed.
3. Long-term behavioural effects: aggressive discounts or clickbait-style recommendations that trained users to wait or churn.
4. Feedback loop: the model now trains on data generated by itself, narrowing what it learns.
5. An unrelated cause: seasonality, a pricing change, a competitor. Correlation with launch isn't proof.

**How to investigate**
- Check whether a long-term holdout exists (a small slice still on the old model). Comparing it with the treated population is the cleanest answer.
- Plot the daily treatment effect from the original test. A lift that shrinks over its duration suggests novelty.
- Break revenue down: traffic × conversion × order value × return rate. Find which term moved.
- Check what else changed in the same window before attributing it to the model.

**Fix:** If the model is the cause, roll back or re-tune its objective to include the long-term metric (e.g. margin or 30-day retention) as a target or guardrail.

**Prevent it next time:** Keep a persistent 1–5% holdout for major models, run tests long enough to see novelty decay, and include revenue and retention as guardrail metrics, not just the primary proxy.

**What the interviewer is listening for:** Knowing that A/B tests measure short-term effects on a proxy, and that long-term holdouts exist for exactly this problem.

### 8. "Shadow-mode predictions don't match the offline scores for the same inputs"

**Likely causes**
1. Different model artifact or version loaded in the service.
2. Different preprocessing: feature order, categorical encoding, missing-value handling, float precision.
3. Nondeterministic inference (dropout left on, batch-dependent layers like batch norm in train mode).
4. The "same inputs" aren't actually the same: features fetched at a different time.

**How to investigate**
- Pick five requests. Save the raw input, the processed feature vector, and the output at each stage in both paths. Find the first stage where they diverge.
- Check the model hash loaded by the service against the registry.
- Score the same request twice in the service. Different outputs mean nondeterminism (`model.eval()` missing).

**Fix:** Correct the divergent stage. Feature ordering by position rather than name is a common culprit — pass features by name.

**Prevent it next time:** Add a golden-request test to CI: a set of fixed inputs with expected outputs that both paths must reproduce within tolerance before deploy.

**What the interviewer is listening for:** Bisecting the pipeline stage by stage rather than guessing.

---

## Production Drift and Monitoring

### 9. "Accuracy has slowly degraded over the last three months"

**Likely causes**
1. Concept drift: the relationship between features and label has changed (new fraud patterns, new user behaviour).
2. Covariate drift: the input distribution moved into regions the model saw rarely in training.
3. A slow upstream data problem (a feature slowly becoming null as a source is deprecated).
4. Label drift or a change in label definition.

**How to investigate**
- Compute drift per feature (PSI or KS) between the training window and each recent month. Rule of thumb: PSI under 0.1 is stable, 0.1–0.25 is moderate, above 0.25 is significant.

```python
import numpy as np
def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

- Retrain the same architecture on the most recent data and evaluate on the latest month. If it recovers most of the gap, it's drift and retraining works. If not, something structural changed.
- Break performance down by segment to see whether the decline is broad or concentrated.

**Fix:** Retrain on recent data; consider a sliding or time-weighted training window. If a feature has decayed, remove or replace it.

**Prevent it next time:** Set a retraining cadence based on how fast performance decays, monitor input drift and prediction distribution (which don't need labels), and alert on performance once delayed labels arrive.

**What the interviewer is listening for:** Distinguishing covariate from concept drift, and knowing which monitors work before labels arrive.

### 10. "The prediction distribution shifted overnight and nobody deployed anything"

**Likely causes**
1. Upstream data change: a schema change, unit change (cents to dollars), renamed category, or a join that started returning nulls.
2. A failed or partial pipeline run: the feature store serving yesterday's or empty values.
3. A traffic mix change: a marketing campaign, a new market, a bot surge, a new app version logging differently.
4. A dependency update in the serving image (a library that changed default behaviour).

**How to investigate**
- "No deploy" usually means "no deploy *by us*." Check upstream table change logs, pipeline run status, and feature freshness timestamps first.
- Compare null rate, mean, and cardinality of every input feature before and after the shift. A sudden change usually points to one or two features.
- Split the prediction shift by traffic source, platform, and app version. If it's all in one segment, it's traffic mix or a client change.

**Fix:** If a feature is broken, fall back (use the last good value, a default, or the previous model version that doesn't depend on it) while the upstream owner fixes it.

**Prevent it next time:** Schema and range checks on inputs (e.g. Great Expectations or pipeline assertions), freshness alerts on the feature store, and a data contract with upstream teams.

**What the interviewer is listening for:** Monitoring *inputs*, not just outputs, and checking the upstream dependencies before the model.

### 11. "The model now predicts the same value for almost every input"

**Likely causes**
1. One or more critical features are all missing and being imputed with the same default.
2. A preprocessing artifact (scaler, encoder) failed to load and a no-op or fresh one was used.
3. A newly retrained model collapsed (bad training data, e.g. one class only).
4. Output clipping or a threshold config changed.

**How to investigate**
- Look at the input feature vectors for a sample of requests. If they're nearly identical, it's the inputs.
- Check the model and preprocessing artifact versions actually loaded, and the service startup logs for fallback warnings.
- If a retrain just happened, check the label distribution of its training set.

**Fix:** Roll back to the last good model and artifacts immediately; then fix the failing load or data.

**Prevent it next time:** Fail loudly on artifact load errors instead of silently falling back. Add a post-deploy check that alerts if prediction variance or entropy drops below a floor. Gate retrained models on a validation set before promotion.

**What the interviewer is listening for:** Rolling back first, and treating silent fallbacks as the real bug.

### 12. "Fraud model precision was fine, then collapsed within a week"

**Likely causes**
1. Adversarial adaptation: fraudsters probed the model and found a pattern it doesn't catch, or started mimicking legitimate behaviour on the top features.
2. A legitimate population change triggering false positives (a holiday sale, a new payment method, a new region).
3. Label delay: chargebacks arrive weeks later, so the "collapse" may be partly measurement artifacts.

**How to investigate**
- Split false positives and false negatives by recency, merchant, device, and payment method. Adversarial attacks are usually concentrated in a narrow pattern.
- Compare feature distributions of recent confirmed fraud with fraud in the training data.
- Check whether volume in any segment spiked recently.

**Fix:** Short term: targeted rules for the new pattern, adjusted thresholds for the affected segment, and more manual review. Medium term: retrain with the new fraud examples and features that are harder to fake (device, network, velocity features).

**Prevent it next time:** Faster retraining loops, analyst feedback feeding labels quickly, rules and model working together, and monitoring on segment-level alert rates.

**What the interviewer is listening for:** Recognising that fraud is adversarial, so drift is intentional and fast, and that rules are a legitimate short-term tool.

---

## Data and Labels

### 13. "We retrained with twice as much data and the model got worse"

**Likely causes**
1. The evaluation set also changed, so the comparison isn't like-for-like.
2. The new data comes from a different distribution (a new source, region, or time period) or has a different label definition.
3. The new data is noisier: a cheaper labelling vendor, weak labels, or duplicates.
4. Class balance shifted, changing the effective threshold.

**How to investigate**
- Evaluate both the old and new model on the *same* fixed evaluation set. Many "regressions" disappear here.
- Train on old data only, new data only, and both. Evaluate each on the fixed set.
- Compare label rates and feature distributions between old and new data, and sample 50 new labels to check by hand.

**Fix:** Filter or reweight the bad slice, fix the label mapping, or train separate handling for the new population if it genuinely differs.

**Prevent it next time:** Keep a frozen, versioned evaluation set that is only changed deliberately. Version training data (DVC, Delta, Iceberg snapshots) so you can reproduce and diff any past training run.

**What the interviewer is listening for:** Checking the evaluation setup before questioning the data, and the ablation of old vs new data.

### 14. "The model plateaus at 80% no matter what we try"

**Likely causes**
1. Label noise sets a ceiling: if annotators only agree 80% of the time, the model can't reliably beat that.
2. The features don't contain enough information to predict the label.
3. The task is ambiguous as defined (unclear guidelines, overlapping classes).

**How to investigate**
- Have multiple annotators label the same few hundred examples and measure agreement.

```python
from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(annotator_a, annotator_b))  # < 0.6 means the labels themselves are unreliable
```

- Look at the model's most confident errors. If many are actually mislabelled, noise is the problem. Tools like cleanlab automate finding likely label errors.
- Check the confusion matrix: if errors concentrate between two classes, the class boundary is probably unclear in the guidelines.

**Fix:** Clarify labelling guidelines, relabel the confusing slice, merge classes that can't be distinguished, or collect the missing signal as a new feature.

**Prevent it next time:** Measure inter-annotator agreement at the start of any labelling project, and use it as the realistic ceiling when setting targets.

**What the interviewer is listening for:** Knowing that model quality is capped by label quality, and how to measure that cap.

### 15. "The model works well overall but is terrible for one customer segment"

**Likely causes**
1. The segment is underrepresented in training data.
2. Key features are missing or have different meaning for that segment (new users have no history; a region uses a different currency).
3. The label means something different for that segment.
4. Aggregate metrics hid the problem all along.

**How to investigate**
- Break down metrics by segment with sample sizes and confidence intervals. Small segments have noisy metrics.
- Compare feature null rates and distributions for the segment versus everyone else.
- Look at 20–30 errors from the segment by hand.

**Fix:** Collect or upweight data for the segment, add features that work for it (content features for cold users), or use a dedicated model or fallback for it. If fairness is involved, involve the relevant stakeholders before choosing a trade-off.

**Prevent it next time:** Report sliced metrics for key segments on every evaluation, and make segment regressions a release-blocking check.

**What the interviewer is listening for:** Slicing metrics as a default habit, and noticing when a segment matters for fairness or business risk.

---

## LLM, RAG, and Agent Systems

### 16. "RAG answers got worse after we re-indexed the documents"

**Likely causes**
1. Embedding model mismatch: documents were re-embedded with a new model or version, but queries still use the old one (or vice versa). Vectors from different models aren't comparable.
2. Chunking changed: different chunk size, overlap, or splitter, breaking answers across chunks.
3. Parsing regressed: a new PDF or HTML parser dropped tables, headers, or whole pages.
4. Index configuration changed: lower `ef_search` (HNSW) or `nprobe` (IVF), a different distance metric, or missing normalisation.
5. Metadata or filters were lost, so filtered queries return the wrong documents or nothing.

**How to investigate**
- Separate retrieval from generation. Run a golden set of queries with known relevant documents and compare recall@k before and after re-indexing.

```python
def recall_at_k(golden, retrieve, k=5):
    hits = [len(set(retrieve(q, k)) & set(rel)) > 0 for q, rel in golden]
    return sum(hits) / len(hits)
```

- Confirm the query encoder and document encoder names and versions match in config and in code.
- Compare chunk counts, average chunk length, and a few sample chunks from the old and new index.
- Search for an exact sentence from a known document. If it doesn't come back top-1, the index or embeddings are broken.

**Fix:** Re-embed with a consistent model, restore the chunking or parsing that worked, or restore index parameters. Keep the old index live until the new one passes evaluation.

**Prevent it next time:** Treat re-indexing as a deploy: build the new index side by side, gate the cutover on retrieval recall and answer-quality evals, and store the embedding model version as index metadata.

**What the interviewer is listening for:** Evaluating retrieval and generation separately, and knowing that mixed embedding models silently break search.

### 17. "The LLM answers confidently with facts that aren't in the retrieved documents"

**Likely causes**
1. Retrieval is returning irrelevant context, so the model falls back on its parametric knowledge.
2. The prompt doesn't require grounding or allow "I don't know."
3. The relevant fact is present but buried in a long context.
4. The question is outside the corpus entirely.

**How to investigate**
- For a sample of bad answers, check whether the supporting fact was in the retrieved context. This splits the problem into a retrieval failure or a generation failure.
- Measure faithfulness on an eval set: an LLM judge or NLI model checks whether each claim is supported by the context. Spot-check the judge against human labels.
- Test the same questions with the relevant passage placed first versus in the middle.

**Fix:** Improve retrieval (hybrid BM25 + dense search, a reranker, fewer but better chunks), instruct the model to answer only from the context and cite passage IDs, and allow an explicit abstain response. Validate that cited IDs exist in the context.

**Prevent it next time:** A regression eval suite with faithfulness and citation accuracy metrics, run on every prompt, model, or index change.

**What the interviewer is listening for:** Splitting "hallucination" into retrieval failure versus generation failure, and measuring faithfulness rather than eyeballing.

### 18. "Our agent keeps calling the same tool in a loop and costs spiked"

**Likely causes**
1. Tool errors are returned in a form the model can't act on (a stack trace, an empty string), so it retries the same call.
2. No step limit or budget, so a stuck agent runs until it hits the context limit.
3. Ambiguous tool descriptions or overlapping tools.
4. Context management dropped earlier tool results, so the agent doesn't know it already made the call.
5. No clear completion criterion in the instructions.

**How to investigate**
- Pull traces for the most expensive sessions. Look at the step where the loop starts and what the tool returned just before.
- Group sessions by number of steps and tool-call sequence; loops show up as repeated identical calls with identical arguments.
- Check whether the loop started after a tool, prompt, or model version change.

**Fix:** Return structured, actionable errors ("invalid date format, expected YYYY-MM-DD"), cap steps and tokens per task, detect repeated identical calls and stop or escalate, and tighten tool descriptions.

**Prevent it next time:** Per-session cost and step limits with alerts, tracing on every agent run, and an eval set of multi-step tasks measuring success rate, steps, and cost.

**What the interviewer is listening for:** Reading traces, hard limits as a safety net, and treating tool error messages as part of the prompt.

### 19. "After upgrading the LLM version, the downstream JSON parser started failing"

**Likely causes**
1. The new model formats output differently: markdown code fences, extra commentary, different key names or casing.
2. The prompt relied on quirks of the old model.
3. Longer outputs now hit `max_tokens` and are truncated mid-JSON.

**How to investigate**
- Collect failing raw outputs and categorise them: wrapped in fences, extra text, missing fields, truncated.
- Check the stop reason on failed responses; a length stop means truncation.
- Run the prompt regression suite against both model versions and diff.

**Fix:** Use the provider's structured output or JSON schema mode, or constrained decoding, instead of parsing free text. Validate with a schema (Pydantic) and retry once with the validation error in the prompt. Raise `max_tokens` if truncating.

**Prevent it next time:** Pin model versions explicitly, and gate every upgrade on a regression eval that includes format validity, not just answer quality.

**What the interviewer is listening for:** Pinning versions, schema-enforced outputs, and treating a model upgrade as a deploy that needs its own evals.

---

## Recommendation and Ranking

### 20. "Offline NDCG improved, but online CTR fell"

**Likely causes**
1. Exposure bias: offline evaluation uses logs from the old policy, so it only rewards re-ranking items the old model already showed. The new model's novel picks have no labels and count as negatives.
2. Position bias: clicks in logs are partly caused by position, so the model learned "what was ranked high" rather than "what's relevant."
3. The offline metric covers positions or candidates that differ from what's displayed (NDCG@50 when users see 5 items).
4. Reduced diversity: the new model shows near-duplicate items, which scores fine per item but performs worse as a page.
5. Skew or latency issues, as with any model.

**How to investigate**
- Compare offline NDCG at the actual display depth, not the default.
- Measure what share of the new model's top-k items had any exposure in the logs. A low share means the offline metric can't judge it.
- Check diversity on the served pages: distinct categories or sellers in the top-k, pairwise similarity.
- Use interleaving to compare the two rankers online with far less traffic than an A/B test.

**Fix:** Train with position debiasing (position as a feature set to a constant at serving, or inverse propensity weighting), evaluate with counterfactual estimators, and add a diversity re-rank (e.g. MMR) if pages became homogeneous.

**Prevent it next time:** Log propensities or randomise a small slice of traffic to get unbiased evaluation data. Track the historical correlation between offline and online metrics and don't trust an offline metric that doesn't predict online results.

**What the interviewer is listening for:** Understanding that logged data is biased by the policy that generated it, and naming position bias and counterfactual evaluation.

### 21. "Recommendations became homogeneous — everyone sees the same popular items"

**Likely causes**
1. Feedback loop: the model recommends popular items, they get more clicks, the next model learns they're even more popular.
2. Training objective rewards clicks only, and popular items have the highest base click rate.
3. Candidate generation narrowed (a smaller retrieval set, or a popularity-based candidate source dominating).
4. Personalisation features broken, so everyone is scored as the average user.

**How to investigate**
- Measure catalogue coverage (share of items recommended at least once) and the Gini coefficient of impressions over time.
- Check whether personalised features are populated: if user embeddings are all zero or default, everyone gets the same list.
- Measure the overlap of top-10 lists between random pairs of users.

**Fix:** Fix broken personalisation if that's the cause. Otherwise add exploration (a small share of slots for less-exposed items, or a bandit), apply a popularity penalty or diversity re-rank, and include longer-term engagement in the objective.

**Prevent it next time:** Monitor coverage, diversity, and novelty alongside CTR, and keep an exploration budget as a permanent part of the system.

**What the interviewer is listening for:** Recognising the feedback loop, and that recommendation health metrics go beyond CTR.

### 22. "New items never get recommended"

**Likely causes**
1. Cold start: the model depends on item ID embeddings or interaction counts that new items don't have.
2. Candidate retrieval excludes items below an interaction threshold.
3. Training data freshness: new items only enter the model after the next retrain.

**How to investigate**
- Plot impressions per item against item age. Check how long it takes a new item to reach typical exposure.
- Trace a new item through the pipeline: is it in the candidate index, does it get a score, where does it rank?
- Check what embedding a never-seen item ID maps to (often a shared "unknown" vector).

**Fix:** Add content-based features (text, image, category embeddings) so new items can be scored without interactions, reserve exploration slots for new items, and refresh the candidate index more frequently than the ranker is retrained.

**Prevent it next time:** Monitor time-to-first-impression and new-item exposure share as product metrics.

**What the interviewer is listening for:** Tracing the item through retrieval and ranking, and content features plus exploration as the standard cold-start answer.

---

## Infrastructure and Latency

### 23. "p99 latency doubled after a model update"

**Likely causes**
1. The new model is larger or deeper, and requests queue behind it under load.
2. Dynamic input shapes trigger recompilation or re-optimisation (TensorRT, XLA, `torch.compile`) for unseen shapes.
3. Dynamic batching: padding to the longest sequence in a batch, or a longer batch wait timeout.
4. An op fell back to CPU, or a new feature requires an extra network call.
5. Resource contention: thread oversubscription, memory pressure, garbage collection pauses.

**How to investigate**
- Check whether p50 moved too. If only p99 moved, suspect tail causes: queueing, recompilation, GC, padding, a slow dependency. If p50 moved as well, the model itself is slower.
- Break latency down by stage using traces: feature fetch, preprocessing, queue wait, inference, postprocessing.
- Correlate slow requests with input length or shape.
- Profile the model offline with the same batch size and input distribution.

**Fix:** Roll back if the SLO is breached. Then: bucket input shapes and warm them up at startup, tune batching timeout and max batch size, fix CPU fallbacks, cache or parallelise feature calls, or reduce model cost (quantisation, distillation, pruning).

**Prevent it next time:** A load test with production-like traffic in CI that fails on p99 regressions, and a canary rollout with automatic rollback on latency SLOs.

**What the interviewer is listening for:** Comparing p50 against p99 to split the causes, and a per-stage latency breakdown instead of assuming it's the model.

### 24. "GPU utilisation is 30% during training"

**Likely causes**
1. The data loader is the bottleneck: too few workers, heavy CPU augmentation, slow storage or decoding.
2. Frequent host-device synchronisation (`.item()`, `.cpu()`, printing tensors every step).
3. Batch size too small to fill the GPU.
4. In distributed training: communication overhead or stragglers.

**How to investigate**
- Watch utilisation over time with `nvidia-smi dmon -s u`. Saw-tooth patterns suggest the GPU is waiting for data.
- Time a loop that only iterates the data loader, with no model. If it's slower than a training step, data loading is the bottleneck.
- Run `torch.profiler` for a few steps and look for large gaps between kernels and sync points.

**Fix:** Increase `num_workers`, set `pin_memory=True` and `persistent_workers=True`, preprocess offline into an efficient format, move augmentation to GPU, remove per-step syncs, raise batch size, and use mixed precision.

**Prevent it next time:** Log step time split into data time and compute time on every run, so input bottlenecks are visible immediately.

**What the interviewer is listening for:** Isolating the data loader from the model, and knowing the usual PyTorch fixes.

### 25. "The nightly batch scoring job went from 1 hour to 6 hours"

**Likely causes**
1. Data skew: one key (a large customer, a null key) now holds a huge share of rows, so one Spark task runs far longer than the rest.
2. Input volume grew, or a join started multiplying rows.
3. The model is loaded per row or per small partition instead of once per executor.
4. A row-at-a-time Python UDF replaced a vectorised one.
5. Cluster changes: fewer executors, spot instance preemption, a different instance type.

**How to investigate**
- Compare input row counts and output row counts to earlier runs. A row explosion points to a join.
- Open the Spark UI and compare task durations within the slow stage. One task at 5 hours while others take 2 minutes is skew.
- Check recent code changes to the scoring job and the cluster config history.

**Fix:** Salt or split skewed keys, fix the join, broadcast the model once per executor, use a pandas UDF (Arrow-vectorised) for inference, and repartition to match the cluster.

**Prevent it next time:** Alert on job runtime and on input and output row counts against recent history, and assert join cardinality in the pipeline.

**What the interviewer is listening for:** Checking the data volume and shape first, and reading the task-duration distribution to find skew.

---

## Common Answer Mistakes

| Mistake | Why it hurts | Better approach |
|---|---|---|
| Jumping straight to "retrain the model" | Retraining hides the cause and often reproduces it | Diagnose first; retrain only when drift is confirmed |
| Listing every possible cause with no order | Shows recall, not judgment | Rank by likelihood × cost-to-check, start with the top one |
| No clarifying questions | You'll solve the wrong problem | Ask what changed, when, how much, and which segments |
| Debugging a live regression without mitigating | Users keep paying for your investigation | Roll back, fall back, or kill-switch first |
| Blaming the model before checking the data | Most production failures are data and pipeline failures | Check inputs, freshness, and upstream changes early |
| Trusting a surprisingly good metric | Leakage is far more common than breakthroughs | Treat too-good results as a bug to explain |
| Changing the eval set and the model at once | Makes the comparison meaningless | Compare on a frozen, versioned evaluation set |
| Treating an LLM system as one black box | Can't tell retrieval failures from generation failures | Evaluate each stage separately with its own metric |
| Vague answers ("check the data", "look at the logs") | Gives the interviewer nothing to assess | Name the specific check, query, or metric |
| Ending at the fix | Misses the senior-level signal | Add the test, monitor, or process change that prevents recurrence |
| Assuming "no deploy" means "nothing changed" | Upstream data, traffic, and dependencies change constantly | Check upstream changelogs and traffic mix |
| Refusing to commit to a hypothesis | Seems indecisive | State your best guess, then say how you'd confirm it |

---

## Related Topics

- [Behavioral and Project Deep-Dive Guide](./behavioral-interview-guide.md)
- [ML Take-Home Projects](./take-home-projects.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [Data Quality](../mlops/intro_data_quality.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
- [Feature Stores](../mlops/intro_feature_store.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Data Labeling and Active Learning](../mlops/intro_data_labeling_active_learning.md)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Neural Network Training](../deep_learning/intro_neural_network_training.md)
- [Distributed Training](../deep_learning/intro_distributed_training.md)
- [RAG Engineering](../ai_genai/intro_rag_engineering.md)
- [Agent Tool Use](../ai_genai/intro_agent_tool_use.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Recommendation System Design](../system_design/recommendation_system.md)
- [Observability](../devops/intro_observability.md)
