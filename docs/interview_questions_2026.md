# Common ML Interview Questions — 2026 Edition

> Focus areas in 2026: LLM applications, agent systems, RAG, production AI, evaluation, and classical ML at depth.

---

## Section 1: Agent Systems Design

**Q1: Design a research agent that can answer questions about a company's internal documents.**

**Strong Answer:**
```
Architecture:
1. Knowledge base indexing: chunk docs → embed → Pinecone (with department namespace)
2. Agent loop (LangGraph StateGraph):
   - Query rewriting: expand ambiguous queries
   - Hybrid search: BM25 + dense retrieval
   - Reranking: Cohere Rerank to select top-3
   - Generation: Claude Sonnet with citations
3. Memory: LangGraph checkpointing per session
4. Guardrails: filter docs by user access level before retrieval
5. Observability: LangSmith tracing for every run

Key tradeoffs:
- More retrieval diversity (hybrid) vs. higher latency
- More context (better answers) vs. more hallucination risk
- Streaming responses vs. waiting for complete answer
```

---

**Q2: What are the risks of using multi-agent systems in production, and how do you mitigate them?**

**Strong Answer:**
- **Non-determinism**: Same input, different output. Mitigate: set temperature=0 for routing decisions, add output validators.
- **Token cost explosion**: 5 agents × 3 tool calls = 15x tokens. Mitigate: set `max_iter` limits, use cheaper models for simple sub-tasks.
- **Error cascades**: Agent A fails → Agent B gets bad input → Agent C produces garbage. Mitigate: validate outputs at each handoff, fail fast with circuit breakers.
- **Coordination overhead**: More time coordinating than doing work. Mitigate: use single agent unless parallelism or specialization is genuinely needed.
- **Debugging difficulty**: Which agent caused the bad output? Mitigate: full trace with LangSmith, unique run IDs per agent.

---

**Q3: Explain the difference between ReAct and function calling in agent systems.**

**Strong Answer:**
> **ReAct** (Reasoning + Acting): The LLM explicitly outputs a "Thought:" step, then an "Action:" step, then observes the result. Human-readable reasoning chain, but verbose and slower. Works with any LLM that can follow instructions.

> **Function calling (tool use)**: The LLM outputs structured JSON specifying which tool to call and with what arguments. The runtime executes it and passes results back. Faster, more reliable, but requires model support. Claude, GPT-4o, and Gemini all support native tool calling. In production, always prefer tool calling over ReAct.

---

## Section 2: RAG Optimization

**Q4: Your RAG chatbot gives wrong answers 30% of the time. Walk me through your debugging process.**

**Strong Answer:**
```
Step 1: Measure faithfulness vs. accuracy separately
  - Is the answer faithful to retrieved context? (hallucination check)
  - Is the retrieved context actually correct? (retrieval check)

Step 2: Sample 50 failures — categorize:
  a) Context retrieved but answer wrong → Generation problem
  b) Context not retrieved → Retrieval problem
  c) Context retrieved but irrelevant → Reranking problem
  d) Context correct, answer correct, but formatting off → Parser problem

Step 3: Fix retrieval problems:
  - Use hybrid search (BM25 + dense)
  - Improve chunking (semantic vs fixed-size)
  - Better embedding model (Voyage-3 > OpenAI)
  - Query rewriting / HyDE (Hypothetical Document Embeddings)

Step 4: Fix generation problems:
  - Stronger grounding prompt: "Answer ONLY from context. Say 'I don't know' if not in context."
  - Reduce context to top-3 (lost-in-middle effect)
  - Upgrade model (Sonnet → Opus)
  - Add chain-of-thought ("First identify what the question is asking, then find it in the context")
```

---

**Q5: What is HyDE and when would you use it?**

**Strong Answer:**
> HyDE (Hypothetical Document Embeddings): Instead of embedding the user's query directly, use an LLM to generate a *hypothetical* answer, then embed that. The hypothetical answer's embedding is closer to real answer embeddings in the vector space than the raw question.

> Use when: queries are short/ambiguous and documents are long/detailed. Example: "GDP 2024" → HyDE generates a paragraph about GDP statistics → finds more relevant chunks.

> Drawback: 1 extra LLM call per query (2x cost, 2x latency). Worth it for difficult queries; skip for simple lookup.

---

**Q6: Explain chunking strategy selection for different document types.**

| Document Type | Best Strategy | Reasoning |
|--------------|---------------|-----------|
| Markdown/Wiki | Header-based chunking | Respect document structure |
| Legal contracts | Clause-based + semantic | Preserve clause integrity |
| Code files | Function/class level | Don't split mid-function |
| Research papers | Section-based | Abstract, methods, results are separate |
| Customer support Q&A | One Q&A per chunk | Natural atomic unit |
| Books/long-form | Recursive + semantic | Balance structure and coherence |

---

## Section 3: LLM Scaling Challenges

**Q7: How does the attention mechanism scale, and what are the implications for RAG design?**

**Strong Answer:**
> Attention is O(n²) in time and space with sequence length n. Doubling the context = 4x compute. At 128K tokens, this is expensive. Implications for RAG:
> 1. Don't stuff 50 chunks into context — retrieve 3-5 high-quality ones
> 2. Place most relevant chunks at the beginning and end (lost-in-middle effect)
> 3. Compress: summarize retrieved docs before inserting
> 4. Use context-efficient models (Claude 3.5 Haiku handles long context well for cost)

---

**Q8: What is prompt caching and how does it reduce costs?**

**Strong Answer:**
> Anthropic's prompt caching lets you mark parts of a prompt as cacheable. If the same prefix is reused across requests, the cached tokens are processed at 10% of normal cost.

> Best for: system prompts, long documents (e.g., a 50K token codebase), few-shot examples that don't change.

> Example: RAG with a large fixed knowledge base → cache the knowledge base portion, only pay full price for the user's question.

```python
# Anthropic cache_control example
client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": system_prompt_50k_tokens,
                "cache_control": {"type": "ephemeral"}  # Cache this
            },
            {"type": "text", "text": user_question}  # Not cached
        ]
    }]
)
```

---

**Q9: How do you handle LLM rate limits in production?**

**Strong Answer:**
```
1. Retry with exponential backoff + jitter:
   - First retry: 1s, Second: 2s, Third: 4s (up to 60s)
   - Add random jitter to avoid thundering herd
   - Only retry on 429 and 503, not 400s

2. Semaphore for concurrency control:
   - asyncio.Semaphore(N) limits concurrent requests
   - N = (rate_limit_rpm / expected_latency_s / 60)

3. Request queuing:
   - Use Redis queue + worker pool
   - Priority queue: paid users > free users

4. Model fallback:
   - Primary: Claude Opus → Fallback: Claude Sonnet → Fallback: cached response

5. Proactive rate limit management:
   - Track tokens per minute (TPM) not just requests
   - Back-pressure: reject new requests when 80% of limit reached
```

---

## Section 4: Production AI Systems

**Q10: How do you monitor an LLM application in production?**

**Strong Answer:**
```
Four categories of metrics:

1. System metrics:
   - P50/P95/P99 latency
   - Error rate (by error type: timeout, 429, 502)
   - Throughput (RPS)
   - Cost per request (tokens × price)

2. Quality metrics:
   - Automated: faithfulness score, hallucination rate
   - User feedback: thumbs up/down, follow-up question rate
   - Task completion rate (did user get their answer?)

3. Safety metrics:
   - Harmful content detection rate
   - Policy violation rate
   - Prompt injection attempt detection

4. Business metrics:
   - User retention correlated with AI quality
   - Support ticket deflection rate
   - Feature adoption rate

Tooling: LangSmith for LLM traces, Prometheus+Grafana for system metrics, custom dashboards for business metrics.
```

---

**Q11: What is model drift in production LLM systems, and how do you detect it?**

**Strong Answer:**
> Unlike classical ML models (where data drift causes accuracy decline), LLMs face:

> 1. **Upstream model drift**: The LLM provider updates their model (Claude 3.5 → 3.6) — output distribution changes even without you doing anything. Detect: run your eval suite after any model update.

> 2. **Prompt drift**: Works initially, breaks after adding features. Track prompt versions, run evals on all prompt changes.

> 3. **Data drift in RAG**: The knowledge base becomes stale — users ask about events after the last index update. Monitor: track "I don't know" rate; high rate = stale knowledge.

> 4. **Distribution shift in user queries**: Users start asking different types of questions. Monitor: query embedding drift using cosine similarity to training distribution.

---

**Q12: Explain the CAP theorem's relevance to ML feature stores.**

**Strong Answer:**
> Feature stores must choose between Consistency and Availability during partition:

> - **Strong consistency needed**: Fraud detection, dynamic pricing (must use latest features; stale features cause errors)
> - **Eventual consistency acceptable**: Recommendations, content ranking (slightly stale user history is fine)

> In practice: feature stores use two separate stores:
> - **Online store (Redis/DynamoDB)**: Low latency, eventually consistent, for inference
> - **Offline store (S3/BigQuery)**: Consistent, high throughput, for training

> The challenge: keep them synchronized (double-write patterns, Kafka-based sync).

---

## Section 5: Classical ML Depth

**Q13: You have a dataset with 1M samples but only 50 positives for fraud detection. How do you handle this?**

**Strong Answer:**
```
1. Don't use accuracy as metric → use Precision-Recall AUC, F1 at threshold, ROI

2. At the data level:
   - SMOTE / ADASYN (oversample minority)
   - Undersampling majority (random or Tomek links)
   - Class weights in loss function (preferred in practice)

3. At the model level:
   - Tree models (XGBoost) handle imbalance naturally with scale_pos_weight
   - Calibrate probabilities: Platt scaling or isotonic regression
   - Adjust decision threshold based on business cost matrix

4. Evaluation:
   - Never use random train/test split for fraud (temporal split)
   - Report at multiple thresholds (precision@k)
   - Business metric: false positive cost vs. false negative cost

Example:
scale_pos_weight = (1M - 50) / 50 = 19,998  # XGBoost
→ Treats each fraud as 20K non-frauds in the loss function
```

---

**Q14: What's the difference between online learning and continual learning?**

**Strong Answer:**
> **Online learning**: Model updates on each example one-at-a-time (or small batches) as they arrive. Used in recommendation systems, ad bidding. Example: SGD with `partial_fit()` in sklearn.

> **Continual learning**: Training on new tasks/data without forgetting old knowledge. Challenge: catastrophic forgetting. Techniques: EWC (Elastic Weight Consolidation), learning without forgetting, replay buffers.

> In production, most systems use "periodic retraining" (not true online learning): retrain on last N days of data every week. True online learning is used only where sub-second model updates are needed.

---

**Q15: Explain gradient boosting and why it often beats deep learning on tabular data.**

**Strong Answer:**
> Gradient boosting trains an ensemble of decision trees sequentially. Each tree corrects the errors of the previous ensemble. The "gradient" refers to fitting each tree to the negative gradient of the loss function (not the raw residuals).

> Why it beats deep learning on tabular data:
> 1. **Tabular data has irregular structure** — trees naturally handle mixed types, missing values, and non-monotonic relationships without preprocessing
> 2. **Needs less data** — deep learning requires 100K+ samples to outperform GBM; GBM works well with 1K-100K
> 3. **Training speed** — XGBoost/LightGBM train in seconds; deep learning takes hours
> 4. **Feature importance is interpretable** — SHAP values work well with GBM
> 5. **Less hyperparameter sensitivity** — GBM defaults are competitive; deep learning requires careful tuning

> Exception: When tabular data has image/text columns, or when sample size is 1M+, deep learning can compete.

---

## Section 6: LLM Fine-tuning vs RAG

**Q16: When would you fine-tune an LLM vs use RAG?**

| Scenario | Fine-tuning | RAG |
|----------|-------------|-----|
| **Teaching facts** | No (facts change, fine-tuning is expensive) | Yes |
| **Teaching style/format** | Yes (consistent output format) | Possible but less reliable |
| **Domain vocabulary** | Yes (medical/legal terminology) | Partially |
| **Private knowledge base** | No (retraining too slow for updates) | Yes |
| **Reducing prompt length** | Yes (embed instructions into weights) | No |
| **Citations/sources needed** | No (can't cite training data) | Yes (can cite retrieved docs) |
| **Cost** | High upfront, low per-query | No upfront, per-query retrieval cost |

---

**Q17: Explain RLHF and its limitations.**

**Strong Answer:**
> RLHF (Reinforcement Learning from Human Feedback):
> 1. Collect human preference data (which of 2 responses is better?)
> 2. Train a reward model on preferences
> 3. Fine-tune LLM with PPO to maximize reward model score
> 4. Result: LLM more aligned with human preferences

> Limitations:
> 1. **Reward hacking**: LLM learns to game the reward model (write verbose, sycophantic answers)
> 2. **Annotation cost**: Human preferences are expensive to collect
> 3. **Annotator disagreement**: Different humans have different preferences
> 4. **Distribution shift**: Reward model trained on distribution A fails on distribution B

> DPO (Direct Preference Optimization) is now preferred — same result, no RL needed, more stable training.

---

## Section 7: System Design Scenarios

**Q18: Design a real-time product recommendation system for an e-commerce site with 10M users.**

```
Requirements:
- Latency: <100ms P99
- Scale: 10M users, 1M products
- Freshness: session behavior should affect recommendations
- Coverage: handle new users (cold start)

Two-stage architecture:

Stage 1: Candidate Generation (<10ms)
  - Collaborative filtering (user-item matrix) → ANN search
  - Content-based (item embeddings by category/description) → ANN search
  - Popularity-based → pre-computed lists (cold start)
  - Session-based → last 5 clicks → similar items (ANN)
  → Merge and deduplicate → 200 candidates

Stage 2: Ranking (<50ms)
  - LightGBM/deep model with features:
    * User history features (from feature store)
    * Item features (price, rating, category)
    * Context features (time of day, device)
    * Cross features (user_category × item_category)
  → Score all 200 candidates → Top 10

Serving:
  - Feature store: Redis (online) + BigQuery (offline)
  - ANN search: FAISS (GPU) or Pinecone
  - Model serving: Ray Serve or TorchServe
  - Cache: Redis for repeated user-session pairs

Monitoring:
  - CTR (click-through rate)
  - Conversion rate
  - Diversity / serendipity
  - Coverage (long-tail exposure)
```

---

**Q19: How would you A/B test a new LLM model vs the existing one?**

**Strong Answer:**
```
1. Define success metrics upfront:
   - Primary: task completion rate, user satisfaction score
   - Guardrails: latency (must be < 5s), cost (must be < $0.02/query), error rate

2. Traffic allocation:
   - Start 5% new model / 95% control
   - Monitor guardrails for 24h
   - If safe, ramp to 50/50

3. Randomization:
   - Hash user_id → consistent assignment (same user always sees same model)
   - Exclude edge cases: API users, internal users, users in other experiments

4. Statistical analysis:
   - Wait for sufficient sample (power analysis: typically 1000+ conversions per variant)
   - Use two-tailed z-test for proportions (CTR, task completion)
   - Use Mann-Whitney for non-normal distributions (latency)
   - Set α = 0.05, power = 0.8

5. LLM-specific considerations:
   - Test on diverse query types (not just common ones)
   - Check for model-specific failure modes
   - Validate on your eval suite before A/B test

6. Decision:
   - Ship if: primary metric improved at statistical significance AND all guardrails met
   - Do not ship if: guardrail violated, even if primary metric improved
```

---

## Section 8: AI Workflow Automation

**Q20: When would you use n8n in an AI system, and when would you avoid it?**

**Strong Answer:**
> Use **n8n** when the system is driven by operational workflows: webhooks, SaaS integrations, approvals, notifications, CRM updates, or internal back-office automation. It is especially strong when product and operations teams need visibility into the workflow.

> Avoid using n8n as the only orchestration layer when the AI core needs complex state machines, agent loops, custom recovery logic, deep testing, or highly optimized retrieval and reasoning. In those cases, use n8n as the outer workflow shell and call a code-first backend such as a FastAPI or LangGraph service.

> A good interview answer also mentions idempotency, approval gates, retries, audit logs, and policy checks before executing any AI-suggested action.

---

## Quick Reference: Interview Pattern Cheatsheet

| Topic | 30-second answer | Full answer depth |
|-------|-----------------|-------------------|
| Overfitting | High train, low test accuracy; fix with regularization, more data, simpler model | Bias-variance tradeoff, cross-val, early stopping, dropout, L1/L2 |
| RAG vs fine-tuning | RAG for dynamic knowledge, fine-tuning for style/format | Cost, freshness, citation needs, task type |
| Attention complexity | O(n²) time and space | Implications for long context, FlashAttention, sparse attention |
| Hallucination | LLM generates false information confidently | Detection (NLI, LLM judge), mitigation (RAG, CoT, grounding prompts) |
| Class imbalance | Class weights, resampling, threshold adjustment | Metrics: PR-AUC, F1; SMOTE, focal loss |
| Agent vs LLM chain | Agents have loops, conditionals, tool use; chains are fixed | LangGraph vs LCEL vs direct API calls |
| n8n vs LangGraph vs Airflow | n8n for operational automation, LangGraph for agent control, Airflow for scheduled data pipelines | Tradeoffs in state, approvals, integrations, retries, and ownership |
| Embedding vs fine-tuning | Embedding: fast, cheap, good for similarity; FT: slow, expensive, changes model behavior | When each wins |
