# ML System Design Patterns — 2026 Production Guide

## The ML System Design Framework

For every system design interview, structure your answer using this 6-step framework:

```
1. CLARIFY REQUIREMENTS
   ├── Functional: What must the system do?
   └── Non-functional: Scale, latency, accuracy, cost, reliability

2. DEFINE THE ML PROBLEM
   ├── Problem type (classification, regression, ranking, generation)
   ├── Input/output specification
   └── Success metrics (business + technical)

3. DATA DESIGN
   ├── Data sources and collection
   ├── Feature engineering
   └── Training/validation/test splits

4. MODEL DESIGN
   ├── Model architecture choices
   ├── Training approach
   └── Evaluation strategy

5. SERVING ARCHITECTURE
   ├── Online vs batch vs hybrid
   ├── Latency/throughput tradeoffs
   └── Caching, preprocessing, postprocessing

6. PRODUCTION & OPERATIONS
   ├── Monitoring and alerting
   ├── Model refresh strategy
   └── A/B testing and gradual rollout
```

---

## Pattern 1: RAG Pipeline

### When to Use

- Q&A over private/proprietary knowledge bases
- Document search and synthesis
- Customer support automation
- Code assistance with private codebase context

### Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              RAG SYSTEM                      │
                    │                                              │
                    │   INDEXING (Offline)                        │
                    │   Documents → Chunker → Embedder → VectorDB │
                    │                                              │
                    │   SERVING (Online)                          │
                    │   Query                                      │
                    │     ↓                                        │
                    │   [Query Rewriter]  ← (optional)            │
                    │     ↓                                        │
                    │   [Hybrid Search]  (BM25 + Dense)           │
                    │     ↓                                        │
                    │   [Reranker]  (cross-encoder)               │
                    │     ↓                                        │
                    │   [Context Assembly]                         │
                    │     ↓                                        │
                    │   [LLM Generation]                          │
                    │     ↓                                        │
                    │   [Answer + Citations]                       │
                    └─────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Chunking** | Fixed, recursive, semantic | Recursive for general, semantic for technical docs |
| **Embedding model** | OpenAI, Voyage, Cohere | Voyage-3 for highest quality, OpenAI for cost |
| **Vector DB** | Pinecone, Weaviate, pgvector | Weaviate for hybrid, pgvector if you have Postgres |
| **Search** | Dense only, hybrid | Always hybrid in production |
| **Reranking** | None, cross-encoder, Cohere | Cohere Rerank v3 for best quality |
| **Generation model** | Any LLM | Claude Sonnet for quality/cost balance |

### Scaling Considerations

```
Bottleneck 1: Embedding at indexing time
→ Batch embedding API calls (1000 texts/request)
→ Cache embeddings for unchanged documents
→ Use cheaper embedding model for initial indexing

Bottleneck 2: Vector search latency
→ HNSW index (vs IVF) for lower latency
→ Pre-filter by metadata before vector search
→ Cache frequent queries (Redis with TTL)

Bottleneck 3: LLM generation latency
→ Streaming responses (don't make user wait for full output)
→ Faster model (Haiku) for simple queries, Sonnet for complex
→ Reduce context size (fewer, better chunks)
```

### Cost Optimization

```python
# Per-query cost breakdown (approximate 2026 pricing):
# Embedding: $0.00002 per query (OpenAI small)
# Vector search: $0.0001 per query (Pinecone serverless)
# Reranking: $0.001 per 1000 docs (Cohere)
# LLM generation: $0.003-0.015 per query (Claude Sonnet, ~1K tokens)
# Total: ~$0.005-0.02 per query

# Reduce cost:
# 1. Cache top-1000 common queries → saves LLM cost
# 2. Use embedding cache → avoid re-embedding identical queries
# 3. Route simple queries to Haiku → 10x cheaper
# 4. Limit context size → fewer output tokens
```

---

## Pattern 2: Agent Architecture

### When to Use

- Multi-step tasks requiring planning
- Tasks needing external data/tool access
- Dynamic workflows where next step depends on results
- User-facing assistants with real-world actions

### Architecture

```
                    ┌────────────────────────────────────────────┐
                    │          AGENT SYSTEM                       │
                    │                                             │
                    │  User Input                                 │
                    │       ↓                                     │
                    │  [Task Planner / Router]                    │
                    │       ↓                                     │
                    │  ┌────────────────────────────┐            │
                    │  │      Agent Loop             │            │
                    │  │                             │            │
                    │  │  [LLM] → think → [Tool]    │            │
                    │  │    ↑                 ↓      │            │
                    │  │    └─── observe ────┘       │            │
                    │  │                             │            │
                    │  │  Until: task complete       │            │
                    │  │     or: max iterations      │            │
                    │  └────────────────────────────┘            │
                    │       ↓                                     │
                    │  [Output Validator]                         │
                    │       ↓                                     │
                    │  [Human Review?] (optional HITL)            │
                    │       ↓                                     │
                    │  Final Response / Action                    │
                    └────────────────────────────────────────────┘
```

### Tool Design Principles

```python
# Good tool design:
@tool
def search_knowledge_base(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the internal knowledge base for relevant information.

    Args:
        query: Natural language search query
        max_results: Number of results to return (1-10)

    Returns:
        List of {title, content, relevance_score} dicts
    """
    # Clear description helps LLM know WHEN to use this tool
    # Typed inputs help LLM know HOW to call this tool
    # Return type helps LLM interpret results

# Bad tool design:
@tool
def do_stuff(x):  # Vague name, no description, untyped
    """Does stuff."""
    return some_api_call(x)
```

### Multi-Agent vs Single-Agent Decision

```
Is the task parallelizable?
├── Yes → Multi-agent (fan-out pattern)
└── No → Single agent is usually better

Does it require multiple specialized domains?
├── Yes (e.g., legal + technical + financial) → Multi-agent
└── No → Single agent

Is total token count within 200K?
├── Yes → Single agent (simpler, cheaper)
└── No → Multi-agent (divide and conquer)

Is the team willing to debug multi-agent coordination?
├── Yes → Multi-agent if needed
└── No → Single agent with good tools
```

---

## Pattern 3: Batch ML System

### When to Use

- Offline predictions for all users (recommendations, risk scores)
- Daily/weekly model retraining
- Large-scale feature computation
- ETL with ML transforms

### Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           BATCH ML SYSTEM                     │
                    │                                               │
                    │  TRAINING (weekly/daily)                     │
                    │  Raw Data → Feature Store → Train → Registry │
                    │                                               │
                    │  INFERENCE (hourly/daily)                    │
                    │  Trigger (cron/event)                        │
                    │       ↓                                       │
                    │  [Feature Fetch] ← Feature Store            │
                    │       ↓                                       │
                    │  [Batch Inference] (Spark/Ray)               │
                    │       ↓                                       │
                    │  [Score Storage] → Database/Feature Store    │
                    │       ↓                                       │
                    │  [Downstream Services] (use cached scores)   │
                    └──────────────────────────────────────────────┘
```

### Batch vs Real-Time Tradeoff

| Dimension | Batch | Real-Time |
|-----------|-------|-----------|
| **Latency** | Pre-computed: <10ms | Computed: 10ms-10s |
| **Cost** | Low (compute once) | High (compute per request) |
| **Freshness** | Stale (minutes-hours) | Fresh (milliseconds) |
| **Scalability** | Trivially scalable | Requires serving infra |
| **Complexity** | Low | High |
| **Best for** | Recommendations, risk scores | Fraud detection, pricing |

### Hybrid: Lambda Architecture for ML

```
                    ┌────────────────────────────────────────┐
                    │         LAMBDA ML ARCHITECTURE          │
                    │                                         │
                    │  Batch Layer:                          │
                    │  Historical data → Full model retrain  │
                    │  → Precomputed scores for all users    │
                    │                                         │
                    │  Speed Layer:                          │
                    │  Real-time events → Lightweight model  │
                    │  → Fresh adjustments (recency signals) │
                    │                                         │
                    │  Serving Layer:                        │
                    │  Batch score + Speed adjustment = Final│
                    │  score shown to user                   │
                    └────────────────────────────────────────┘
```

---

## Pattern 4: Real-Time ML System

### When to Use

- Fraud detection (must decide in <100ms)
- Dynamic pricing (real-time demand signals)
- Content moderation (before post goes live)
- Personalized ranking (user-specific, session-specific)

### Architecture

```
Request → [API Gateway]
               ↓
         [Feature Store] ← [Real-time events (Kafka)]
               ↓
         [Model Server] (TorchServe, vLLM, Triton)
               ↓
         [Post-processing] (threshold, ranking)
               ↓
         Response (<100ms total)
```

### Latency Budget (Example: Fraud Detection <100ms)

```
Network in:         5ms
Feature fetch:      10ms  (Redis/Feature Store)
Model inference:    20ms  (GPU-accelerated)
Post-processing:    5ms
Network out:        5ms
─────────────────────────
Total:              45ms  (45ms budget left for safety margin)
```

### Feature Store Integration

```python
from feast import FeatureStore, FeatureService

store = FeatureStore(repo_path=".")

# Online feature retrieval (sub-millisecond)
feature_vector = store.get_online_features(
    features=FeatureService("fraud_detection_features"),
    entity_rows=[{"user_id": "user_123", "transaction_id": "txn_456"}]
).to_dict()

# Features available:
# - user_30d_avg_transaction
# - user_country_mismatch_flag
# - merchant_fraud_rate_7d
# - device_seen_before
```

---

## Common Interview Questions (2026)

### Agent Systems Design

**Q: Design a code review agent for a software team.**

```
Requirements:
- Review PRs automatically
- Flag bugs, security issues, style violations
- Explain issues in developer-friendly language
- Learn from feedback over time

Architecture:
1. Trigger: GitHub webhook on PR creation
2. Fetch PR diff → chunk by file
3. Agent tools:
   - static_analysis(code) → AST-based checks
   - search_codebase(pattern) → find similar code
   - check_security(code) → SAST rules
   - get_pr_history(author) → author's typical patterns
4. Multi-agent: separate agents per concern (security, style, logic)
5. Aggregator: prioritize and format findings
6. Output: structured review comments via GitHub API
7. Feedback loop: track which comments devs accept/dismiss
```

**Q: Design a customer support agent with escalation.**

```
Tiers:
Tier 1: LLM agent (RAG over docs) → handles 70% of tickets
Tier 2: Specialized agent (order lookup, refund processing)
Tier 3: Human agent (complex/sensitive issues)

Routing logic:
- Simple FAQ → Tier 1
- Account-specific → Tier 2 (with tool access)
- Emotional signals detected → Tier 3 immediately
- Tier 1 fails after 2 attempts → escalate to Tier 2
- Tier 2 policy limits exceeded → escalate to Tier 3

Key metrics:
- Containment rate (% resolved without human)
- CSAT score
- First contact resolution rate
- Escalation rate
```

---

### RAG Optimization

**Q: Your RAG system has low faithfulness scores. How do you fix it?**

```
Diagnosis:
1. Check if context actually contains the answer
   → If no: retrieval problem (improve chunking, hybrid search)
   → If yes: generation problem (LLM ignoring context)

2. Retrieval problems:
   - Wrong chunks retrieved → better embedding model, reranking
   - Relevant chunks missed → increase top-k, hybrid search
   - Chunk boundary cuts key info → better chunking strategy

3. Generation problems:
   - LLM hallucinating despite context → stronger prompt with
     "Answer ONLY from the provided context. Say 'I don't know'
     if the answer isn't in the context."
   - Context too long (lost-in-middle) → reduce to top-3 chunks,
     reorder (most relevant first and last)
   - Wrong model → try larger model (Claude Opus vs Sonnet)
```

**Q: How do you scale a RAG system to 10M documents?**

```
Indexing:
- Parallel embedding generation (batch API + multiple workers)
- Pinecone serverless or Weaviate cluster (horizontal scaling)
- Incremental updates vs full re-index strategy

Query time:
- Pre-filter by metadata (reduces search space 10-100x)
- Caching: Redis cache for frequent queries (TTL=1h)
- Approximate ANN (HNSW) vs exact search
- Namespace isolation for multi-tenancy

Cost:
- Tiered storage: hot docs in main index, cold docs in cheaper archive
- Embedding cache: avoid re-embedding same query twice
- Model routing: simple queries → fast/cheap model
```

---

### LLM Scaling Challenges

**Q: How do you reduce LLM inference cost by 10x?**

```
1. Model selection: Claude Haiku vs Opus = 30x cost difference
   → Route simple tasks to Haiku, complex to Opus

2. Prompt optimization: Shorter prompts = fewer input tokens
   → Remove redundant instructions
   → Use system prompts efficiently
   → Avoid verbose few-shot examples

3. Caching:
   → Semantic cache (cache by embedding similarity)
   → Exact cache for repeated identical queries
   → Prompt prefix caching (Anthropic supports this)

4. Output length control:
   → Specify max_tokens precisely
   → Structured output (JSON) is often shorter than prose
   → Streaming allows early termination

5. Batching:
   → Batch multiple requests into one API call
   → Reduces per-call overhead
```

---

### Cost vs Latency Tradeoffs

**Q: Walk me through how you'd decide between Claude Opus and Claude Haiku for an application.**

```
Build a routing layer based on:

1. Task complexity signals:
   - Token count of input
   - Number of reasoning steps needed
   - Presence of code / math
   - Confidence required

2. Cost budget:
   - Haiku: ~$0.00025/1K input, $0.00125/1K output
   - Sonnet: ~$0.003/1K input, $0.015/1K output
   - Opus: ~$0.015/1K input, $0.075/1K output

3. Latency requirements:
   - Real-time chat: Haiku or Sonnet (2-5s)
   - Background processing: Opus acceptable (5-30s)

4. Quality requirements:
   - Factual lookup → Haiku sufficient
   - Complex reasoning → Opus required
   - Code generation → Sonnet usually enough

Implementation:
classifier = small_model.predict_task_complexity(prompt)
if classifier.score < 0.3:
    return haiku_response(prompt)
elif classifier.score < 0.7:
    return sonnet_response(prompt)
else:
    return opus_response(prompt)
```

---

## System Design Interview Tips

1. **Start with requirements** — ask 2-3 clarifying questions before diving in
2. **Define success metrics** — both business (revenue, engagement) and technical (precision, latency)
3. **Make tradeoffs explicit** — "I'm choosing X over Y because of Z, with the tradeoff that..."
4. **Start simple** — baseline design first, then optimize
5. **Quantify** — "This handles 10K RPS at P99 < 100ms with 99.9% availability"
6. **Know failure modes** — what happens when the LLM returns garbage? When the vector DB is down?
7. **Cost awareness** — LLM costs are non-trivial; show you've thought about them
8. **Evolve the design** — "In V1 we do X, in V2 we add Y when we hit Z scale"
