# Context Engineering

Prompt engineering is about wording. **Context engineering is about what goes into the window at all** — which facts, in what order, at what cost, and what gets dropped when the conversation outgrows the budget. It is now the dominant design activity in LLM applications, because in a RAG or agent system the model is rarely the bottleneck; what you chose to show it is.

---

## Table of Contents
1. [Prompt Engineering vs Context Engineering](#prompt-engineering-vs-context-engineering)
2. [The Context Budget](#the-context-budget)
3. [Context Ordering and the Lost-in-the-Middle Effect](#context-ordering-and-the-lost-in-the-middle-effect)
4. [Prompt Caching and Prefix Stability](#prompt-caching-and-prefix-stability)
5. [Conversation Memory](#conversation-memory)
6. [Compaction Strategies](#compaction-strategies)
7. [Context for Agents](#context-for-agents)
8. [Context Rot and Failure Modes](#context-rot-and-failure-modes)
9. [Measuring Context Quality](#measuring-context-quality)
10. [Interview Q&A](#interview-qa)
11. [Common Pitfalls](#common-pitfalls)
12. [Related Topics](#related-topics)

---

## Prompt Engineering vs Context Engineering

| | Prompt engineering | Context engineering |
|---|---|---|
| Unit of work | Wording of an instruction | Composition of the whole window |
| Typical question | "How do I phrase this?" | "What does the model need to see, and what must I leave out?" |
| Failure it fixes | Model misunderstands the task | Model lacks the fact, or drowns in irrelevant text |
| Cost impact | Marginal | Dominant — input tokens are most of the bill |
| Scales with | Task complexity | Corpus size, conversation length, tool count |

The shift happened because context windows grew. When the window was 4k tokens, the constraint was obvious and everyone economized. At 200k+, it became possible to stuff everything in — and teams discovered that **more context often makes answers worse**, not just more expensive. Deciding what to exclude turned into the real skill.

---

## The Context Budget

Treat the window as a budget with named line items, not as a bucket.

```
┌─────────────────────────────────────────────────────┐
│ System prompt + tool definitions      stable, cached │
│ Few-shot examples                     stable, cached │
├─────────────────────────────────────────────────────┤
│ Long-term memory / user profile       semi-stable    │
│ Conversation history                  growing        │
│ Retrieved documents                   per-request    │
│ Current user message                  per-request    │
├─────────────────────────────────────────────────────┤
│ Reserved headroom for the response    never fill     │
└─────────────────────────────────────────────────────┘
```

Two rules follow directly:

1. **Never plan to fill the window.** Reserve room for the output plus a safety margin. A request that fits at turn 10 and overflows at turn 11 is a production incident, and truncation errors surface as user-visible failures.
2. **Order by stability, not by importance.** Everything that rarely changes goes first so the cached prefix stays valid; volatile content goes last.

```python
def assemble_context(system, tools, examples, memory, history, docs, question, budget=100_000):
    """Fill from a fixed skeleton, trimming the elastic sections to fit."""
    fixed = count_tokens(system) + count_tokens(tools) + count_tokens(examples)
    reserved_for_output = 4_000
    elastic = budget - fixed - reserved_for_output - count_tokens(question)

    # Priority order when space is tight: memory > recent history > documents
    memory_part  = truncate(memory, min(count_tokens(memory), int(elastic * 0.15)))
    docs_part    = truncate(docs,   int(elastic * 0.50))
    history_part = truncate(history, elastic - count_tokens(memory_part) - count_tokens(docs_part))

    return [system, tools, examples, memory_part, history_part, docs_part, question]
```

**Count tokens, never characters.** A 4-chars-per-token rule of thumb is fine for a rough estimate and wrong at the boundary — code, JSON, non-English text, and long identifiers tokenize far denser. Use the provider's tokenizer for any check that gates a request.

---

## Context Ordering and the Lost-in-the-Middle Effect

Models attend unevenly across a long context. Accuracy on retrieving a fact is highest when it sits near the **beginning or end** of the window and measurably lower in the middle — the "lost in the middle" finding. It has been reproduced across model families, and it holds even for models advertising very long windows.

Practical consequences:

- **Put the most relevant retrieved chunk last**, immediately before the question. Rerankers output a relevance order; placing rank 1 nearest the query beats placing it first in a long list.
- **Repeat the instruction at the end** for very long contexts. A task instruction given 80k tokens earlier competes with everything since.
- **Fewer, better chunks beat more chunks.** Passing 20 documents when 4 suffice both costs more and dilutes attention across irrelevant text.
- **Structure the context.** Delimiters, headers, and per-source labels help the model locate what it needs and make citation possible.

```python
# Reranked descending by relevance — reverse so the strongest sits closest to the question
context = "\n\n".join(
    f"<document id=\"{d.id}\" source=\"{d.source}\" date=\"{d.date}\">\n{d.text}\n</document>"
    for d in reversed(reranked[:4])
)
prompt = f"{context}\n\nAnswer using only the documents above, citing document ids.\n\nQuestion: {question}"
```

---

## Prompt Caching and Prefix Stability

Providers cache a request's prefix and charge a fraction of the normal input price for the cached portion; self-hosted servers skip the prefill compute entirely. The cache matches on an **exact prefix**, so a single changed byte early in the prompt invalidates everything after it.

This makes prompt *ordering* an economic decision:

| Layout | Cache behavior |
|---|---|
| `system → tools → examples → history → docs → question` | Prefix stable across turns; large cache hit |
| `timestamp → system → docs → question` | A leading timestamp busts the cache on **every** request |
| `user profile → system → ...` | Cache is per-user at best; no sharing across users |

Concrete anti-patterns that silently destroy cache hit rate: injecting the current time or a request ID at the top; putting the user's name or tenant into the system prompt when it could go later; reordering tool definitions non-deterministically (a `set` or dict iteration that isn't stable); and re-summarizing conversation history every turn, which rewrites the middle of the prefix.

For agents this matters enormously: an agent loop re-sends the whole conversation on each step, so a stable prefix can cut cost by an order of magnitude over a long run.

---

## Conversation Memory

| Strategy | Mechanism | Best for | Weakness |
|---|---|---|---|
| **Full history** | Send everything | Short sessions | Grows without bound |
| **Sliding window** | Keep last N turns | Chat with local context | Silently forgets early commitments |
| **Summary buffer** | Summarize old turns, keep recent verbatim | Long conversations | Summarization loses detail and costs a call |
| **Retrieval over history** | Embed turns, retrieve relevant ones | Very long or resumed sessions | Needs an index; can miss context |
| **Structured memory** | Extract facts to a store (key-value or graph) | Assistants with persistent users | Extraction errors persist and compound |

The strategy that works in practice is layered rather than singular: keep a small structured profile of durable facts, keep the last few turns verbatim, and summarize or retrieve the rest.

```python
class LayeredMemory:
    """Durable facts + verbatim recent turns + summarized older turns."""

    def __init__(self, keep_recent=6, summarize_after=20):
        self.facts = {}            # durable, extracted: {"timezone": "CET", "role": "MLE"}
        self.turns = []            # verbatim
        self.summary = ""          # compacted older history
        self.keep_recent = keep_recent
        self.summarize_after = summarize_after

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.summarize_after:
            older, self.turns = self.turns[:-self.keep_recent], self.turns[-self.keep_recent:]
            # Summarize into the EXISTING summary so it stays append-only and cache-friendly
            self.summary = summarize(self.summary, older)

    def render(self):
        parts = []
        if self.facts:
            parts.append("Known facts:\n" + "\n".join(f"- {k}: {v}" for k, v in self.facts.items()))
        if self.summary:
            parts.append(f"Earlier conversation summary:\n{self.summary}")
        return parts + self.turns
```

**What must survive compaction**: decisions and commitments made, constraints the user stated, corrections the user issued (an assistant that re-makes a corrected mistake destroys trust faster than any other failure), identifiers and names, and open questions. What can be dropped: pleasantries, superseded intermediate reasoning, and tool output already reflected in a conclusion.

---

## Compaction Strategies

When a conversation approaches the budget, something must go. Options, roughly in order of fidelity:

1. **Drop tool call bodies, keep conclusions.** Tool outputs are usually the largest and most disposable content — a 3,000-token API response reduces to "the query returned 14 rows; the top account is X."
2. **Summarize the oldest span** into the existing summary, leaving recent turns verbatim.
3. **Deduplicate.** Retrieved documents repeat across turns in RAG chats; keep one copy.
4. **Externalize.** Write long artifacts to a file or store and keep a reference the model can re-read on demand, rather than carrying them inline.
5. **Hard-truncate the middle**, keeping the head (system, task) and tail (recent turns). Crude but predictable, and better than an overflow error.

Compaction should be **idempotent and append-only** where possible. Re-summarizing the whole history each turn produces a different prefix every time, which both busts the cache and lets facts drift as summaries of summaries degrade.

---

## Context for Agents

Agent loops re-send accumulated context at every step, so context growth is the main driver of both cost and failure over a long run.

- **Tool definitions are permanent context.** Twenty tools with verbose schemas can consume several thousand tokens on every single step. Trim descriptions, and load tools by task phase rather than exposing all of them always.
- **Tool results are the fastest-growing segment.** Truncate large outputs at the boundary, with an explicit marker, and give the agent a way to request more.
- **Subagents isolate context.** Delegating a research subtask to a subagent that returns only its conclusion keeps the parent's window clean — this is the main practical argument for multi-agent designs, more than any notion of specialization.
- **Externalize state.** A scratchpad file or task list the agent reads and writes beats carrying all intermediate reasoning inline.

```python
MAX_TOOL_RESULT_TOKENS = 2_000

def add_tool_result(messages, tool_name, result):
    text = str(result)
    if count_tokens(text) > MAX_TOOL_RESULT_TOKENS:
        text = (truncate(text, MAX_TOOL_RESULT_TOKENS)
                + f"\n\n[truncated — {count_tokens(str(result))} tokens total. "
                  f"Call {tool_name} with a narrower filter for specifics.]")
    messages.append({"role": "tool", "name": tool_name, "content": text})
```

---

## Context Rot and Failure Modes

| Failure | What it looks like | Cause | Fix |
|---|---|---|---|
| **Distraction** | Answer drifts to an irrelevant retrieved document | Too many low-relevance chunks | Rerank; pass fewer |
| **Lost in the middle** | A fact that *is* in context is ignored | Poor position | Put key content at the end |
| **Context poisoning** | A hallucination early on is treated as fact later | Model's own error persists in history | Validate before appending; drop bad turns |
| **Instruction dilution** | Model stops following the format after many turns | Instruction is far away and outnumbered | Restate constraints near the end |
| **Stale context** | Answers from an outdated retrieved doc | No recency signal | Timestamp metadata; prefer recent sources |
| **Conflicting sources** | Confidently picks one of two contradictory docs | No precedence rule | Rank by authority and recency; ask the model to flag conflicts |
| **Silent truncation** | Answer omits something that was "in" the prompt | Middle-out truncation by the client or provider | Count tokens yourself; fail loudly |

**Context poisoning** deserves emphasis because it is specific to multi-turn systems and easy to miss: once a model states something false and that turn stays in history, the falsehood is now "established context" the model will defend and build on. Validating tool results and model claims *before* they enter durable history is far cheaper than detecting the downstream damage.

---

## Measuring Context Quality

Do not evaluate the prompt; evaluate what the context enables.

| Metric | Question it answers |
|---|---|
| **Retrieval recall@k** | Is the needed fact even in the window? |
| **Context precision** | What fraction of included tokens are relevant? |
| **Faithfulness / groundedness** | Are claims supported by what was provided? |
| **Citation accuracy** | Do cited spans exist and support the claim? |
| **Cache hit rate** | Is the prefix actually stable? |
| **Tokens per resolved request** | The real cost metric |
| **Truncation rate** | How often is content silently dropped? |

The highest-signal diagnostic is a **needle test on your own corpus**: place a known fact at varying depths in a realistic context and measure retrieval accuracy by position. Published long-context benchmarks use synthetic haystacks and consistently overstate performance relative to real, semantically dense documents.

The ablation worth running before shipping: hold the question set fixed and vary only the number of retrieved chunks (2, 4, 8, 16). Accuracy usually peaks well before the maximum, and the peak is your `top_k`.

---

## Interview Q&A

#### What is context engineering and how does it differ from prompt engineering?

Prompt engineering optimizes the *wording* of instructions — phrasing, examples, output format. Context engineering optimizes *what information enters the window at all*: which documents, how much history, in what order, within what token and cost budget, and what gets evicted when it overflows.

The distinction became important once windows got large enough that "include everything" was possible. It turns out that including everything degrades quality — irrelevant context distracts the model, and content in the middle of a long window is attended to less reliably. So the job shifted from writing better instructions to curating a smaller, better-ordered context. In a RAG or agent system it is also where nearly all the cost lives, since input tokens dominate.

#### The context window is 200k tokens. Should you just put everything in?

No, for three separate reasons.

**Quality**: accuracy degrades with irrelevant content. The lost-in-the-middle effect means a fact buried at 50% depth is retrieved less reliably than one at the edges, and distraction from low-relevance chunks measurably worsens answers.

**Cost**: input tokens dominate a RAG bill. Sending 100k tokens to answer a question that needed 3k is a ~30x cost multiplier on every request, and it recurs on every agent step.

**Latency**: prefill is compute-bound and scales with prompt length, so a huge prompt directly inflates time-to-first-token.

The right approach is retrieve broadly, rerank, and pass the smallest context that reliably contains the answer — usually far fewer chunks than people expect, and worth determining empirically with a `top_k` ablation.

#### How does prompt caching change how you structure a prompt?

The cache keys on an exact prefix, so anything that varies must go **after** everything that doesn't. The layout becomes: system prompt, tool definitions, few-shot examples, then long-term memory, then conversation history, then retrieved documents, then the user's question.

The consequences are concrete. A timestamp or request ID at the top of the system prompt invalidates the cache on every request — one of the most expensive one-line bugs in LLM applications. Non-deterministic tool ordering (iterating a set) does the same. Re-summarizing the whole history each turn rewrites the middle of the prefix and destroys the hit rate.

Done right, cached input is typically around 10% of normal price, and for an agent loop that re-sends context every step it is often the single largest cost reduction available.

#### What is the lost-in-the-middle effect and how do you design around it?

Models retrieve facts from the start and end of a long context substantially more reliably than from the middle — a U-shaped accuracy curve by position, reproduced across model families and present even in long-context models.

Design responses: put the highest-relevance retrieved chunk immediately before the question rather than first in the list; restate critical instructions at the end of a long context; reduce the number of chunks so there is less middle to get lost in; and add structural markers so content is locatable rather than uniform prose. Then verify on your own data with a needle test at varying depths, because synthetic benchmarks overstate real performance on semantically dense documents.

#### How would you manage memory for an assistant used daily over months?

Layered, because no single strategy covers it. A **structured profile** for durable facts — preferences, constraints, role, timezone — extracted deliberately and stored outside the conversation. **Recent turns verbatim**, since immediate context needs full fidelity. **A rolling summary** of older turns, updated append-only so the prefix stays cache-stable. **Retrieval over an archive** of past sessions for anything older, so a question about a conversation from March pulls in just that.

The critical design detail is what survives compaction: decisions, commitments, stated constraints, and especially **user corrections**. An assistant that repeats a mistake the user already corrected loses trust faster than one that simply forgets. I'd also make memory writes auditable and user-editable, because extraction errors otherwise persist indefinitely and compound.

#### What is context poisoning?

When a model states something false and that turn remains in the conversation history, the falsehood becomes established context. On subsequent turns the model treats its own prior claim as a given, builds on it, and will often defend it against correction — a single hallucination compounds across a long session. The same happens with an erroneous tool result that is never validated.

Mitigations are all about the write path: validate tool outputs before appending them, check claims against retrieved sources before they enter durable history, allow removing or correcting a poisoned turn rather than only appending, and keep durable memory extraction separate from raw conversation so a bad turn does not become a permanent "fact."

#### An agent's cost grows superlinearly over a long run. Why, and what do you do?

Because each step re-sends the entire accumulated context. If the conversation grows by roughly a constant amount per step, total tokens across `N` steps grow as `O(N²)`. Tool results are usually the fastest-growing segment, and verbose tool schemas are re-sent every single step.

Fixes, in order of impact: **prompt caching** with a strictly stable prefix, so re-sent context costs a fraction; **truncate tool results** at a boundary with a marker and a way to request more; **compact** older steps into a summary once past a threshold; **externalize** long artifacts to files the agent reads on demand; **delegate to subagents** that return only conclusions, keeping the parent window small; and **trim tool definitions** or load them by phase. Plus a hard step limit and a per-request cost ceiling, because an agent stuck in a loop is the failure mode that produces the surprising invoice.

#### How do you decide how many retrieved chunks to include?

Empirically, with an ablation. Fix a labeled question set, vary only `top_k` (2, 4, 8, 16, 32), and measure end-to-end answer accuracy along with cost and latency. Accuracy almost always peaks at a modest value and then declines as irrelevant context dilutes attention — the peak is the answer, and it is usually lower than intuition suggests.

Two refinements matter. Retrieve broadly (say 30) and **rerank** down to the final few, so the candidate set is wide but the context is tight. And measure retrieval recall separately: if recall@30 is poor, no amount of `top_k` tuning helps, and the problem is chunking or the embedding model rather than context assembly.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Timestamp or request ID at the top of the prompt | Busts the prefix cache on every request | Move all volatile content to the end |
| Filling the window to its limit | Overflow at turn N+1 becomes a user-visible failure | Reserve explicit output headroom |
| Estimating tokens from character count | Code, JSON, and non-English tokenize far denser | Use the provider's tokenizer for any gating check |
| Passing every retrieved chunk "just in case" | Higher cost *and* measurably worse answers | Rerank; ablate `top_k`; pass fewer |
| Highest-relevance chunk placed first | Lands in the weakly-attended middle of a long list | Put it last, adjacent to the question |
| Re-summarizing the entire history each turn | Rewrites the prefix, destroys caching, drifts facts | Append-only summary of the oldest span |
| Appending unvalidated tool output to history | Errors become established context and compound | Validate before it enters durable history |
| Non-deterministic tool definition order | Silent cache misses | Sort tool definitions deterministically |
| Dropping user corrections during compaction | Assistant repeats a corrected mistake | Pin corrections and commitments as durable facts |
| Trusting published long-context benchmarks | Synthetic haystacks overstate real performance | Needle test on your own corpus at varying depths |

---

## Related Topics

- [Prompt Engineering](./intro_prompt_engineering.md)
- [Intro to RAG](./intro_rag.md)
- [RAG Engineering](./intro_rag_engineering.md)
- [Embeddings](./intro_embeddings.md)
- [Agentic AI](./intro_agentic_ai.md)
- [Multi-Agent Systems](./intro_multi_agent_systems.md)
- [LLM Inference Optimization](./intro_llm_inference_optimization.md)
- [LLM Fundamentals](./intro_llm_fundamentals.md)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md)
