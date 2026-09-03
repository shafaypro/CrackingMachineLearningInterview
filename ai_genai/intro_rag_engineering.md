# Retrieval-Augmented Generation Engineering

This guide focuses on the engineering decisions behind production RAG systems rather than only the concept.

---

## Overview

Retrieval-Augmented Generation combines search with generation. Instead of asking the model to answer from parametric memory alone, the system retrieves relevant documents and injects them into the prompt.

RAG matters because it improves:

- factual grounding
- access to private knowledge
- freshness of information
- debuggability compared with pure prompting

---

## Core Concepts

### Document ingestion

The ingestion pipeline determines what the system can retrieve later. In practice this includes parsing, cleaning, metadata extraction, deduplication, and indexing.

### Chunking strategies

Chunking controls the tradeoff between recall and context quality. Small chunks improve retrieval granularity; large chunks preserve context but can dilute relevance.

### Retrieval methods

Common retrieval approaches include:

- dense vector search
- keyword search
- hybrid retrieval
- reranking after initial retrieval

### Context injection

Retrieved evidence must be placed into prompts cleanly. Good context injection preserves source boundaries, prioritizes relevant passages, and avoids overwhelming the model with noise.

---

## Key Skills

### Designing retrieval pipelines

A strong engineer can decide:

- how data is ingested
- which metadata matters
- when to use hybrid retrieval
- how many documents to return

### Optimizing recall vs precision

This means tuning chunk size, overlap, `top_k`, metadata filters, and reranking until the system returns enough relevant evidence without excessive noise.

### Handling long-context data

In practice, this involves hierarchical retrieval, map-reduce summarization, and context compression rather than blindly increasing prompt size.

### Hybrid search

Hybrid search combines lexical and semantic retrieval. It is especially useful for exact terms, IDs, code snippets, and acronyms that pure vector search can miss.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| Pinecone | Managed vector database | Hosted semantic search at scale |
| Weaviate | Vector database with hybrid search support | Rich metadata filters and semantic retrieval |
| FAISS | Local high-performance vector index | Prototyping or self-managed retrieval |
| Chroma | Lightweight embedding store | Local experiments and simple apps |
| BM25 / keyword search | Lexical search baseline | Exact-match-heavy corpora and hybrid systems |

---

## Projects

### Document Q&A system

- Goal: Answer questions from a document set with citations.
- Key components: parser, chunker, retriever, prompt template, answer formatter.
- Suggested tech stack: Python, Chroma or FAISS, FastAPI.
- Difficulty: Intermediate.

### Knowledge base assistant

- Goal: Build an assistant over internal docs, runbooks, and onboarding material.
- Key components: ingestion pipeline, metadata filters, access control, feedback loop.
- Suggested tech stack: vector DB, FastAPI, SSO-aware backend, evaluation harness.
- Difficulty: Advanced.

### Internal company search tool

- Goal: Search across policies, docs, and tickets using hybrid retrieval.
- Key components: BM25 + vector retrieval, reranker, query rewriting, result snippets.
- Suggested tech stack: Weaviate or Elasticsearch + vectors, Python.
- Difficulty: Advanced.

### Multi-source RAG pipeline

- Goal: Combine structured and unstructured data in one retrieval flow.
- Key components: SQL tool access, document retrieval, ranking, source-aware answer generation.
- Suggested tech stack: Postgres, vector store, FastAPI, LangGraph or custom orchestration.
- Difficulty: Advanced.

---

## Example Code

```python
def build_rag_context(chunks: list[str], max_chunks: int = 4) -> str:
    selected = chunks[:max_chunks]
    return "\n\n".join(
        f"[Source {i + 1}]\n{chunk}"
        for i, chunk in enumerate(selected)
    )
```

---

## Suggested Project Structure

```text
knowledge-base-assistant/
├── ingest/
├── retriever/
├── prompts/
├── evals/
├── api/
└── README.md
```

---

## Interview Q&A

#### Walk me through the components of a production RAG system.

Ingestion (parse, clean, deduplicate, extract metadata), chunking, embedding, indexing into a vector store, then at query time: query understanding or rewriting, retrieval (usually hybrid), reranking, context assembly, generation with citations, and finally logging and evaluation.

The parts candidates usually omit are the ones that decide whether the system works: **reranking**, which fixes the "right document, wrong position" problem cheaply; **metadata filtering**, which is usually a larger accuracy win than a better embedding model and is where access control lives; and the **evaluation loop**, without which you cannot tell whether any change helped.

#### Retrieval returns documents that look relevant but the answer is wrong. What's happening?

Several distinct causes, and they need different fixes:
- **Topically similar but not answer-bearing.** Embeddings measure relatedness, not whether a passage contains the answer. A reranker (cross-encoder) scores query-document pairs jointly and fixes most of this.
- **The chunk is truncated.** The answer spans a boundary and each half looks partially relevant. Increase overlap or use small-to-big retrieval.
- **Stale content.** The index still holds a document that was updated or deleted. Check the ingest pipeline's handling of updates.
- **Contradictory sources.** Two documents disagree and the model picked one arbitrarily. Add recency and authority metadata, and prefer the newest authoritative source in context assembly.
- **The model ignored the context.** Verify by feeding the gold chunk directly; if the answer is still wrong, it's a prompt or model problem, not retrieval.

#### How do you evaluate a RAG system?

Separately at each stage, or you cannot localize a regression.

**Retrieval**: recall@k and MRR against a labeled set of questions with known-relevant chunk IDs. Recall@k is the ceiling on everything downstream — if the evidence isn't retrieved, no prompt fixes it.

**Generation given context**: faithfulness (is every claim supported by the retrieved text?), answer relevance, and citation correctness. LLM-as-judge works here if the judge is validated against human labels on a sample.

**End to end**: task success rate on a fixed regression suite, run on every prompt, model, or index change.

**Production**: user feedback signals, escalation rate, and the frequency of "I don't know" responses — a sudden rise usually means retrieval broke, not that the model got more cautious.

#### How do you handle queries the corpus cannot answer?

Explicitly, because the default behavior is confident fabrication. Set a relevance-score floor and return "I don't have information on that" below it. Instruct the model in the prompt to answer only from the provided context and to say when it can't. Require citations, and validate post-hoc that the cited spans exist in the retrieved text. Then monitor the abstention rate as a first-class metric: too low means hallucination, too high means retrieval is failing.

#### Why use hybrid retrieval instead of pure vector search?

Because they fail in complementary ways. Dense retrieval handles paraphrase and synonymy but blurs exact identifiers — error codes, SKUs, function names, rare proper nouns — since those carry little distributional meaning. BM25 nails exact terms but returns nothing when the query shares no vocabulary with the document.

Fusing with Reciprocal Rank Fusion (`score = Σ 1/(k + rank)`, k≈60) avoids having to calibrate BM25 scores against cosine similarities, which live on incomparable scales.

#### Your RAG system is too expensive. Where do you cut?

Input tokens dominate RAG cost, so start there:
1. **Retrieve more, send less** — retrieve 20 candidates, rerank, pass the top 3–5. Usually cheaper *and* more accurate, since irrelevant context degrades answers.
2. **Prompt caching** — put the stable system prompt and instructions first so the cached prefix is reused; cached input is typically ~10% of the price.
3. **Semantic caching** for repeated questions, with a tuned threshold and a real invalidation strategy.
4. **Model routing** — a small model handles simple lookups, escalating only for synthesis-heavy questions.
5. **Smaller embedding dimensions or quantized vectors** to cut index cost, verified against recall@k.

#### How do you keep the index fresh?

Drive index mutations from the source of truth's change stream rather than periodic full reindexes: on create/update, re-chunk and upsert; on delete, remove by document ID. Store a content hash per chunk so unchanged chunks are skipped. Keep the ingest timestamp in metadata so retrieval can prefer recent content and so you can audit staleness.

For an embedding model upgrade, the whole corpus must be re-embedded — vectors from two models are not comparable. Standard practice is a blue/green reindex into a new collection with an atomic cutover.

#### When is RAG the wrong tool?

When the knowledge is small and static enough to fit in the prompt — just put it in the context and skip the infrastructure. When the task needs reasoning over the *entire* corpus (aggregate questions like "how many contracts expire this quarter"), which is a database query, not retrieval. When the model needs to internalize a style or format rather than facts — that's fine-tuning. And when the data is highly structured, where SQL or a graph query gives exact answers that top-k similarity cannot.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Chunks too large | The vector averages several topics and matches none well | 300–800 tokens, or small-to-big retrieval |
| No reranking stage | Right document retrieved but ranked below the cutoff | Cross-encoder rerank of the top 20–50 |
| Skipping metadata | No filtering, no access control, no recency preference | Attach source, section, timestamp, permissions at ingest |
| Post-filtering by permissions | Empty result pages and leaked document existence | Filter inside the vector search |
| Different embedding models for index and query | Vectors are incomparable; results are noise | Pin the model version; reindex on change |
| No abstention path | The model fabricates when nothing relevant is retrieved | Score floor + explicit "not in context" instruction + citation validation |
| Evaluating end to end only | You cannot tell retrieval failures from generation failures | Measure recall@k and faithfulness separately |
| Stuffing all retrieved chunks in "just in case" | Higher cost and measurably worse answers | Rerank and pass fewer, better chunks |
| Ignoring updates and deletes | The system cites documents that no longer exist | Drive index writes from the source change stream |
| Variable content at the top of the prompt | Destroys prefix cache hits and inflates cost | Stable prefix first, retrieved context last |

---

## Related Topics

- [Intro to RAG](./intro_rag.md)
- [Vector Databases](./intro_vector_databases.md)
- [Vector Databases Advanced](./intro_vector_databases_advanced.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
