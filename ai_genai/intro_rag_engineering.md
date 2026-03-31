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

## Related Topics

- [Intro to RAG](./intro_rag.md)
- [Vector Databases](./intro_vector_databases.md)
- [Vector Databases Advanced](./intro_vector_databases_advanced.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
