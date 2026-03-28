# RAG – Retrieval-Augmented Generation (2026 Edition)

**RAG** (Retrieval-Augmented Generation) is the technique of enhancing LLM responses by retrieving relevant documents from a knowledge base and including them in the prompt. It's the primary way to give LLMs access to private, up-to-date, or domain-specific knowledge.

---

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [RAG Pipeline Architecture](#rag-pipeline-architecture)
3. [Indexing (Offline)](#indexing-offline)
4. [Retrieval (Online)](#retrieval-online)
5. [Generation](#generation)
6. [Advanced RAG Techniques](#advanced-rag-techniques)
7. [Agentic RAG](#agentic-rag)
8. [Evaluation](#evaluation)
9. [End-to-End Example](#end-to-end-example)
10. [RAG in 2026](#rag-in-2026)
11. [Related Topics](#related-topics)

---

## What is RAG?

### The Problem RAG Solves

LLMs have limitations:
- **Knowledge cutoff** — don't know recent events
- **No private data** — can't access your internal docs
- **Hallucination** — may make up facts confidently
- **Context limits** — can't fit millions of docs in one prompt

RAG solves all of these by **retrieving relevant documents at query time** and adding them to the prompt.

```
Without RAG:
"What's our Q4 revenue?" → LLM: "I don't have access to your financial data."

With RAG:
"What's our Q4 revenue?"
→ Search financial_reports/ → find Q4_2025_report.pdf
→ Extract relevant section → add to prompt
→ LLM: "Based on your Q4 2025 report: $4.2M, up 23% YoY"
```

### RAG vs Fine-tuning

| Approach | Use For | Pros | Cons |
|----------|---------|------|------|
| **RAG** | Private/changing knowledge | Easy to update, transparent | Retrieval latency, depends on retrieval quality |
| **Fine-tuning** | Style, format, domain skills | Faster inference, baked-in knowledge | Expensive to update, knowledge can fade |
| **Both** | Best of both worlds | High quality + current knowledge | Complex, expensive |

---

## RAG Pipeline Architecture

```
INDEXING (offline)                    RETRIEVAL (online)
─────────────────────────────         ──────────────────────────────────────

Documents                             User Query
    │                                     │
    ▼                                     ▼
Load & Parse                         Embed Query
    │                                     │
    ▼                                     ▼
Chunk                              Vector Search ──────────────────────┐
    │                                     │                             │
    ▼                                     ▼                             │
Embed (text → vector)            Top-K Documents                       │
    │                                     │                             │
    ▼                                     ▼                             ▼
Store in Vector DB             [Rerank] → Context Building → LLM → Answer
```

---

## Indexing (Offline)

### Step 1: Load Documents

```python
from pathlib import Path
import PyPDF2
import docx

def load_document(path: str) -> str:
    path = Path(path)

    if path.suffix == ".pdf":
        reader = PyPDF2.PdfReader(path)
        return "\n".join(page.extract_text() for page in reader.pages)

    elif path.suffix in [".docx", ".doc"]:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif path.suffix == ".md":
        return path.read_text()

    elif path.suffix == ".txt":
        return path.read_text()

    raise ValueError(f"Unsupported format: {path.suffix}")

# LangChain loaders (handles 50+ formats)
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, GitLoader,
    ConfluenceLoader, NotionDirectoryLoader
)
```

### Step 2: Chunking

The key insight: **chunk size is a trade-off**. Small chunks are more precise, large chunks have more context.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Recursive character splitter (most common)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # characters per chunk
    chunk_overlap=200,     # overlap to preserve context across chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document_text)

# Semantic chunking (splits at semantic boundaries)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
semantic_splitter = SemanticChunker(OpenAIEmbeddings())
chunks = semantic_splitter.split_text(document_text)

# Markdown-aware chunking
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "section"),
        ("##", "subsection"),
        ("###", "subsubsection"),
    ]
)
md_chunks = md_splitter.split_text(markdown_text)
```

### Step 3: Embedding

Embeddings convert text into vectors for similarity search.

```python
from anthropic import Anthropic
import numpy as np

# Voyage AI (Anthropic's embedding models)
import voyageai
voyage = voyageai.Client()

def embed_texts(texts: list[str]) -> list[list[float]]:
    result = voyage.embed(
        texts,
        model="voyage-3",           # voyage-3, voyage-code-3, voyage-finance-2
        input_type="document",      # "document" for indexing, "query" for search
    )
    return result.embeddings

# OpenAI embeddings
from openai import OpenAI
openai_client = OpenAI()

def embed_openai(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [e.embedding for e in response.data]

# Sentence Transformers (local, free)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

### Step 4: Store in Vector DB

```python
import chromadb
from chromadb.utils import embedding_functions

# ChromaDB (local development)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=chunks,
    embeddings=embed_texts(chunks),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "report.pdf", "page": i} for i in range(len(chunks))]
)
```

---

## Retrieval (Online)

### Basic Vector Search

```python
def retrieve(query: str, collection, n_results: int = 5) -> list[dict]:
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": 1 - dist  # cosine distance → similarity
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
```

### Hybrid Search (BM25 + Vector)

Best of both worlds: keyword precision + semantic understanding.

```python
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.embeddings = embed_texts(documents)

        # BM25 (keyword)
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())

        # Vector similarity scores
        q_emb = embed_texts([query])[0]
        vec_scores = np.dot(self.embeddings, q_emb)

        # Normalize and combine
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() + 1e-8)
        vec_norm = (vec_scores - vec_scores.min()) / (vec_scores.max() + 1e-8)
        combined = alpha * vec_norm + (1 - alpha) * bm25_norm

        # Return top-k
        top_k = np.argsort(combined)[::-1][:k]
        return [{"text": self.documents[i], "score": combined[i]} for i in top_k]
```

### Reranking

After initial retrieval, use a cross-encoder to rerank for better precision.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

---

## Generation

```python
import anthropic

client = anthropic.Anthropic()

def rag_query(user_question: str, collection, n_docs: int = 5) -> str:
    # 1. Retrieve
    relevant_docs = retrieve(user_question, collection, n_docs)

    # 2. Build context
    context = "\n\n---\n\n".join([
        f"Source: {doc['metadata'].get('source', 'unknown')}\n{doc['text']}"
        for doc in relevant_docs
    ])

    # 3. Generate
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="""You are a helpful assistant that answers questions based on provided documents.

Rules:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say "I don't have information about that in the provided documents"
- Cite sources when possible
- Be concise and accurate""",
        messages=[{
            "role": "user",
            "content": f"""Context documents:

{context}

---

Question: {user_question}"""
        }]
    )

    return response.content[0].text

answer = rag_query("What was our revenue in Q4 2025?", collection)
print(answer)
```

---

## Advanced RAG Techniques

### Query Rewriting

```python
def rewrite_query(original_query: str) -> list[str]:
    """Generate multiple search queries for better recall."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="Generate 3 different search queries for the given question. Return as JSON array.",
        messages=[{"role": "user", "content": original_query}]
    )
    import json
    return json.loads(response.content[0].text)

# Search with all rewritten queries, deduplicate results
queries = rewrite_query("What's our churn rate?")
# → ["customer churn rate", "subscriber cancellation percentage", "customer retention metrics"]
```

### HyDE (Hypothetical Document Embeddings)

```python
def hyde_search(question: str, collection) -> list[dict]:
    """Generate a hypothetical answer, embed it, use that for retrieval."""

    # Generate hypothetical answer
    hyp_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Write a paragraph that answers: {question}\n(Be concise, doesn't need to be factual)"
        }]
    )
    hypothetical_answer = hyp_response.content[0].text

    # Use the hypothetical answer for retrieval (better semantic match)
    return retrieve(hypothetical_answer, collection)
```

### Contextual Compression

```python
def compress_context(question: str, raw_context: str) -> str:
    """Extract only the relevant parts of retrieved chunks."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system="Extract only the parts of the document relevant to the question. Be concise.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nDocument:\n{raw_context}"
        }]
    )
    return response.content[0].text
```

---

## Agentic RAG

In 2026, RAG has evolved beyond simple retrieval to **agentic RAG** where the LLM decides when and how to retrieve.

```python
# Tools for agentic RAG
rag_tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "date_after": {"type": "string"},
                        "department": {"type": "string"},
                        "doc_type": {"type": "string"}
                    }
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
]

# Agent can choose to search, re-search with better query, or combine sources
```

---

## Evaluation

Evaluating a RAG system requires assessing both the **retrieval** and **generation** components separately, then the end-to-end pipeline. This is the most overlooked part of RAG and the primary reason RAG systems fail in production.

### RAG Evaluation Framework

```
                    ┌─────────────────────────────────────────┐
                    │           RAG Evaluation                 │
                    └─────────────────────────────────────────┘
                          │                         │
               ┌──────────▼──────────┐    ┌────────▼────────────┐
               │   Retrieval Quality  │    │  Generation Quality  │
               └──────────────────────┘    └─────────────────────┘
               │ - Context Precision  │    │ - Faithfulness       │
               │ - Context Recall     │    │ - Answer Relevancy   │
               │ - MRR / NDCG         │    │ - Correctness        │
               │ - Hit Rate           │    │ - Hallucination Rate │
               └──────────────────────┘    └─────────────────────┘
```

### RAGAS Metrics (Most Common)

| Metric | What It Measures | Formula / Approach | Target |
|--------|-----------------|-------------------|--------|
| **Faithfulness** | Is the answer grounded in retrieved context? No hallucinations? | LLM judges if each claim in the answer is supported by context | > 0.9 |
| **Answer Relevancy** | Does the answer actually address the question? | LLM generates back-questions from answer, measures similarity to original | > 0.85 |
| **Context Precision** | What fraction of retrieved context is actually relevant? | Ground-truth verified: relevant chunks / total retrieved chunks | > 0.8 |
| **Context Recall** | Were all ground-truth relevant docs retrieved? | Ground-truth verified: retrieved relevant / all relevant | > 0.8 |
| **Answer Correctness** | Is the answer factually correct? | F1 between answer and ground truth (factual + semantic) | Task-specific |
| **Context Entity Recall** | Are all key entities from ground truth present in context? | Entity overlap between context and reference | > 0.7 |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

# Evaluation dataset format
data = {
    "question": [
        "What is our Q4 revenue?",
        "Who is the CEO of the company?",
    ],
    "answer": [
        "Q4 revenue was $4.2M, up 23% YoY.",
        "Jane Smith has been CEO since 2023.",
    ],
    "contexts": [
        ["From Q4 2025 report: Total revenue reached $4.2M, a 23% increase year-over-year."],
        ["Leadership page: Jane Smith joined as CEO in January 2023."],
    ],
    "ground_truth": [
        "Q4 2025 revenue was $4.2M, 23% YoY growth.",
        "Jane Smith is the CEO.",
    ],
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
)
print(results)
# Output: {'faithfulness': 0.92, 'answer_relevancy': 0.88, ...}
```

### Retrieval-Only Evaluation

```python
# Evaluate just the retrieval component using standard IR metrics
def evaluate_retrieval(questions, relevant_doc_ids, retrieved_doc_ids_list, k=5):
    """
    questions: list of questions
    relevant_doc_ids: dict of {question: [relevant_doc_id, ...]}
    retrieved_doc_ids_list: list of [retrieved_doc_id, ...] per question
    """
    hit_rate = 0
    mrr_total = 0
    precision_total = 0

    for q, retrieved in zip(questions, retrieved_doc_ids_list):
        relevant = set(relevant_doc_ids[q])
        retrieved_k = retrieved[:k]

        # Hit Rate@k: Did at least one relevant doc appear in top-k?
        hit = any(doc_id in relevant for doc_id in retrieved_k)
        hit_rate += int(hit)

        # MRR (Mean Reciprocal Rank): Rank of first relevant result
        for rank, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant:
                mrr_total += 1 / rank
                break

        # Precision@k: fraction of retrieved that are relevant
        relevant_retrieved = sum(1 for d in retrieved_k if d in relevant)
        precision_total += relevant_retrieved / k

    n = len(questions)
    return {
        f"hit_rate@{k}": hit_rate / n,
        "mrr": mrr_total / n,
        f"precision@{k}": precision_total / n,
    }
```

### Building an Evaluation Dataset

```python
# Option 1: LLM-generated synthetic QA pairs (RAGAS TestsetGenerator)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./docs", glob="**/*.md")
documents = loader.load()

generator = TestsetGenerator.with_anthropic(
    generator_llm=ChatAnthropic(model="claude-sonnet-4-6"),
    critic_llm=ChatAnthropic(model="claude-sonnet-4-6"),
)

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=50,
    distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2}
)
testset.to_pandas().to_csv("rag_eval_dataset.csv", index=False)
```

### TruLens Evaluation (Alternative to RAGAS)

```python
from trulens_eval import Tru, TruChain, Feedback
from trulens_eval.feedback.provider import Anthropic as TruAnthropic

tru = Tru()
provider = TruAnthropic(model_engine="claude-sonnet-4-6")

# Define feedback functions
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruChain.Select.RecordCalls.retriever.get_relevant_documents.rets.page_content[:].collect())
    .on_output()
    .aggregate(provider.grounded_statements_aggregator)
)

f_qa_relevance = Feedback(provider.relevance, name="QA Relevance").on_input_output()

tru_rag = TruChain(
    rag_chain,
    app_id="rag-v1",
    feedbacks=[f_groundedness, f_qa_relevance]
)

# Run and view results
with tru_rag as recording:
    rag_chain.invoke("What is our Q4 revenue?")

tru.get_leaderboard(app_ids=["rag-v1"])
```

### Continuous Evaluation in Production

```python
# Sample a percentage of queries for automated evaluation
import random

class ProductionRAGMonitor:
    def __init__(self, rag_system, eval_sample_rate=0.05):
        self.rag = rag_system
        self.sample_rate = eval_sample_rate
        self.eval_buffer = []

    def query(self, question: str) -> str:
        answer = self.rag.query(question)

        # Sample for evaluation
        if random.random() < self.sample_rate:
            self.eval_buffer.append({
                "question": question,
                "answer": answer,
                "contexts": self.rag.last_retrieved_contexts,
                "timestamp": datetime.now().isoformat()
            })

        return answer

    def flush_evals(self):
        """Periodically run RAGAS on the buffer and log to monitoring system."""
        if len(self.eval_buffer) >= 50:
            results = evaluate(
                Dataset.from_list(self.eval_buffer),
                metrics=[faithfulness, answer_relevancy]
            )
            # Log to Datadog, Grafana, etc.
            log_metrics(results)
            self.eval_buffer.clear()
```

### Evaluation Best Practices

| Practice | Why It Matters |
|----------|---------------|
| **Separate retrieval and generation evals** | Isolate where failures occur (bad retrieval vs bad generation) |
| **Use ground-truth datasets for precision/recall** | LLM-only metrics can miss factual correctness |
| **Human eval for a sample** | Automated metrics miss nuance; validate with human judges |
| **A/B test retrieval strategies** | Compare embedding models, chunk sizes, top-k values |
| **Track metrics over time** | Catch embedding drift and dataset staleness |
| **Test adversarial queries** | Off-topic, ambiguous, or unanswerable questions |

---

## End-to-End Example

```python
import chromadb
import anthropic
import os
from pathlib import Path

client = anthropic.Anthropic()
chroma_client = chromadb.PersistentClient("./rag_db")
collection = chroma_client.get_or_create_collection("docs")

class RAGSystem:
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self._index_documents()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import voyageai
        voyage = voyageai.Client()
        return voyage.embed(texts, model="voyage-3", input_type="document").embeddings

    def _index_documents(self):
        """Load and index all documents."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

        all_chunks = []
        all_metadata = []

        for file in self.docs_dir.glob("**/*.md"):
            text = file.read_text()
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": str(file), "type": "markdown"}] * len(chunks))

        if all_chunks:
            embeddings = self._embed(all_chunks)
            collection.add(
                documents=all_chunks,
                embeddings=embeddings,
                ids=[f"doc_{i}" for i in range(len(all_chunks))],
                metadatas=all_metadata
            )
            print(f"Indexed {len(all_chunks)} chunks from {self.docs_dir}")

    def query(self, question: str) -> str:
        # Retrieve
        q_emb = self._embed([question])[0]
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        docs = results["documents"][0]

        # Generate
        context = "\n---\n".join(docs)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="Answer questions based on the provided context. Be concise and cite sources.",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )
        return response.content[0].text

# Usage
rag = RAGSystem("./docs")
print(rag.query("How do I set up dbt incrementally?"))
```

---

## RAG in 2026

### Ecosystem

| Tool | Role |
|------|------|
| **Voyage AI** | Anthropic's embedding models (best-in-class for retrieval) |
| **LangChain** | RAG pipelines, document loaders, retrievers |
| **LlamaIndex** | Advanced RAG, knowledge graphs, query engines |
| **RAGAS** | RAG evaluation framework |
| **Weaviate** | Vector DB with built-in hybrid search |
| **Pinecone** | Managed vector DB |
| **pgvector** | PostgreSQL extension for vectors |
| **Chroma** | Open-source, developer-friendly vector DB |
| **Qdrant** | Rust-based high-performance vector DB |
| **Cohere Rerank** | Best-in-class reranking API |

### Trends in 2026

| Trend | Description |
|-------|-------------|
| **Graph RAG** | Build knowledge graphs from docs for better multi-hop reasoning |
| **Multimodal RAG** | Retrieve images, charts, tables alongside text |
| **Streaming RAG** | Stream context from DB as generation happens |
| **Corrective RAG** | Agent validates retrieved context relevance, retries if needed |
| **Self-RAG** | Model decides when to retrieve, what to retrieve, and validates answers |
| **Long-context alternatives** | For smaller corpora, just stuff everything in 200K context |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Vector Databases](./intro_vector_databases.md) | RAG depends on vector DBs for semantic retrieval |
| [LLMOps](./intro_llmops.md) | Monitoring, evaluation, and deployment of RAG pipelines |
| [Agentic AI](./intro_agentic_ai.md) | Agentic RAG extends retrieval into multi-step reasoning |
| [LangChain](./intro_langchain.md) | Primary framework for building RAG pipelines |
| [Anthropic Overview](./intro_anthropic.md) | Claude models as the generation backbone in RAG |
| [MLflow (MLOps)](../mlops/intro_mlflow.md) | Experiment tracking for RAG evaluation runs |
