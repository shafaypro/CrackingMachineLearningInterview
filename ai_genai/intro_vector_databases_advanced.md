# Vector Databases & Advanced RAG — 2026 Production Guide

## Vector Database Comparison

| Feature | Pinecone | Weaviate | FAISS | pgvector | Chroma |
|---------|----------|----------|-------|----------|--------|
| **Type** | Managed cloud | Self-hosted / cloud | Library | Postgres extension | Library / cloud |
| **Scale** | 100M+ vectors | 100M+ vectors | Billions (GPU) | 10M-50M vectors | Prototype-scale |
| **Hybrid search** | Yes (sparse+dense) | Yes (BM25+dense) | No (dense only) | Partial | No |
| **Metadata filtering** | Yes | Yes (GraphQL) | No | Yes (SQL) | Yes |
| **Replication** | Managed | Manual / cloud | No | Postgres replication | No |
| **Cost** | $$$-$$$$ | $$-$$$ | Free (self-hosted) | $ (Postgres cost) | Free / $ |
| **Best for** | Production SaaS | Hybrid search, rich schema | Research, on-prem | Existing Postgres apps | Prototyping |

---

## Pinecone

### Architecture

```
Your App → Pinecone API → Pinecone Pod/Serverless
                              ├── Index (namespace)
                              │   ├── Vectors (id, values, metadata)
                              │   └── Namespaces (multi-tenant isolation)
                              └── Replicas (HA)
```

### Key Concepts

- **Serverless**: Pay per query/storage, no pod management
- **Pod-based**: Dedicated compute, predictable latency, better for high-QPS
- **Namespaces**: Logical partitions within an index (e.g., per user, per document set)
- **Metadata filtering**: Pre-filter before ANN search (reduces candidate set)

### Code Example

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create serverless index
pc.create_index(
    name="ml-knowledge-base",
    dimension=1536,  # OpenAI text-embedding-3-small
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("ml-knowledge-base")

# Upsert vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = ["RAG combines retrieval with generation", "Fine-tuning updates model weights"]
vectors = embeddings.embed_documents(texts)

index.upsert(vectors=[
    {"id": "doc-1", "values": vectors[0], "metadata": {"topic": "rag", "source": "guide"}},
    {"id": "doc-2", "values": vectors[1], "metadata": {"topic": "finetuning", "source": "guide"}},
], namespace="ml-guide")

# Query with metadata filter
query_vector = embeddings.embed_query("How does RAG work?")
results = index.query(
    vector=query_vector,
    top_k=5,
    namespace="ml-guide",
    filter={"topic": {"$eq": "rag"}},  # Metadata pre-filter
    include_metadata=True
)

for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | {match['metadata']}")
```

### Pinecone Pitfalls

1. **Cold start**: Serverless indexes have cold start latency (100-500ms). Use pod-based for latency-sensitive apps.
2. **Metadata bloat**: Storing large metadata slows queries. Keep metadata small; store full documents in a separate DB.
3. **Namespace isolation**: Data in different namespaces is NOT shared. Use namespaces for multi-tenancy, not topic grouping.
4. **Dimension lock**: Index dimension cannot be changed after creation. Plan embedding model choice upfront.

---

## Weaviate

### Architecture

```
Weaviate Instance
├── Schema (Classes = like tables)
│   ├── Class: Document
│   │   ├── Properties: title, content, source
│   │   └── Vectorizer: text2vec-openai
│   └── Class: Chunk
│       └── Reference: belongsTo → Document
├── Modules
│   ├── text2vec-openai (auto-vectorize on insert)
│   ├── reranker-cohere (cross-encoder reranking)
│   └── generative-anthropic (RAG generation)
└── HNSW Index (per class)
```

### Code Example

```python
import weaviate
import weaviate.classes as wvc

# Connect
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="your-cluster.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("your-api-key")
)

# Create collection
client.collections.create(
    name="MLConcept",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    generative_config=wvc.config.Configure.Generative.anthropic(
        model="claude-sonnet-4-6"
    ),
    properties=[
        wvc.config.Property(name="concept", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="difficulty", data_type=wvc.config.DataType.INT),
    ]
)

collection = client.collections.get("MLConcept")

# Insert (auto-vectorized)
collection.data.insert_many([
    {"concept": "Attention mechanism scales quadratically with sequence length",
     "category": "transformers", "difficulty": 3},
    {"concept": "RLHF uses human preferences to align LLMs",
     "category": "alignment", "difficulty": 4},
])

# Hybrid search (BM25 + dense vectors)
results = collection.query.hybrid(
    query="how does attention work",
    alpha=0.5,  # 0=pure BM25, 1=pure vector, 0.5=balanced
    limit=5,
    filters=wvc.query.Filter.by_property("difficulty").less_than(4)
)

# RAG with generative module
rag_results = collection.generate.hybrid(
    query="explain attention mechanism",
    grouped_task="Summarize these concepts for a senior ML engineer",
    alpha=0.75,
    limit=3
)
print(rag_results.generated)

client.close()
```

### Weaviate Hybrid Search

```
Query: "attention mechanism transformers"
         │
    ┌────┴────────────┐
    │                 │
 BM25 (keyword)   Vector (semantic)
    │                 │
 "attention"      cos_sim(q, d)
 "mechanism"          │
    │                 │
    └────────┬────────┘
             │
         Fusion (RRF or weighted alpha)
             │
         Reranker (cross-encoder)
             │
         Top-K results
```

---

## FAISS (Facebook AI Similarity Search)

### Index Types

| Index | Description | Speed | Memory | Accuracy | Best For |
|-------|-------------|-------|--------|----------|----------|
| `IndexFlatL2` | Exact brute-force L2 | Slow | High | 100% | Ground truth, small datasets |
| `IndexFlatIP` | Exact inner product | Slow | High | 100% | Cosine (after normalization) |
| `IndexIVFFlat` | Inverted file, exact cells | Fast | Medium | ~95% | Medium scale (1M-10M) |
| `IndexIVFPQ` | IVF + Product Quantization | Fastest | Low | ~90% | Large scale, memory-constrained |
| `IndexHNSW` | Hierarchical NSW graph | Very fast | Medium-High | ~98% | Production, latency-critical |

### Code Example

```python
import faiss
import numpy as np
from langchain_anthropic import ChatAnthropic

# Prepare vectors
d = 1536  # Embedding dimension
n = 100000  # Number of vectors

# Normalize for cosine similarity (FAISS uses inner product)
vectors = np.random.random((n, d)).astype('float32')
faiss.normalize_L2(vectors)

# Build HNSW index (best for production)
M = 32  # Connections per node (higher = more accurate, more memory)
ef_construction = 200  # Build-time search depth
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 100  # Query-time search depth (tune for accuracy/speed)
index.add(vectors)

# Save/load
faiss.write_index(index, "ml_knowledge.faiss")
index = faiss.read_index("ml_knowledge.faiss")

# Query
query = np.random.random((1, d)).astype('float32')
faiss.normalize_L2(query)

distances, indices = index.search(query, k=10)  # Top-10 results
print(f"Nearest neighbors: {indices[0]}")
print(f"Similarity scores: {distances[0]}")  # 1.0 = identical

# GPU acceleration
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    distances, indices = gpu_index.search(query, k=10)
```

### FAISS IVF for Large Scale

```python
# IVF: Partition vectors into cells, search only top-N cells
nlist = 1000  # Number of Voronoi cells
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Must train before adding vectors
train_vectors = vectors[:50000]  # 10-20x nlist samples recommended
index.train(train_vectors)
index.add(vectors)

index.nprobe = 50  # Check 50 cells (higher = more accurate, slower)
D, I = index.search(query, k=10)
```

---

## pgvector

### Setup and Usage

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE ml_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI dimension
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index
CREATE INDEX ON ml_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Or IVFFlat index
CREATE INDEX ON ml_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Similarity search
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM ml_embeddings
WHERE metadata->>'source' = 'ml-guide'  -- Metadata filter (uses regular index)
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

```python
# Python with SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text, Index
from sqlalchemy.orm import DeclarativeBase
import numpy as np

class Base(DeclarativeBase):
    pass

class MLEmbedding(Base):
    __tablename__ = "ml_embeddings"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(1536))

# pgvector operators:
# <->  Euclidean distance
# <#>  Negative inner product
# <=>  Cosine distance

results = session.query(MLEmbedding).order_by(
    MLEmbedding.embedding.cosine_distance(query_embedding)
).limit(5).all()
```

**When to use pgvector**: You already have Postgres, dataset < 10M vectors, need complex SQL joins with metadata, don't want another service to manage.

---

## Chunking Strategies

### Fixed-Size Chunking

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=512,       # Characters per chunk
    chunk_overlap=50,     # Overlap to preserve context
    separator="\n"
)
chunks = splitter.split_text(document)
```

**Pros**: Simple, predictable. **Cons**: Cuts mid-sentence, loses context.

### Recursive Chunking (Recommended Default)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try each separator in order
)
chunks = splitter.split_text(document)
```

**Pros**: Respects paragraph/sentence boundaries. Better semantic coherence.

### Semantic Chunking (Best Quality)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # Split when similarity drops
    breakpoint_threshold_amount=95           # 95th percentile of similarity drops
)
chunks = splitter.split_text(long_document)
```

**Pros**: Each chunk is semantically coherent. **Cons**: Expensive (embed every sentence), variable chunk sizes.

### Document-Structure-Aware Chunking

```python
# For markdown/code: split by headers and code blocks
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_doc)
# Each chunk retains its header hierarchy as metadata
```

### Chunking Strategy Decision Guide

```
Document type?
├── Structured (markdown, code) → Document-structure-aware
├── Long-form prose (books, reports) → Semantic chunking
├── Mixed content → Recursive character splitter
└── Simple use case / prototyping → Fixed-size with overlap

Embedding model context limit?
├── < 512 tokens → chunk_size ≈ 300-400 tokens
├── 512-8192 tokens → chunk_size ≈ 500-1000 tokens
└── 8192+ tokens (Voyage, OpenAI-3) → chunk_size ≈ 1000-4000 tokens
```

---

## Hybrid Search (BM25 + Embeddings)

### Why Hybrid Search?

| Query Type | Dense Vector | BM25 | Hybrid |
|------------|-------------|------|--------|
| "explain attention mechanism" | Excellent | Good | Best |
| "BERT model F1 score 94.3" | Poor | Excellent | Best |
| "what is GPT-4" | Good | Good | Best |
| Exact product codes/IDs | Poor | Excellent | Best |

### Implementation

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma

# Dense retriever
dense_retriever = Chroma.from_documents(
    documents, embeddings
).as_retriever(search_kwargs={"k": 10})

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Ensemble with Reciprocal Rank Fusion (RRF)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]  # BM25:40%, Dense:60%
)

results = ensemble_retriever.invoke("BERT accuracy on GLUE benchmark")
```

---

## Reranking with Cross-Encoders

### Why Reranking?

Bi-encoders (embedding models) encode query and document **independently** → fast but approximate. Cross-encoders encode them **together** → slower but much more accurate.

```
Standard Retrieval:
Query Embedding ──→ [ANN Search] ──→ Top-100 docs
                                           ↓
                                     [Reranker]  (cross-encoder)
                                           ↓
                                       Top-5 docs (high precision)
```

### Implementation

```python
from sentence_transformers import CrossEncoder
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Option 1: Local cross-encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[str], top_n: int = 5) -> list:
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# Option 2: Cohere Rerank API
compressor = CohereRerank(model="rerank-v3.5", top_n=5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=dense_retriever
)
results = reranking_retriever.invoke("What is attention mechanism?")
```

---

## RAG Evaluation Metrics

### RAGAS Framework

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,     # Are retrieved docs relevant to the question?
    context_recall,        # Did we retrieve all needed docs?
    faithfulness,          # Is the answer grounded in retrieved context?
    answer_relevancy,      # Does the answer address the question?
)
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": ["What is RAG?", "Explain RLHF"],
    "answer": ["RAG retrieves documents...", "RLHF uses human feedback..."],
    "contexts": [["retrieved doc 1", "retrieved doc 2"], ["retrieved doc 3"]],
    "ground_truth": ["RAG stands for...", "RLHF stands for..."]
})

results = evaluate(eval_data, metrics=[
    context_precision, context_recall, faithfulness, answer_relevancy
])
print(results.to_pandas())
```

### Key RAG Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Context Precision** | relevant retrieved / total retrieved | Retrieval precision |
| **Context Recall** | retrieved relevant / all relevant | Retrieval recall |
| **Faithfulness** | supported claims / total claims | Hallucination rate |
| **Answer Relevancy** | LLM score of query-answer alignment | Answer quality |
| **Hit Rate@K** | ∑ (relevant in top-K) / N | Basic retrieval health |
| **MRR** | ∑ 1/rank_of_first_relevant | Ranking quality |

---

## Context Window Optimization

### Problem

Stuffing too many retrieved chunks causes:
- LLM "lost in the middle" effect (ignores middle context)
- Higher cost and latency
- Lower answer quality

### Solutions

```python
# 1. Rerank and keep only top-3 instead of top-10
# 2. Contextual compression — extract only relevant sentences
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 3. Map-reduce for very long documents
from langchain.chains import MapReduceDocumentsChain
# Split across multiple LLM calls, then combine

# 4. Hierarchical retrieval — coarse then fine
summary_retriever = ...  # Retrieve document summaries
chunk_retriever = ...    # Then retrieve specific chunks from those docs
```

---

## System Design: Production RAG Pipeline

```
                        ┌─────────────────────────────────────┐
                        │         INDEXING PIPELINE            │
                        │                                      │
Documents → [Chunker] → [Embedder] → [Vector DB]              │
                │             │           │                    │
           {metadata}    (batch API)  (upsert)                │
                └─────────────────────────┘                   │
                        └─────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │         QUERY PIPELINE               │
                        │                                      │
User Query → [Query Rewriter] → [Hybrid Search] → [Reranker] │
                                  (BM25 + Dense)      │        │
                                                      ↓        │
                                               [Top-5 chunks] │
                                                      ↓        │
                                               [LLM + context]│
                                                      ↓        │
                                                  Response     │
                        └─────────────────────────────────────┘
```

---

## Interview Questions

**Q: What is hybrid search and when would you use it over pure vector search?**
> Hybrid search combines BM25 (keyword/exact match) with dense vector search (semantic similarity), fused via Reciprocal Rank Fusion. Use it when your queries include: exact IDs, product codes, names, dates, or technical terms that dense search struggles with. Pure vector search is great for semantic queries, but hybrid is almost always better.

**Q: Explain the tradeoffs between Pinecone, pgvector, and FAISS.**
> Pinecone: managed, fast, expensive, great for production SaaS. pgvector: free if you have Postgres, SQL joins, but limited scale (~10M vectors). FAISS: free, blazing fast on GPU, billions of vectors, but no built-in metadata filtering and requires custom infrastructure.

**Q: What are chunking strategies and which would you recommend for a legal document RAG system?**
> Legal documents have hierarchical structure (sections, clauses). Use document-structure-aware chunking (split by headers/numbering) with moderate overlap (100-200 tokens). Avoid fixed-size — it breaks clause boundaries. Consider semantic chunking for the final split within sections.

**Q: Why is reranking important and how does it work?**
> Bi-encoders produce approximate rankings — fast but imprecise. A cross-encoder evaluates query + document together, scoring true relevance. Reranking takes top-100 from ANN search and re-scores with a cross-encoder to get top-5 with much higher precision. The cost is worthwhile because you only rerank a small candidate set.

**Q: How do you evaluate a RAG system?**
> Use RAGAS metrics: context precision (are retrieved docs relevant?), context recall (did we retrieve all needed info?), faithfulness (is the answer grounded in context?), and answer relevancy. Create a golden dataset manually. Automate eval in CI/CD and compare scores before deploying changes to retrieval or prompts.
