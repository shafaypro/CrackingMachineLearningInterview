# Vector Databases – Complete Guide (2026 Edition)

**Vector databases** store and search high-dimensional vector embeddings — the mathematical representations of text, images, audio, and code produced by AI models. They're foundational to RAG, semantic search, recommendation systems, and AI applications.

---

## What is a Vector Database?

Traditional databases store structured data and search by exact match or range:
```sql
SELECT * FROM products WHERE price < 100;   -- exact/range
```

Vector databases store **embeddings** and search by **similarity** (nearest neighbors):
```python
# Find the 5 most semantically similar documents to a query
results = collection.query(query_embedding=embed("AI safety"), n_results=5)
```

### How Embeddings Work

```
Text → Embedding Model → Vector [0.23, -0.15, 0.87, ..., 0.04]  (1536 dimensions)
                                        ↑
                              Encodes semantic meaning
                              Similar texts → similar vectors
                              Cosine similarity measures closeness
```

---

## Similarity Metrics

| Metric | Formula | Best For |
|--------|---------|---------|
| **Cosine similarity** | cos(θ) between vectors | Text, normalized embeddings |
| **Euclidean distance** | L2 distance | Image features, dense vectors |
| **Dot product** | u · v | When vectors are normalized |
| **Manhattan distance** | L1 distance | Sparse vectors |

---

## ANN Algorithms

Exact nearest neighbor search is O(n) — too slow at scale. Vector DBs use **Approximate Nearest Neighbor (ANN)** algorithms:

| Algorithm | Description | DB |
|-----------|-------------|-----|
| **HNSW** | Hierarchical Navigable Small World — graph-based, fast queries | Most DBs |
| **IVF** | Inverted File Index — cluster-based, efficient for large datasets | Faiss, pgvector |
| **ScaNN** | Google's ANN library | Vertex AI |
| **DiskANN** | Disk-based ANN for billion-scale | Azure, Qdrant |

---

## Top Vector Databases in 2026

### 1. Pinecone (Managed, Cloud-native)

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")

# Create index
pc.create_index(
    name="my-index",
    dimension=1024,          # match your embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "doc-1", "values": [0.1, 0.2, ...], "metadata": {"text": "...", "source": "report.pdf"}},
    {"id": "doc-2", "values": [0.3, 0.4, ...], "metadata": {"text": "...", "source": "wiki.md"}},
])

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={"source": {"$eq": "report.pdf"}}  # metadata filtering
)

for match in results.matches:
    print(f"Score: {match.score:.3f} | {match.metadata['text'][:100]}")
```

**Pros:** Fully managed, scales to billions, fast. **Cons:** Paid, vendor lock-in.

---

### 2. Weaviate (Open-source, Hybrid Search Built-in)

```python
import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()   # or connect_to_weaviate_cloud()

# Create collection (schema)
client.collections.create(
    name="Document",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    generative_config=wvc.config.Configure.Generative.anthropic(),
    properties=[
        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="date", data_type=wvc.config.DataType.DATE),
    ]
)

collection = client.collections.get("Document")

# Insert objects
collection.data.insert_many([
    {"text": "dbt is a transformation tool", "source": "docs.md"},
    {"text": "Kubernetes orchestrates containers", "source": "k8s.md"},
])

# Semantic search
results = collection.query.near_text(
    query="data transformation",
    limit=3,
    return_metadata=wvc.query.MetadataQuery(score=True)
)

# Hybrid search (BM25 + vector)
results = collection.query.hybrid(
    query="data transformation",
    limit=5,
    alpha=0.75    # 0 = pure BM25, 1 = pure vector
)

# Generative search (RAG built-in!)
results = collection.generate.near_text(
    query="data transformation tools",
    limit=3,
    grouped_task="Summarize these documents in 2 sentences."
)
print(results.generated)
```

**Pros:** Hybrid search out-of-box, built-in RAG, GraphQL API. **Cons:** Complex setup.

---

### 3. Chroma (Developer-friendly, Local-first)

```python
import chromadb
from chromadb.utils import embedding_functions

# In-memory (development)
client = chromadb.Client()

# Persistent (production-lite)
client = chromadb.PersistentClient(path="./chroma_db")

# Remote (server mode)
client = chromadb.HttpClient(host="localhost", port=8000)

# Create collection with embedding function
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key="...",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Add documents (Chroma auto-embeds if you pass documents)
collection.add(
    documents=["dbt transforms data", "Kubernetes runs containers"],
    ids=["doc-1", "doc-2"],
    metadatas=[{"type": "data_eng"}, {"type": "devops"}]
)

# Query
results = collection.query(
    query_texts=["data transformation"],   # auto-embeds
    n_results=3,
    where={"type": "data_eng"},            # metadata filter
    include=["documents", "distances", "metadatas"]
)

# Or query with pre-computed embedding
results = collection.query(
    query_embeddings=[my_vector],
    n_results=5
)
```

**Pros:** Simple, local, great for development. **Cons:** Not for billion-scale production.

---

### 4. pgvector (PostgreSQL Extension)

Use your existing Postgres as a vector database. Best choice if you're already on Postgres.

```sql
-- Install extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id        BIGSERIAL PRIMARY KEY,
    content   TEXT,
    source    TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    embedding vector(1536)    -- dimension matches your model
);

-- Create index for fast ANN search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
-- OR
CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Insert with embedding
INSERT INTO documents (content, source, embedding)
VALUES ('dbt transforms data', 'docs.md', '[0.1, 0.2, ...]'::vector);

-- Semantic search (cosine similarity)
SELECT content, source, 1 - (embedding <=> query_embedding) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
-- <=> cosine distance, <-> L2 distance, <#> negative dot product

-- Hybrid search: combine with full-text search
SELECT content, source,
    ts_rank(to_tsvector(content), query) AS text_rank,
    1 - (embedding <=> $1) AS vector_rank
FROM documents, plainto_tsquery('data transformation') query
WHERE to_tsvector(content) @@ query
   OR embedding <=> $1 < 0.3
ORDER BY vector_rank + text_rank DESC
LIMIT 10;
```

```python
# Python with pgvector
from pgvector.psycopg import register_vector
import psycopg
import numpy as np

conn = psycopg.connect("postgresql://localhost/mydb")
register_vector(conn)

# Insert
embedding = np.array([0.1, 0.2, ...])
conn.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    ("my document", embedding)
)

# Query
results = conn.execute(
    "SELECT content, 1-(embedding<=>%s) as sim FROM documents ORDER BY embedding<=>%s LIMIT 5",
    (embedding, embedding)
).fetchall()
```

**Pros:** No new infrastructure, ACID, full SQL power, free. **Cons:** Not built for vector-first workloads.

---

### 5. Qdrant (Performance-focused, Rust-based)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

client = QdrantClient(":memory:")                          # in-memory
# client = QdrantClient("localhost", port=6333)           # local server
# client = QdrantClient(url="https://...", api_key="...") # cloud

# Create collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# Insert points
client.upsert(
    collection_name="docs",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"text": "dbt guide", "category": "data_eng"}
        ),
    ]
)

# Search
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    limit=5,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="data_eng"))]
    )
)

for hit in results:
    print(f"Score: {hit.score:.3f} | {hit.payload['text']}")
```

**Pros:** High performance, payload filtering, sparse vector support. **Cons:** Self-hosted complexity.

---

## Choosing a Vector Database

```
Are you already on PostgreSQL?  → pgvector
Need simplest dev setup?        → Chroma
Need managed cloud service?     → Pinecone
Need hybrid search?             → Weaviate
Need billion-scale + speed?     → Qdrant or Pinecone
Enterprise / AWS integration?   → OpenSearch (k-NN), RDS pgvector
```

| DB | Open Source | Managed Cloud | Hybrid Search | Scale |
|----|-------------|---------------|---------------|-------|
| Pinecone | ✗ | ✓ | Limited | Billions |
| Weaviate | ✓ | ✓ | ✓ (built-in) | Hundreds of millions |
| Chroma | ✓ | Limited | ✗ | Millions |
| pgvector | ✓ | ✓ (Supabase, Neon) | Manual | Millions |
| Qdrant | ✓ | ✓ | ✓ | Billions |
| Milvus | ✓ | ✓ | ✓ | Billions |

---

## Embedding Models in 2026

| Model | Provider | Dimensions | Best For |
|-------|----------|-----------|----------|
| `voyage-3` | Voyage AI (Anthropic) | 1024 | General retrieval |
| `voyage-code-3` | Voyage AI | 1024 | Code search |
| `voyage-finance-2` | Voyage AI | 1024 | Financial docs |
| `text-embedding-3-large` | OpenAI | 3072 | General, multilingual |
| `text-embedding-3-small` | OpenAI | 1536 | Fast/cheap |
| `embed-v3.0` | Cohere | 1024 | Multilingual |
| `all-MiniLM-L6-v2` | Sentence Transformers | 384 | Local, fast |
| `nomic-embed-text` | Nomic | 768 | Open-source, local |

---

## Production Patterns

### Metadata Filtering

Always filter by metadata before vector search for efficiency:

```python
# Pinecone
results = index.query(
    vector=embedding,
    top_k=10,
    filter={
        "department": {"$eq": "engineering"},
        "date": {"$gte": "2025-01-01"},
        "doc_type": {"$in": ["runbook", "architecture"]}
    }
)
```

### Chunking Strategy

```python
# Rule of thumb:
# chunk_size=512    → precise retrieval, less context
# chunk_size=1024   → balanced (recommended)
# chunk_size=2048   → more context, less precise
# chunk_overlap=10-20% of chunk_size
```

### Namespace/Tenant Isolation

```python
# Pinecone namespaces (per-customer isolation)
index.upsert(vectors=[...], namespace="customer-123")
results = index.query(vector=emb, namespace="customer-123")

# Chroma collections (one per tenant)
collection = client.get_or_create_collection(f"tenant_{customer_id}")
```
