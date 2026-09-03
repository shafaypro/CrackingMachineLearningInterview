# Embeddings

Embeddings turn text, images, and other data into dense vectors where geometric closeness means semantic similarity. They are the substrate under RAG, semantic search, recommendation, clustering, and deduplication — and the layer where most retrieval quality problems actually live. Interviewers ask about them because getting embeddings wrong silently degrades everything downstream.

---

## Table of Contents
1. [What an Embedding Is](#what-an-embedding-is)
2. [From Word2Vec to Modern Text Embeddings](#from-word2vec-to-modern-text-embeddings)
3. [Similarity Metrics](#similarity-metrics)
4. [Choosing an Embedding Model](#choosing-an-embedding-model)
5. [Chunking for Embeddings](#chunking-for-embeddings)
6. [Fine-Tuning Embeddings](#fine-tuning-embeddings)
7. [Dimensionality, Quantization, and Cost](#dimensionality-quantization-and-cost)
8. [Evaluating Embeddings](#evaluating-embeddings)
9. [Operational Concerns](#operational-concerns)
10. [Interview Q&A](#interview-qa)
11. [Common Pitfalls](#common-pitfalls)
12. [Related Topics](#related-topics)

---

## What an Embedding Is

An embedding is a learned map from an object to a vector in `R^d` such that semantically related objects land close together. "Learned" is the operative word: the geometry comes from a training objective, and the objective determines what "close" means.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
vecs = model.encode(
    ["How do I reset my password?",
     "I forgot my login credentials",
     "What is the refund policy?"],
    normalize_embeddings=True,     # unit vectors → dot product == cosine similarity
)
print(vecs.shape)                  # (3, 768)
print(vecs[0] @ vecs[1])           # ~0.85 — same intent
print(vecs[0] @ vecs[2])           # ~0.35 — unrelated
```

Crucially, embedding similarity is **topical/semantic relatedness**, not entailment or truth. "The drug is effective" and "The drug is not effective" embed very close together — they share nearly all their content words and topic. Any system that needs to distinguish those needs a reranker or an NLI model, not cosine similarity.

---

## From Word2Vec to Modern Text Embeddings

| Era | Approach | Key property | Limitation |
|---|---|---|---|
| 2013 | **Word2Vec / GloVe** | One static vector per word; `king - man + woman ≈ queen` | No context — "bank" has one vector |
| 2018 | **ELMo / BERT (contextual)** | Vector depends on the sentence | Raw BERT `[CLS]` is a poor sentence embedding |
| 2019 | **Sentence-BERT** | Siamese training with a similarity objective | Needs labeled pairs |
| 2021+ | **Contrastive at scale** (E5, BGE, GTE) | Trained on massive (query, positive, negatives) triples | Task-prefix conventions vary by model |
| 2023+ | **LLM-based embedders** (`text-embedding-3`, Voyage, Cohere v3) | Strong multilingual, long inputs, Matryoshka dimensions | API cost, vendor lock-in |

**Why raw BERT embeddings are bad for similarity**: BERT is trained on masked-token prediction, not on making sentence vectors comparable. Mean-pooled BERT vectors occupy a narrow cone in which almost every pair scores above 0.8 cosine — the space is anisotropic, so the numbers carry little discriminative signal. Sentence-BERT fixed this by fine-tuning with a contrastive/triplet objective so distance actually corresponds to semantic difference.

**Contrastive learning**, the modern recipe: pull a query and its relevant document together while pushing apart in-batch negatives, typically with InfoNCE loss:

```
L = -log( exp(sim(q, d⁺)/τ) / Σ_j exp(sim(q, d_j)/τ) )
```

Large batches matter here because every other item in the batch acts as a negative — more negatives per step means a sharper decision boundary. **Hard negative mining** (retrieving plausible-but-wrong documents to include as negatives) is the single biggest quality lever when fine-tuning.

---

## Similarity Metrics

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **Cosine** | `a·b / (‖a‖‖b‖)` | [-1, 1] | Ignores magnitude; the default for text |
| **Dot product** | `a·b` | Unbounded | Equals cosine when vectors are normalized; magnitude can encode popularity |
| **Euclidean (L2)** | `‖a-b‖` | [0, ∞) | Monotonically equivalent to cosine *for normalized vectors* |

For unit-normalized vectors, `‖a-b‖² = 2 - 2·cos(a,b)`, so cosine, dot product, and L2 rank identically. Once vectors are not normalized they diverge, and mismatching the metric to how the model was trained is a real and common bug: if a model was trained with cosine similarity and you query a dot-product index of unnormalized vectors, long documents win purely for being long.

**Rule**: normalize at write time and at query time, then use dot product — it's the fastest and provably equals cosine.

---

## Choosing an Embedding Model

| Consideration | Question to ask |
|---|---|
| **Quality** | Where does it sit on MTEB *for my task type* (retrieval ≠ classification ≠ STS)? |
| **Dimension** | 384 / 768 / 1024 / 3072 — storage and search cost scale linearly |
| **Max sequence length** | 512 tokens truncates most documents; 8k models change chunking strategy |
| **Domain** | Legal, medical, code — a domain model often beats a bigger general one |
| **Multilingual** | Does it share a space across languages, so a French query retrieves English docs? |
| **Hosting** | API (no ops, per-token cost, data leaves your network) vs self-hosted (GPU, control) |
| **Asymmetric support** | Does it need query/document prefixes, and are you applying them? |

```python
# Many modern models are ASYMMETRIC and expect instruction prefixes.
# Omitting them is a top-3 cause of "our retrieval is mysteriously mediocre".
query_vec = model.encode("query: how do I reset my password", normalize_embeddings=True)
doc_vecs  = model.encode(["passage: To reset your password, go to..."], normalize_embeddings=True)
```

`e5` and `bge` families use prefixes like `query:` / `passage:` (or a BGE query instruction). Check the model card — the same model can lose 5–10 points of recall@10 when prefixes are applied inconsistently between indexing and querying.

**MTEB caveat for interviews**: the leaderboard is useful for a shortlist, not a decision. Models are sometimes tuned on tasks close to the benchmark, and your corpus is not MTEB. Always run a small internal eval set (100–300 real queries with known-relevant documents) before committing — the ranking often changes.

---

## Chunking for Embeddings

A chunk is what gets embedded, retrieved, and shown to the model, so the chunking strategy is a retrieval-quality decision, not preprocessing trivia.

| Strategy | How | Best for |
|---|---|---|
| **Fixed-size + overlap** | N tokens, 10–20% overlap | Baseline; simple and surprisingly hard to beat |
| **Recursive/structural** | Split on `\n\n`, then `\n`, then sentences | Prose documents with structure |
| **Document-aware** | Split on markdown headers, code functions, table rows | Technical docs, code, structured content |
| **Semantic** | Split where consecutive-sentence similarity drops | Long unstructured text; costs an extra embedding pass |
| **Parent-document / small-to-big** | Embed small chunks, return the enclosing parent | Best of both: precise retrieval, complete context |
| **Contextual retrieval** | Prepend an LLM-generated summary of the document to each chunk | Highest quality; costs one LLM call per chunk at ingest |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,          # keeps sentences spanning a boundary retrievable
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],  # structure first
    length_function=len,
)
chunks = splitter.split_text(document)
```

The tradeoff, stated plainly: **small chunks give precise retrieval and poor context; large chunks give rich context and diluted embeddings.** A 2,000-token chunk covering five topics produces an average vector matching none of them well. Small-to-big retrieval resolves the tension — index 300-token chunks for matching, return the 1,500-token parent section for generation.

Always carry metadata on the chunk (source, section, timestamp, permissions, document ID). Metadata filters are usually a bigger accuracy win than any embedding upgrade, and access control depends on them.

---

## Fine-Tuning Embeddings

Worth doing when your domain vocabulary differs from web text (internal product names, legal citations, medical codes, ticket jargon) or when you have real query-document click data.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# (anchor, positive) pairs — MultipleNegativesRankingLoss uses the rest of the batch as negatives
train_examples = [
    InputExample(texts=["how to reset password", "Password reset: visit Settings > Security..."]),
    InputExample(texts=["refund window", "Refunds are accepted within 30 days of purchase..."]),
]
loader = DataLoader(train_examples, shuffle=True, batch_size=64)   # larger batch = more negatives
loss = losses.MultipleNegativesRankingLoss(model)

model.fit(train_objectives=[(loader, loss)], epochs=3, warmup_steps=100)
```

Where to get pairs without a labeling budget:
- **Click logs** — (query, clicked document) is a positive; log them from day one.
- **Synthetic queries** — have an LLM generate 3–5 questions each chunk answers, then train query → chunk.
- **Existing structure** — FAQ question/answer pairs, ticket title/resolution, doc title/body.

**Hard negatives** are the difference between a small gain and a large one: retrieve the top-20 for each query with the current model, drop the true positive, and use the remaining plausible-but-wrong documents as explicit negatives. Random negatives are too easy — the model learns to separate "unrelated topic", which it already could.

Critical operational point: **fine-tuning changes the vector space, so the entire index must be re-embedded.** Plan for a dual-write or blue/green reindex, and never mix vectors from two model versions in one index — the geometry is incomparable and results will be silently wrong.

---

## Dimensionality, Quantization, and Cost

Storage for `N` vectors of dimension `d` in float32 is `N × d × 4` bytes, plus index overhead.

| Corpus | Dim | float32 | int8 | binary |
|---|---|---|---|---|
| 1M chunks | 768 | 2.9 GB | 0.73 GB | 92 MB |
| 10M chunks | 1536 | 59 GB | 15 GB | 1.8 GB |
| 100M chunks | 1024 | 390 GB | 98 GB | 12 GB |

**Matryoshka Representation Learning (MRL)** trains a model so that truncating the vector still works: dimensions are ordered by importance, so the first 256 of a 1536-dim OpenAI `text-embedding-3` vector retain most of the quality. This makes a two-stage search practical — retrieve with truncated vectors, rescore the top candidates with full vectors.

```python
# Adaptive retrieval: cheap wide pass, then exact rescoring
candidates = index_256d.search(query_256, top_k=200)     # fast, small memory footprint
final = rescore_with_full_vectors(query_1536, candidates)[:10]
```

**Quantization** trades a little recall for large savings: int8 (scalar quantization) typically loses ~1–2% recall for 4x compression; binary quantization loses more but is 32x smaller and enables Hamming-distance search, usually paired with a float rescoring pass over the top few hundred candidates.

---

## Evaluating Embeddings

Never evaluate an embedding model by eyeballing similarity scores. Build a small labeled set and measure retrieval directly.

| Metric | What it tells you |
|---|---|
| **Recall@k** | Did the right document make the candidate set? The ceiling for everything downstream |
| **MRR** | How high the first correct result ranks |
| **NDCG@k** | Ranking quality with graded relevance |
| **Latency p95** | Whether it's usable |

```python
def recall_at_k(retriever, eval_set, k=10):
    """eval_set: list of (query, set_of_relevant_doc_ids)"""
    hits = 0
    for query, relevant in eval_set:
        retrieved = {d.id for d in retriever.search(query, top_k=k)}
        hits += len(retrieved & relevant) > 0
    return hits / len(eval_set)
```

The diagnostic that matters in a RAG debugging interview: **measure retrieval and generation separately.** If recall@10 is 0.55, no amount of prompt engineering will fix the answers — the evidence isn't in the context. Fix retrieval first (hybrid search, reranking, better chunking, metadata filters), then look at generation.

---

## Operational Concerns

- **Versioning.** Store the model name and version alongside every vector. When you upgrade, reindex into a new collection and cut over atomically; mixing spaces produces plausible-looking garbage that no test catches.
- **Incremental updates.** Deletes and updates must propagate to the index, or the system confidently cites documents that no longer exist. Tie index mutations to the source-of-truth change stream.
- **Caching.** Embedding the same text repeatedly is pure waste — cache by content hash. Query embeddings for frequent queries are worth caching too.
- **Batching.** Encoding throughput is dominated by batch size; encode in batches of 32–256 rather than one at a time. This is often a 10x+ ingest speedup.
- **Access control.** Filter by permission metadata *inside* the vector search, not after — post-filtering can return an empty page when a user lacks access to the top-k, and silently leaks the existence of documents through result counts.
- **Multi-tenancy.** Either separate collections per tenant (clean isolation, more overhead) or a mandatory `tenant_id` filter (cheaper, one bug away from a data leak). For regulated data, choose separation.

---

## Interview Q&A

#### What is an embedding and why does semantic search need one?

An embedding maps text to a dense vector such that semantically similar text lands nearby in the space. Keyword search matches surface tokens, so "how do I get my money back" fails to retrieve a document titled "Refund Policy" — no shared terms. Embeddings encode meaning, so the two land close and retrieval succeeds.

The tradeoff runs the other way too: embeddings are weak at exact matching — product SKUs, error codes, function names, rare proper nouns — because those carry little distributional meaning. That's why production systems use hybrid retrieval rather than choosing one.

#### Cosine similarity vs dot product vs Euclidean distance — which and why?

For unit-normalized vectors all three produce the same ranking, since `‖a-b‖² = 2 - 2cos(a,b)`. So the practical answer is: normalize at index and query time, then use dot product, which is the cheapest to compute and hardware-accelerated everywhere.

The choice matters when vectors are *not* normalized. Then dot product rewards magnitude, which for text embeddings usually just means document length — an artifact, not relevance. The other case is recommendation systems, where magnitude is sometimes deliberately trained to encode popularity, and dot product is then the correct metric.

#### How do you choose a chunk size?

I'd start from the retrieval unit, not a number: a chunk should be the smallest span that fully answers a typical question. Then I'd measure. Set up 100 real queries with known-good answers, sweep chunk sizes (256/512/1024 tokens) with 10–20% overlap, and compare recall@10.

Structure beats size where it exists — splitting on markdown headers or function boundaries preserves coherence far better than a token count. When both context and precision are needed, I'd use small-to-big: embed small chunks for matching, return the parent section for generation. Empirically 300–800 tokens is the usual sweet spot for prose.

#### Your RAG system gives wrong answers. How do you tell if it's retrieval or generation?

Measure them separately, which requires an eval set of questions with the known-relevant chunk IDs.

First, compute recall@k on the retriever. If the correct chunk isn't being retrieved, generation is irrelevant — fix retrieval: check embedding model and prefix conventions, chunk size, add hybrid search and a reranker, verify metadata filters aren't excluding documents.

If recall is high, feed the *gold* chunks directly to the LLM and see whether the answer is correct. If it is, retrieval ranks poorly (right document, low position — add a reranker). If the answer is still wrong with perfect context, it's a generation problem: prompt structure, context ordering, model capability, or the model ignoring the provided evidence.

#### When would you fine-tune an embedding model instead of using an off-the-shelf one?

When the domain vocabulary is far from web text — internal product codenames, legal citation formats, medical abbreviations, industrial part numbers — a general model treats those tokens as near-noise. Also when I have real query-document relevance data from click logs, which is essentially free supervision.

I'd first exhaust cheaper options: a domain-specific off-the-shelf model, hybrid search with BM25 (which handles the exact-match problem fine-tuning is often chasing), and a cross-encoder reranker, which typically buys more than fine-tuning the bi-encoder for far less effort. If I do fine-tune, hard negative mining is where the gains come from, and I'd budget for a full reindex on every model version.

#### Why can't you mix vectors from two embedding models in one index?

Each model defines its own coordinate system; the axes have no shared meaning. Two vectors of the same dimension from different models are not comparable, so cosine similarity between them is a meaningless number — and it will still return results, ranked by nothing. There's no error, just silently wrong retrieval, which makes it a nasty production bug.

The operational consequence: any embedding model change requires re-embedding the whole corpus. Standard practice is a blue/green reindex into a new collection with the model version recorded in metadata, then an atomic cutover.

#### How do you handle documents longer than the model's context window?

Chunk them, since truncation silently discards content that then can never be retrieved. Beyond that:
- **Small-to-big**: embed sub-chunks, return the parent for generation.
- **Hierarchical indexing**: embed a document-level summary as well as chunks, retrieve at the document level first and then within it.
- **Late chunking** with long-context embedders: run the whole document through the model, then pool per chunk, so each chunk vector carries document-wide context.
- **Contextual retrieval**: prepend a short LLM-written description of the document to each chunk before embedding — expensive at ingest, and among the largest reported recall gains.

#### What is hybrid search and why does it beat pure vector search?

Hybrid search runs lexical retrieval (BM25) and dense vector retrieval in parallel and fuses the result lists, commonly with Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank_i(d))`, with `k≈60`.

It wins because the two methods fail in complementary ways. Vectors handle paraphrase and synonymy but blur exact identifiers — error codes, SKUs, function names, rare names. BM25 nails exact terms but returns nothing for a paraphrase with no shared vocabulary. RRF needs no score calibration between the two systems, since it fuses ranks rather than raw scores, which is why it's the common default.

#### How would you reduce embedding storage and search cost for 100M documents?

Layered, from cheapest to most invasive:
1. **Smaller dimension** — a 768-dim model instead of 3072 is a 4x cut and often loses very little; with a Matryoshka model, truncation is free.
2. **Scalar (int8) quantization** — 4x smaller for roughly 1–2% recall loss.
3. **Binary quantization + rescoring** — 32x smaller, Hamming search, then rescore the top few hundred with full vectors to recover most of the accuracy.
4. **ANN index tuning** — HNSW's `M` and `ef_construction` trade memory and build time for recall; IVF-PQ is far more memory-efficient at large scale.
5. **Tiered storage** — keep hot partitions in memory and cold ones on disk, partitioned by tenant or recency.
6. **Deduplicate** — near-duplicate chunks are common in real corpora and waste both storage and result slots.

I'd measure recall@10 at each step against a fixed eval set rather than assuming the published numbers transfer.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Different embedding models for indexing and querying | Vectors are not comparable; results are noise | Pin the model version in metadata; reindex on change |
| Omitting `query:` / `passage:` prefixes | Asymmetric models lose significant recall | Follow the model card exactly on both paths |
| Unnormalized vectors with dot-product search | Long documents win on magnitude, not relevance | Normalize at write and query time |
| Chunks too large | The vector averages several topics and matches none | 300–800 tokens, or small-to-big retrieval |
| No overlap between chunks | Answers spanning a boundary become unretrievable | 10–20% overlap |
| Evaluating by eyeballing similarity scores | Scores aren't comparable across models or queries | Recall@k / NDCG on a labeled eval set |
| Pure vector search for identifiers | Exact codes and names get blurred | Hybrid BM25 + vector with RRF |
| Post-filtering by permissions | Empty pages and leaked existence of documents | Filter inside the vector search |
| Forgetting deletes and updates | The system cites documents that no longer exist | Drive index mutations from the source change stream |
| Assuming the MTEB leader is best for you | Benchmarks aren't your corpus | Run an internal eval set before committing |

---

## Related Topics

- [Intro to RAG](./intro_rag.md)
- [RAG Engineering](./intro_rag_engineering.md)
- [Vector Databases](./intro_vector_databases.md)
- [Vector Databases — Advanced](./intro_vector_databases_advanced.md)
- [LLM Fundamentals](./intro_llm_fundamentals.md)
- [Dimensionality Reduction](../classical_ml/intro_dimensionality_reduction.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
