# Visual Search System Design

"Design search by image" (or "shop the look", or "show me similar products") comes up in e-commerce, marketplace, and photo-platform interviews. It looks like a computer vision question but is mostly a retrieval question: an embedding model, an approximate nearest neighbour index over hundreds of millions of vectors, and a re-ranker. Candidates who spend the whole interview on the CNN architecture and never mention index memory, embedding versioning, or how the training pairs are collected usually do poorly.

---

## Table of Contents
1. [Clarify the Problem First](#clarify-the-problem-first)
2. [High-Level Architecture](#high-level-architecture)
3. [Query Processing and Object Detection](#query-processing-and-object-detection)
4. [Embedding Models](#embedding-models)
5. [Training Data](#training-data)
6. [ANN Indexing](#ann-indexing)
7. [Re-Ranking](#re-ranking)
8. [Near-Duplicate Detection](#near-duplicate-detection)
9. [Embedding Versioning and Backfills](#embedding-versioning-and-backfills)
10. [Evaluation](#evaluation)
11. [Serving and Latency Budget](#serving-and-latency-budget)
12. [Capacity and Cost](#capacity-and-cost)
13. [Privacy and Safety](#privacy-and-safety)
14. [Monitoring](#monitoring)
15. [Interview Q&A](#interview-qa)
16. [Common Pitfalls](#common-pitfalls)
17. [Related Topics](#related-topics)

---

## Clarify the Problem First

The answers to these questions change the model, the index, and the latency budget, so ask them before drawing anything.

**What is the query?**
- **Full image**: a user uploads a photo of a product and wants the same or similar items.
- **Cropped region**: the photo contains several objects (a street-style photo with a jacket, bag, and shoes) and the user wants results for one of them. This is "shop the look".
- **Image plus text**: "this dress but in red" or "this sofa, under $500". The text modifies the image query or adds filters.
- **Item-to-item**: "similar products" on a product page. The query is a catalog image, so it is already clean and its embedding can be precomputed.

**What counts as a good result?** Exact match (the same SKU), visually similar (same style, different brand), or complementary (items that go with it)? Exact match and similar are both retrieval problems with different training labels. Complementary items are closer to a recommendation problem and should be scoped out unless the interviewer asks for it.

**Scale numbers to ask for:**
- Catalog size: 1M items fits in one machine's memory at full precision; hundreds of millions need compression and sharding
- Images per item: products often have 5-10 photos, which multiplies the index size
- QPS: uploaded-photo queries are usually a small fraction of text search traffic; "similar items" widgets can be much higher because they fire on page views
- Latency: users tolerate more for a photo upload than for typing, but end-to-end should still be under roughly 500 ms after the upload completes
- Freshness: how quickly must a newly listed item be findable? Minutes for a marketplace, a day may be fine for a retailer with a slow catalog

**Constraints:** Must results be in stock and ship to the user's region? Are user-uploaded photos stored, and for how long? Are there images of people (faces) in queries or catalog?

For the rest of this guide, assume: **e-commerce marketplace, 200M active items with an average of 3 indexed images each (600M vectors), 2k QPS peak for photo queries plus a precomputed "similar items" widget, p99 under 400 ms server-side, new items searchable within 15 minutes, optimizing for add-to-cart and purchase.**

---

## High-Level Architecture

There are two paths. The offline path turns every catalog image into a vector and puts it in an index. The online path turns the query into a vector, looks up neighbours, and re-ranks them.

```
OFFLINE / NEAR-LINE (catalog side)
─────────────────────────────────────────────────────────────────────
Catalog DB ──► CDC stream ──► Image fetch ──► Safety + dedup filters
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Detector + embedder    │  GPU batch jobs
                       │ (same model version as │
                       │  the query encoder)    │
                       └───────────┬────────────┘
                                   ▼
             Embedding store (item_id, image_id, model_version, vector)
                                   │
                                   ▼
             Index builder ──► ANN shards (HNSW or IVF-PQ) + attribute filters

ONLINE (query side)
─────────────────────────────────────────────────────────────────────
Upload ──► Decode/resize ──► Safety check ──► Object detection
                                                   │ user picks a box
                                                   ▼
                                          Crop ──► Query encoder (GPU)
                                                   │ + optional text
                                                   ▼
                                  ANN search (all shards, filtered) ~1000
                                                   │
                                                   ▼
                              Re-ranker (heavier model + business signals) ~50
                                                   │
                                                   ▼
                              Dedup, diversity, stock filter ──► Results
```

Two rules hold the design together:

1. **The query encoder and the catalog encoder must be the same model version.** Vectors from different model versions are not comparable, even if the architectures match. Versioning is covered below.
2. **Recall at the ANN stage is the ceiling.** If the right item is not in the top 1000 candidates, the re-ranker cannot recover it.

---

## Query Processing and Object Detection

User photos are messy: cluttered backgrounds, several objects, bad lighting, odd angles. Catalog photos are usually clean product shots on a white background. That gap between query domain and catalog domain is the main modelling problem.

**Object detection.** Run a detector (a Faster R-CNN, DETR, or YOLO-family model fine-tuned on your product categories) on the query image. It returns boxes with category labels such as `dress`, `handbag`, `sneaker`. Then:
- Show the boxes to the user and let them pick one. Default to the most prominent box by area and centrality.
- Crop the chosen box with some padding and embed the crop, not the full image.
- Use the predicted category as a soft filter or re-ranking feature. A hard filter is risky because detector mistakes then remove every correct result.

**Run the detector on the catalog too.** Catalog lifestyle photos (a model wearing the jacket) contain the same clutter as user photos. Detecting and cropping the product in catalog images makes the catalog side look more like the query side, and it lets you index several crops per image.

**Preprocessing:** fix EXIF orientation, resize on the client before upload (saves bandwidth and latency), and normalize colour space. Keep resizing identical between training and serving; a mismatch in interpolation method or aspect-ratio handling is a common silent cause of lower recall.

**Text + image queries.** Two options:
- **Structured text becomes filters.** "Under $500, in red" becomes `price < 500` and a colour attribute filter. Cheap and precise.
- **Free text modifies the embedding.** With a joint image-text model (CLIP-style), you can combine the image and text embeddings, for example a weighted sum of normalized vectors, or train a small composition model that takes both and outputs a target embedding. The weighted sum is a reasonable baseline; a trained composition model does better on "this but different" queries because it learns which image features the text should override.

---

## Embedding Models

The embedding model maps an image to a vector where similar items are close under cosine similarity. It is a backbone plus a projection head, trained with a metric learning loss.

### Backbone

| Backbone | Notes |
|---|---|
| **CNN (ResNet, EfficientNet, ConvNeXt)** | Strong, cheap to serve, well understood. Good default when latency and GPU cost are tight |
| **ViT** | Better at global context and scales well with data; usually better accuracy at larger sizes |
| **Pretrained CLIP-style image tower** | Strong general features and a text tower for free; a good starting point for fine-tuning |
| **Self-supervised (DINO-style)** | Good fine-grained visual features without labels; useful when labelled pairs are scarce |

Start from a pretrained backbone and fine-tune on your own pairs. Generic ImageNet features capture object category well but not the fine distinctions that matter in a catalog (this pattern vs that pattern, this cut vs that cut).

Output dimension is usually 128-512. Lower dimension shrinks the index and speeds up search; measure recall at each size rather than assuming.

### Metric learning losses

| Loss | Idea | Trade-off |
|---|---|---|
| **Contrastive (pairwise)** | Pull positive pairs together; push negative pairs apart beyond a margin | Simple; needs careful pair sampling |
| **Triplet** | `d(a, p) + margin < d(a, n)` for anchor, positive, negative | Intuitive; most random triplets are already satisfied and give zero gradient, so it depends on mining |
| **InfoNCE / in-batch softmax** | Each anchor must pick its positive out of all other items in the batch | Every batch gives B-1 negatives for free; large batches help. This is the CLIP loss |
| **Proxy-based (Proxy-NCA, Proxy-Anchor)** | Learn one proxy vector per class; pull samples to their proxy | Fast convergence, no pair mining; needs class labels (e.g. product IDs) |
| **Normalized softmax / ArcFace-style** | Classification over product IDs with normalized weights and an angular margin | Strong baseline when you have many images per product ID; weight matrix grows with the number of classes |

A practical recipe: start with an in-batch softmax loss over (query crop, catalog image) pairs with a learned temperature, add mined hard negatives, and compare against a normalized-softmax classifier over product IDs. Both are strong; the right choice depends on whether your labels look more like pairs or like classes.

### Hard-negative mining

Random negatives are too easy: a dress versus a lawnmower teaches nothing. Useful negatives look similar but are the wrong item.

- **In-batch negatives**: free, but mostly easy unless batches are built by category.
- **Category-aware batching**: fill each batch with items from the same category so in-batch negatives are harder.
- **Offline mined negatives**: use the current model to retrieve top neighbours for each anchor and take the ones that are not positives.
- **Semi-hard negatives**: negatives farther than the positive but inside the margin. The hardest negatives are often label noise (an unlabelled duplicate of the positive), and training on them can collapse the embedding.

Filter mined negatives against known duplicates and same-product variants before using them. Otherwise you teach the model to separate two photos of the same item.

### Joint image-text embeddings

A CLIP-style dual encoder trains an image tower and a text tower so that matching (image, text) pairs have high cosine similarity. It gives you:
- **Text-to-image search** over the same image index, without a separate index.
- **Multimodal queries** by combining image and text vectors.
- **Zero-shot attribute tagging** (compare an image to "a photo of a striped shirt").

Fine-tune on your own (product image, title + attributes) pairs. Product titles are noisy but plentiful, and they carry the vocabulary your users use.

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query_emb, item_emb, temperature=0.05):
    """In-batch softmax: row i's positive is column i; all other columns are negatives."""
    q = F.normalize(query_emb, dim=-1)
    k = F.normalize(item_emb, dim=-1)
    logits = q @ k.T / temperature                     # (B, B) cosine similarities
    labels = torch.arange(q.size(0), device=q.device)
    # Symmetric loss: query->item and item->query
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

If the same product appears twice in a batch, the loss treats one copy as a negative for the other. Deduplicate by product ID within each batch or mask those entries.

---

## Training Data

The quality of the embedding depends mostly on the positive pairs. Several sources, each with different noise:

| Source | What it gives | Noise and caveats |
|---|---|---|
| **Catalog multi-view** | Different photos of the same product ID are positives | Clean, but all studio shots; doesn't teach the user-photo domain |
| **Catalog duplicates** | The same product listed by different sellers | Must be found first (perceptual hashing, title matching); good exact-match signal |
| **User photos linked to purchases** | Review photos attached to a purchased item | Very valuable: real user-photo domain paired with the correct catalog item |
| **Co-click / co-purchase** | Items clicked or bought in the same visual-search session | Large volume; reflects "similar" but also "complementary" and position bias |
| **Query-to-click logs** | Uploaded query crop paired with the item the user clicked or added to cart | Direct signal for the task; biased toward what the current system already retrieves |
| **Human-labelled relevance** | Graded judgments (exact / similar / irrelevant) on query-result pairs | Expensive; use mainly for evaluation and a small high-quality fine-tuning set |
| **Synthetic augmentation** | Crop, colour jitter, blur, perspective warp, background pasting on catalog images | Cheap way to simulate user-photo conditions; too much colour jitter destroys colour as a signal |

Points worth making in an interview:

- **Domain gap.** Studio-to-studio pairs alone produce a model that works for "similar items" but fails on user photos. Get some real user-photo pairs (review photos, click logs) and use augmentation that pastes catalog products onto realistic backgrounds.
- **Feedback loop.** Click logs only contain items the current system showed. As with text search, train on them with care: mix in other sources and keep a human-labelled set for evaluation.
- **Augmentations must respect the task.** If colour matters to users (it does for fashion), don't apply strong hue shifts. Horizontal flips are fine for most products but not for items with text or logos.
- **Splits by product, not by image.** If photos of the same product appear in both train and test, offline recall is inflated.

---

## ANN Indexing

Brute-force search over 600M vectors per query is too slow and too expensive, so use approximate nearest neighbour search. The trade-off is always between **recall, latency, and memory**.

### HNSW vs IVF-PQ

| | HNSW | IVF-PQ |
|---|---|---|
| **Structure** | Multi-layer proximity graph; greedy search from the top layer down | Coarse k-means clusters (IVF) plus compressed residuals (product quantization) |
| **Recall** | High (often 0.95+ at reasonable settings) | Lower, because PQ distances are approximate; recovered partly by re-scoring |
| **Latency** | Very low at moderate scale | Low; tunable through `nprobe` (clusters searched) |
| **Memory** | Full vectors plus graph links, the most memory-hungry option | Very compact: tens of bytes per vector |
| **Updates** | Inserts supported; deletes are awkward (tombstones, periodic rebuild) | Easy to add vectors to existing clusters; centroids go stale if the data distribution shifts |
| **Knobs** | `M` (links per node), `efConstruction`, `efSearch` | `nlist`, `nprobe`, number of PQ sub-vectors `m`, bits per code |
| **Best when** | Index fits in RAM and recall matters most | Hundreds of millions to billions of vectors, memory is the constraint |

A common hybrid at large scale: IVF-PQ for candidate generation, then re-score the top few hundred candidates with full-precision vectors fetched from a separate store (sometimes on SSD). This gets most of HNSW's recall at a fraction of the memory. Graph indexes stored mostly on SSD (DiskANN-style) are another option when RAM is the constraint.

### Sharding

At 600M vectors the index does not fit on one machine, so shard it.

- **Random / hash sharding by item ID**: every query goes to every shard, results are merged. Load is balanced and recall is not affected by the sharding. p99 is set by the slowest shard, so use hedged requests and per-shard timeouts.
- **Sharding by category**: a query only hits the shards for its predicted category. Cheaper, but a wrong category prediction means zero recall. Use it only with a fallback to broad search.

Replicate each shard for throughput and availability.

### Filtering by attributes and availability

Results must usually be in stock, ship to the user's region, and sometimes match a price range or category.

| Approach | Problem |
|---|---|
| **Post-filter** (search top-k, then drop) | If most of the top-k are out of stock, the page comes back nearly empty |
| **Pre-filter** (brute force over the matching subset) | Works when the filter is very selective; too slow when it is broad |
| **Filtered ANN** (filter checked during graph/cluster traversal) | Best general answer; recall can drop for very selective filters because the graph gets disconnected |
| **Partitioned indexes** (one index per region or top-level category) | Fast and exact on the partition key; multiplies memory if items belong to many partitions |

A practical setup: partition by a few coarse keys (region), filter during search on the rest, and over-fetch (ask for 3-5x more candidates than needed) when the filter is selective. Stock status changes constantly, so keep it in a fast attribute store checked at search time rather than baked into the index.

### Index updates for new items

- New items flow through CDC, get embedded within minutes, and are inserted into a **small fresh index** that is searched alongside the main index. Results from both are merged.
- The main index is rebuilt periodically (daily or weekly) with the fresh items folded in. For IVF-PQ, rebuilding also retrains the centroids and codebooks on current data.
- Deleted or delisted items are tombstoned immediately and removed at the next rebuild. Monitor the tombstone fraction; a graph index with many tombstones gets slower and less accurate.

---

## Re-Ranking

The ANN stage returns ~1000 candidates ranked by a single cosine similarity. The re-ranker has time for more.

**Heavier visual model.** Options include a larger embedding model applied only to the candidates, or a cross-attention model that takes the query crop and the candidate image together and outputs a match score. The cross model is more accurate for exact-match decisions (same pattern, same logo placement) but cannot be precomputed, so it only runs on the top few dozen.

**Business and behavioural features.** A gradient boosted tree or small neural ranker combining:

| Group | Examples |
|---|---|
| **Visual similarity** | ANN cosine score, re-ranker score, category match with the detector's prediction |
| **Item quality** | Rating, review count, return rate, image quality score, seller quality |
| **Commerce** | Price, price relative to the query item's price (if known), discount, stock level, shipping time |
| **Behavioural** | Historical CTR and add-to-cart rate from visual search, smoothed for low counts |
| **User context** | Region, price sensitivity, past brand preferences (use carefully; see below) |

Train it on click and add-to-cart logs from visual search, with position debiasing as in text search.

**Diversity.** Visual search tends to return ten nearly identical items, often the same product from different sellers. Apply:
- Near-duplicate collapsing (group identical products, show the best offer)
- A per-seller or per-brand cap
- MMR (maximal marginal relevance), which trades off relevance against similarity to results already selected

Keep personalization in the re-ranker and keep it mild. A user who uploads a photo has stated what they want; personalization should not override the visual match.

---

## Near-Duplicate Detection

Marketplaces contain many copies of the same image: the same product from many sellers, re-uploaded stock photos, lightly edited copies. Without dedup, they fill the results and waste index memory.

**Perceptual hashing** (aHash, dHash, pHash) produces a short fingerprint (often 64 bits) that changes little under resizing, compression, and small edits. Two images are near-duplicates if the Hamming distance between their hashes is below a threshold.

- **Cheap and exact enough** for re-uploads and re-compressions.
- **Weak against crops, rotations, and large edits.** For those, use the embedding itself: pairs with cosine similarity above a high threshold, verified on a labelled sample.
- **Lookup at scale:** split the 64-bit hash into bands and index each band, so candidate pairs share at least one band exactly (a locality-sensitive hashing trick), then check the full Hamming distance.

Uses:
- **At ingestion:** cluster duplicates, index one representative image per cluster, and keep the mapping to all item IDs.
- **At serving:** collapse duplicates in results and show the best offer.
- **In training:** remove duplicates from negative sets and add them as positives.
- **For safety:** match against hash lists of known prohibited images.

---

## Embedding Versioning and Backfills

Every time the embedding model changes, every vector in the index is invalid. Vectors from model v1 and v2 live in different spaces, and comparing a v2 query against v1 items gives meaningless scores. This is the operational problem interviewers most often probe.

**Rules:**
- Store `model_version` with every vector, and route each query to the index built with the same version as its encoder.
- Never mix versions in one index.

**Migration with dual indexes:**

```
1. Train v2; validate offline recall on the labelled set.
2. Backfill: re-embed the full catalog with v2 (batch GPU job).
   New/updated items during the backfill are embedded with BOTH v1 and v2.
3. Build the v2 index alongside v1. Both are live; v1 serves all traffic.
4. Shadow: send a copy of queries through v2 (query encoder + index);
   compare recall, latency, and result overlap.
5. A/B test: route a slice of users to the v2 encoder + v2 index.
6. Ramp to 100%, keep v1 warm for rollback, then delete v1.
```

The cost is roughly twice the index memory during migration and a full re-embedding job. For 600M images that is a large GPU batch job, which is one reason teams don't retrain the embedding model weekly. The re-ranker, which sits on top of the embeddings, can be retrained much more often without a backfill.

**Backward-compatible embeddings** are an alternative: train v2 with an extra loss that keeps its query embeddings comparable to v1 item embeddings, so the new query encoder can be deployed before the backfill finishes. It avoids the dual-index period but constrains how much v2 can improve, so most teams use it only for small updates.

---

## Evaluation

### Offline

Build a labelled evaluation set of query images (real user uploads and crops, not just catalog images) with graded relevance judgments for their results: exact match, similar, not relevant.

| Metric | What it measures |
|---|---|
| **Recall@k** | Fraction of relevant items that appear in the top k. Use k=1000 for the ANN stage, k=10 or 20 for the final list |
| **mAP** | Mean over queries of average precision; rewards putting all relevant items high |
| **NDCG@k** | Graded relevance with a position discount; the main metric for the re-ranked list |
| **Exact-match hit rate@k** | Whether the exact product appears in the top k; the key metric for "find this item" |
| **ANN recall vs brute force** | Overlap of ANN top-k with exact top-k for the same embedding; isolates index error from model error |

Keep the last two separate. If quality drops, you need to know whether the embedding model got worse or the index lost recall.

The snippet below shows exact cosine search (the ground truth for measuring ANN recall) and recall@k against labelled relevant items.

```python
import numpy as np

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

def brute_force_search(queries, catalog, k=10):
    """Exact top-k by cosine similarity. queries: (Q, d), catalog: (N, d)."""
    q, c = normalize(queries), normalize(catalog)
    sims = q @ c.T                                          # (Q, N)
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]  # unordered top-k
    order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)           # (Q, k), best first

def recall_at_k(retrieved, relevant, k):
    """retrieved: (Q, >=k) item ids; relevant: list of sets of relevant ids."""
    scores = []
    for row, rel in zip(retrieved, relevant):
        if rel:
            scores.append(len(set(row[:k]) & rel) / len(rel))
    return float(np.mean(scores))

# Example: measure ANN recall against exact search on the same vectors
rng = np.random.default_rng(0)
catalog = rng.normal(size=(100_000, 256)).astype(np.float32)
queries = rng.normal(size=(100, 256)).astype(np.float32)
exact = brute_force_search(queries, catalog, k=10)
ground_truth = [set(row) for row in exact]
# ann_results = ann_index.search(queries, k=10)   # from HNSW / IVF-PQ
# print(recall_at_k(ann_results, ground_truth, k=10))
```

For large N, brute force is done in chunks over the catalog or on a GPU, and only on a sample of queries.

### Online

| Metric | Why |
|---|---|
| **CTR on results** | Engagement with what was shown |
| **Add-to-cart and purchase rate** | The business outcome |
| **Zero-result / low-confidence rate** | Share of queries where the top score is below a threshold or filters removed everything |
| **Box re-selection rate** | Users picking a different box than the default suggests the detector's ranking is off |
| **Query abandonment** | Uploads with no interaction at all |

**Guardrails:** p99 latency, result diversity (share of the page from one seller), and the safety-filter block rate.

---

## Serving and Latency Budget

An example server-side budget for a photo query at p99 under 400 ms (upload time excluded):

| Step | Budget | Notes |
|---|---|---|
| Decode, resize, validate | 15 ms | Client already resized to ~512 px |
| Safety check (NSFW, prohibited content) | 25 ms | Small classifier, can run in parallel with detection |
| Object detection | 40 ms | GPU; batched with other requests |
| Query embedding | 20 ms | GPU; small batches for latency |
| ANN search (scatter-gather over shards) | 40 ms | Includes filtered search and merge; hedged requests |
| Full-precision re-scoring of top candidates | 20 ms | If using IVF-PQ |
| Feature fetch for re-ranker | 25 ms | Batched key-value lookups for price, stock, stats |
| Re-ranking | 40 ms | GBDT on ~500 items plus optional cross-model on top 30 |
| Dedup, diversity, response | 15 ms | |
| **Total** | **~240 ms** | Leaves headroom for tail latency and retries |

Tactics:
- **Parallelize** the safety check with detection, and the feature fetch with the ANN search where possible.
- **Dynamic batching** on GPU servers (collect requests for a few milliseconds) raises throughput a lot at a small latency cost.
- **Precompute item-to-item results.** "Similar items" on a product page uses catalog images, so compute neighbours offline for every item and serve from a key-value store. Only live filters (stock) are applied at request time.
- **Cache by image hash.** Popular images get uploaded repeatedly (screenshots of the same influencer post); cache results keyed on the perceptual hash of the query.

---

## Capacity and Cost

### Index memory: worked example

Assume 600M vectors at 256 dimensions.

```
Full precision (fp32):
  600e6 × 256 × 4 bytes = 614.4e9 bytes ≈ 614 GB
fp16:
  600e6 × 256 × 2 bytes ≈ 307 GB
HNSW graph overhead (M = 32, ~2M links on layer 0, 4-byte ids):
  600e6 × 64 × 4 bytes ≈ 154 GB  on top of the vectors

IVF-PQ with m = 32 sub-vectors, 8 bits each (1 byte per sub-vector):
  600e6 × 32 bytes ≈ 19.2 GB for codes
  + 8-byte ids: 600e6 × 8 ≈ 4.8 GB
  + centroids and codebooks: negligible (e.g. 65,536 × 256 × 4 ≈ 67 MB)
  ≈ 24 GB total → compression ratio vs fp32 ≈ 1024 / 32 = 32×
```

So HNSW over fp32 vectors needs roughly 770 GB of RAM across shards (before replication), while IVF-PQ fits in about 24 GB. That is the core trade-off: PQ loses some recall, which is why the top few hundred candidates are re-scored with full-precision vectors kept on SSD or in a separate store.

Other levers: reduce dimension (256 → 128 halves everything), dedupe images before indexing, and index only one or two images per item rather than all of them.

### Compute

- **Query-side GPU inference:** detection plus embedding per query. At 2k QPS with dynamic batching, a modest GPU pool handles it; the exact count depends on model size and should be measured with a load test, not guessed.
- **Catalog-side inference:** new and updated items (steady, small) plus full backfills on model changes (large, occasional). Run backfills on spare or cheaper batch capacity.
- **Cheaper models:** distil a large teacher embedder into a smaller student for the query path, and quantize to int8 or fp16 for inference. Keep query and catalog encoders compatible: if you distil the query encoder, the student must be trained to map into the teacher's embedding space, or the catalog must be re-embedded with the student.

---

## Privacy and Safety

**Faces and people.** User photos often contain people. Decide explicitly:
- Do not build or allow person or face search unless that is the product and it has legal review. Visual search for products should not match faces.
- Detect the product region and embed the crop, not the whole image; this also reduces how much identity information goes into the embedding.
- Store uploaded query images briefly, or not at all, and state the retention policy. Using them for training needs consent under most privacy regimes.

**NSFW and prohibited content.** Run a safety classifier on both uploaded queries and catalog images. Block or down-rank flagged catalog items, and don't return results for flagged queries. Match against hash lists of known illegal content at ingestion.

**Counterfeits.** Exact-match visual search is very good at finding counterfeits of branded items, which is useful for trust-and-safety teams but a risk in the user-facing product. Seller quality and brand authorization belong in the re-ranker.

**Adversarial listings.** Sellers may copy popular product photos to attract traffic. Near-duplicate detection across sellers, combined with other listing signals, flags this.

---

## Monitoring

| What | How | Signal |
|---|---|---|
| **Embedding drift** | Track distribution of query embeddings (mean vector, norm, distance to catalog centroids) over time | New camera types, new trends, a preprocessing bug |
| **Top-1 similarity distribution** | Histogram of the best cosine score per query | A drop means queries are finding weaker matches |
| **ANN recall regression** | Daily job: sample queries, compute brute-force top-k, compare to ANN top-k | Index degradation from tombstones, stale IVF centroids, bad parameters |
| **Model-version consistency** | Assert the query encoder version matches the index version at startup and per request | Mixed-version bugs, which otherwise fail silently |
| **Indexing lag** | Time from item listing to searchable | Stalled CDC or embedding pipeline |
| **Detector health** | Distribution of detected categories and box counts; box re-selection rate | Detector failing on new product types |
| **Online metrics** | CTR, add-to-cart, zero-result rate, by category and by query source | Business regressions, often concentrated in one category |
| **Latency** | p50/p99 per stage | Slow shards, GPU saturation |

The recall regression job is cheap and catches problems no online metric will attribute correctly. Run it daily and alert on a drop.

---

## Interview Q&A

#### Why not just use a pretrained CLIP or ImageNet model and skip fine-tuning?

It is a good baseline and a reasonable first launch, but it is usually not enough for commerce. Generic models learn features that separate object categories (shoe vs bag) and broad concepts. Product search needs finer distinctions: this floral pattern vs that one, this sneaker model vs a lookalike, this cut of jeans vs another. Those differences matter to shoppers and are not what generic pretraining optimizes.

Fine-tuning with metric learning on your own pairs (catalog multi-view, review photos matched to purchases, click logs) moves the embedding space toward those distinctions and closes the gap between user photos and studio catalog photos. The practical plan is to launch with the pretrained model to collect click data, then use that data to fine-tune.

#### How do you choose between HNSW and IVF-PQ?

By memory budget first. HNSW stores full vectors plus graph links, so at hundreds of millions of vectors it needs hundreds of gigabytes of RAM. In return it gives high recall at low latency with simple tuning. IVF-PQ compresses each vector to a few dozen bytes, so the whole index can fit in tens of gigabytes, but PQ distances are approximate and recall drops.

At moderate scale (tens of millions) or when recall is the top priority and the budget allows, HNSW is the simpler choice. At hundreds of millions or more, IVF-PQ with full-precision re-scoring of the top candidates is common. Either way, the choice should come from measured recall@k vs latency vs memory curves on your data, using brute-force search on a sample as ground truth.

#### What happens when you ship a new embedding model?

Every vector in the index becomes incompatible with the new query encoder, so you cannot swap the model in place. You re-embed the whole catalog with the new model, build a second index, and run both side by side. New items are embedded with both models during the transition. You shadow test the new stack, A/B test it, ramp it up, and only then delete the old index.

This costs a full GPU backfill and double index memory for the migration period, so embedding model updates are infrequent. Tuning that can happen without a backfill (the re-ranker, filters, business logic) is where most iteration happens. If updates need to be frequent, a backward-compatible training loss lets a new query encoder work against old item vectors, at some cost to how much the new model can improve.

#### How do you handle a query photo with several products in it?

Run an object detector on the query, show the detected boxes, and let the user choose; default to the largest and most central box. Crop that region with a bit of padding and embed the crop. Embedding the whole image mixes features from every object and the results match none of them well.

The detector's category label helps too, as a re-ranking feature or soft filter. A hard filter is risky because a detector mistake then removes all correct results. Run the same detector on catalog lifestyle photos, so the catalog side contains crops that look like user crops.

#### How do you get training data for user photo to catalog item matching?

The best source is data where a real user photo is already linked to a catalog item: review photos attached to a purchase, and uploaded queries followed by an add-to-cart or purchase of a specific item. These cover the domain gap directly.

These sources are smaller and biased. Click logs only include what the current system showed, and review photos skew toward certain categories. So I would combine them with catalog multi-view pairs, duplicates across sellers as extra positives, and synthetic pairs created by augmenting catalog images (cropping, pasting on realistic backgrounds, blur, lighting changes). A human-labelled set of real queries is needed for evaluation regardless, because every other source is biased in some way.

#### Results are relevant but half of them are out of stock. How do you fix it?

This is a filtering problem, not a model problem. Post-filtering the ANN top-k removes most of the list when many near neighbours are out of stock. Options: filter during ANN traversal using an attribute bitmap for in-stock items, over-fetch candidates when filters are selective, and keep stock status in a fast attribute store updated in near real time rather than in the index itself, since it changes too often to rebuild for.

I would also check whether out-of-stock items should be removed at all. For exact-match queries, showing "this exact item, out of stock" alongside similar in-stock alternatives can be better than hiding it.

#### How would you detect that search quality dropped, and whether the model or the index is to blame?

Separate the two with different measurements. For the index, run a daily job that samples queries, computes exact top-k with brute force, and compares with ANN results; a drop in overlap points at the index (tombstone buildup, stale IVF centroids, a parameter change). For the model, run the labelled evaluation set regularly and watch recall@k and NDCG; a drop there with stable ANN recall points at the model or the preprocessing.

Online, watch the distribution of top-1 similarity scores and the zero-result rate by category. A sudden shift usually means a pipeline bug, such as a changed resize, a colour-space change, or a query encoder whose version doesn't match the index. Assert version compatibility at startup and log it with every request so that last case cannot happen silently.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Embedding the whole query image | Features from several objects blur together | Detect, let the user pick, embed the crop |
| Training only on studio catalog photos | Fails on real user photos | Review photos, click pairs, realistic augmentation |
| Mixing embedding versions in one index | Scores become meaningless, silently | Version every vector; dual index during migration |
| Treating the ANN index as exact | Recall loss is invisible without measurement | Monitor ANN recall against brute force on a sample |
| Post-filtering stock and region | Empty or near-empty result pages | Filtered ANN, over-fetch, partitioned indexes |
| Random negatives only | Model learns easy distinctions | Category-aware batches, mined semi-hard negatives |
| Mined negatives that are actually duplicates | Model learns to separate identical items | Dedup with perceptual hashes before mining |
| Train/test split by image | Same product in both sets inflates recall | Split by product ID |
| Strong colour augmentation for fashion | Model stops using colour | Mild colour jitter; keep colour when users care |
| No near-duplicate collapsing | Page full of the same product from different sellers | Hash-based clustering, show best offer, seller caps |
| Ignoring index memory until the end | Design doesn't fit in any realistic budget | Do the N × d × bytes math early; plan PQ or lower dimension |
| Retraining the embedder often | Full backfill each time | Iterate on the re-ranker; update embeddings less often |

---

## Related Topics

- [ML System Design Framework](./README.md)
- [Search and Ranking System Design](./search_ranking_system.md)
- [Recommendation System Design](./recommendation_system.md)
- [Content Moderation System Design](./content_moderation_system.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
- [Computer Vision](../deep_learning/intro_computer_vision.md)
- [Model Compression](../deep_learning/intro_model_compression.md)
- [Embeddings](../ai_genai/intro_embeddings.md)
- [Multimodal AI](../ai_genai/intro_multimodal_ai.md)
- [Vector Databases — Advanced](../ai_genai/intro_vector_databases_advanced.md)
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [A/B Testing](../mlops/intro_ab_testing.md)
