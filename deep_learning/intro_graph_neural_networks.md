# Graph Neural Networks

Most real data is relational — social networks, fraud rings, molecules, recommendation bipartite graphs, knowledge bases, code. GNNs are how you learn on it without flattening away the structure. They come up in interviews at companies with graph-shaped problems (payments, social, biotech, recommendations) and as a "do you know anything beyond transformers?" probe.

---

## Table of Contents
1. [When a Graph Is the Right Model](#when-a-graph-is-the-right-model)
2. [Graph Basics and Notation](#graph-basics-and-notation)
3. [Message Passing](#message-passing)
4. [GCN](#gcn)
5. [GraphSAGE](#graphsage)
6. [GAT](#gat)
7. [Task Types](#task-types)
8. [Over-Smoothing and Depth](#over-smoothing-and-depth)
9. [Scaling to Large Graphs](#scaling-to-large-graphs)
10. [Expressiveness Limits](#expressiveness-limits)
11. [Practical Training Notes](#practical-training-notes)
12. [Model Comparison](#model-comparison)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## When a Graph Is the Right Model

Use a GNN when **the relationships carry signal that node features alone don't**.

| Problem | Why a graph helps |
|---|---|
| Fraud detection | Fraud rings share devices, addresses, cards — the *structure* is the evidence |
| Recommendations | User-item interactions form a bipartite graph; multi-hop reveals taste similarity |
| Molecular property prediction | A molecule *is* a graph; bonds determine properties |
| Social / abuse detection | Coordinated accounts cluster structurally |
| Knowledge graphs | Reasoning over typed relations |
| Code analysis | ASTs and call graphs |

**When not to.** If a few aggregate features (`friend_count`, `mean_neighbor_score`) capture most of the signal, engineer those and use gradient boosting — it'll be faster, easier to serve, and often just as accurate. The honest framing for an interview: GNNs win when *multi-hop* structure matters and hand-crafted neighbourhood aggregates plateau. Try the boring baseline first.

---

## Graph Basics and Notation

A graph `G = (V, E)` with `n = |V|` nodes. Representations:

- **Adjacency matrix** `A ∈ R^{n×n}` — dense, `O(n²)` memory, impractical above ~50k nodes.
- **Edge list / sparse COO** — `[2, num_edges]` tensor, what every real library uses.
- **Node features** `X ∈ R^{n×d}`; optionally edge features.

Graphs may be directed or undirected, weighted, heterogeneous (multiple node and edge types), or dynamic (evolving over time).

**The defining property is permutation invariance**: relabelling nodes must not change the output. That's why every aggregation function in a GNN is a *set* function — sum, mean, max — rather than something order-dependent. A CNN can assume a fixed grid neighbourhood; a GNN cannot, because neighbourhoods have arbitrary size and no canonical order.

---

## Message Passing

Nearly every GNN is an instance of one template. Getting this framework across is usually enough to answer "explain how GNNs work":

```
For each layer k:
  1. MESSAGE:   m_uv = Message(h_u, h_v, e_uv)        for each edge u→v
  2. AGGREGATE: a_v  = Aggregate({m_uv : u ∈ N(v)})   permutation-invariant
  3. UPDATE:    h_v  = Update(h_v, a_v)               combine with own state
```

**Each layer extends the receptive field by one hop.** After `k` layers, a node's representation depends on its `k`-hop neighbourhood — which is the intuition behind both the power and the depth problem of GNNs.

```python
import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    """The generic template, written out with scatter-add aggregation."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.message = nn.Linear(2 * d_in, d_out)
        self.update = nn.Linear(d_in + d_out, d_out)

    def forward(self, x, edge_index):
        # x: (n, d_in)   edge_index: (2, E) with rows [source, target]
        src, dst = edge_index

        # 1. Build a message per edge from the source and target states
        msg = self.message(torch.cat([x[src], x[dst]], dim=-1))     # (E, d_out)

        # 2. Sum messages into their destination nodes — permutation invariant
        agg = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        agg.index_add_(0, dst, msg)                                  # (n, d_out)

        # 3. Update each node from its own state plus the aggregate
        return self.update(torch.cat([x, agg], dim=-1))
```

**Choice of aggregator matters and is a common question:**

| Aggregator | Keeps | Loses |
|---|---|---|
| **Sum** | Degree information; most expressive | Sensitive to degree scale |
| **Mean** | Scale invariance | Cannot distinguish degree |
| **Max** | Salient features | Distribution shape |

Sum is the most expressive — mean and max cannot distinguish a node with two identical neighbours from one with four — which is exactly the argument behind Graph Isomorphism Networks.

---

## GCN

The Graph Convolutional Network simplifies message passing to a normalized neighbourhood average:

```
H^(k+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(k) W^(k) )
```

where `Ã = A + I` adds self-loops and `D̃` is its degree matrix.

**Why symmetric normalization `D^(-1/2) A D^(-1/2)`** rather than plain `D^(-1)A`: it keeps the eigenvalues of the propagation matrix bounded, which stabilizes training and prevents high-degree nodes from dominating the representation of their neighbours. Self-loops matter because without them a node's own features are discarded at every layer.

GCN is **transductive** in its original form — it operates on the full fixed adjacency matrix, so a new node arriving after training has no representation without recomputation. That limitation is precisely what GraphSAGE addresses.

---

## GraphSAGE

Two changes that make GNNs practical at scale.

**Neighbour sampling**: instead of aggregating over *all* neighbours, sample a fixed number per layer (say 25 at layer 1, 10 at layer 2). This bounds computation per node regardless of degree, so a celebrity node with 10 million followers doesn't blow up the batch.

**Inductive learning**: it learns aggregation *functions* rather than per-node embeddings, so an unseen node can be embedded at inference from its features and neighbourhood. That's what makes it deployable in production systems where the graph grows continuously.

```
h_v^(k) = σ( W · CONCAT( h_v^(k-1), AGGREGATE({h_u^(k-1) : u ∈ SAMPLE(N(v))}) ) )
```

The `CONCAT` (rather than summing the node into the aggregate) preserves a distinction between "what I am" and "what my neighbourhood is", which measurably helps.

For most production graph problems — fraud, recommendations, evolving social graphs — GraphSAGE or a variant of it is the realistic answer, and saying so signals practical experience over paper familiarity.

---

## GAT

Graph Attention Networks learn *how much* to weight each neighbour rather than treating them uniformly:

```
e_uv = LeakyReLU( aᵀ [W h_u ‖ W h_v] )
α_uv = softmax_u( e_uv )                    # normalized over v's neighbours
h_v  = σ( Σ_u α_uv · W h_u )
```

Multi-head attention, as in transformers, stabilizes it.

**When attention helps**: heterogeneous neighbourhoods where some neighbours are far more informative than others — in fraud, a shared device is much stronger evidence than a shared city. When all neighbours are roughly equivalent, GAT adds parameters and compute for little gain over GCN.

The relationship to transformers is worth stating: **a transformer is essentially a GAT on a fully-connected graph**, with positional encodings supplying the structure that a graph provides explicitly. Attention is the aggregation function in both.

---

## Task Types

| Task | Example | Output layer |
|---|---|---|
| **Node classification** | Is this account fraudulent? | Per-node MLP head |
| **Link prediction** | Will these users connect? Recommend an item | Dot product or MLP on node-pair embeddings |
| **Graph classification** | Is this molecule toxic? | Pooling (mean/sum/attention) → MLP |
| **Edge classification** | Is this transaction suspicious? | MLP on edge + endpoint embeddings |

```python
# Link prediction — score a candidate pair, train against sampled negatives
def link_score(h, u, v):
    return (h[u] * h[v]).sum(-1)          # dot product; or an MLP on concat

# Graph classification — pool node embeddings into one graph vector
def graph_readout(h, batch_index):
    return global_mean_pool(h, batch_index)   # permutation-invariant over nodes
```

**Splitting is task-specific and easy to get wrong.** For node classification you mask nodes, not rows — the graph structure stays whole and you simply don't compute loss on held-out nodes. For link prediction you must *remove* test edges from the message-passing graph, or the model sees the answer during aggregation. That's the classic leakage bug in GNN work.

---

## Over-Smoothing and Depth

GNNs don't get deeper the way CNNs do. Beyond 2–4 layers, performance usually degrades.

**Over-smoothing**: each layer averages a node with its neighbours. Repeat enough times and every node's representation converges toward the same value — the graph equivalent of blurring an image until it's uniform grey. Nodes become indistinguishable, so classification collapses.

**Over-squashing** is the complementary problem: after `k` layers a node's receptive field contains exponentially many nodes, all compressed into one fixed-size vector. Information from distant nodes is squashed through bottleneck edges and effectively lost.

Mitigations: **residual/skip connections** (as in ResNets), **jumping knowledge** (concatenate or max-pool representations from all layers so the model can use shallow ones), normalization layers, and **graph rewiring** to add shortcut edges. But the honest answer is that 2–3 layers is the norm, and most real graphs have small diameter so 2–3 hops covers most of the useful signal anyway.

---

## Scaling to Large Graphs

The core difficulty: a graph doesn't decompose into independent examples the way images do. A node's computation depends on its neighbours, whose computation depends on *their* neighbours — the **neighbourhood explosion** problem. With average degree 100 and 3 layers, one node's full computation touches a million nodes.

| Strategy | How | Trade-off |
|---|---|---|
| **Neighbour sampling** (GraphSAGE) | Fixed fan-out per layer | Variance from sampling; the standard approach |
| **Subgraph sampling** (Cluster-GCN) | Partition into clusters, train per cluster | Loses cross-cluster edges |
| **Layer-wise sampling** (FastGCN) | Sample nodes per layer, not per node | Better complexity, fiddlier |
| **Historical embeddings** (GNNAutoScale) | Cache stale neighbour embeddings | Memory for staleness |
| **Full-batch on GPU** | Just fit it | Only up to a few million nodes |

**Serving is the harder half**, and it's where interviews go if the interviewer has shipped one. Real-time inference needs `k`-hop neighbourhoods fetched at request time — a graph database query with a tight latency budget. The common production pattern is to **precompute embeddings in batch** and refresh periodically, accepting staleness, with real-time computation only for nodes that must be fresh (a brand-new account, for instance). Say that, and you're clearly speaking from deployment experience.

---

## Expressiveness Limits

Standard message-passing GNNs are **at most as powerful as the 1-Weisfeiler-Lehman graph isomorphism test**. They cannot distinguish certain non-isomorphic graphs — the standard example being two triangles versus one hexagon, where every node has degree 2 and identical local neighbourhoods, so message passing produces identical embeddings forever.

**GIN (Graph Isomorphism Network)** achieves the 1-WL bound by using **sum** aggregation plus an MLP, since sum is injective over multisets in a way mean and max are not:

```
h_v = MLP( (1 + ε)·h_v + Σ_{u ∈ N(v)} h_u )
```

Ways past the limit: add **structural features** (node degree, triangle counts, positional encodings), use **higher-order** GNNs operating on node tuples, or add random node identifiers. In practice the theoretical limit rarely binds on real tasks — but knowing it exists, and that sum aggregation is more expressive than mean, is exactly the kind of detail that distinguishes a strong answer.

---

## Practical Training Notes

**Libraries**: PyTorch Geometric (PyG) is the default; DGL is the main alternative and scales well.

```python
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGENet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.conv1 = SAGEConv(d_in, d_hidden)
        self.conv2 = SAGEConv(d_hidden, d_out)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
```

**Class imbalance** is severe in the common use cases — fraud is well under 1% of nodes. Use weighted loss and evaluate with PR-AUC, not accuracy or ROC-AUC.

**Negative sampling for link prediction**: random negatives are too easy (most random pairs are trivially unconnected). Sample *hard* negatives — nodes 2 hops away, or high-degree nodes — so the model learns something beyond degree.

**Always benchmark against a non-graph baseline**: gradient boosting on node features plus hand-engineered neighbourhood aggregates (neighbour count, mean neighbour label rate, distinct shared devices). If the GNN doesn't clearly beat it, the added complexity in training and serving isn't justified.

---

## Model Comparison

| | GCN | GraphSAGE | GAT | GIN |
|---|---|---|---|---|
| Aggregation | Normalized mean | Mean / max / LSTM | Attention-weighted | **Sum + MLP** |
| Inductive? | No (transductive) | **Yes** | Yes | Yes |
| Scales to large graphs | Poorly | **Yes** (sampling) | Moderate | Moderate |
| Neighbour weighting | Uniform | Uniform | **Learned** | Uniform |
| Expressiveness | Below 1-WL | Below 1-WL | Below 1-WL | **1-WL** |
| Best for | Small fixed graphs, benchmarks | **Production, evolving graphs** | Heterogeneous neighbourhoods | Graph classification |

---

## Interview Q&A

#### Explain how a GNN works.

Almost all of them are message passing, repeated in layers. In each layer, every node builds a message from each of its neighbours, aggregates those messages with a permutation-invariant function like sum or mean, and updates its own representation by combining that aggregate with its previous state.

The key structural fact is that **each layer extends the receptive field by one hop** — after two layers a node's embedding encodes its 2-hop neighbourhood. That's the source of the power (structure gets baked into representations) and of the main limitation (you can't stack many layers).

Aggregation must be permutation-invariant because nodes have no canonical ordering and neighbourhoods have arbitrary size — which is exactly why a CNN's fixed-grid convolution doesn't transfer directly.

#### Why can't you just stack 20 GNN layers?

**Over-smoothing.** Each layer averages a node with its neighbours, so repeated application drives all node representations toward the same value — the graph equivalent of blurring an image until it's uniform. Once nodes are indistinguishable, classification collapses.

There's a second, complementary problem: **over-squashing**. A node's `k`-hop receptive field grows exponentially, and all that information must be compressed into a fixed-size vector, often passing through bottleneck edges. Distant information is effectively lost regardless of depth.

Mitigations exist — residual connections, jumping knowledge, normalization, graph rewiring — but the practical answer is that 2–3 layers is standard, and most real graphs have small diameter, so 2–3 hops already covers the useful signal. Depth simply isn't the axis you scale on with GNNs.

#### What does GraphSAGE add over GCN?

Two things, both aimed at production.

**Inductive capability.** GCN in its original form is transductive — it operates on the full fixed adjacency matrix, learning representations tied to specific nodes, so a new node arriving after training has no embedding without recomputing over the whole graph. GraphSAGE learns aggregation *functions*, so an unseen node can be embedded from its features and sampled neighbourhood at inference. For any graph that grows — users, transactions, items — that's the difference between deployable and not.

**Neighbour sampling.** Instead of aggregating over every neighbour, it samples a fixed fan-out per layer. This bounds per-node computation regardless of degree, which matters enormously because real graphs have heavy-tailed degree distributions — one celebrity node with millions of edges would otherwise dominate the batch.

#### How do you scale GNN training and, harder, GNN serving?

**Training**: the problem is neighbourhood explosion — a node's computation depends on neighbours, whose computation depends on theirs, so with degree 100 and 3 layers one node touches a million others. Neighbour sampling (GraphSAGE) bounds the fan-out per layer and is the standard fix. Cluster-GCN partitions the graph and trains on subgraphs, trading away cross-cluster edges. Historical-embedding methods cache stale neighbour representations to trade memory for freshness.

**Serving is harder**, and it's usually where the real constraint bites. Real-time inference requires fetching a `k`-hop neighbourhood per request, which is a graph database query inside a tight latency budget. The common production design is to **precompute embeddings in batch** on a schedule, serve them from a key-value store, and only compute on demand for nodes that genuinely need freshness — a new account, say. That accepts embedding staleness in exchange for predictable latency, and for most applications the graph structure changes slowly enough that it's fine.

#### When would you *not* use a GNN?

When simpler neighbourhood features capture most of the signal. Very often, computing `neighbour_count`, `mean_neighbour_label_rate`, `count_of_shared_devices` and feeding them to gradient boosting gets you most of the way — with far better training speed, easier serving, and better interpretability.

I'd reach for a GNN when **multi-hop structure matters** and those hand-crafted aggregates plateau. Fraud rings are the canonical case: the signal isn't in any single account's features, it's in the pattern of shared attributes two or three hops out, which is combinatorially awkward to hand-engineer.

I'd also weigh the serving cost seriously. A GNN needs graph infrastructure, neighbourhood fetching, and embedding refresh pipelines. That's a real operational commitment, and it should be justified by a measured lift over the boring baseline — which I'd always build first.

#### What's the expressiveness limit of message-passing GNNs?

They're bounded above by the **1-Weisfeiler-Lehman** graph isomorphism test, meaning there are non-isomorphic graphs they provably cannot distinguish. The standard example is two disjoint triangles versus a single hexagon: every node has degree 2 with identical local neighbourhoods, so message passing produces identical embeddings no matter how many layers you add.

**GIN** reaches that 1-WL bound by using sum aggregation followed by an MLP, because sum is injective over multisets in a way mean and max are not — mean can't tell two identical neighbours from four, and max discards multiplicity entirely.

To go beyond, you add information message passing can't derive: structural features like degree or triangle counts, positional encodings, random node identifiers, or higher-order GNNs operating on node tuples. In practice the limit rarely binds on real tasks, but it explains why aggregator choice matters and why sum is the more expressive default.

#### How do you split data for link prediction without leaking?

This is the classic GNN leakage bug. For link prediction, test edges must be **removed from the message-passing graph**, not merely excluded from the loss. If a test edge remains in the adjacency used for aggregation, the model literally observes the connection it's being asked to predict, and validation scores become meaningless.

So the procedure is: hold out a set of edges as positive test examples, build the message-passing graph from the *remaining* edges only, and score the held-out pairs at evaluation. For temporal graphs, split by time rather than randomly — train on edges before a cutoff, test on edges after — because random splits let the model use future connections to predict past ones.

Negative sampling matters too: random node pairs are trivially unconnected, so the model can succeed by learning degree. Hard negatives — nodes two hops away, or degree-matched samples — force it to learn something real.

Node classification is different and simpler: you mask *nodes* rather than removing structure, since the graph itself isn't the label.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Test edges left in the message-passing graph | Model sees the answer; scores are meaningless | Build the graph from training edges only |
| Random split on a temporal graph | Uses future edges to predict the past | Split by time |
| Stacking many layers | Over-smoothing collapses representations | 2–3 layers; residuals; jumping knowledge |
| Random negatives for link prediction | Too easy; model learns degree | Hard negatives (2-hop, degree-matched) |
| Full-neighbourhood aggregation on a hub node | One celebrity node explodes the batch | Neighbour sampling with fixed fan-out |
| Mean aggregation when multiplicity matters | Can't distinguish 2 neighbours from 4 | Sum aggregation (GIN) |
| Accuracy on an imbalanced node task | Fraud is <1%; accuracy is meaningless | PR-AUC; weighted loss |
| No non-graph baseline | Can't justify the operational cost | Gradient boosting on neighbourhood aggregates |
| Ignoring serving latency until the end | `k`-hop fetch may be impossible in budget | Precompute embeddings; refresh on a schedule |
| Forgetting self-loops in GCN | Node's own features dropped each layer | `Ã = A + I` |
| Dense adjacency matrices | `O(n²)` memory; infeasible past ~50k nodes | Sparse COO edge index |

---

## Related Topics

- [Neural Network Training](./intro_neural_network_training.md)
- [Transformers](./intro_transformers.md)
- [Sequence Models](./intro_sequence_models.md)
- [Generative Models](./intro_generative_models.md)
- [Recommender Systems](../classical_ml/intro_recommender_systems.md)
- [Anomaly Detection](../classical_ml/intro_anomaly_detection.md)
- [Fraud Detection System Design](../system_design/fraud_detection.md)
- [PyTorch](../frameworks/intro_pytorch.md)
- [Deep Learning Overview](./README.md)
