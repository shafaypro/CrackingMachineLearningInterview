# Transformers Deep Dive

A comprehensive guide to the Transformer architecture — the foundation of modern NLP and AI.

---

## Table of Contents

1. ["Attention is All You Need" Overview](#attention-is-all-you-need-overview)
2. [Self-Attention Mechanism](#self-attention-mechanism)
3. [Multi-Head Attention](#multi-head-attention)
4. [Positional Encoding](#positional-encoding)
5. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
6. [Encoder-Only, Decoder-Only, Encoder-Decoder](#encoder-only-decoder-only-encoder-decoder)
7. [Key Innovations](#key-innovations)
8. [Comparison with RNNs/LSTMs](#comparison-with-rnnslstms)
9. [PyTorch Implementation](#pytorch-implementation)
10. [Interview Q&A](#interview-qa)
11. [References](#references)

---

## "Attention is All You Need" Overview

Published in 2017 by Vaswani et al. at Google, this paper introduced the Transformer architecture — replacing recurrence entirely with attention mechanisms.

**Key contributions:**
- Eliminated sequential processing (RNNs) with parallel attention
- Scaled to much larger models than RNNs due to parallelizability
- Introduced multi-head attention and positional encoding
- Became the foundation for BERT, GPT, T5, and all modern LLMs

**Original use case:** Machine translation (English → German/French). Outperformed all previous RNN/CNN-based systems.

---

## Self-Attention Mechanism

Self-attention allows each token in a sequence to attend to all other tokens, capturing long-range dependencies in a single step.

### Intuition

For the sentence "The animal didn't cross the street because it was too tired":
- When encoding "it", self-attention can directly relate it to "animal" with high weight
- RNNs must propagate this relationship through intermediate tokens (slow, information degradation)

### Mathematical Formulation

**Step 1:** Create Query, Key, Value matrices from input embeddings X:
```
Q = X · Wq    (query: what am I looking for?)
K = X · Wk    (key: what do I have to offer?)
V = X · Wv    (value: what information do I contain?)
```

**Step 2:** Compute attention scores:
```
scores = Q · Kᵀ              (dot product similarity)
scores_scaled = scores / √dₖ  (scale by sqrt of key dimension)
weights = softmax(scores_scaled)  (normalize to probabilities)
```

**Step 3:** Apply weights to values:
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

**Why divide by √dₖ?**

Without scaling, the dot products grow large in magnitude as dimension dₖ increases, pushing softmax into regions of very small gradients (saturation). Dividing by √dₖ keeps the scale manageable.

### Example

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)

    # Scaled dot product scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, seq_len, seq_len)

    # Apply optional mask (for causal attention in decoder)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Attention weights
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, V)
    return output, weights
```

---

## Multi-Head Attention

Instead of one large attention, run multiple smaller attention operations in parallel (heads), allowing the model to attend to different aspects simultaneously.

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · Wₒ

where headᵢ = Attention(Q·Wqᵢ, K·Wkᵢ, V·Wvᵢ)
```

**Intuition:** Different heads can specialize:
- Head 1: Syntactic dependencies
- Head 2: Semantic relationships
- Head 3: Coreference (pronoun resolution)
- Head 4: Local context

**Dimensions:** If model dim is d_model = 512 and h = 8 heads, each head operates on d_k = d_v = 64 dimensions.

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from (batch, seq, d_model) to (batch, heads, seq, d_k)."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq, d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.split_heads(self.W_q(query))   # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = self.dropout(torch.softmax(scores, dim=-1))
        attention_output = torch.matmul(weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        return self.W_o(attention_output)
```

---

## Positional Encoding

Self-attention is order-agnostic — it treats "cat sat mat" the same as "mat sat cat". Positional encodings add position information to token embeddings.

### Sinusoidal Positional Encoding (original paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties:
- Unique encoding for each position
- Model can learn relative positions from linear combinations
- Works for sequences longer than seen in training

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create position encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to token embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Learned Positional Embeddings

Modern models like BERT use learned positional embeddings:

```python
# Simple learned positional embedding
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len).to(x.device)
x = x + self.pos_embedding(positions)
```

### RoPE (Rotary Positional Embedding)

Used in Llama, Mistral, Gemma. Encodes relative position by rotating Q and K vectors in the attention calculation. Generalizes better to longer sequences than seen in training.

---

## Encoder-Decoder Architecture

The original Transformer architecture for seq2seq tasks (translation):

```
Source Sequence (English)
        ↓
[Token Embedding + Positional Encoding]
        ↓
┌──────────────────────────────┐
│         ENCODER              │ ← N layers (N=6 in original)
│  [Self-Attention]            │
│  [Feed-Forward Network]      │
└──────────────────────────────┘
        ↓ (encoder memory)
Target Sequence (German) → [Embedding + PE]
        ↓
┌──────────────────────────────┐
│         DECODER              │ ← N layers
│  [Masked Self-Attention]     │ (causal: can't see future tokens)
│  [Cross-Attention]           │ (Q from decoder, K/V from encoder)
│  [Feed-Forward Network]      │
└──────────────────────────────┘
        ↓
    Linear + Softmax
        ↓
    Output Probabilities
```

### Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LayerNorm variant (more stable in practice)
        # Self-attention with residual connection
        attn_output = self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
```

---

## Encoder-Only, Decoder-Only, Encoder-Decoder

### Encoder-Only (BERT-style)

- Bidirectional: each token attends to all other tokens
- Pretraining: Masked Language Model (MLM) — predict masked tokens
- Use case: Classification, NER, QA, embeddings

```python
# BERT: bidirectional, best for understanding tasks
# Input: [CLS] token [MASK] more text [SEP]
# Masked positions predicted from bidirectional context
```

### Decoder-Only (GPT-style)

- Causal/autoregressive: each token attends only to previous tokens
- Pretraining: next token prediction
- Use case: Text generation, chat, code completion

```python
# GPT: causal mask prevents attending to future tokens
# At each step, predict the next token given all previous tokens
# Used in: GPT-4, Claude, Llama, Mistral, Gemma

def create_causal_mask(seq_len):
    """Upper triangular mask — prevents attending to future positions."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
```

### Encoder-Decoder (T5/BART-style)

- Encoder: bidirectional (processes source)
- Decoder: causal + cross-attention to encoder
- Use case: Translation, summarization, code generation from description

---

## Key Innovations

### Layer Normalization

Applied before (Pre-LN, modern) or after (Post-LN, original) each sublayer.

```python
# Pre-LN: more stable training (used in GPT-3, Llama)
x = x + attention(LayerNorm(x))
x = x + ffn(LayerNorm(x))

# Post-LN: original paper
x = LayerNorm(x + attention(x))
x = LayerNorm(x + ffn(x))
```

### Residual Connections

Every sublayer has a skip connection: `output = LayerNorm(x + sublayer(x))`. Allows gradients to flow directly through, enabling training of deep networks (100+ layers).

### Feed-Forward Sublayer

Two linear transformations with ReLU in between:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

d_ff is typically 4× d_model (512 → 2048 in original paper). Modern models use SwiGLU activation:

```
FFN_SwiGLU(x) = (SiLU(xW₁) ⊙ xW₂) W₃
```

### Flash Attention

A memory-efficient attention implementation that uses tiling to avoid materializing the full N×N attention matrix in memory. Up to 7x faster and 2x more memory efficient. Widely used in production (FlashAttention-2 in Llama, Mistral, etc.).

---

## Comparison with RNNs/LSTMs

| Aspect | RNN / LSTM | Transformer |
|--------|-----------|-------------|
| **Parallelization** | Sequential (can't parallelize) | Fully parallel |
| **Long-range dependencies** | Poor (vanishing gradient) | Excellent (direct attention) |
| **Training speed** | Slow | Fast (parallelizable) |
| **Inference speed** | Fast (fixed state) | Slower (recompute attention) |
| **Memory (training)** | O(sequence length) | O(sequence length²) |
| **Positional information** | Built-in (sequential) | Requires explicit encoding |
| **Context window** | Theoretically unlimited | Limited by context window |
| **Scale** | Hard to scale effectively | Scales excellently (more params → better) |

**Why Transformers won:** The parallelization advantage on GPUs, combined with better handling of long-range dependencies, made it possible to train on vastly more data, achieving much better performance.

---

## PyTorch Implementation

A minimal but complete Transformer implementation:

```python
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Scale embeddings
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        return self.output_projection(output)

# Create causal mask for decoder
def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns upper triangular mask for causal attention."""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )

# Usage
model = TransformerModel(vocab_size=30000)
src = torch.randint(0, 30000, (2, 20))  # (batch, src_len)
tgt = torch.randint(0, 30000, (2, 15))  # (batch, tgt_len)

tgt_mask = generate_causal_mask(tgt.size(1), tgt.device)
output = model(src, tgt, tgt_mask=tgt_mask)
print(output.shape)  # (2, 15, 30000)
```

---

## Interview Q&A

**Q1: Explain the self-attention mechanism.** 🟡 Intermediate

Self-attention computes `Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V`. Each token creates a Query (what it's looking for), Key (what it offers), and Value (its information). Dot products between Q and K give similarity scores; softmax normalizes them to attention weights; the output is a weighted sum of Values. Dividing by √dₖ prevents vanishing gradients from large dot products. This allows each token to directly attend to any other token — unlike RNNs that must propagate information through intermediate steps.

---

**Q2: Why divide attention scores by √dₖ?** 🔴 Advanced

For random Q and K vectors of dimension dₖ, the dot product Q·K has expected magnitude ≈ √dₖ (variance scales with dₖ). Without scaling, large dot products push softmax into regions of near-zero gradients (the function is nearly 0 or 1 everywhere), making learning difficult. Dividing by √dₖ keeps the input to softmax in a range where gradients are meaningful (~variance 1).

---

**Q3: What is the difference between self-attention and cross-attention?** 🟡 Intermediate

In **self-attention**, Q, K, and V all come from the same sequence — each position attends to all other positions in the same sequence. Used in encoders and decoder's first sublayer.

In **cross-attention**, Q comes from one sequence (decoder) and K, V come from another (encoder output). This allows the decoder to use encoder representations when generating the output. Used in the second sublayer of encoder-decoder architectures.

---

**Q4: What is masked self-attention and why is it needed in the decoder?** 🟡 Intermediate

During training, the decoder receives the target sequence all at once. Without masking, each decoder position could attend to future positions — cheating by seeing the answer. Masked self-attention applies a causal mask that sets attention scores for future positions to -inf (after softmax → weight ≈ 0). This ensures each decoder position only attends to previous positions, matching the autoregressive inference behavior.

---

**Q5: What are the computational complexity differences between self-attention and RNNs?** 🔴 Advanced

| Operation | Self-Attention | RNN |
|-----------|---------------|-----|
| Per-layer computation | O(n²·d) | O(n·d²) |
| Sequential operations | O(1) | O(n) |
| Max path length between positions | O(1) | O(n) |

Self-attention: O(n²) in sequence length (attention matrix) but O(1) sequential depth — enables full parallelization. RNN: O(n) sequential depth — cannot parallelize across the sequence. For short sequences, RNNs are competitive; for long sequences, Transformers dominate due to parallelism.

---

**Q6: Explain multi-head attention and why it's useful.** 🟡 Intermediate

Instead of one large attention computation, multi-head attention splits the model dimension into h smaller attention heads, each operating on d_model/h dimensions. Each head can learn different types of relationships — syntactic, semantic, coreference, local context. The heads' outputs are concatenated and projected back. This is more expressive than single-head attention of the same total dimension and at the same computational cost.

---

**Q7: What is the feed-forward sublayer in a Transformer and what is its role?** 🟡 Intermediate

The FFN sublayer applies two linear transformations with a ReLU (original) or GELU/SwiGLU (modern) between them: `FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂`. It is applied independently to each position. It acts as a position-wise learned transformation — where attention mixes information across positions, the FFN processes each position's representation independently. It typically has 4× the model dimension (d_ff = 4·d_model), providing most of the model's capacity in modern LLMs.

---

**Q8: What is the difference between sinusoidal and learned positional encodings?** 🟡 Intermediate

Sinusoidal PE uses fixed mathematical functions (sin/cos at different frequencies) — no learned parameters, generalizes to sequence lengths longer than training, relative positions have predictable relationships. Learned PE trains an embedding matrix — potentially better for the specific training distribution but doesn't generalize to longer sequences. Modern models use RoPE (Rotary Positional Embedding) which encodes relative positions by rotating Q/K vectors and generalizes better to long sequences through techniques like YaRN and RoPE scaling.

---

**Q9: Why do Transformers scale better than RNNs?** 🔴 Advanced

1. **Parallelization:** Transformers process all positions simultaneously on GPUs; RNNs are sequential
2. **Gradient flow:** Direct attention connections provide O(1) gradient paths between any positions; RNNs suffer vanishing gradients over long sequences
3. **Parameter efficiency:** Transformer layers are largely parallelizable; adding layers doesn't hurt training stability as much
4. **Empirical observation (scaling laws):** Model performance scales predictably as a power law with parameters, compute, and data — enabling systematic scaling

---

**Q10: What is Flash Attention?** 🔴 Advanced

Flash Attention is an IO-aware exact attention implementation that avoids materializing the full N×N attention matrix. Standard attention writes the N×N matrix to HBM (slow) then reads it back. Flash Attention uses tiling: processes blocks of Q, K, V in SRAM (fast), computes attention incrementally using the softmax trick to maintain running statistics, and writes only the final output to HBM. Result: 2-4x memory reduction, 3-7x speed improvement. Critical for training on long context windows (>4K tokens).

---

**Q11: What is the context window of a Transformer and what limits it?** 🟡 Intermediate

The context window is the maximum number of tokens a Transformer can attend to at once. Limits: (1) quadratic memory and compute of attention (`O(n²)`), (2) positional encoding generalization — models trained on short sequences may perform poorly on longer ones. Modern techniques: sliding window attention (Mistral), sparse attention (Longformer), efficient attention approximations (Linformer), and RoPE with extended scaling (enabling 128K+ context). GPT-4 supports 128K tokens; Claude 3.5 Sonnet supports 200K.

---

**Q12: How does BERT's pretraining differ from GPT's?** 🟡 Intermediate

**BERT** uses Masked Language Model (MLM): randomly masks 15% of tokens and trains to predict them using bidirectional context. Also uses Next Sentence Prediction (NSP). Result: bidirectional representations excellent for understanding tasks.

**GPT** uses Causal Language Model (CLM): predicts the next token given all previous tokens. Trained on standard left-to-right text. Result: strong autoregressive generation capability.

The key difference: BERT sees the full context bidirectionally (knows the "future"), GPT sees only past context. BERT is better for classification/understanding; GPT is better for generation.

---

**Q13: What is attention head pruning and why might you do it?** 🔴 Advanced

Research (Michel et al. 2019) showed many attention heads in trained Transformers are redundant — some heads have little effect on output. Attention head pruning removes low-importance heads to create smaller, faster models. Importance is measured by the effect of masking a head on the loss. Can reduce model size by 20-30% with minimal performance loss. Used in model compression for deployment on resource-constrained hardware.

---

**Q14: What is the difference between absolute and relative position encodings?** 🔴 Advanced

**Absolute positional encoding** assigns a unique encoding to each absolute position (1, 2, 3, ...). Can struggle with generalization to positions not seen in training and doesn't explicitly model relative distances.

**Relative positional encoding** (Shaw et al., ALiBi, RoPE) encodes the relative distance between two positions rather than absolute positions. Key advantage: "5 positions apart" means the same regardless of where in the sequence. RoPE (used in Llama, Mistral) achieves this by rotating Q and K vectors by position-dependent angles — the dot product naturally depends only on the relative position.

---

**Q15: What is the scaling law for Transformers?** 🔴 Advanced

Kaplan et al. (OpenAI, 2020) showed model performance (cross-entropy loss) scales as power laws with:
- Model parameters N: `L ~ N^(-0.076)`
- Training tokens D: `L ~ D^(-0.095)`
- Compute budget C: `L ~ C^(-0.050)`

These laws allow prediction of model performance before training. Chinchilla (DeepMind, 2022) refined this: for a given compute budget, you should train a ~4x smaller model on 4x more data than previously thought optimal. This led to data-efficient models like Llama, which prioritize large training datasets over large model sizes.

---

## References

- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al. (2018)](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3) — Brown et al. (2020)](https://arxiv.org/abs/2005.14165)
- [FlashAttention — Dao et al. (2022)](https://arxiv.org/abs/2205.14135)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
- [The Illustrated Transformer — Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [CS224n Stanford NLP — Lecture on Transformers](http://web.stanford.edu/class/cs224n/)
- [Scaling Laws for Neural Language Models — Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361)
