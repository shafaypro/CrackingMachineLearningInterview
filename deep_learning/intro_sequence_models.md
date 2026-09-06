# Sequence Models: RNNs, LSTMs, GRUs, and Seq2Seq

Transformers replaced recurrent networks for most language tasks, but sequence models remain standard interview material — they are where the vanishing-gradient problem becomes concrete, where the attention mechanism was invented to fix a real bottleneck, and where the "why transformers won" story actually starts. They are also still the right tool for streaming, low-latency, and small-data sequence problems.

---

## Table of Contents
1. [Why Sequences Need Different Architectures](#why-sequences-need-different-architectures)
2. [The Vanilla RNN](#the-vanilla-rnn)
3. [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
4. [LSTM](#lstm)
5. [GRU](#gru)
6. [Bidirectional and Stacked RNNs](#bidirectional-and-stacked-rnns)
7. [Seq2Seq and the Bottleneck](#seq2seq-and-the-bottleneck)
8. [Attention: The Fix That Became the Architecture](#attention-the-fix-that-became-the-architecture)
9. [RNNs vs Transformers in Practice](#rnns-vs-transformers-in-practice)
10. [Practical Training Notes](#practical-training-notes)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## Why Sequences Need Different Architectures

A feed-forward network on sequence data has three problems: it needs a fixed input size, it has no notion of order, and it learns a separate weight for every position, so a pattern learned at position 3 does not transfer to position 40.

Recurrent networks solve all three with **weight sharing across time**. One set of parameters is applied at every step, carrying a hidden state forward:

```
h_t = f(h_{t-1}, x_t)
```

This is the sequence analogue of what convolution does for images: the same feature detector applies everywhere, so the parameter count is independent of sequence length and patterns generalize across positions.

---

## The Vanilla RNN

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

```python
import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    """One recurrent step, written out — this is the whiteboard version."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t, h_prev):
        # x_t: (batch, input_size)   h_prev: (batch, hidden_size)
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


# Unrolling over a sequence
def run_sequence(cell, xs, hidden_size):
    # xs: (batch, seq_len, input_size)
    batch = xs.size(0)
    h = torch.zeros(batch, hidden_size, device=xs.device)
    outputs = []
    for t in range(xs.size(1)):
        h = cell(xs[:, t, :], h)      # same weights reused at every step
        outputs.append(h)
    return torch.stack(outputs, dim=1), h   # (batch, seq_len, hidden), (batch, hidden)
```

**Backpropagation Through Time (BPTT)** is ordinary backprop on this unrolled graph. Its cost is what motivates everything that follows: the gradient at step 1 must pass through `T` multiplications by `W_hh`.

**Truncated BPTT** caps that by backpropagating only `k` steps (commonly 32–256) instead of the full sequence. It bounds memory and compute at the price of never learning dependencies longer than `k`.

---

## Vanishing and Exploding Gradients

Differentiating the loss at step `T` with respect to `h_1` produces a product of Jacobians:

```
∂h_T/∂h_1 = Π_{t=2..T} W_hh^T · diag(tanh'(z_t))
```

Two things go wrong, both exponential in sequence length:

| Failure | Condition | Symptom | Fix |
|---|---|---|---|
| **Vanishing** | Largest singular value of `W_hh` < 1, or `tanh'` saturated | Early steps get no gradient; model learns only recent context | Gated units (LSTM/GRU), orthogonal init, skip connections |
| **Exploding** | Largest singular value > 1 | Loss spikes to NaN; weights blow up | **Gradient clipping**, lower LR |

Note `tanh'(z) = 1 - tanh²(z) ≤ 1`, and it approaches 0 as the unit saturates — so the activation derivative alone shrinks the signal at every step. This is why vanishing is the *typical* case and exploding is the dramatic one.

Exploding gradients are easy to fix — clip the global norm and move on:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Vanishing gradients are not fixable by clipping; they need an architectural change. That change is gating.

---

## LSTM

The LSTM adds a **cell state** `C_t` that flows through time with only elementwise operations — no repeated matrix multiplication. Gates decide what to erase, what to write, and what to expose.

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        forget gate  — what to keep from C_{t-1}
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        input gate   — how much new info to write
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)     candidate    — the new content
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t            cell update  — the additive path
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        output gate  — what to expose
h_t = o_t ⊙ tanh(C_t)                      hidden state
```

**Why this fixes vanishing gradients.** The cell update is *additive*: `∂C_t/∂C_{t-1} = f_t`. If the forget gate stays near 1, the gradient flows back through many steps essentially unattenuated — there is no repeated matrix multiplication on the cell path. This is the same trick as a residual connection, discovered for sequences two decades earlier.

```python
lstm = nn.LSTM(
    input_size=300,
    hidden_size=512,
    num_layers=2,
    batch_first=True,
    dropout=0.3,          # applied BETWEEN layers only, not across time steps
    bidirectional=False,
)
# output: (batch, seq_len, hidden)   h_n, c_n: (num_layers, batch, hidden)
output, (h_n, c_n) = lstm(embedded)
```

**Initialize the forget-gate bias to 1.** At initialization the gates sit near 0.5, so the cell state halves at every step and long-range gradient still decays. Biasing the forget gate positive makes the network default to *remembering*, and it learns to forget where useful. This is a small change with a large effect on convergence, and naming it is a strong interview signal.

```python
for name, param in lstm.named_parameters():
    if 'bias_ih' in name or 'bias_hh' in name:
        n = param.size(0)
        param.data[n // 4 : n // 2].fill_(1.0)   # PyTorch gate order: i, f, g, o
```

---

## GRU

The GRU merges the forget and input gates into a single **update gate** and drops the separate cell state.

```
z_t = σ(W_z · [h_{t-1}, x_t])                update gate — interpolation weight
r_t = σ(W_r · [h_{t-1}, x_t])                reset gate  — how much past to use
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])         candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t        convex blend of old and new
```

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| State | Separate cell + hidden | Hidden only |
| Parameters | ~4 × (d_in + d_h) × d_h | ~3 × (d_in + d_h) × d_h (**25% fewer**) |
| Speed | Slower | ~20–30% faster |
| Typical edge | Long sequences, large data | Small data, tight compute |

Empirically the two are close, and which wins is task-dependent rather than principled. The defensible interview answer: try GRU first because it is cheaper and trains faster; move to LSTM if the task has long dependencies and you have the data to support the extra parameters.

---

## Bidirectional and Stacked RNNs

**Bidirectional** runs one RNN forward and another backward, concatenating the states, so every position sees both left and right context. It typically gives a solid accuracy gain on classification and tagging — but it requires the **entire sequence up front**, so it is unusable for streaming, real-time transcription, or autoregressive generation. That constraint is the interview point, not the accuracy number.

**Stacking** layers lets lower layers capture local patterns and higher layers capture longer-range structure. Two to four layers is the practical range; beyond that, returns diminish and optimization gets harder without residual connections.

```python
nn.LSTM(input_size=300, hidden_size=256, num_layers=3,
        bidirectional=True, batch_first=True, dropout=0.3)
# output width is 2 × hidden_size = 512 because directions are concatenated
```

---

## Seq2Seq and the Bottleneck

The encoder-decoder architecture maps an input sequence to an output sequence of a different length — translation, summarization, speech recognition.

```
Encoder RNN  →  final hidden state (the "context vector")  →  Decoder RNN
```

The problem is stated in one sentence: **the entire input must be compressed into one fixed-size vector.** For a 5-word sentence that is fine; for a 50-word one, information is lost, and translation quality measurably degrades as source length grows. This bottleneck is the direct motivation for attention.

**Teacher forcing** trains the decoder on ground-truth previous tokens rather than its own predictions. It speeds convergence dramatically, but creates **exposure bias**: at inference the model consumes its own outputs, a distribution it never trained on, so one early mistake compounds. Scheduled sampling — mixing in the model's own predictions with increasing probability — is the classic partial mitigation.

**Decoding** is a separate decision from training. Greedy decoding takes the argmax at each step and is fast but myopic. **Beam search** keeps the `k` highest-probability partial sequences and usually improves quality on translation; it needs length normalization, since raw sequence probability decreases monotonically with length and would otherwise favor short outputs.

---

## Attention: The Fix That Became the Architecture

Attention removed the bottleneck by letting the decoder look back at *every* encoder state, weighted by relevance to the current decoding step.

```
score(s_t, h_i)  →  α_ti = softmax_i(score)  →  context_t = Σ_i α_ti · h_i
```

| Variant | Score function | Note |
|---|---|---|
| **Additive** (Bahdanau, 2014) | `vᵀ tanh(W₁s + W₂h)` | A small MLP; works when dimensions differ |
| **Multiplicative** (Luong, 2015) | `sᵀ W h` or `sᵀh` | Cheaper; one matmul |
| **Scaled dot-product** (2017) | `sᵀh / √d_k` | The transformer's; scaling keeps softmax out of saturation |

Two things followed. First, quality on long sequences improved sharply, because there is no longer a fixed-size summary. Second — and this is the historically important part — the attention weights turned out to carry most of the useful signal, which raised the question the transformer answered: **if attention does the work, is the recurrence needed at all?** Removing it made the whole sequence parallelizable during training, which is what unlocked training at scale.

---

## RNNs vs Transformers in Practice

| Dimension | RNN / LSTM | Transformer |
|---|---|---|
| Training parallelism | Sequential in `T` — cannot parallelize across time | Fully parallel across positions |
| Complexity per layer | `O(T · d²)` | `O(T² · d)` — quadratic in length |
| Path length between distant tokens | `O(T)` | `O(1)` |
| Memory at inference | `O(1)` — fixed state | `O(T)` — KV cache grows |
| Long sequences (T ≫ d) | Cheaper | Expensive without sparse/linear attention |
| Small datasets | Often competitive | Data-hungry |
| Streaming / real-time | Natural fit | Needs windowing or a cache |

**When an RNN is still the right choice**: strict streaming with unbounded input and constant memory (real-time speech, sensor telemetry, anomaly detection on event streams); very long sequences where `T² ` attention is prohibitive; small labeled datasets where a transformer overfits; and tight edge deployments where an `O(1)`-state model beats a growing KV cache. State-space models (S4, Mamba) are the modern revival of exactly this argument — recurrent-style linear scaling with much better long-range modeling.

---

## Practical Training Notes

**Padding and masking.** Batches contain variable-length sequences, so shorter ones are padded. Feeding padding through the RNN corrupts the final hidden state and pollutes the loss. Use packed sequences, and mask the loss:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
packed_out, (h_n, c_n) = lstm(packed)          # h_n now reflects each sequence's TRUE last step
output, _ = pad_packed_sequence(packed_out, batch_first=True)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)   # exclude padding from the loss
```

Without packing, `h_n` is the state after consuming trailing `<pad>` tokens — a genuinely common and hard-to-spot bug that quietly caps accuracy.

**Dropout across time.** PyTorch's `dropout` argument applies *between layers*, not between time steps, and it is silently ignored when `num_layers=1`. Applying independent dropout at each time step is harmful — it resamples the mask every step and destroys the memory the recurrence is meant to carry. Variational dropout (the same mask at every step) is the correct form when you want recurrent regularization.

**Gradient clipping is not optional.** Unlike transformers, where it is good hygiene, RNNs genuinely explode without it. Clip global norm at 1.0–5.0.

**Sort or bucket by length.** Batching sequences of wildly different lengths wastes compute on padding. Length-bucketed batching often gives a 2–3x throughput win for free.

---

## Interview Q&A

#### Why do vanilla RNNs fail on long sequences, and how does an LSTM fix it?

The gradient from step `T` back to step `1` is a product of `T` Jacobians, each containing `W_hh` and the `tanh` derivative. Since `tanh' ≤ 1` and typically well below it, and `W_hh`'s singular values are rarely exactly 1, that product shrinks or grows exponentially in sequence length. Shrinking is the common case: early steps receive effectively no learning signal, so the model only learns short-range dependencies.

The LSTM adds a cell state updated **additively**: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`. The gradient along that path is `∂C_t/∂C_{t-1} = f_t`, an elementwise multiply by a learned gate rather than a repeated matrix multiplication. With the forget gate near 1, gradients flow back over hundreds of steps largely intact. It is the same mechanism as a residual connection — give the gradient an uninterrupted highway.

#### LSTM or GRU — how do you choose?

GRU has two gates instead of three and no separate cell state, so roughly 25% fewer parameters and 20–30% faster training. LSTM's extra output gate gives finer control over what is exposed versus stored, which tends to help on long sequences with lots of data.

In practice the accuracy difference is small and task-dependent, so I'd start with GRU for the cheaper iteration loop and switch to LSTM if I have long dependencies and enough data to support the extra capacity. What I would not do is claim one is universally better — the literature does not support that, and the honest answer is that it is an empirical choice.

#### What is the seq2seq bottleneck, and how did attention solve it?

In a basic encoder-decoder, the encoder compresses the whole input into one fixed-size context vector. Capacity is constant while input length grows, so information is lost and translation quality degrades measurably with source length.

Attention removes the constraint: the decoder computes a relevance score between its current state and *every* encoder hidden state, softmaxes those into weights, and takes a weighted sum. Each decoding step gets a context vector tailored to what it currently needs, so nothing has to be compressed away. The follow-on insight — that attention was doing the heavy lifting and recurrence could be dropped entirely — is what produced the transformer, and with it full training parallelism.

#### What is teacher forcing and what problem does it create?

During training, the decoder is fed the ground-truth previous token rather than its own prediction. This makes training much faster and more stable, since the decoder never has to recover from its own early mistakes while the model is still random.

The problem is **exposure bias**: at inference there is no ground truth, so the model consumes its own outputs — a distribution it never saw in training. One wrong token puts it in an unfamiliar state, and errors compound over the sequence. Scheduled sampling (gradually mixing in the model's own predictions during training) and sequence-level objectives are the standard mitigations, though modern practice largely sidesteps it by training at scale on next-token prediction.

#### When would you use a bidirectional RNN, and when can't you?

Use it whenever the full sequence is available before you need an output and both directions carry signal — text classification, named entity recognition, POS tagging, offline speech transcription. Knowing that a word appears before "Inc." helps classify it, and only the backward pass sees that.

You cannot use it for anything causal or streaming: real-time transcription, autoregressive generation, or online anomaly detection, because the backward pass requires future tokens that do not exist yet. That constraint — not the accuracy difference — is what determines the choice.

#### Why do transformers beat RNNs, and when do RNNs still win?

Two reasons. **Parallelism**: an RNN's step `t` depends on `t-1`, so training cannot be parallelized across time; a transformer processes all positions simultaneously, which is what made training on internet-scale data feasible. **Path length**: information between two distant tokens traverses `O(T)` steps in an RNN, with degradation at each, versus `O(1)` through attention.

RNNs still win when `T` is very large, because attention is `O(T²)` while recurrence is linear; when inference memory must be constant, since the KV cache grows with context; in true streaming settings with unbounded input; and on small datasets where transformers overfit. State-space models like Mamba are a direct modern attempt to keep the linear scaling and constant state while fixing the long-range weakness.

#### Your LSTM's validation accuracy plateaus well below expectations. How do you debug it?

I'd check the mechanical bugs before touching the architecture, because they're common and silent:
1. **Padding handling** — is the model consuming `<pad>` tokens into its final hidden state? Without `pack_padded_sequence`, `h_n` reflects padding, not the real last token. Is the loss masked with `ignore_index`?
2. **Gradient clipping** — missing clipping shows up as loss spikes; check whether the run is silently diverging and recovering.
3. **Can it overfit a single batch?** If not, the bug is in the code, not the hyperparameters.
4. **Sequence truncation** — is the max length cutting off the part of the input that carries the label?
5. **Forget-gate bias** — initialized to 0 means the cell state halves each step and long-range memory never forms.

Then the modeling questions: are the embeddings pretrained or learned from scratch on too little data, is the sequence long enough to need attention, and is a bidirectional or stacked variant warranted.

#### What is truncated BPTT and what does it cost you?

Full BPTT unrolls the entire sequence and backpropagates through all of it, so memory scales with sequence length and long sequences become infeasible. Truncated BPTT backpropagates only `k` steps back, carrying the hidden state forward across chunks but detaching the gradient at chunk boundaries.

The cost is that the model can never learn a dependency longer than `k` steps, because no gradient signal crosses the boundary. So `k` is a direct cap on the range of learnable dependencies, and choosing it means estimating how far back the relevant context actually sits — a modeling decision, not just a memory knob.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Not packing padded sequences | Final hidden state reflects `<pad>` tokens, silently capping accuracy | `pack_padded_sequence` / `pad_packed_sequence` |
| Unmasked loss over padding | Model is rewarded for predicting padding | `CrossEntropyLoss(ignore_index=PAD_IDX)` |
| No gradient clipping | RNNs genuinely explode; loss goes NaN | `clip_grad_norm_(..., 1.0)` |
| Forget-gate bias left at 0 | Cell state decays by half each step; long memory never forms | Initialize forget-gate bias to 1 |
| `dropout=` with `num_layers=1` | Silently does nothing — PyTorch applies it between layers | Stack layers, or apply dropout explicitly |
| Independent dropout per time step | Destroys the recurrent memory being learned | Variational dropout (same mask across time) |
| Bidirectional model for a streaming task | Requires future tokens that do not exist at inference | Unidirectional, or windowed lookahead |
| Beam search without length normalization | Sequence probability shrinks with length, so short outputs win | Divide by length, or use a length penalty |
| Batching wildly different lengths | Most compute spent on padding | Length-bucketed batching |
| Reaching for an RNN on a modern NLP task | Transformers dominate where data and parallelism exist | Use a pretrained transformer unless streaming or tiny data argues otherwise |

---

## Related Topics

- [Transformers](./intro_transformers.md)
- [Neural Network Training](./intro_neural_network_training.md)
- [Applied Deep Learning Roadmap](./intro_applied_deep_learning.md)
- [Time Series & Forecasting](../classical_ml/intro_time_series.md)
- [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [PyTorch](../frameworks/intro_pytorch.md)
- [Deep Learning Overview](./README.md)
