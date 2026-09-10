# Model Compression: Distillation, Pruning, and Quantization

Training a model is a research problem; making it small and fast enough to serve is an engineering one. Compression is where most of the cost and latency of a deployed system is actually won, and it comes up constantly in ML Engineer and MLOps interviews as "how would you make this fit / run faster / cost less?"

---

## Table of Contents
1. [Why Compress](#why-compress)
2. [Where the Size Goes](#where-the-size-goes)
3. [Knowledge Distillation](#knowledge-distillation)
4. [Pruning](#pruning)
5. [Quantization](#quantization)
6. [Low-Rank Factorization](#low-rank-factorization)
7. [Architecture-Level Efficiency](#architecture-level-efficiency)
8. [Compiling and Runtime Optimization](#compiling-and-runtime-optimization)
9. [Combining Techniques](#combining-techniques)
10. [Measuring What Matters](#measuring-what-matters)
11. [Technique Selection](#technique-selection)
12. [Interview Q&A](#interview-qa)
13. [Common Pitfalls](#common-pitfalls)
14. [Related Topics](#related-topics)

---

## Why Compress

Four distinct pressures, and naming which one you're solving determines the technique:

| Pressure | Symptom | Best levers |
|---|---|---|
| **Memory** | Model doesn't fit on the device | Quantization, pruning, distillation |
| **Latency** | p99 misses the SLO | Distillation, structured pruning, compilation |
| **Throughput / cost** | GPU bill or QPS ceiling | Quantization, batching, smaller model |
| **Energy** | Battery or thermal limits on edge | Quantization (especially int8), architecture choice |

The mistake to avoid: assuming "smaller" means "faster". **Unstructured pruning can cut 80% of weights and change latency by nothing at all**, because dense hardware kernels still process the zeros. Compression that reduces parameter count is not automatically compression that reduces time. That distinction is the single most useful thing to know here.

---

## Where the Size Goes

For a transformer with hidden size `d` and `L` layers, parameters are roughly:

```
Attention per layer:  4 · d²           (Q, K, V, O projections)
MLP per layer:        8 · d²           (typically 4× expansion, up and down)
Total:               ~12 · L · d²  + embeddings (vocab × d)
```

Two consequences worth stating in an interview:

- **The MLP is about two-thirds of the parameters**, so compression targeting only attention leaves most of the model untouched.
- **Embeddings dominate for small models** with large vocabularies — a 100k-token vocabulary at `d=768` is 77M parameters before any layers exist. Weight tying (sharing input and output embeddings) is the standard fix.

At inference, memory is weights + activations + (for generative models) the KV cache, which grows with sequence length and batch size and frequently exceeds the weights entirely.

---

## Knowledge Distillation

Train a small **student** to imitate a large **teacher**, using the teacher's full output distribution rather than just hard labels.

**Why soft targets help** is the core interview question. A hard label says "this is a cat". The teacher's soft distribution says "cat 0.85, lynx 0.10, dog 0.03, car 0.0001" — encoding that lynxes resemble cats and cars do not. That similarity structure, sometimes called *dark knowledge*, is a much richer training signal per example than a one-hot vector, so the student learns more from the same data.

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """Blend soft teacher targets with hard ground-truth labels."""
    # Temperature softens both distributions, exposing the small-probability structure
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean',
    ) * (T ** 2)          # rescale: soft gradients shrink as 1/T², this restores their magnitude

    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

The `T²` factor is a detail interviewers like: raising temperature shrinks gradient magnitudes by roughly `1/T²`, so without rescaling the soft term silently loses influence as `T` grows.

**Variants:**

| Type | Matches | Use |
|---|---|---|
| **Response (logit)** | Output distribution | Classification; the classic |
| **Feature** | Intermediate activations | Deeper transfer; needs a projection when widths differ |
| **Attention transfer** | Attention maps | Transformers; cheap and effective |
| **Sequence-level** | Generated sequences | Translation, generation; teacher generates the training data |
| **Self-distillation** | The model teaches itself | Regularization; often improves the teacher-sized model too |

For LLMs, the dominant practical form is **sequence-level distillation on generated data**: have the large model produce outputs for a large prompt set, then fine-tune the small model on those. It's simple, needs no logit access (so it works with API-only teachers, subject to terms of service), and captures reasoning style rather than only token distributions.

Distillation is the one technique that reliably delivers **real latency wins**, because the student is genuinely a smaller architecture — fewer layers, narrower — not a large model with holes in it.

---

## Pruning

Remove weights or structures deemed unimportant.

### Unstructured vs structured — the distinction that matters

| | Unstructured | Structured |
|---|---|---|
| Removes | Individual weights | Whole neurons, channels, heads, layers |
| Sparsity achievable | 80–95% | 30–50% |
| Accuracy at equal sparsity | **Better** | Worse |
| **Actual speedup on GPU** | **~None** without special kernels | **Real** — the tensor is genuinely smaller |
| Result | Sparse matrix, same shape | Smaller dense model |

This is the crux. Unstructured pruning produces a matrix full of zeros; a standard dense GEMM kernel multiplies those zeros just as fast as any other number, so you save storage (if stored sparse) and nothing else. Structured pruning removes an entire channel, so the weight matrix is physically smaller and every kernel benefits automatically.

**Semi-structured (2:4) sparsity** is the middle ground: exactly 2 of every 4 contiguous weights are zero, a pattern NVIDIA Ampere+ tensor cores accelerate natively for roughly 2× on those matmuls.

```python
import torch.nn.utils.prune as prune

# Unstructured magnitude pruning: smaller |w| assumed less important
prune.l1_unstructured(layer, name='weight', amount=0.5)

# Structured: remove entire output channels by L2 norm — this actually shrinks compute
prune.ln_structured(layer, name='weight', amount=0.3, n=2, dim=0)

prune.remove(layer, 'weight')   # make it permanent (otherwise it's a mask at runtime)
```

**Iterative magnitude pruning** works far better than one-shot: prune a fraction, fine-tune to recover, repeat. Removing 50% at once and retraining loses much more accuracy than five rounds of 13%.

**The Lottery Ticket Hypothesis** is worth knowing: within a large randomly-initialized network there exist sparse subnetworks ("winning tickets") that, *when reset to their original initialization*, train to comparable accuracy alone. The rewinding detail is the whole finding — the same sparse structure with fresh random weights does not work, which suggests initialization and structure are jointly what matters.

---

## Quantization

Reduce numerical precision. The highest-leverage technique for LLM inference, because decode is memory-bandwidth-bound — fewer bytes per weight translates almost directly into faster generation.

```
fp32 → fp16/bf16   2× smaller,  ~free
     → int8        4× smaller,  small quality cost
     → int4        8× smaller,  measurable but usually acceptable cost
```

The affine mapping:

```
scale = (max - min) / (2^bits - 1)
q = round(x / scale) + zero_point
x̂ = (q - zero_point) · scale        # dequantized, with error
```

### PTQ vs QAT

| | Post-Training Quantization | Quantization-Aware Training |
|---|---|---|
| Needs training? | No — a small calibration set | Yes, full fine-tune |
| Effort | Hours | Days |
| Quality at int8 | Usually fine | Slightly better |
| Quality at int4 and below | Degrades | **Noticeably better** |
| Use when | Default; 8-bit | Aggressive bit widths, or PTQ failed |

QAT inserts fake quantization during training so the model learns weights robust to the rounding, using a **straight-through estimator** to pass gradients through the non-differentiable rounding operation.

### Granularity and the outlier problem

Per-tensor quantization uses one scale for a whole weight matrix; **per-channel** uses one per output channel and is substantially better for almost no cost. For activations, **per-token** granularity is common in LLMs.

The central difficulty in LLM quantization is **activation outliers**: a small number of feature dimensions carry values orders of magnitude larger than the rest, and they dominate the quantization range, crushing precision for everything else. The named solutions all address exactly this:

- **LLM.int8()** — isolate outlier dimensions and compute them in fp16, the rest in int8.
- **SmoothQuant** — migrate difficulty from activations to weights by rescaling, since weights are easier to quantize.
- **GPTQ** — layer-wise quantization using second-order information to compensate error as it goes.
- **AWQ** — identify salient weight channels (by activation magnitude) and protect them.

```python
# int8 dynamic quantization — one line, works well for CPU-bound linear layers
import torch
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**The framing that lands in interviews**: quantization is usually better spent buying a *bigger* model than shrinking a fixed one. A 4-bit 70B model (~35 GB) generally beats an fp16 13B (~26 GB) on quality at comparable memory. Treat bit-width as a knob for "which model fits", not merely "how small can this get".

---

## Low-Rank Factorization

Replace a `d × d` weight matrix with `A·B` where `A` is `d × r` and `B` is `r × d`, cutting parameters from `d²` to `2dr`. Meaningful only when `r ≪ d/2`.

Applied post-hoc via SVD it tends to lose accuracy, because trained weight matrices are often not low-rank. It shines in two other places: **LoRA**, where the *update* rather than the weight is constrained to low rank (updates genuinely are low-rank, which is why it works so well), and in factorized embedding layers, where large vocabularies make the saving substantial.

---

## Architecture-Level Efficiency

Often better than compressing a bad architecture. Worth naming because it shows breadth:

- **GQA / MQA** — share key/value heads across query heads, shrinking the KV cache by 4–8× with minimal quality loss. This is the highest-impact efficiency change in modern LLMs.
- **Mixture of Experts** — many parameters, few active per token; high capacity at low inference FLOPs, at the cost of memory and routing complexity.
- **Depthwise separable convolutions** — MobileNet's core trick, factorizing a convolution into depthwise + pointwise for ~8–9× fewer operations.
- **Early exit / cascades** — a cheap model handles the easy majority and escalates only uncertain cases. Frequently a 60–80% cost reduction for negligible quality change, and it requires no model surgery at all.

Cascading deserves emphasis: it's the cheapest big win available, purely a serving-layer change, and candidates routinely forget it while reaching for exotic compression.

---

## Compiling and Runtime Optimization

No accuracy cost at all — do these before touching the weights.

| Technique | Typical gain | Mechanism |
|---|---|---|
| **Operator fusion** | 1.2–2× | Fewer kernel launches and memory round-trips |
| **`torch.compile` / TensorRT / ONNX Runtime** | 1.3–3× | Graph optimization, kernel selection, fusion |
| **CUDA graphs** | Meaningful at small batch | Removes per-launch overhead |
| **FlashAttention** | 2–4× on attention | Tiling in SRAM; avoids materializing the score matrix |
| **Batching** | Large on throughput | Amortizes weight reads across requests |

```python
model = torch.compile(model, mode="max-autotune")   # free speedup, no quality change
```

---

## Combining Techniques

They compose, in a sensible order:

```
1. Fix the architecture      (GQA, right model size)
2. Distill                   → genuinely smaller student
3. Structured prune          → remove redundant channels/heads
4. Quantize                  → int8 or int4
5. Compile                   → fusion, kernel selection
6. Serve well                → batching, caching, cascading
```

Do the **free** things first — compilation, batching, cascading cost no accuracy. Then distillation, which gives real architectural savings. Quantization last, because it's easy to apply and easy to reverse if quality drops.

Compounding losses are the risk: each step may cost 1% and the stack costs 5%. Evaluate after every step, not only at the end, so you know which one broke it.

---

## Measuring What Matters

Report all of these, not just size:

| Metric | Why |
|---|---|
| **Task accuracy on your eval set** | Not perplexity — see below |
| **p50 / p95 / p99 latency** | At realistic batch size and sequence length |
| **Throughput (QPS or tokens/s)** | The cost-relevant number |
| **Peak memory** | Weights + activations + KV cache |
| **Cost per 1k requests** | What the business asks about |

**Perplexity is a trap for compression evaluation.** It barely moves under quantization while task accuracy — especially on reasoning, structured output, and tool calling — can drop noticeably. Always run the application's own eval set, and look at the tails rather than the mean.

Benchmark honestly: warm up before timing, use realistic input distributions rather than fixed-length dummies, measure on the target hardware, and include tokenization and pre/post-processing.

---

## Technique Selection

| Goal | First choice | Why |
|---|---|---|
| Model won't fit in memory | **Quantization** | Largest immediate reduction, minimal effort |
| Need lower latency | **Distillation + compilation** | Genuinely fewer FLOPs and better kernels |
| Need higher throughput | **Quantization + batching** | Decode is bandwidth-bound |
| Edge / mobile deployment | **Distill → structured prune → int8** | Compounding, and int8 is well supported |
| Have GPU, tight budget | **Cascading + quantization** | Cheapest wins first |
| Cannot lose any accuracy | **Compilation, batching, caching** | Zero quality cost by construction |
| Large vocabulary model | **Weight tying + factorized embeddings** | Embeddings dominate at small `d` |

---

## Interview Q&A

#### You pruned 80% of the weights and latency didn't change. Why?

Because it was **unstructured** pruning. The weights are zero, but the tensor is the same shape, and a dense GEMM kernel multiplies zeros at exactly the same speed as anything else. You've saved storage — if you actually store it in a sparse format — and no compute.

To get real speedup you need one of: **structured pruning**, which removes whole channels or heads so the matrices are physically smaller and every kernel benefits; **2:4 semi-structured sparsity**, which Ampere+ tensor cores accelerate natively for around 2× on those matmuls; or a genuine sparse kernel, which typically needs very high sparsity (>95%) before it beats dense.

The general lesson is that parameter count and latency are different currencies, and compression targets one or the other.

#### Explain knowledge distillation. Why do soft targets work better than hard labels?

You train a small student to match a large teacher's output distribution, usually blending a KL term against the teacher's softened logits with the ordinary cross-entropy against ground truth.

Soft targets carry more information per example. A one-hot label says "cat" and nothing else. The teacher's distribution says cat 0.85, lynx 0.10, dog 0.03, car 1e-4 — encoding which classes are similar and how confident the teacher is. That relative structure ("dark knowledge") is a far richer signal, effectively teaching the student the teacher's learned similarity metric rather than just the answer.

Temperature exposes that structure: softmax at T=1 can be nearly one-hot, hiding the small probabilities that carry the signal. Raising T flattens the distribution so those relationships influence the loss. The `T²` rescaling matters because soft-target gradients scale as `1/T²`, so without it the soft term fades as you raise the temperature.

#### Walk me through quantization. What makes LLMs hard to quantize?

Quantization maps high-precision values to a small integer grid via a scale and zero point, then dequantizes at compute time. Going fp16 → int8 is 2× smaller, int4 is 4× smaller, and because decode is memory-bandwidth-bound, fewer bytes per weight means proportionally faster generation.

What makes LLMs specifically hard is **activation outliers**: in large transformers, a handful of hidden dimensions carry values orders of magnitude larger than the rest. Since the quantization scale must cover the full range, those few outliers stretch the range and crush the resolution available to every normal value, so quality collapses.

Every well-known method addresses precisely this: LLM.int8() keeps outlier dimensions in fp16 and quantizes the rest; SmoothQuant rescales to shift the difficulty from activations onto weights, which quantize more gracefully; GPTQ uses second-order information to compensate error layer by layer; AWQ protects the channels that matter most by activation magnitude.

#### PTQ or QAT?

PTQ by default. It needs only a small calibration set, takes hours instead of days, and at 8 bits the quality difference is usually negligible. Start there and measure.

QAT when PTQ isn't good enough — typically at 4 bits and below, or in domains sensitive to small numerical changes. It simulates quantization during training with a straight-through estimator so the model learns weights robust to rounding, which recovers much of the gap at aggressive bit widths. The cost is a full fine-tuning run and the pipeline complexity that comes with it, so I'd only pay it after PTQ demonstrably fails on the actual eval set.

#### How do you verify a compressed model is still good enough?

Not with perplexity. It's the classic trap — perplexity barely moves under quantization while task performance can degrade noticeably, especially on reasoning chains, structured output validity, and tool-call correctness.

I'd run the **application's own eval set** and compare task success rate, schema validity, and any domain metric against the uncompressed baseline. I'd look at the tails rather than the mean, since compression often hurts hard examples specifically. Then benchmark p50/p95/p99 latency, throughput, and peak memory on the **target hardware** with realistic input lengths.

If several techniques are stacked, I'd evaluate after each step rather than only at the end — otherwise a 5% total drop gives no clue which stage caused it.

#### What's the cheapest way to cut inference cost without touching the model?

**Cascading**, and it's routinely overlooked. Route every request to a small cheap model first, and escalate to the large one only when a confidence signal, schema check, or verifier says the cheap answer isn't good enough. For typical traffic where most requests are easy, 60–80% of volume never touches the expensive model, and quality is essentially preserved because hard cases still escalate.

Alongside that: **batching**, which amortizes the memory-bandwidth cost of reading weights across requests and is the largest single throughput lever; **caching**, both exact-match and prefix/prompt caching; and **compilation** via `torch.compile` or TensorRT, which is a pure win with no accuracy change. All of these are serving-layer changes with zero risk to model quality, so they should be exhausted before any weight surgery.

#### What is the Lottery Ticket Hypothesis?

The claim that a large randomly-initialized network contains sparse subnetworks — winning tickets — that can be trained in isolation to accuracy comparable to the full network, in similar time.

The essential detail is **rewinding**: the winning subnetwork must be reset to its *original* initialization values. Take the same sparse structure with fresh random weights and it trains poorly. That implies the structure and the specific initialization are jointly responsible, which is why the result was interesting rather than just an observation that networks are overparameterized.

Practically, it's more an explanation of why iterative magnitude pruning works than a deployable recipe, since finding the ticket requires training the dense model first. Its lasting influence is the norm of iterative prune-and-retrain rather than one-shot pruning.

#### A 7B model doesn't fit your GPU. Options in order?

1. **Quantize** — int8 roughly halves it versus fp16, int4 quarters it. A 7B model goes from ~14 GB to ~4 GB at 4-bit, which fits almost anything. Cheapest and most effective first move.
2. **Reduce KV cache** — it frequently exceeds the weights at long context. Cap `max_model_len` to real p99 usage, quantize the cache, and use a GQA model.
3. **Offload** — keep some layers in CPU memory. Works, but the PCIe transfer makes it slow; a last resort for latency-tolerant workloads.
4. **Distill to a smaller student** — real work, but yields a genuinely smaller model rather than a compressed large one.
5. **Tensor parallelism** across GPUs — adds an all-reduce per layer, so it only pays off over fast interconnect.

And the framing point: rather than squeezing the 7B, ask whether a 4-bit 13B in the same memory budget would be *better*. Quantization is usually best spent on fitting a larger model than on shrinking a fixed one.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Expecting unstructured pruning to speed things up | Dense kernels process zeros at full cost | Structured pruning, 2:4 sparsity, or accept storage-only savings |
| Evaluating compression by perplexity | Barely moves while task accuracy drops | Run the application eval set; check the tails |
| One-shot pruning to the target sparsity | Much larger accuracy loss | Iterative prune → fine-tune → repeat |
| Per-tensor quantization for LLM weights | Outliers destroy resolution | Per-channel weights, per-token activations |
| Ignoring activation outliers | int8 quality collapses | SmoothQuant, LLM.int8(), AWQ, GPTQ |
| Stacking techniques and evaluating once | Can't attribute the accuracy loss | Evaluate after each stage |
| Forgetting `prune.remove()` | Pruning stays a runtime mask; no benefit | Make it permanent before export |
| Benchmarking without warm-up | First-call overhead dominates the number | Warm up, then measure p50/p95/p99 |
| Benchmarking on fixed-length dummy inputs | Hides real-world variance | Use the production length distribution |
| Compressing before compiling and batching | Spends accuracy on wins that were free | Do zero-cost optimizations first |
| Dropping the `T²` factor in distillation | Soft loss silently loses weight as T rises | Multiply the soft term by `T²` |

---

## Related Topics

- [Neural Network Training](./intro_neural_network_training.md)
- [Fine-Tuning](./intro_fine_tuning.md)
- [Transformers](./intro_transformers.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [vLLM](../frameworks/intro_vllm.md)
- [Ollama](../frameworks/intro_ollama.md)
- [Deep Learning Overview](./README.md)
