# LLM Inference Optimization

Training an LLM is a one-time cost; serving it is a bill that arrives every month. This is now one of the most-asked areas in AI engineering interviews, because it is where most production LLM budgets and latency SLOs are won or lost. This guide covers the two-phase nature of inference, KV caching, batching, quantization, speculative decoding, and how to reason about latency and cost.

---

## Table of Contents
1. [The Two Phases of Inference](#the-two-phases-of-inference)
2. [Latency Metrics That Matter](#latency-metrics-that-matter)
3. [KV Cache](#kv-cache)
4. [Batching Strategies](#batching-strategies)
5. [Attention Kernel Optimizations](#attention-kernel-optimizations)
6. [Quantization](#quantization)
7. [Speculative Decoding](#speculative-decoding)
8. [Serving Frameworks](#serving-frameworks)
9. [Cost Modeling](#cost-modeling)
10. [Application-Level Optimizations](#application-level-optimizations)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## The Two Phases of Inference

Autoregressive generation has two phases with completely different hardware characteristics. Almost every optimization exists because of this split, and saying so is the fastest way to show you understand the area.

| Phase | What happens | Bottleneck | Parallelism |
|---|---|---|---|
| **Prefill** | Process the whole prompt in one forward pass | **Compute-bound** (large matmuls) | All prompt tokens at once |
| **Decode** | Generate one token at a time | **Memory-bandwidth-bound** | One token per step, per sequence |

During decode, generating a single token requires reading *every model weight* from HBM to compute one small matrix-vector product. For a 13B model in fp16, that's ~26 GB read per token. On an A100 with ~2 TB/s bandwidth, the floor is ~13 ms/token regardless of how fast the GPU's FLOPs are — the arithmetic intensity is terrible.

This single fact explains the field:
- **Batching helps enormously in decode** — the weights are read once and amortized across all sequences in the batch, so throughput scales nearly linearly until compute finally binds.
- **Quantization helps decode directly** — half the bytes to read is roughly half the time per token.
- **Speculative decoding helps** — it verifies several tokens in one weight-read pass.
- **More FLOPs alone barely helps decode** — which is why raw TFLOPs is a misleading spec for a serving GPU.

---

## Latency Metrics That Matter

| Metric | Definition | Driven by |
|---|---|---|
| **TTFT** (time to first token) | Request arrival → first token emitted | Prefill: prompt length, queueing, batch scheduling |
| **TPOT / ITL** (time per output token) | Steady-state inter-token latency | Decode: memory bandwidth, batch size |
| **End-to-end latency** | `TTFT + TPOT × output_tokens` | Both |
| **Throughput** | Tokens/sec across all requests | Batch size, GPU utilization |
| **Goodput** | Throughput of requests that met their SLO | The number that actually matters |

```
Total latency = TTFT + (output_tokens - 1) × TPOT
```

For a chat UI, TTFT dominates perceived responsiveness — stream tokens and users tolerate a slower TPOT as long as text appears quickly. For a batch summarization job, only throughput matters and you should push batch size until TTFT is awful, because nobody is waiting.

**There is a fundamental throughput/latency tradeoff.** Larger batches raise throughput and raise per-request latency. Serving one model to both an interactive chat product and an offline pipeline from the same endpoint is usually a mistake — run separate deployments with different batch configurations.

---

## KV Cache

In attention, each new token attends to all previous tokens. Recomputing keys and values for the whole prefix at every step would be `O(n²)` work per sequence. The KV cache stores them instead, making each decode step `O(n)`.

```
KV cache bytes = 2 (K and V)
              × num_layers
              × num_kv_heads × head_dim
              × sequence_length
              × batch_size
              × bytes_per_element
```

Worked example — Llama-3-8B (32 layers, 8 KV heads via GQA, head_dim 128) in fp16, 8k context:

```
2 × 32 × 8 × 128 × 8192 × 2 bytes ≈ 1.07 GB per sequence
```

At batch size 32 that's **34 GB of KV cache** — larger than the 16 GB of model weights. On long-context workloads the KV cache, not the model, is what limits how many concurrent requests fit on a GPU.

### Reducing KV cache size

| Technique | Mechanism | Saving |
|---|---|---|
| **MQA** (multi-query attention) | All query heads share one K/V head | `num_heads`x smaller |
| **GQA** (grouped-query attention) | Heads share K/V in groups | 4–8x, with better quality than MQA |
| **MLA** (multi-head latent attention) | Compress K/V to a low-rank latent | Large; used by DeepSeek models |
| **KV cache quantization** | Store cache in int8/fp8 | 2–4x |
| **PagedAttention** | Non-contiguous paged allocation | Eliminates fragmentation waste (~2x effective) |
| **Sliding window / eviction** | Keep only recent + sink tokens | Bounded memory, some quality loss |

**PagedAttention** (vLLM's core contribution) is worth explaining precisely: naive serving preallocates a contiguous buffer sized to the maximum possible sequence length for every request, so a request that generates 100 tokens with a 4096 reservation wastes 97% of its allocation. PagedAttention borrows OS virtual memory: the cache is split into fixed-size blocks allocated on demand and tracked through a block table. Fragmentation drops to near zero, and identical prefixes can share blocks via copy-on-write — which is what makes prefix caching for shared system prompts nearly free.

---

## Batching Strategies

| Strategy | How it works | Problem it has |
|---|---|---|
| **No batching** | One request at a time | Wastes almost all GPU throughput |
| **Static batching** | Wait for N requests, run them together | Whole batch waits for the longest generation |
| **Dynamic batching** | Batch what has arrived within a time window | Still blocks on the longest sequence |
| **Continuous / in-flight batching** | Evict finished sequences and admit new ones every step | The standard; 10–20x throughput over static |

Continuous batching matters because generation lengths in a real workload vary by 100x. With static batching, a batch of 32 where one request generates 2,000 tokens and the rest generate 20 keeps 31 slots idle for the whole run. Continuous batching frees each slot the moment its sequence emits EOS and immediately admits a queued request. This is the single largest throughput win available in LLM serving, and it is why vLLM/TGI/TensorRT-LLM exist rather than a plain `model.generate()` loop.

**Chunked prefill** solves a related problem: a long prompt's prefill pass occupies the GPU and stalls every decoding request behind it, spiking their inter-token latency. Splitting prefill into chunks interleaved with decode steps smooths TPOT at a small cost to TTFT.

---

## Attention Kernel Optimizations

**FlashAttention** does not change the math; it changes the memory traffic. Standard attention materializes the `n × n` score matrix in HBM, so memory traffic is `O(n²)`. FlashAttention tiles the computation in SRAM and uses the online-softmax trick to never write the full matrix out, making traffic `O(n²/M)` for SRAM size `M`. The result is 2–4x faster attention and memory that scales linearly in sequence length — which is what made long-context training and inference practical at all.

**PagedAttention** (memory allocation) and **FlashAttention** (kernel efficiency) are complementary and both used in modern servers. A common interview slip is treating them as alternatives.

Other kernel-level levers: CUDA graphs to remove per-step launch overhead (significant when each decode step is only a few milliseconds), fused kernels for the norm+attention+MLP sequence, and tensor parallelism to split a model that doesn't fit on one GPU (with the caveat that it adds an all-reduce on every layer, so it only pays off across fast interconnect like NVLink).

---

## Quantization

Quantization reduces the numerical precision of weights and sometimes activations. Since decode is memory-bandwidth-bound, fewer bytes per weight translates almost directly into faster generation.

| Method | Bits | Type | Quality impact | Notes |
|---|---|---|---|---|
| **fp16 / bf16** | 16 | Baseline | — | Standard serving precision |
| **fp8** | 8 | Post-training | Very small | H100+ native support |
| **INT8 (SmoothQuant, LLM.int8)** | 8 | Post-training | Small | Handles outlier activation channels |
| **GPTQ** | 4 | Post-training, calibrated | Small–moderate | Layer-wise second-order error correction |
| **AWQ** | 4 | Post-training, activation-aware | Small | Protects salient weight channels; fast kernels |
| **GGUF (llama.cpp)** | 2–8 | Post-training | Varies by level | The CPU/edge standard |
| **QLoRA (NF4)** | 4 | Training-time | Small | For fine-tuning, not primarily serving |

Rules of thumb worth stating in an interview: 8-bit is essentially free in quality terms and should be the default. 4-bit costs a small but measurable amount of quality — usually acceptable, and worth it when it lets you fit a larger model on the same GPU. Below 4-bit, degradation becomes obvious on reasoning tasks.

**A larger 4-bit model usually beats a smaller fp16 model of the same memory footprint.** A 4-bit 70B model (~35 GB) generally outperforms an fp16 13B (~26 GB) on quality. That framing — quantization as a way to buy a bigger model rather than as a way to shrink a fixed one — is the answer interviewers are looking for.

**Verify quality yourself.** Perplexity barely moves under quantization while task accuracy can drop noticeably, especially on structured output, tool calling, and multi-step reasoning. Run your own eval set on the quantized model before shipping.

---

## Speculative Decoding

A small, fast **draft model** proposes `k` tokens; the large **target model** verifies all `k` in a single forward pass. Accepted tokens are kept, and the first rejection resamples from a corrected distribution.

Why it works: verifying `k` tokens costs almost the same as generating 1, because decode is memory-bandwidth-bound — you read the weights once either way. The speedup is roughly the mean number of accepted tokens per verification step, typically 2–3x on predictable text.

```
Expected speedup ≈ (1 - α^(k+1)) / ((1 - α)(1 + c·k))
   α = draft acceptance rate,  c = draft cost relative to target
```

Key properties:
- **Output distribution is provably unchanged** with the standard rejection-sampling acceptance rule. This is the crucial point: it is not an approximation and does not trade quality for speed.
- Acceptance rate matters most. A draft model too weak gets rejected constantly and you pay its cost for nothing; too strong and it's expensive to run.
- Variants that avoid a separate draft model: **Medusa** (extra prediction heads on the target model), **EAGLE** (feature-level drafting), **n-gram / prompt lookup decoding** (draft by copying from the prompt — remarkably effective for summarization, RAG, and code editing where output copies input heavily).

Speculative decoding improves **latency at low batch sizes**. At high batch sizes the GPU is already compute-saturated and there is no spare capacity to spend on verification, so gains shrink or vanish — a nuance worth mentioning.

---

## Serving Frameworks

| Framework | Strengths | Best for |
|---|---|---|
| **vLLM** | PagedAttention, continuous batching, wide model support, OpenAI-compatible API | The default self-hosted choice |
| **TensorRT-LLM** | Fastest on NVIDIA with compiled kernels; FP8 support | Maximum performance, willing to pay in build complexity |
| **TGI** (HuggingFace) | Production-ready, good observability, easy HF integration | HF-centric stacks |
| **SGLang** | RadixAttention prefix caching, strong structured output | Heavy prefix sharing, agents, constrained decoding |
| **llama.cpp / Ollama** | CPU and Apple Silicon, GGUF quantization | Local development, edge |
| **Managed APIs** | Zero ops, elastic scale | Variable load, small teams, frontier models |

```python
# vLLM: offline batch inference
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,   # fraction of VRAM for weights + KV cache
    max_model_len=8192,            # caps KV cache per sequence
    enable_prefix_caching=True,    # big win for shared system prompts
    tensor_parallel_size=1,
)
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, params)   # continuous batching is automatic
```

**Self-host vs API** is a cost-and-control question, not an ideology. Self-hosting wins on high, steady volume (a dedicated GPU amortizes well above roughly 20–30% utilization), on data residency requirements, and when you need a custom fine-tune. APIs win on spiky or low volume, on access to frontier models, and on avoiding an on-call rotation for GPU infrastructure. Model the break-even explicitly: an A100 at roughly $1–2/hour is ~$1,000/month, which buys a substantial number of API tokens.

---

## Cost Modeling

```python
def monthly_cost_api(requests_per_day, in_tokens, out_tokens, price_in, price_out):
    """price_* in dollars per 1M tokens."""
    daily = requests_per_day * (in_tokens * price_in + out_tokens * price_out) / 1e6
    return daily * 30

def monthly_cost_self_hosted(gpu_hourly, num_gpus, utilization=1.0):
    return gpu_hourly * num_gpus * 24 * 30 / max(utilization, 1e-9)

# Break-even sanity check
api  = monthly_cost_api(50_000, 2000, 500, price_in=0.25, price_out=1.25)   # ≈ $1,690/mo
host = monthly_cost_self_hosted(gpu_hourly=1.20, num_gpus=1)                # ≈ $864/mo
```

Two structural points that matter more than the arithmetic:

1. **Input tokens usually dominate** in RAG and agent systems. A 4,000-token retrieved context with a 200-token answer is 95% input. Optimizing output length is the wrong lever; trimming retrieved context, reranking to fewer chunks, and caching the prompt prefix are the right ones.
2. **Prompt/prefix caching changes the economics.** Providers charge a fraction (often ~10%) for cached input tokens, and self-hosted prefix caching skips the prefill compute entirely. Putting the stable system prompt and few-shot examples first, with the variable part last, is a design decision worth several times the effort of micro-optimizing wording.

---

## Application-Level Optimizations

Often larger wins than anything at the kernel level, and cheaper to implement:

- **Prompt caching** — order prompts stable-prefix-first so the cache hits. Frequently a 50–90% input cost reduction for agents and RAG.
- **Semantic caching** — embed the query, and if a near-identical past query exists, return its answer. Excellent for FAQ-style traffic; needs a similarity threshold tuned carefully and an invalidation strategy, or it will confidently serve stale answers.
- **Model routing** — send easy requests to a small model and hard ones to a large one, with a classifier or heuristic in front. Typically 60–80% cost reduction with minimal quality impact when the routing signal is decent.
- **Shorter outputs** — instruct the model to be concise, cap `max_tokens`, and return structured data rather than prose. Output tokens are usually 4–5x the price of input.
- **Streaming** — doesn't change cost or total time, but it slashes *perceived* latency, which is often the actual complaint.
- **Batch APIs** — providers offer roughly 50% discounts for asynchronous batch jobs with a 24-hour window. Free money for offline workloads.
- **Stop sequences and early termination** — stop generating the moment the useful content ends.
- **Context trimming** — retrieve 20, rerank, pass 5. Fewer tokens, and usually *better* answers, since irrelevant context measurably degrades quality.

---

## Interview Q&A

#### Why is LLM inference memory-bandwidth-bound rather than compute-bound?

During decode, one token is generated per step per sequence, so each weight matrix participates in a matrix-*vector* product — a tiny amount of arithmetic per byte loaded. The GPU must stream every model weight from HBM for each step: ~26 GB for a 13B fp16 model. On an A100 at ~2 TB/s that's ~13 ms of pure memory traffic per token, while the arithmetic takes a fraction of that. The GPU sits idle waiting on memory.

Prefill is different: the whole prompt is processed at once, so it's a matrix-matrix product with high arithmetic intensity and it is compute-bound. This is why prefill and decode have different optimizations, why batching multiplies decode throughput almost for free, and why quantization directly speeds up generation.

#### Explain the KV cache and why it can be larger than the model.

Attention needs the keys and values of every previous token. Without a cache, each new token requires recomputing them for the whole prefix — `O(n²)` per sequence. The KV cache stores them, making each step `O(n)`.

Its size is `2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes`, which grows linearly in *both* sequence length and batch size while the model weights are fixed. For Llama-3-8B at 8k context, one sequence costs ~1 GB; 32 concurrent sequences cost ~34 GB against 16 GB of fp16 weights. That's why long-context, high-concurrency serving is a KV cache capacity problem, and why GQA, cache quantization, and PagedAttention exist.

#### What is continuous batching and why is it such a large win?

Static batching waits for a fixed group of requests, runs them together, and returns when the *longest* generation finishes. Real workloads have generation lengths varying by 100x, so most slots sit idle for most of the batch's life.

Continuous (in-flight) batching operates at the token level: after every decode step, finished sequences are evicted and queued requests are admitted into the free slots. The GPU stays saturated regardless of length variance. Measured throughput improvements over static batching are typically 10–20x, which is why it's the defining feature of vLLM, TGI, and TensorRT-LLM rather than an optional tuning knob.

#### How does speculative decoding preserve output quality?

The draft model proposes `k` tokens and the target model verifies them in one forward pass, which is nearly free because decode is bandwidth-bound. Each drafted token is accepted with probability `min(1, p_target(x)/p_draft(x))`; on rejection, the token is resampled from the normalized residual distribution `max(0, p_target - p_draft)`.

That acceptance rule is exactly modified rejection sampling, and it makes the resulting sequence distribution **provably identical** to sampling from the target model alone. So it is a pure latency optimization with no quality tradeoff — the tradeoff is compute spent on the draft model and the fact that gains shrink at high batch sizes, where the GPU has no idle capacity to spend on verification.

#### Your chat app has a 4-second time to first token. How do you diagnose and fix it?

TTFT is prefill plus queueing, so I'd first measure which one dominates — instrument queue wait, prefill time, and network separately before changing anything.

If it's **queueing**, the server is saturated: add replicas, enable continuous batching if it isn't on, or use chunked prefill so long prompts stop head-of-line-blocking short ones.

If it's **prefill compute**, the prompt is too long: enable prefix caching so the stable system prompt is computed once, trim retrieved context (rerank 20 down to 5), and drop verbose few-shot examples. Prefill scales with prompt length, so halving the prompt roughly halves TTFT.

If it's **model size**, route easy requests to a smaller model, or use a smaller model with a larger context budget.

Independently of all this, I'd make sure the response streams — a 4-second wall before any text is a much worse experience than 800 ms to first token and a slower stream, at the same total latency.

#### How would you cut LLM serving costs by 70% without hurting quality much?

Layered, measuring at each step:
1. **Prompt/prefix caching** — often the single biggest win, since input tokens dominate RAG and agent workloads and cached input is roughly 10% of the price.
2. **Trim context** — retrieve 20, rerank, send 5. Cheaper *and* usually more accurate, since irrelevant context degrades answers.
3. **Model routing** — a small model handles the 70–80% of easy requests; escalate only when a classifier or confidence signal says so.
4. **Semantic caching** for repeated or near-repeated queries.
5. **Cap output length** and prefer structured output over prose; output tokens cost several times input.
6. **Batch API** (~50% discount) for anything that isn't interactive.
7. **Quantize to int8/fp8** if self-hosting, or move to 4-bit to fit a larger model on the same GPU.

I'd hold a fixed eval set throughout, because every one of these can quietly degrade quality, and the point is cost reduction *at constant quality*.

#### When should you self-host a model instead of using an API?

Self-host when volume is high and steady enough to amortize a GPU (a dedicated A100 is ~$1,000/month, so the break-even is a real calculation, not a preference), when data cannot leave your network for regulatory reasons, when you need a custom fine-tune or LoRA adapters served at scale, when you need latency guarantees a shared API can't promise, or when you require a specific quantization or sampling behavior.

Use an API when load is spiky (you pay for idle GPUs otherwise), when you need frontier-model quality you can't self-host, when the team has no GPU operations capacity, or during early product development where the model choice is still changing weekly. The honest answer usually starts with an API and moves specific high-volume workloads in-house once the traffic shape is known.

#### What quality risk does 4-bit quantization carry, and how do you validate it?

The risk is uneven: perplexity often moves very little, which makes quantization look safe, while task-level accuracy on reasoning, structured output, and tool calling can drop noticeably. Outlier activation channels are the usual culprit — a few dimensions with very large magnitudes dominate the quantization range, which is exactly what AWQ and SmoothQuant are designed to protect against.

Validation: run the *application's* eval set, not a generic benchmark. Compare fp16 and quantized on task success rate, JSON schema validity, and tool-call correctness; check the tails, not just the mean. If quality holds, 4-bit is usually the right call — and a 4-bit 70B typically beats an fp16 13B at the same memory, which is the real reason to quantize.

#### What's the difference between FlashAttention and PagedAttention?

They solve different problems and are used together. **FlashAttention** is a kernel optimization: it tiles attention computation in SRAM and uses online softmax so the `n × n` attention matrix is never written to HBM, reducing memory traffic and making attention 2–4x faster with memory linear rather than quadratic in sequence length. **PagedAttention** is a memory *allocator*: it stores the KV cache in fixed-size non-contiguous blocks with a block table, eliminating the internal fragmentation of preallocating max-length buffers and enabling copy-on-write sharing of common prefixes.

FlashAttention makes each attention computation faster; PagedAttention lets more sequences fit in memory concurrently. Framing them as alternatives is a common mistake.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Optimizing FLOPs for decode | Decode is bandwidth-bound; FLOPs barely matter | Optimize bytes moved: quantization, GQA, batching |
| Using `model.generate()` in a loop for serving | No continuous batching, ~10-20x throughput lost | vLLM, TGI, or TensorRT-LLM |
| Setting `max_model_len` to the model maximum by default | KV cache reservation collapses concurrency | Set it to the real p99 context you serve |
| Sizing a GPU by model weights only | KV cache often exceeds weights at scale | Budget weights + KV for target concurrency |
| Variable content at the start of the prompt | Destroys prefix cache hits | Stable system prompt first, variable input last |
| One deployment for interactive and batch traffic | The batch config ruins interactive latency | Separate deployments with different batch settings |
| Measuring average latency only | Tail latency is what users and SLOs feel | Track p50/p95/p99 for TTFT and TPOT separately |
| Trusting perplexity to validate quantization | Task accuracy can drop while perplexity barely moves | Run the application eval set on the quantized model |
| Stuffing all retrieved context "just in case" | More cost *and* worse answers | Rerank and pass fewer, better chunks |
| Semantic cache with no invalidation | Confidently serves stale answers | TTL, source-change invalidation, tuned threshold |

---

## Related Topics

- [LLM Fundamentals](./intro_llm_fundamentals.md)
- [LLMOps](./intro_llmops.md)
- [Multi-Model Orchestration](./intro_multi_model_orchestration.md)
- [Embeddings](./intro_embeddings.md)
- [vLLM](../frameworks/intro_vllm.md)
- [Ollama](../frameworks/intro_ollama.md)
- [Transformers](../deep_learning/intro_transformers.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Backend AI System Design](../system_design/intro_backend_ai_system_design.md)
