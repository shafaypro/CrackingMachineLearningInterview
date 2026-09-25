# Distributed and Large-Scale Model Training

Once a model or its batch stops fitting on one GPU, training becomes a systems problem: how to split memory, compute, and communication across hundreds of devices without the network becoming the bottleneck. This comes up in interviews at any company training foundation models or fine-tuning large ones, and increasingly as a general probe of whether you understand what "we trained a 70B model on 512 GPUs" actually involves. The questions are predictable: where does the memory go, what does ZeRO shard, why is tensor parallelism kept inside a node, what is a pipeline bubble, and what do you do when the job hangs at step 40,000.

---

## Table of Contents
1. [Where the Memory Goes](#where-the-memory-goes)
2. [Communication Collectives and Interconnect](#communication-collectives-and-interconnect)
3. [Data Parallelism and DDP](#data-parallelism-and-ddp)
4. [ZeRO and FSDP](#zero-and-fsdp)
5. [Tensor Parallelism](#tensor-parallelism)
6. [Pipeline Parallelism](#pipeline-parallelism)
7. [Sequence and Context Parallelism](#sequence-and-context-parallelism)
8. [Expert Parallelism for MoE](#expert-parallelism-for-moe)
9. [3D Parallelism and How to Choose](#3d-parallelism-and-how-to-choose)
10. [Mixed Precision](#mixed-precision)
11. [Activation Checkpointing and Gradient Accumulation](#activation-checkpointing-and-gradient-accumulation)
12. [Large-Batch Training](#large-batch-training)
13. [Fault Tolerance and Checkpointing](#fault-tolerance-and-checkpointing)
14. [Throughput Metrics](#throughput-metrics)
15. [Debugging Distributed Jobs](#debugging-distributed-jobs)
16. [Interview Q&A](#interview-qa)
17. [Common Pitfalls](#common-pitfalls)
18. [Related Topics](#related-topics)

---

## Where the Memory Goes

Every distributed training strategy is an answer to one question: which of these tensors do we split, and across which devices?

| Component | Size (mixed-precision Adam) | Notes |
|---|---|---|
| Parameters (bf16/fp16) | 2 bytes/param | Used in forward and backward |
| Gradients (bf16/fp16) | 2 bytes/param | Same shape as parameters |
| fp32 master weights | 4 bytes/param | Optimizer updates these, then casts down |
| Adam first moment `m` | 4 bytes/param | fp32 |
| Adam second moment `v` | 4 bytes/param | fp32 |
| **Model states total** | **16 bytes/param** | Before any activations |
| Activations | Depends on batch, sequence length, hidden size, depth | Often the largest term for long sequences |
| Temporary buffers, fragmentation | Several GB | Communication buffers, allocator overhead |

The **16 bytes/param** rule is the number to have memorized. Note that the optimizer (12 bytes) dominates, not the weights themselves.

**Worked example: a 7B-parameter model.**

```
Model states:   7e9 × 16 bytes = 112 GB
  bf16 params      14 GB
  bf16 grads       14 GB
  fp32 master      28 GB
  Adam m, v        56 GB

Inference only (bf16 weights): 7e9 × 2 = 14 GB
```

112 GB does not fit on one 80 GB GPU before a single activation is stored. So full fine-tuning of a 7B model with Adam already needs sharding (ZeRO/FSDP), CPU offload, or a memory-lighter method (LoRA, 8-bit optimizers). With ZeRO-3 across 8 GPUs, model states drop to 14 GB per GPU, leaving room for activations.

**Activations.** For a transformer layer with sequence length `s`, micro-batch `b`, hidden size `h`, and `a` heads, the Megatron-LM activation analysis gives roughly:

```
activation bytes per layer ≈ s · b · h · (34 + 5 · a · s / h)     (16-bit activations, no recomputation)
```

The `5as/h` term is the attention score matrix, which is quadratic in `s`. FlashAttention never materializes it, which removes that term and is a large part of why long-context training became practical. Multiply by the number of layers and activations easily exceed model states at long sequence lengths, which is why activation checkpointing and sequence parallelism exist.

```python
def training_memory_gb(n_params, bytes_per_param=16):
    """Model-state memory for mixed-precision Adam, excluding activations."""
    return n_params * bytes_per_param / 1e9

training_memory_gb(7e9)    # 112.0
training_memory_gb(70e9)   # 1120.0 -> needs sharding across at least ~16 x 80GB GPUs
```

---

## Communication Collectives and Interconnect

Distributed training is built from a handful of collective operations. Knowing which strategy uses which collective is what lets you reason about cost.

| Collective | What it does | Used by |
|---|---|---|
| **All-reduce** | Every rank ends with the sum (or mean) of all ranks' tensors | DDP gradient sync, tensor-parallel layer outputs |
| **Reduce-scatter** | Sum across ranks, each rank keeps one shard of the result | ZeRO-2/3 and FSDP gradient sync |
| **All-gather** | Each rank contributes a shard, every rank ends with the full tensor | ZeRO-3/FSDP parameter gathering, sequence parallelism |
| **All-to-all** | Each rank sends a distinct chunk to every other rank | MoE token dispatch, Ulysses sequence parallelism |
| **Broadcast** | One rank sends to all | Initial weight sync |
| **Send / Recv (P2P)** | Point-to-point | Pipeline parallelism between stages |

A useful identity: **all-reduce = reduce-scatter + all-gather.** That is exactly why ZeRO-2 costs the same bandwidth as DDP: it performs the two halves separately and keeps only its shard in between.

**Ring all-reduce cost.** With `N` ranks and a tensor of `M` bytes, ring all-reduce runs a reduce-scatter phase and an all-gather phase, each with `N-1` steps moving `M/N` bytes per step. Each rank sends and receives:

```
2 · (N - 1) / N · M   bytes   ≈ 2M for large N
```

The per-rank bandwidth cost is nearly independent of `N`, which is why data parallelism scales. Latency, however, grows with `N` (there are `2(N-1)` steps), so at large scale NCCL uses tree or hierarchical algorithms for small messages.

**Interconnect.** The bandwidth gap between links drives every placement decision.

| Link | Typical scope | Order of magnitude bandwidth |
|---|---|---|
| **NVLink / NVSwitch** | GPUs within a node (e.g. 8 per node) | Hundreds of GB/s per GPU (H100: 900 GB/s aggregate bidirectional) |
| **InfiniBand / RoCE** | Across nodes | ~25-50 GB/s per NIC (200-400 Gb/s), typically one NIC per GPU |
| **PCIe** | GPU to CPU, or GPUs without NVLink | Tens of GB/s |
| **Ethernet (commodity)** | Cloud without RDMA | Often the bottleneck |

The intra-node link is roughly an order of magnitude faster than the inter-node one. Rule of thumb: put the chattiest parallelism (tensor parallelism) inside the node and the least chatty (data parallelism, pipeline P2P) across nodes.

---

## Data Parallelism and DDP

Every GPU holds a full replica of the model, processes a different slice of the batch, and gradients are averaged so all replicas take the same step.

```
1. Each rank: forward + backward on its local micro-batch
2. All-reduce gradients (average across ranks)
3. Each rank: identical optimizer step -> replicas stay in sync
```

PyTorch **DistributedDataParallel (DDP)** makes this efficient by grouping gradients into **buckets** (default 25 MB) and launching an all-reduce for each bucket as soon as its gradients are ready during backward. Communication overlaps with the rest of the backward pass, so for a well-sized model it is mostly hidden.

**DDP vs DataParallel**: `nn.DataParallel` is single-process, multi-threaded, scatters inputs and gathers outputs through GPU 0 every step, and is limited by the Python GIL. DDP runs one process per GPU with no central bottleneck. Always use DDP.

```python
# train_ddp.py  -- launch with: torchrun --nproc_per_node=8 train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    dist.init_process_group(backend="nccl")          # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = build_model().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)                     # otherwise every epoch uses the same shuffle
        for x, y in loader:
            x, y = x.cuda(local_rank, non_blocking=True), y.cuda(local_rank, non_blocking=True)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()                          # gradient all-reduce overlaps with backward
            opt.step()

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "ckpt.pt")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

The limitation is obvious from the memory table: **every rank stores all 16 bytes/param.** DDP scales throughput but does nothing for model size. That is what ZeRO fixes.

---

## ZeRO and FSDP

ZeRO (Zero Redundancy Optimizer, from DeepSpeed) observes that in DDP every rank stores identical optimizer states, gradients, and parameters. It shards them across the data-parallel group in three progressive stages.

Using `Ψ` for parameter count and `N` for data-parallel degree, per-GPU model-state memory is:

| Stage | Shards | Memory per GPU | Communication vs DDP |
|---|---|---|---|
| DDP (baseline) | Nothing | `16Ψ` | 1x (all-reduce of grads) |
| **ZeRO-1** | Optimizer states | `4Ψ + 12Ψ/N` | 1x |
| **ZeRO-2** | + Gradients | `2Ψ + 14Ψ/N` | 1x (reduce-scatter grads, all-gather params after step) |
| **ZeRO-3** | + Parameters | `16Ψ/N` | ~1.5x (all-gather params in forward and again in backward, reduce-scatter grads) |

For the 7B model on 8 GPUs: DDP needs 112 GB/GPU, ZeRO-1 about 38.5 GB, ZeRO-2 about 26 GB, ZeRO-3 about 14 GB.

**How ZeRO-3 runs a layer**: before the layer's forward, all-gather its parameter shards into a full copy; compute; free the gathered copy. Repeat in backward, then reduce-scatter the gradients so each rank keeps only the gradient shard for the parameters it owns, and runs the optimizer on that shard. Prefetching the next layer's all-gather while the current layer computes hides most of the cost.

**FSDP** (Fully Sharded Data Parallel) is PyTorch's native implementation of the same idea.

| FSDP sharding strategy | Equivalent |
|---|---|
| `FULL_SHARD` | ZeRO-3 |
| `SHARD_GRAD_OP` | ZeRO-2 |
| `NO_SHARD` | DDP |
| `HYBRID_SHARD` | ZeRO-3 within a node, replicate across nodes (keeps all-gathers on NVLink) |

The **wrapping unit** matters: FSDP gathers one unit at a time, so wrapping each transformer block separately bounds peak memory to roughly one block's full parameters plus the shards. Wrapping the whole model as one unit gathers everything at once and saves nothing.

```python
# FSDP2 (per-parameter sharding, PyTorch 2.x). Launch with torchrun as above.
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

model = build_transformer()                    # can be built on the "meta" device for huge models
for block in model.layers:                     # shard each transformer block as its own unit
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)               # root: embeddings, final norm, head

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)   # optimizer only ever sees local shards

for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()                            # reduce-scatter of grads happens here
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
```

The older FSDP1 API (`FullyShardedDataParallel(model, auto_wrap_policy=..., sharding_strategy=...)`) does the same thing with a module wrapper; many codebases still use it.

**ZeRO-Offload / ZeRO-Infinity** move optimizer states (and optionally parameters) to CPU RAM or NVMe. It lets you fit much larger models on few GPUs, at the price of PCIe traffic and CPU-side optimizer steps. It suits fine-tuning on a small cluster, not throughput-critical pretraining.

---

## Tensor Parallelism

Tensor parallelism (TP) splits individual weight matrices across GPUs, so each GPU computes part of every layer. The Megatron-LM scheme pairs two splits so that only one collective is needed per block.

**MLP block** (`Y = GeLU(X A) B`):

```
A split by columns:  A = [A1 | A2]       -> GPU i computes GeLU(X Ai)       (no communication; GeLU is elementwise)
B split by rows:     B = [B1 ; B2]       -> GPU i computes GeLU(X Ai) Bi    (partial sums)
Y = Σ_i GeLU(X Ai) Bi                    -> one all-reduce
```

Column-parallel first is what makes this work: each GPU holds complete output columns, so the nonlinearity can be applied locally. If you split `A` by rows, you would need an all-reduce before GeLU.

**Attention block**: split heads across GPUs (Q, K, V projections column-parallel, each GPU owns `a/t` heads and computes attention for them independently), then the output projection is row-parallel, followed by one all-reduce.

| Property | Value |
|---|---|
| Collectives per transformer layer | 2 all-reduces in forward (attention + MLP), 2 in backward |
| What it shards | Parameters, gradients, optimizer states, and much of the activation memory |
| Communication pattern | Blocking, on the critical path, every layer |
| Typical degree | 2-8, within one NVLink node |

Because those all-reduces sit on the critical path of every layer and scale with activation size, **TP is almost always confined to a single node.** Crossing InfiniBand with TP usually destroys throughput. Embedding and output layers are split along the vocabulary dimension, with a parallel cross-entropy that avoids gathering the full logits.

---

## Pipeline Parallelism

Pipeline parallelism (PP) splits the model by layers into `p` stages on different GPUs. Activations flow forward stage to stage via point-to-point sends; gradients flow back.

The naive version leaves all but one GPU idle at any moment. The fix is to split the batch into `m` **micro-batches** so stages can work on different micro-batches at once.

**The bubble.** At the start and end of each step, some stages are idle while the pipeline fills and drains. For GPipe-style schedules:

```
bubble fraction = (p - 1) / (m + p - 1)
```

With `p = 8` stages and `m = 8` micro-batches the bubble is 7/15, or 47% idle. With `m = 64` it drops to about 10%. More micro-batches means a smaller bubble, but each micro-batch must still be large enough to use the GPU efficiently, and the global batch grows.

| Schedule | How it works | Bubble | Activation memory |
|---|---|---|---|
| **GPipe** | All `m` forwards, then all `m` backwards | `(p-1)/(m+p-1)` | Stores activations for all `m` micro-batches |
| **1F1B** (PipeDream-Flush) | After warmup, alternate one forward, one backward | Same as GPipe | Bounded by `p` in-flight micro-batches, independent of `m` |
| **Interleaved 1F1B** | Each GPU holds `v` non-contiguous chunks of layers | Reduced by ~`v`x | Similar to 1F1B, more P2P messages |
| **Zero-bubble variants** | Split backward into input-grad and weight-grad, reschedule | Near zero | More bookkeeping |

The key interview point: **1F1B does not shrink the bubble, it shrinks memory.** Because memory no longer grows with `m`, you can afford many more micro-batches, and that is what shrinks the bubble in practice.

PP's strengths: communication is small (only activations at stage boundaries) and point-to-point, so it tolerates the slower inter-node link. Its weaknesses: the bubble, load balancing between stages (the embedding and output layers make first and last stages uneven), and implementation complexity.

---

## Sequence and Context Parallelism

Long sequences make activations, not parameters, the binding constraint. Two related ideas split along the sequence dimension.

**Sequence parallelism (Megatron-SP)** complements TP. Inside the TP region, LayerNorm and dropout activations are replicated on every TP rank, which wastes memory. Megatron-SP shards those regions along the sequence dimension and replaces each all-reduce with a reduce-scatter (entering the sequence-parallel region) and an all-gather (entering the tensor-parallel region). Same communication volume, activation memory in those regions divided by `t`.

**Context parallelism (CP)** shards the whole sequence across GPUs for all layers, which is needed for sequences of hundreds of thousands of tokens. Everything except attention is per-token and needs no communication. Attention needs every query to see every key, handled in one of two ways:

| Approach | Mechanism | Constraint |
|---|---|---|
| **Ring attention** | Each GPU keeps its query block and passes K/V blocks around a ring, accumulating attention with an online softmax | Overlaps P2P with compute; causal masking needs load balancing (e.g. each rank takes a chunk from the start and one from the end) |
| **All-to-all (DeepSpeed Ulysses)** | All-to-all switches from sequence-sharded to head-sharded, runs full attention per head group, switches back | Degree limited by the number of attention heads (or KV heads with GQA) |

---

## Expert Parallelism for MoE

A mixture-of-experts layer replaces the dense MLP with `E` expert MLPs and a router that sends each token to its top-`k` experts. Parameters grow with `E`; compute per token grows only with `k`.

**Expert parallelism (EP)** places different experts on different GPUs:

```
1. Router picks top-k experts per token
2. All-to-all: dispatch tokens to the GPUs that own their experts
3. Each GPU runs its local experts on the tokens it received
4. All-to-all: return outputs to the tokens' original GPUs, combine with router weights
```

| Issue | Why it matters | Mitigation |
|---|---|---|
| Two all-to-alls per MoE layer | Latency-bound, sensitive to inter-node bandwidth | Keep EP within a node where possible; overlap with compute |
| Load imbalance | A popular expert makes its GPU the straggler for everyone | Auxiliary load-balancing loss, or bias-based balancing without an auxiliary loss |
| Capacity limits | Fixed buffers per expert; overflow tokens are dropped or rerouted | Tune capacity factor (e.g. 1.0-1.25); dropless implementations |
| Memory | All experts' parameters and optimizer states must live somewhere | Combine EP with ZeRO/FSDP over the non-expert parameters |

EP is typically composed with data parallelism: the non-expert layers are replicated across the EP group, and experts are sharded across it.

---

## 3D Parallelism and How to Choose

Large runs combine strategies along separate axes of a device mesh: `total GPUs = DP × TP × PP` (plus CP and EP when used).

A standard placement on 8-GPU nodes:

```
TP  (degree ≤ 8)       inside a node        chatty, per-layer all-reduces, needs NVLink
PP  (degree 2-16+)     across nodes         small P2P messages, tolerates InfiniBand
DP / FSDP (the rest)   outermost            gradient sync overlaps with backward
```

**Decision procedure** (roughly the order practitioners reason in):

| Situation | Start with |
|---|---|
| Model + optimizer fit on one GPU | DDP. Nothing else needed |
| Model fits for inference but not training | FSDP / ZeRO-2 or ZeRO-3 |
| Single layer too large, or FSDP all-gathers dominate at high GPU counts | Add TP within the node |
| Model too large even with TP=8 and FSDP, or cross-node bandwidth is weak | Add PP across nodes |
| Very long sequences, activations dominate | Add sequence / context parallelism, activation checkpointing |
| MoE model | Add EP for the expert layers |

Guiding principles: use the simplest scheme that fits, since each added axis brings complexity and new failure modes. Keep the global batch within what the optimizer tolerates, because DP degree multiplies it. Measure MFU after each change rather than assuming the textbook configuration is best for your cluster.

---

## Mixed Precision

Training in 16-bit roughly halves activation and weight memory and uses tensor cores, which are much faster than fp32 math.

| Format | Exponent bits | Mantissa bits | Max value | Notes |
|---|---|---|---|---|
| fp32 | 8 | 23 | ~3.4e38 | Master weights, optimizer states, reductions |
| **fp16** | 5 | 10 | 65,504 | Narrow range; needs loss scaling |
| **bf16** | 8 | 7 | ~3.4e38 | Same range as fp32, lower precision; no loss scaling needed |
| fp8 E4M3 | 4 | 3 | 448 | Forward activations and weights |
| fp8 E5M2 | 5 | 2 | 57,344 | Gradients (need range more than precision) |

**Loss scaling (fp16 only).** Small gradients underflow to zero in fp16. Multiply the loss by a scale factor `S` before backward, so gradients are `S×` larger. Unscale before the optimizer step, and skip the step and halve `S` if any gradient is inf/NaN. Dynamic scaling increases `S` again after a run of clean steps. bf16 has fp32's exponent range, so it does not need this. That is the main reason it became the default on A100/H100/TPU.

**What stays in fp32** even in "bf16 training": master weights, optimizer states, and usually softmax, LayerNorm statistics, loss computation, and gradient reductions. Updating bf16 weights directly loses small updates because bf16 has only about 3 significant decimal digits.

**fp8** (H100 and later) needs per-tensor (or finer-grained) scaling factors, usually computed from a history of recent absolute maxima ("delayed scaling"), because its range is tiny. Libraries such as NVIDIA Transformer Engine handle this for matmuls; sensitive operations stay in higher precision. It can give a substantial speedup over bf16 for large matmuls, but it needs careful validation against a bf16 baseline.

```python
# bf16 autocast (no scaler needed)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward()

# fp16 requires a GradScaler
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(opt)                          # unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)                              # skips the step if grads are inf/NaN
scaler.update()
```

---

## Activation Checkpointing and Gradient Accumulation

**Activation checkpointing** (gradient checkpointing, recomputation) stores only the inputs to selected blocks during forward and recomputes the internals during backward.

| | No checkpointing | Full checkpointing per layer | Selective checkpointing |
|---|---|---|---|
| Activation memory | All intermediates | Only layer inputs | Keep cheap-to-store, drop cheap-to-recompute (e.g. attention scores) |
| Extra compute | 0 | ~1 extra forward (~33% more total, since backward ≈ 2x forward) | A few percent |

Checkpointing trades compute for memory. The memory saved is often reinvested in a larger micro-batch, which can recover much of the lost throughput.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.layers:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

**Gradient accumulation** runs `k` micro-batches before each optimizer step, so the effective batch is `k × micro_batch × DP`. It is how you reach a target global batch when memory limits the micro-batch. In DDP, skip the all-reduce on the non-final micro-batches:

```python
import contextlib

for i, (x, y) in enumerate(loader):
    sync = (i + 1) % accum_steps == 0
    ctx = contextlib.nullcontext() if sync else model.no_sync()   # DDP: no all-reduce until last micro-batch
    with ctx:
        loss = loss_fn(model(x), y) / accum_steps                 # divide, or gradients are k times too large
        loss.backward()
    if sync:
        opt.step()
        opt.zero_grad(set_to_none=True)
```

Two things to remember: accumulation gives the same gradient as a bigger batch only for batch-independent layers (BatchNorm statistics still see the micro-batch), and the loss normalization must be right when micro-batches have different token counts. Sum token losses and divide by total tokens across the accumulated micro-batches (and ranks), not by the number of micro-batches.

---

## Large-Batch Training

Data parallelism multiplies the batch size, and a bigger batch does not automatically train as well.

- **Linear scaling rule** (SGD): when the batch grows by `k`, multiply the learning rate by `k`. It holds up to a point and was used to train ImageNet ResNets with batches in the thousands.
- **Square-root scaling** is often a better starting point for Adam-family optimizers.
- **Warmup** is essential at large batch or high LR. Ramp the LR linearly over the first hundreds to thousands of steps, while early gradients are large and Adam's second-moment estimates are unreliable.
- **Layer-wise adaptive optimizers** (LARS for SGD, LAMB for Adam) scale each layer's update by the ratio of its weight norm to its update norm, which stabilized very large-batch training for ResNets and BERT.
- **Critical batch size**: beyond some batch size (which depends on the task, and grows as loss falls), extra batch size gives diminishing reductions in steps. Past it you spend more compute for the same result. The gradient noise scale is one way to estimate it.

Practical consequence: adding GPUs through DP is only free up to the critical batch size. Beyond that, adding TP/PP to use more GPUs at a fixed global batch is often better than growing the batch.

---

## Fault Tolerance and Checkpointing

On thousands of GPUs, hardware failures (a GPU falling off the bus, ECC errors, NIC flaps, node reboots) happen routinely over a multi-week run. The question is not whether the job will crash but how much work you lose each time.

| Practice | Why |
|---|---|
| **Sharded / distributed checkpoints** (`torch.distributed.checkpoint`, DeepSpeed) | Each rank writes its own shard in parallel; no rank-0 gather of a terabyte-scale state |
| **Async checkpointing** | Copy to CPU memory, write to storage in the background; training resumes in seconds |
| **Save everything needed for an exact resume** | Model, optimizer, LR scheduler, step count, data loader position, RNG states, loss scaler |
| **Resharding on load** | Resume on a different number of GPUs or parallel layout |
| **Checkpoint interval from failure rate** | Balance checkpoint overhead against expected lost work (interval ≈ sqrt(2 × checkpoint cost × mean time between failures) is a classic approximation) |
| **Elastic launch** (`torchrun --max-restarts`, spare nodes) | Replace a failed node and restart automatically |
| **Health checks** | Run NCCL and GPU burn-in tests on nodes before admitting them; blacklist repeat offenders |

Silent data corruption is the nastier failure: a faulty GPU produces wrong numbers without crashing. Symptoms are unexplained loss spikes or divergence between replicas that should be identical. Some teams periodically compare parameter checksums across data-parallel replicas to catch it.

---

## Throughput Metrics

| Metric | Definition | Use |
|---|---|---|
| **Tokens/sec (per GPU)** | Training tokens processed per second, divided by GPU count | Day-to-day throughput; compare configs on the same hardware |
| **MFU** (Model FLOPs Utilization) | Model FLOPs per second achieved ÷ hardware peak FLOPs | Hardware-independent efficiency; comparable across clusters |
| **HFU** (Hardware FLOPs Utilization) | Like MFU but counts recomputation FLOPs too | Always ≥ MFU; can flatter a checkpointed run |
| **Step time breakdown** | Compute vs exposed communication vs data loading vs bubble | Tells you what to fix |

For a dense transformer, training costs about **6 FLOPs per parameter per token** (2 forward, 4 backward), ignoring attention:

```python
def mfu(n_params, tokens_per_sec, n_gpus, peak_flops_per_gpu):
    achieved = 6 * n_params * tokens_per_sec
    return achieved / (n_gpus * peak_flops_per_gpu)

# 7B model, 64 H100s (~989e12 dense bf16 peak), 0.5M tokens/s total
mfu(7e9, 0.5e6, 64, 989e12)   # ≈ 0.33
```

Doing this sanity check out loud is worth it in an interview. Well-tuned large dense runs typically report MFU in the roughly 35-55% range. Numbers far above that usually mean a counting bug. Use MFU rather than tokens/sec when comparing model sizes, since tokens/sec naturally falls as models grow.

---

## Debugging Distributed Jobs

| Symptom | Likely causes | What to do |
|---|---|---|
| **Hang with no error** | Ranks issued collectives in a different order or count (conditional code path on one rank, uneven number of batches, a rank skipped backward on an unused parameter) | `py-spy dump` every rank to see where each is stuck; `TORCH_DISTRIBUTED_DEBUG=DETAIL`; `DistributedSampler(drop_last=True)` |
| **NCCL timeout / watchdog error** | A hang that hit the timeout, a crashed peer, network faults, one rank far slower | `NCCL_DEBUG=INFO`; check which rank failed first (the others' errors are downstream); PyTorch's flight recorder dumps recent collectives per rank |
| **"Expected to mark a variable ready only once" / unused parameters** | Parts of the model not used on every step | Fix the model, or `find_unused_parameters=True` (costs an extra graph traversal) |
| **Stragglers** (step time set by the slowest rank) | Thermal throttling, a bad NIC, noisy neighbor, data-loading skew, MoE load imbalance | Per-rank step timing; `nvidia-smi` clocks and ECC errors; NCCL tests on suspect nodes; replace the node |
| **Loss spikes** | Bad data batch, LR too high, fp16 overflow, attention logit growth, a faulty GPU | Log the batch at each spike; clip gradients; lower LR / longer warmup; QK-norm or z-loss; skip the batch and rewind to the last checkpoint |
| **Replicas diverge** | Nondeterministic ops, missing gradient sync, silent data corruption | Compare parameter checksums across DP ranks |
| **OOM on one rank only** | Uneven sequence lengths, a larger pipeline stage, rank 0 doing extra work (logging, eval) | Pack or bucket sequences; rebalance stages; move extras off the critical rank |
| **Low MFU with no errors** | Exposed communication, data-loader starvation, too-small micro-batch, large bubble | Profile with `torch.profiler` / Nsight Systems; look for gaps between kernels |

The single most useful habit: **when one rank errors, find the first rank to fail.** In a collective job everything else times out as a consequence, and hundreds of identical timeout messages hide the one real stack trace.

---

## Interview Q&A

#### How much GPU memory does it take to train a 7B model, and why?

With mixed-precision Adam the model states alone cost about 16 bytes per parameter: 2 for bf16 weights, 2 for bf16 gradients, 4 for the fp32 master copy, and 4 each for Adam's two moments. For 7B parameters that is 112 GB before activations. It does not fit on a single 80 GB GPU, even though the same model needs only 14 GB for bf16 inference.

On top of that come activations, which scale with micro-batch size, sequence length, hidden size, and depth, and can exceed model states at long context. So the realistic options are sharding across GPUs (ZeRO-3/FSDP brings it to 14 GB per GPU across 8 GPUs), offloading optimizer states to CPU, using an 8-bit optimizer, or avoiding full fine-tuning entirely with LoRA. I would also mention activation checkpointing to control the activation term.

#### Explain ring all-reduce and why data parallelism scales.

The `N` ranks form a ring and the gradient tensor is split into `N` chunks. In the reduce-scatter phase each rank sends one chunk to its neighbor and adds the chunk it receives, for `N-1` steps. After that, each rank holds the fully summed version of one chunk. The all-gather phase then circulates the finished chunks for another `N-1` steps so every rank has everything.

Each rank sends and receives `2(N-1)/N` times the tensor size, which approaches `2M` as `N` grows. Per-rank bandwidth is essentially constant, so adding GPUs does not increase the communication each GPU does. DDP also overlaps these all-reduces with backward by bucketing gradients. The catch is latency: the number of steps grows with `N`, so for small messages at large scale NCCL switches to tree-based algorithms.

#### What do ZeRO stages 1, 2, and 3 shard, and what does each cost?

Stage 1 shards the optimizer states, the biggest term at 12 bytes/param, across the data-parallel ranks. Stage 2 also shards gradients: instead of an all-reduce, each rank reduce-scatters so it only keeps the gradient shard for the parameters it updates. Then an all-gather redistributes updated parameters. Since all-reduce equals reduce-scatter plus all-gather, stages 1 and 2 cost the same bandwidth as DDP.

Stage 3 also shards the parameters themselves, so each layer's weights must be all-gathered before its forward and again before its backward. That is about 1.5x DDP's communication volume, in exchange for memory of `16Ψ/N`, which scales down linearly with GPU count. FSDP's `FULL_SHARD` is ZeRO-3 and `SHARD_GRAD_OP` is ZeRO-2. The practical question is whether the extra all-gathers can be hidden behind compute by prefetching, which depends on having enough compute per layer and fast interconnect.

#### How does Megatron-style tensor parallelism split an MLP, and why is TP kept inside a node?

For `Y = GeLU(XA)B`, split `A` by columns and `B` by rows. Each GPU computes `GeLU(X A_i)` without communicating, because it holds whole output columns and GeLU is elementwise. It then multiplies by its row shard `B_i`, producing a partial sum. One all-reduce combines the partials. Attention works the same way: heads are split across GPUs through column-parallel QKV, and the output projection is row-parallel, followed by one all-reduce.

That makes two all-reduces in forward and two in backward per transformer layer. They are on the critical path (the next layer cannot start without them), and their size scales with the activations, not the parameters. Over NVLink that is affordable. Over InfiniBand, roughly an order of magnitude slower per GPU, the GPUs would mostly wait. So TP degree is usually capped at the GPUs per node, typically 8.

#### What is the pipeline bubble and how do you reduce it?

With `p` stages, the pipeline needs time to fill at the start of a step and drain at the end, and during those phases some stages are idle. With `m` micro-batches under a GPipe or 1F1B schedule, the idle fraction is `(p-1)/(m+p-1)`. Eight stages with eight micro-batches wastes almost half the time.

The levers: more micro-batches per step (limited by global batch size and by keeping each micro-batch big enough to use the GPU), interleaved schedules where each GPU hosts several non-contiguous layer chunks, which cuts the bubble roughly by that factor at the cost of more P2P messages, and zero-bubble schedules that split backward into input-gradient and weight-gradient work to fill the gaps. Using fewer stages, meaning less PP and more of another parallelism, also helps.

#### GPipe vs 1F1B: what is the difference?

GPipe runs all forward micro-batches, then all backwards, so it must hold activations for all `m` micro-batches at once. 1F1B, after a short warmup, alternates one forward and one backward per stage, so each stage holds at most about `p` micro-batches of activations regardless of `m`.

The bubble fraction is the same for both. The benefit of 1F1B is memory, and the memory headroom is what lets you increase `m`, which is what actually shrinks the bubble. Candidates often say 1F1B reduces the bubble directly, and that is wrong.

#### You have 512 GPUs and a 70B dense model. How would you parallelize it?

Start from the constraints. Model states are about 1.1 TB at 16 bytes/param, so they must be sharded. Nodes have 8 GPUs on NVLink, and nodes connect over InfiniBand.

A reasonable starting point is TP=8 within each node, which splits every layer and cuts activation memory too. Then either FSDP/ZeRO across the remaining 64-way data-parallel dimension, or a modest PP degree (say 4) if cross-node bandwidth is weak or FSDP's all-gathers show up as exposed communication, with DP=16 on the rest. I would add activation checkpointing (selective if possible), and sequence parallelism alongside TP for long contexts. Then I would check the global batch: DP degree times micro-batch times accumulation must stay under what the optimizer tolerates, and the LR schedule needs warmup. Finally, measure MFU and the step-time breakdown, and iterate. The "right" config depends on the cluster's actual bandwidth, so I would benchmark two or three layouts rather than trust one on paper.

#### Why is bf16 preferred over fp16, and where does fp8 fit?

fp16 has 5 exponent bits, max about 65,504, and small gradients underflow. It needs dynamic loss scaling, and overflows still cause skipped steps and occasional instability. bf16 keeps fp32's 8 exponent bits, trading mantissa precision for range, so gradients neither underflow nor overflow in practice and no loss scaling is needed. The lower precision is tolerable because master weights and optimizer states stay in fp32.

fp8 on H100-class hardware roughly doubles matmul throughput again. E4M3 is typically used for weights and activations, E5M2 for gradients. With so few bits, it needs per-tensor or block-wise scaling factors maintained from recent absolute maxima. Sensitive operations such as softmax, norms, and the loss stay in higher precision. It works for large models but should be validated against a bf16 baseline, since subtle divergence can appear late in training.

#### How do you train with very long sequences?

Activations become the problem: they scale linearly with sequence length, plus a quadratic term for attention scores if those are materialized. In order of adoption: FlashAttention removes the quadratic memory term; activation checkpointing trades compute for memory; Megatron sequence parallelism shards the LayerNorm/dropout activations that TP otherwise replicates.

Beyond that comes context parallelism, which shards the sequence across GPUs for every layer. Per-token operations need no communication. Attention is handled by ring attention (K/V blocks circulate while each GPU keeps its queries, combined with an online softmax) or by all-to-all methods like Ulysses that switch to head sharding for the attention computation. With causal masks, ring attention needs a load-balanced sequence assignment or the ranks holding later tokens do more work.

#### How does expert parallelism work, and what goes wrong?

Experts are distributed across GPUs. After routing, an all-to-all sends each token to the GPU owning its chosen expert, the experts run locally, and a second all-to-all returns the results. The non-expert layers are data-parallel.

The typical problems: all-to-all is latency-sensitive and suffers across nodes. Load imbalance means one hot expert makes its GPU everyone's straggler, which is why MoE training uses load-balancing losses or bias adjustments. Capacity limits drop tokens when an expert overflows. And parameter memory is large even though compute per token is small, so EP is usually combined with sharding of the dense parameters.

#### A training job on 256 GPUs hangs with no error. How do you debug it?

A silent hang almost always means ranks disagree about which collective comes next. Some rank is waiting in an all-reduce the others never reach. Common causes are data-dependent control flow (one rank skips a layer, or takes an early `continue`), uneven data so one rank runs out of batches first, a rank that crashed or is stuck in I/O, or a checkpoint/eval path that only some ranks execute.

Diagnosis: `py-spy dump` across ranks to see each stack, looking for the odd one out. Enable `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` to check collective sequences, and use PyTorch's NCCL flight recorder if available. If a timeout eventually fires, find the first rank that errored, since the rest are consequences. Fixes are usually making control flow identical on all ranks, using `drop_last=True` or equalizing the number of batches, and making sure save/eval code calls collectives from every rank.

#### How do you measure whether your distributed training is efficient?

Model FLOPs utilization. Compute the FLOPs the model needs (about `6 × params × tokens` for a dense transformer, plus attention), divide by elapsed time, and divide by the aggregate peak FLOPs of the hardware. Unlike tokens/sec, it compares across model sizes and hardware, and it excludes recomputation, so activation checkpointing does not inflate it (that is HFU).

Good large runs land somewhere around 35-55%. If MFU is low, break the step time into compute, exposed communication, pipeline bubble, and data loading with a profiler, then fix whichever dominates. I would also sanity-check the number itself: an MFU above the plausible range usually means the token count or FLOPs formula is wrong.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Forgetting optimizer states in memory estimates | Adam is 12 of the 16 bytes/param; plans off by 4-8x | Budget 16 bytes/param plus activations |
| Using `nn.DataParallel` | GIL-bound, GPU 0 bottleneck | DDP (or FSDP) with one process per GPU |
| Not calling `sampler.set_epoch(epoch)` | Same shuffle every epoch | Call it at the start of each epoch |
| Tensor parallelism across nodes | Per-layer all-reduces over InfiniBand stall compute | TP within NVLink domain; PP/DP across nodes |
| FSDP wrapping the whole model as one unit | Gathers all parameters at once; no memory saving | Wrap each transformer block |
| Too few micro-batches with PP | Bubble wastes a large fraction of GPU time | `m` well above `p`; 1F1B or interleaved schedules |
| fp16 without loss scaling | Gradients underflow; training stalls or diverges | bf16, or `GradScaler` with fp16 |
| Clipping gradients before unscaling | Clip threshold applies to scaled gradients | `scaler.unscale_(opt)` before `clip_grad_norm_` |
| Gradient accumulation without `no_sync` | All-reduce on every micro-batch; wasted bandwidth | `model.no_sync()` on non-final micro-batches |
| Not dividing loss by accumulation steps | Effective gradient `k`x too large | Normalize by accumulated tokens or steps |
| Scaling DP without adjusting LR / warmup | Large-batch instability or wasted steps | LR scaling rule plus warmup; stay near the critical batch size |
| Rank-dependent control flow | Mismatched collectives cause hangs | Identical collective sequence on every rank |
| Saving only model weights | Resume changes optimizer and data order; loss jumps | Save optimizer, scheduler, RNG, data position |
| Gathering full state to rank 0 for checkpoints | Slow and can OOM at scale | Sharded / async distributed checkpoints |
| Reading only the last NCCL timeout in logs | Real error hidden behind hundreds of downstream timeouts | Find the first rank that failed |

---

## Related Topics

- [Neural Network Training](./intro_neural_network_training.md)
- [Transformers](./intro_transformers.md)
- [Fine-Tuning](./intro_fine_tuning.md)
- [Model Compression](./intro_model_compression.md)
- [PyTorch](../frameworks/intro_pytorch.md)
- [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Cloud ML Platforms](../cloud_ml/intro_cloud_ml_platforms.md)
- [Kubernetes](../devops/intro_kubernetes.md)
- [Deep Learning Overview](./README.md)
