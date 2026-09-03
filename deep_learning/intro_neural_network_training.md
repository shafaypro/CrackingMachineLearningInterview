# Neural Network Training: Optimization and Regularization

Architecture gets the headlines; training dynamics decide whether the model works. This guide covers what interviewers actually probe when they ask "your loss is NaN, what now?" — initialization, optimizers, learning rate schedules, normalization, regularization, and the debugging playbook for a model that won't converge.

---

## Table of Contents
1. [Backpropagation and the Gradient Problems](#backpropagation-and-the-gradient-problems)
2. [Weight Initialization](#weight-initialization)
3. [Activation Functions](#activation-functions)
4. [Optimizers](#optimizers)
5. [Learning Rate Schedules](#learning-rate-schedules)
6. [Normalization Layers](#normalization-layers)
7. [Regularization](#regularization)
8. [Batch Size and Its Effects](#batch-size-and-its-effects)
9. [Mixed Precision and Memory](#mixed-precision-and-memory)
10. [A Debugging Playbook](#a-debugging-playbook)
11. [Interview Q&A](#interview-qa)
12. [Common Pitfalls](#common-pitfalls)
13. [Related Topics](#related-topics)

---

## Backpropagation and the Gradient Problems

Backprop is the chain rule applied over a computation graph. For an `L`-layer network the gradient at layer 1 contains a product of `L` Jacobians:

```
∂Loss/∂W₁ = ∂Loss/∂aL · ∂aL/∂a(L-1) · ... · ∂a2/∂a1 · ∂a1/∂W₁
```

Products of many terms are unstable:

| Problem | Cause | Symptom | Fix |
|---|---|---|---|
| **Vanishing gradients** | Jacobian norms < 1 compound toward 0 | Early layers stop learning; loss plateaus high | ReLU-family activations, residual connections, normalization, careful init, LSTM/GRU gates |
| **Exploding gradients** | Jacobian norms > 1 compound upward | Loss spikes to NaN/Inf; weights blow up | Gradient clipping, lower LR, normalization, better init |

Residual connections are the most important structural fix: `y = x + F(x)` gives the gradient an identity path (`∂y/∂x = I + ∂F/∂x`), so gradients reach early layers even when `F`'s Jacobian is small. This is why 100+ layer networks became trainable at all.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # after backward(), before step()
```

---

## Weight Initialization

Initialization sets the scale of activations and gradients at step 0. Get it wrong and the network is dead or diverging before training starts.

| Scheme | Variance | Use with |
|---|---|---|
| **Xavier / Glorot** | `2 / (fan_in + fan_out)` | tanh, sigmoid, linear |
| **He / Kaiming** | `2 / fan_in` | ReLU, LeakyReLU, GELU |
| **LeCun** | `1 / fan_in` | SELU |
| **Orthogonal** | orthogonal matrix | RNNs, where repeated multiplication needs norm preservation |

He initialization uses `2/fan_in` rather than `1/fan_in` because ReLU zeroes roughly half its inputs, halving the output variance; the factor of 2 compensates.

```python
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

Never initialize all weights to the same constant: every neuron in a layer would receive identical gradients and stay identical forever (the symmetry-breaking problem). Zero-initializing *biases* is fine and standard.

---

## Activation Functions

| Activation | Formula | Pros | Cons |
|---|---|---|---|
| Sigmoid | `1/(1+e^-x)` | Bounded, probabilistic reading | Saturates → vanishing gradient; not zero-centered |
| Tanh | `(e^x - e^-x)/(e^x + e^-x)` | Zero-centered | Still saturates |
| **ReLU** | `max(0, x)` | Cheap, no positive saturation, sparse | Dying ReLU; not zero-centered |
| LeakyReLU | `max(αx, x)` | No dead units | Extra hyperparameter |
| **GELU** | `x · Φ(x)` | Smooth, strong in transformers | Slightly more expensive |
| **SiLU/Swish** | `x · σ(x)` | Smooth, strong empirically | Same |
| Softmax | normalized exponentials | Output distribution | Output layer only |

**Dying ReLU**: a unit whose pre-activation is negative for every input outputs 0 and receives zero gradient forever — permanently dead. Usually caused by too-high a learning rate driving a large negative bias. LeakyReLU, GELU, or lowering the LR all fix it.

Modern transformers use GELU or SwiGLU; the smoothness near zero gives slightly better gradients than ReLU's kink and empirically improves language-model quality at the same parameter count.

---

## Optimizers

| Optimizer | Update rule (essence) | Strengths | Weaknesses |
|---|---|---|---|
| **SGD** | `w ← w - η∇` | Simple; often best final accuracy in vision | Slow; LR-sensitive |
| **SGD + momentum** | `v ← βv + ∇; w ← w - ηv` | Dampens oscillation, escapes plateaus | Two hyperparameters |
| **RMSProp** | Per-parameter scaling by RMS of gradients | Handles varying gradient scales | No momentum |
| **Adam** | Momentum + RMSProp + bias correction | Fast, robust defaults | Can generalize worse than SGD; weight decay is wrong |
| **AdamW** | Adam with **decoupled** weight decay | The default for transformers | — |
| **Adafactor / 8-bit Adam** | Factored or quantized optimizer state | Large memory savings at scale | Slight quality cost |

### Adam, precisely

```
m_t = β₁ m_{t-1} + (1-β₁) g_t            # first moment (momentum)
v_t = β₂ v_{t-1} + (1-β₂) g_t²           # second moment (per-parameter scale)
m̂_t = m_t/(1-β₁^t),  v̂_t = v_t/(1-β₂^t)  # bias correction — m,v start at 0
w_t = w_{t-1} - η · m̂_t / (√v̂_t + ε)
```

Bias correction matters most in the first few hundred steps: without it, `m` and `v` are biased toward zero, making early effective steps far too small.

### Why AdamW instead of Adam

In Adam, L2 regularization added to the loss gets divided by `√v̂`, so parameters with large historical gradients get *less* decay — the opposite of the intent. AdamW decouples it:

```
w ← w - η(m̂/(√v̂ + ε) + λw)     # decay applied directly to the weight
```

This is not a minor detail; it is why AdamW is the standard for transformer training. Typical settings: `lr=1e-4` to `3e-4` for transformers from scratch, `1e-5` to `5e-5` for fine-tuning, `betas=(0.9, 0.95)` for large language models (lower `β₂` than the 0.999 default responds faster to loss spikes), `weight_decay=0.1`, and **no weight decay on biases or LayerNorm parameters**.

```python
decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim < 2 or 'bias' in name or 'norm' in name.lower():
        no_decay.append(param)      # biases and norm scales: no decay
    else:
        decay.append(param)

optimizer = torch.optim.AdamW(
    [{'params': decay, 'weight_decay': 0.1},
     {'params': no_decay, 'weight_decay': 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
)
```

**Memory cost**: Adam stores two extra float32 tensors per parameter. A 7B model needs ~14 GB for fp16 weights plus ~56 GB for fp32 master weights, gradients, and optimizer state — which is why full fine-tuning of a 7B model does not fit on a 24 GB GPU and LoRA does.

---

## Learning Rate Schedules

The learning rate is the single most important hyperparameter. Tune it first, and by orders of magnitude (1e-5, 1e-4, 1e-3), not by 10%.

| Schedule | Shape | Use |
|---|---|---|
| **Linear warmup** | 0 → peak over N steps | Essential for transformers; Adam's variance estimate is unreliable early |
| **Cosine decay** | Peak → ~0 on a cosine curve | Default for LLM pretraining and fine-tuning |
| **Step decay** | Drop by 10x at milestones | Classic vision recipes |
| **ReduceLROnPlateau** | Drop when validation stalls | When you can't predict the schedule |
| **One-cycle** | Up then down, with momentum inverted | Fast convergence on small budgets |

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

scheduler = cosine_with_warmup(optimizer, warmup_steps=500, total_steps=20_000)
# step the scheduler every optimizer step, not every epoch
```

**Why warmup?** At step 0, Adam's second-moment estimate `v` is based on almost no data and is noisy. Large early steps in a random direction can push the model into a bad region it never escapes. Warmup also matters more with large batches, where each step is more confident and therefore more damaging if wrong.

**Finding the LR empirically**: run an LR range test — increase the LR exponentially over a few hundred steps and plot loss. Pick roughly an order of magnitude below where the loss starts diverging.

---

## Normalization Layers

| Layer | Normalizes over | Batch dependence | Typical use |
|---|---|---|---|
| **BatchNorm** | Batch dimension, per channel | Yes — different train/eval behavior | CNNs |
| **LayerNorm** | Feature dimension, per sample | No | Transformers, RNNs |
| **RMSNorm** | Feature dimension, no mean subtraction | No | Modern LLMs (Llama, Mistral) — cheaper |
| **GroupNorm** | Groups of channels | No | Small-batch vision, segmentation |
| **InstanceNorm** | Per sample, per channel | No | Style transfer |

**Why transformers use LayerNorm, not BatchNorm**: sequences have variable length, so batch statistics are computed over a ragged, padded tensor and become noisy and length-dependent. LayerNorm normalizes within one token's feature vector, so it behaves identically for batch size 1 and batch size 512 — critical when training and inference batch sizes differ wildly.

**BatchNorm's train/eval trap**: at training time it uses batch statistics; at eval it uses running averages. Forgetting `model.eval()` at inference means predictions depend on whatever else is in the batch — a genuinely common production bug that also breaks batch-size-1 serving.

**Pre-norm vs post-norm**: the original transformer applied LayerNorm after the residual add (post-norm), which needs careful warmup to train deeply. Modern models use pre-norm (`x + Attention(LN(x))`), which keeps a clean identity path through the residual stream and trains stably at depth. Pre-norm is why you can train a 70-layer transformer without an elaborate warmup schedule.

---

## Regularization

| Technique | Mechanism | Notes |
|---|---|---|
| **L2 / weight decay** | Penalizes large weights | Use AdamW's decoupled form; exclude biases and norms |
| **L1** | Penalizes absolute weights | Induces sparsity; rare in deep nets |
| **Dropout** | Randomly zeroes activations, scales the rest | `p=0.1` in transformers, `0.5` in old MLP heads |
| **Early stopping** | Halts at best validation | Effectively limits capacity; always worth it |
| **Data augmentation** | Expands the effective dataset | Usually the highest-leverage option |
| **Label smoothing** | Target `1-ε` instead of `1` | Reduces overconfidence, improves calibration |
| **Stochastic depth** | Randomly skips residual blocks | Very deep vision networks |
| **Mixup / CutMix** | Trains on convex combinations of examples | Strong vision regularizer |
| **Gradient noise / EMA of weights** | Smooths the optimization path | EMA is nearly free and often helps |

**Dropout's inverted scaling**: at train time, surviving activations are divided by `(1-p)` so the expected value matches inference, where dropout is off. This is why `model.eval()` is required — without it you keep dropping units at inference and get noisy, degraded predictions.

Dropout and BatchNorm interact badly (dropout changes the variance BatchNorm estimated), which is part of why modern architectures use much less dropout than 2015-era networks and rely on weight decay, augmentation, and scale instead.

---

## Batch Size and Its Effects

| Batch size | Gradient noise | Steps/epoch | Generalization | Hardware use |
|---|---|---|---|---|
| Small (8–64) | High | Many | Often better (noise regularizes) | Underutilized GPU |
| Large (1k–8k+) | Low | Few | Needs LR scaling and warmup | Efficient |

The **linear scaling rule**: multiply the batch size by `k`, multiply the learning rate by `k` (with warmup). It holds well up to a point and then breaks — beyond a critical batch size, extra examples per step stop improving the gradient estimate and you're just burning compute.

**Gradient accumulation** simulates a large batch on small hardware:

```python
accum_steps = 8
optimizer.zero_grad(set_to_none=True)
for i, batch in enumerate(loader):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss / accum_steps       # divide to keep the gradient scale right
    loss.backward()
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

Note the `/ accum_steps`: forgetting it makes the effective gradient `accum_steps` times too large, which usually presents as a mysterious divergence after switching to accumulation.

---

## Mixed Precision and Memory

| Format | Bits | Range | Notes |
|---|---|---|---|
| fp32 | 32 | Wide | Baseline, 4 bytes/param |
| **fp16** | 16 | Narrow | 2x faster, needs loss scaling to avoid gradient underflow |
| **bf16** | 16 | Same exponent range as fp32 | Preferred on A100/H100 — no loss scaling needed |
| fp8 | 8 | Very narrow | H100+ training, still specialized |

```python
# bf16 — the modern default where supported
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(**batch).loss
loss.backward()

# fp16 requires a GradScaler because small gradients underflow to zero
scaler = torch.amp.GradScaler()
with torch.autocast('cuda', dtype=torch.float16):
    loss = model(**batch).loss
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**Where training memory goes** (per parameter, mixed-precision AdamW): 2 bytes fp16 weights + 4 bytes fp32 master weights + 4 bytes gradients + 8 bytes optimizer state ≈ **18 bytes/param**, before activations. Activations often dominate and scale with `batch × sequence_length × hidden × layers`.

Levers when you hit OOM, in order of cost to quality: reduce batch size with gradient accumulation (free), gradient checkpointing (recompute activations — ~30% slower, big memory win), bf16 (free on modern GPUs), 8-bit optimizer states, LoRA/QLoRA instead of full fine-tuning, then model/tensor parallelism.

---

## A Debugging Playbook

**Loss is NaN.**
Check in this order: learning rate too high (most common); fp16 overflow (switch to bf16 or add loss scaling); division by zero or `log(0)` in a custom loss (add epsilon); a corrupt batch with NaN inputs; missing gradient clipping. Bisect by running with `torch.autograd.set_detect_anomaly(True)` to find the offending op.

**Loss doesn't decrease at all.**
First: can the model overfit a single batch to near-zero loss? If not, the bug is in the code, not the hyperparameters — check that the optimizer sees the parameters, that `loss.backward()` is called, that `zero_grad()` isn't clearing after backward, that labels align with inputs, and that the LR isn't ~0.

**Training loss falls, validation loss rises.**
Standard overfitting: more data or augmentation, more regularization, smaller model, early stopping. If validation loss rises from the very first epoch, suspect a train/validation distribution mismatch or a broken validation transform instead.

**Training loss is much worse than validation loss.**
Usually dropout/BatchNorm accounting (training loss includes regularization noise; eval doesn't) — normal early on. If persistent, the validation set may be easier or leaking.

**Loss spikes mid-training.**
Common in LLM training: reduce `β₂` to 0.95, clip gradients harder, skip the batch, or resume from the last checkpoint with a lower LR. Repeated spikes at the same data offset means a bad shard.

**GPU utilization is low.**
Data loading is the bottleneck: raise `num_workers`, enable `pin_memory=True` and `persistent_workers=True`, pre-tokenize offline, and check you aren't synchronizing with `.item()` or `.cpu()` inside the loop.

---

## Interview Q&A

#### Explain vanishing gradients and three ways to fix them.

In a deep network, the gradient at an early layer is a product of many Jacobians. When those factors have norm below 1 — as with saturated sigmoid or tanh units, whose derivative maxes at 0.25 — the product shrinks exponentially with depth, and early layers receive effectively no learning signal.

Fixes: (1) **ReLU-family activations**, whose derivative is exactly 1 on the positive side, so no shrinkage per layer; (2) **residual connections**, which add an identity path so the gradient reaches early layers regardless of what the block does; (3) **normalization layers**, which keep activation scales stable across depth so Jacobians stay near unit norm. Careful initialization (He/Xavier) and, for RNNs specifically, gated architectures (LSTM/GRU) with additive cell-state updates are the other standard answers.

#### Adam vs SGD with momentum — which do you pick?

AdamW for transformers and anything with sparse or badly scaled gradients; it adapts per-parameter step sizes and works near-default, which matters when a training run costs real money. SGD with momentum for convolutional vision models trained long, where it still often reaches a better final test accuracy — the usual explanation is that its noisier, non-adaptive updates favor flatter minima.

I'd note that the comparison is often unfair: Adam works well with almost no tuning, while SGD needs a well-tuned LR and schedule to show its advantage. If tuning budget is small, Adam wins by default.

#### Why do transformers need learning rate warmup?

Two reasons. First, Adam's second-moment estimate is computed from very few samples at the start, so the per-parameter scaling is unreliable and early updates can be wildly wrong in magnitude. Warmup keeps steps small until the estimate stabilizes. Second, with post-norm architectures the residual path is poorly conditioned early in training, and large steps push the model into a region it doesn't recover from. Pre-norm architectures need less warmup, which is one reason they became standard, but warmup still helps at large batch sizes where each step is taken with high confidence.

#### What is the difference between BatchNorm and LayerNorm, and why do transformers use LayerNorm?

BatchNorm normalizes each feature across the batch dimension; LayerNorm normalizes each sample across its feature dimension. The consequence is that BatchNorm's output for one example depends on the other examples in the batch, and it must maintain running statistics for a separate inference-time behavior.

Transformers use LayerNorm because sequences are variable-length and padded, making batch statistics noisy and length-dependent; because inference often runs at batch size 1 where batch statistics are meaningless; and because it removes the train/eval discrepancy entirely. Modern LLMs go further to RMSNorm, dropping the mean-subtraction step for a small speedup with no measured quality loss.

#### Your training loss goes to NaN after 200 steps. Walk me through the diagnosis.

I'd bisect systematically rather than guess. First reproduce deterministically with a fixed seed and find the exact step. Then:
1. **Lower the LR by 10x** — if it survives, the LR was too high. This is the most common cause by a wide margin.
2. **Check precision** — in fp16, activations or gradients can overflow; switching to bf16 (same exponent range as fp32) rules this in or out immediately.
3. **Inspect the batch at the failure step** — NaNs or infinities in the inputs, or an empty/degenerate label.
4. **Check the loss for `log(0)` or division by zero**, especially in custom losses; add epsilon.
5. **Add gradient clipping** at norm 1.0 and log the pre-clip gradient norm — a spike right before the NaN confirms exploding gradients.

If none of that fixes it, `torch.autograd.set_detect_anomaly(True)` will name the operation that first produced a NaN in the backward pass.

#### How do you train a model that doesn't fit in GPU memory?

Escalating in order of cost:
1. **Gradient accumulation** — small micro-batches with the same effective batch size; free.
2. **Mixed precision (bf16)** — halves weight and activation memory, usually faster too.
3. **Gradient checkpointing** — store only some activations and recompute the rest in the backward pass; roughly 30% slower for a large memory saving.
4. **8-bit optimizer states** (bitsandbytes) — cuts the 8 bytes/param of Adam state to 2.
5. **Parameter-efficient fine-tuning (LoRA/QLoRA)** — train small adapter matrices while the base model stays frozen and quantized; this is what makes 7B–70B fine-tuning possible on consumer GPUs.
6. **Sharding across GPUs** — ZeRO/FSDP partitions optimizer state, gradients, then parameters; tensor and pipeline parallelism for models too large for one device even at inference.

#### What does weight decay actually do, and why exclude biases from it?

Weight decay shrinks weights toward zero each step, penalizing large-magnitude parameters and favoring simpler functions. In AdamW it is applied directly to the weight rather than through the loss, so the adaptive denominator doesn't distort it.

Biases and normalization scale parameters are excluded because they don't contribute to model complexity in the same way — a bias just shifts the activation, and decaying LayerNorm's gain toward zero actively fights the normalization it's supposed to perform. Decaying them typically costs a little accuracy for no regularization benefit.

#### How does batch size interact with learning rate?

Larger batches give lower-variance gradient estimates, so you can safely take larger steps. The linear scaling rule (`k×` batch → `k×` LR, with warmup) captures this and works well up to a critical batch size that depends on the task. Beyond it, gradient noise is already negligible and further increases buy no convergence speedup per example — you're paying compute for nothing.

Small batches inject noise that acts as a regularizer and often generalizes slightly better; large batches are hardware-efficient. In practice I'd pick the largest batch that fits and trains stably, then tune the LR around the scaling rule's prediction.

#### What is gradient checkpointing and what does it cost?

During the forward pass, activations are normally kept for the backward pass, and they often dominate memory. Gradient checkpointing stores activations only at chosen boundaries and recomputes the intermediate ones during backward. Memory for activations drops from `O(n)` to roughly `O(√n)` with optimal placement; the cost is one extra forward pass through the checkpointed segments, typically 25–40% slower wall-clock. It's the standard first move when you want a larger batch or longer sequences and are activation-bound rather than parameter-bound.

#### Why is bf16 usually preferred over fp16 for training?

bf16 has the same 8-bit exponent as fp32, so it covers the same dynamic range with less mantissa precision. fp16 has a much narrower range, so small gradients underflow to zero and large activations overflow to infinity — which is why fp16 training requires a `GradScaler` that multiplies the loss, checks for infinities, and adapts the scale. bf16 needs none of that machinery and just works, at the cost of ~3 fewer mantissa bits that neural network training turns out not to need. On hardware that supports both (A100 and later), bf16 is the default.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Forgetting `model.eval()` at inference | Dropout stays on and BatchNorm uses batch stats — predictions depend on batch contents | `model.eval()` + `torch.no_grad()` |
| `zero_grad()` after `backward()` | Wipes the gradients you just computed | Zero at the start of the step |
| Not dividing loss by accumulation steps | Effective gradient is `k×` too large | `loss = loss / accum_steps` |
| Stepping the scheduler per epoch when it expects steps | LR decays far too slowly | Match the scheduler's assumed unit |
| Weight decay on biases and LayerNorm | Fights normalization, costs accuracy | Parameter groups with `weight_decay=0` for those |
| Tuning LR in 10% increments | The useful range spans orders of magnitude | Search 1e-5 / 1e-4 / 1e-3, then refine |
| No gradient clipping on transformers | A single bad batch NaNs the run | `clip_grad_norm_(..., 1.0)` |
| Validating with training-time augmentation | Validation loss is noise, not signal | Separate eval transform pipeline |
| Not checking you can overfit one batch | Hours spent tuning a model with a code bug | Overfit 1 batch to ~0 loss before any real run |
| Data loader starving the GPU | 20% utilization, 5x longer runs | More workers, `pin_memory`, pre-tokenize |

---

## Related Topics

- [Applied Deep Learning Roadmap](./intro_applied_deep_learning.md)
- [Transformers](./intro_transformers.md)
- [Fine-Tuning](./intro_fine_tuning.md)
- [Computer Vision](./intro_computer_vision.md)
- [PyTorch](../frameworks/intro_pytorch.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Deep Learning Overview](./README.md)
