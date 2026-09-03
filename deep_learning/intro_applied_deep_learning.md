# Applied Deep Learning Roadmap

This guide turns deep learning from theory into portfolio-ready engineering. Use it alongside the core [Deep Learning overview](./README.md) and [Transformers guide](./intro_transformers.md) when preparing for modern ML engineer and applied AI interviews.

---

## Overview

Deep learning is the practice of learning hierarchical representations from data using neural networks trained with gradient-based optimization. It matters because modern AI systems for vision, language, ranking, recommendation, speech, and multimodal reasoning are all built on deep learning foundations.

In interviews, the expected bar is no longer "define a neural network." You should be able to explain:

- why one architecture is a better fit than another
- how training becomes stable at scale
- where GPU bottlenecks appear
- how to move from notebook experiments to reliable inference

---

## Core Concepts

### Neural networks and representation learning

A deep network learns multiple layers of features. Early layers learn simple patterns; later layers learn task-specific abstractions. In practice, this means the network replaces hand-written feature engineering with learned representations.

### Backpropagation

Backpropagation computes gradients of the loss with respect to parameters using the chain rule. Practically, this is what makes end-to-end training possible across millions or billions of parameters.

### CNNs, RNNs, and Transformers

- CNNs are efficient for local spatial structure, especially images and video frames.
- RNNs process sequences recurrently and are still relevant for some streaming or low-resource sequence tasks.
- Transformers dominate modern NLP and many multimodal tasks because self-attention models long-range dependencies better than recurrent architectures.

### Training pipelines

A production-grade training pipeline includes:

- dataset versioning
- train/validation/test splits
- preprocessing and augmentation
- distributed training or mixed precision
- checkpointing
- evaluation
- model packaging for inference

---

## Key Skills

### Model architecture design

In practice, this means selecting the smallest architecture that can meet the quality target. You should know when to use a CNN backbone, a simple MLP, a sequence model, or a transformer encoder/decoder setup.

### GPU training

This includes understanding:

- batch size vs memory tradeoffs
- mixed precision
- gradient accumulation
- data loader throughput
- multi-GPU scaling limitations

### Hyperparameter tuning

Strong candidates can explain which hyperparameters matter first:

- learning rate
- weight decay
- batch size
- scheduler choice
- dropout or regularization strength

### Model optimization

This means reducing latency or cost without destroying accuracy. Common techniques include quantization, pruning, distillation, compilation, and smaller serving-friendly architectures.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| PyTorch | Flexible deep learning framework with strong research and production adoption | Default choice for most modern DL and LLM work |
| TensorFlow | Graph-based ML ecosystem with good deployment integrations | Teams already standardized on TensorFlow or TFX |
| Keras | High-level API for rapid model iteration | Fast prototyping and teaching |
| PyTorch Lightning | Structured training loops and experiment organization | When training code becomes repetitive |
| Optuna | Hyperparameter search automation | Systematic tuning beyond manual sweeps |

---

## Projects

### Image classifier

- Goal: Build a classifier for a custom image dataset and ship both training and inference.
- Key components: data augmentation, CNN backbone, experiment tracking, evaluation report, REST or batch inference.
- Suggested tech stack: PyTorch, torchvision, Weights & Biases, FastAPI.
- Difficulty: Intermediate.

### Text classifier

- Goal: Classify support tickets, reviews, or forum posts using a transformer encoder.
- Key components: tokenization, class balancing, fine-tuning, threshold tuning, confusion-matrix analysis.
- Suggested tech stack: Hugging Face Transformers, PyTorch, scikit-learn.
- Difficulty: Intermediate.

### Transformer from scratch

- Goal: Implement a minimal transformer encoder-decoder to understand attention mechanics.
- Key components: token embeddings, positional encoding, multi-head attention, masking, training loop.
- Suggested tech stack: PyTorch only.
- Difficulty: Advanced.

### Training and inference pipeline

- Goal: Show how a model moves from raw data to packaged inference artifact.
- Key components: data preprocessing script, training script, checkpoint export, inference service, monitoring hooks.
- Suggested tech stack: PyTorch, MLflow, Docker, FastAPI.
- Difficulty: Advanced.

---

## Example Code

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleClassifier(input_dim=768, hidden_dim=256, num_classes=5).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for batch in DataLoader(train_dataset, batch_size=64, shuffle=True):
    features, labels = [x.cuda() for x in batch]
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Suggested Project Structure

```text
image-classifier/
├── data/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
├── configs/
├── Dockerfile
└── README.md
```

---

## Interview Q&A

#### How do you decide between training from scratch, fine-tuning, and using a pretrained model as-is?

Work down from cheapest. **Pretrained as-is (or with a frozen backbone + new head)** whenever the pretraining domain is close to yours and you have a few thousand labeled examples or fewer — this covers most real problems. **Fine-tuning** when you have enough labeled data (typically tens of thousands of examples) or when the domain differs meaningfully from the pretraining distribution — medical imaging, satellite imagery, industrial defect detection. **From scratch** almost never: only when the input modality has no relevant pretrained model at all, or when you have millions of labeled examples and a research budget.

The related decision is *how much* to fine-tune: freezing early layers and training later ones is the standard middle ground, since early layers learn general features (edges, textures) and later layers learn task-specific ones. With very little data, freezing more prevents overfitting.

#### Your model gets 95% training accuracy and 65% validation accuracy. What do you do?

Confirm the gap is real before treating it. Check that the validation set isn't distributionally different (different source, different preprocessing, different time period) and that validation-time augmentation isn't applied — those look like overfitting and aren't.

If it's genuine overfitting, in order of expected impact: **more data or stronger augmentation** (almost always the biggest win in vision), **early stopping** on validation loss, **more regularization** (weight decay, dropout, label smoothing), **a smaller or more constrained model**, and **transfer learning with more layers frozen** if you're training too much capacity for the data available.

If the training accuracy itself is suspicious, check for leakage — duplicate or near-duplicate images across the split is extremely common in scraped datasets and inflates training metrics specifically.

#### How do you structure a training pipeline that others can reproduce?

Configuration-driven rather than notebook-driven: one config file holding data paths, hyperparameters, and seeds; a deterministic data pipeline with a fixed, versioned split; seeded random number generators for Python, NumPy, and the framework; and an experiment tracker (MLflow, Weights & Biases) recording config, metrics, the code commit, the data version, and the resulting artifact.

Checkpointing matters more than people expect: save on every improvement with the optimizer and scheduler state included, so a run can resume rather than restart. And separate the pipeline into stages — data prep, train, evaluate, export — so a failure in one doesn't cost the others.

Full determinism on GPU costs speed (deterministic kernels, no cuDNN autotuning), so the practical standard is seeded and versioned, with a note that exact bitwise reproduction requires the deterministic flags.

#### How do you choose a batch size and learning rate together?

Pick the largest batch that fits in memory and trains stably, then set the learning rate using the linear scaling rule: relative to a known-good configuration, scaling the batch by `k` means scaling the LR by roughly `k`, with warmup. Verify with a short LR range test — increase the LR exponentially over a few hundred steps and take roughly an order of magnitude below where the loss diverges.

Two caveats worth raising: beyond a task-dependent critical batch size, larger batches stop improving the gradient estimate and you're spending compute for nothing; and small batches inject gradient noise that often generalizes slightly better, so the largest possible batch is not automatically the best batch.

#### How do you speed up a training run that's taking too long?

Profile first — most "slow training" is a starved GPU, not a slow model. Check GPU utilization: if it's below 80%, the bottleneck is data loading, and the fixes are more `num_workers`, `pin_memory=True`, `persistent_workers=True`, pre-decoded or pre-tokenized data, and removing synchronization points like `.item()` inside the loop.

Once the GPU is saturated: mixed precision (bf16) for a near-free 1.5–2x, larger batches, `torch.compile` for kernel fusion, and gradient accumulation only if you need a larger effective batch. Then algorithmic wins — a smaller model that meets the requirement, fewer epochs with early stopping, or a subset of the data for hyperparameter search before the full run. Distributed training is the last resort because it adds the most complexity per unit of speedup.

#### What do you monitor during a long training run?

Training and validation loss on the same axis (divergence tells you when to stop), learning rate (to confirm the schedule is doing what you think), gradient norm (spikes precede divergence and justify clipping), GPU utilization and memory, and throughput in samples per second so you notice a slowdown.

For anything beyond a short run, add automatic alerting on NaN loss and on validation loss failing to improve for N evaluations, plus periodic checkpointing — discovering at hour 40 that the run diverged at hour 3 is an avoidable and expensive mistake.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Not verifying the model can overfit one batch | Hours of tuning spent on a code bug | Overfit a single batch to near-zero loss first |
| Augmentation applied to the validation set | Validation loss becomes noise, not signal | Separate train and eval transform pipelines |
| Duplicate or near-duplicate images across splits | Inflated metrics; the model memorized | Deduplicate by perceptual hash before splitting |
| Forgetting `model.eval()` at inference | Dropout on and BatchNorm using batch statistics | `model.eval()` + `torch.no_grad()` |
| Tuning hyperparameters on the test set | Reported performance is optimistic | Separate validation and test sets |
| Checkpointing weights without optimizer state | A resumed run restarts the optimizer cold | Save model, optimizer, scheduler, epoch, and RNG state |
| Notebook-only training | Not reproducible, not reviewable, not schedulable | Config-driven scripts with experiment tracking |
| Ignoring GPU utilization | Runs take 3x longer than necessary | Profile; fix data loading before touching the model |
| No early stopping on a long run | Wasted compute past the best checkpoint | Early stopping on validation, plus alerting |
| Class imbalance ignored in the loss | The model collapses to the majority class | Class weights, focal loss, or balanced sampling |

---

## Related Topics

- [Deep Learning Overview](./README.md)
- [Transformers](./intro_transformers.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
