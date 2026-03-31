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

## Related Topics

- [Deep Learning Overview](./README.md)
- [Transformers](./intro_transformers.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
