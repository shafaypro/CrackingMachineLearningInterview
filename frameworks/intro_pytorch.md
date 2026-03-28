# PyTorch Guide

A comprehensive guide to PyTorch — the leading deep learning framework for research and production.

---

## Table of Contents

1. [Tensor Operations](#tensor-operations)
2. [nn.Module and Building Models](#nnmodule-and-building-models)
3. [Training Loop](#training-loop)
4. [DataLoader and Dataset](#dataloader-and-dataset)
5. [GPU Usage](#gpu-usage)
6. [Model Saving and Loading](#model-saving-and-loading)
7. [torch.compile — PyTorch 2.0](#torchcompile--pytorch-20)
8. [Interview Q&A](#interview-qa)
9. [References](#references)

---

## Tensor Operations

```python
import torch
import numpy as np

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.zeros(3, 4)           # 3x4 zero tensor
x = torch.ones(3, 4)            # 3x4 ones tensor
x = torch.rand(3, 4)            # Uniform [0, 1)
x = torch.randn(3, 4)           # Normal distribution
x = torch.arange(0, 10, step=2) # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]

# Data types
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = x.float()   # Convert to float32
x = x.double()  # Convert to float64
x = x.long()    # Convert to int64

# Shape operations
x = torch.randn(2, 3, 4)
print(x.shape)          # torch.Size([2, 3, 4])
print(x.size(0))        # 2

x = x.view(2, 12)       # Reshape (shares memory)
x = x.reshape(6, 4)     # Reshape (may copy)
x = x.transpose(0, 1)   # Swap dims
x = x.permute(1, 0, 2)  # Reorder dims
x = x.squeeze()         # Remove size-1 dims
x = x.unsqueeze(0)      # Add size-1 dim at position 0
x = x.flatten()         # Flatten to 1D

# Basic math
a = torch.randn(3, 4)
b = torch.randn(3, 4)

c = a + b               # Element-wise add
c = a * b               # Element-wise multiply
c = torch.mm(a, b.T)    # Matrix multiply
c = a @ b.T             # Matrix multiply (operator)
c = torch.matmul(a, b.T)

c = torch.sum(a)             # Sum all
c = torch.sum(a, dim=0)      # Sum along dim
c = torch.mean(a)
c = torch.std(a)
c = torch.max(a)
c = torch.argmax(a, dim=1)   # Index of max along dim

# NumPy interop
arr = np.array([1.0, 2.0, 3.0])
x = torch.from_numpy(arr)    # Share memory!
x = torch.tensor(arr)        # Copy
arr = x.numpy()              # Back to numpy (CPU only)
arr = x.detach().cpu().numpy()  # Safe conversion
```

---

## nn.Module and Building Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feedforward network
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# CNN for image classification
class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 14x14 → 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # Global average pooling → 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

# Inspect model
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

---

## Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(data)

        # 3. Compute loss
        loss = criterion(outputs, targets)

        # 4. Backward pass
        loss.backward()

        # 5. Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Update weights
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# Full training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_acc = 0.0

for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"Best Validation Accuracy: {best_acc:.2f}%")
```

---

## DataLoader and Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np

# Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# Create dataset and splits
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

dataset = TabularDataset(X, y)

# Split 80/10/10
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True   # Faster GPU transfer
)

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

# Torchvision datasets
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cifar10_train = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
```

---

## GPU Usage

```python
import torch

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")

# Move tensors and models to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
x = x.to(device)
x = x.cuda()  # Shorthand

# Move back to CPU
x = x.cpu()

# Apple Silicon (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")

# Mixed Precision Training (FP16/BF16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, targets in train_loader:
    data, targets = data.to(device), targets.to(device)
    optimizer.zero_grad()

    with autocast():  # Automatically casts to FP16 where safe
        outputs = model(data)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Multi-GPU with DataParallel (simple)
model = nn.DataParallel(model)

# Multi-GPU with DistributedDataParallel (recommended)
import torch.distributed as dist

dist.init_process_group(backend="nccl")
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
```

---

## Model Saving and Loading

```python
import torch

# Save only model weights (recommended)
torch.save(model.state_dict(), "model_weights.pth")

# Load weights
model = ConvNet(num_classes=10)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# Save full model (not recommended — fragile)
torch.save(model, "full_model.pth")
model = torch.load("full_model.pth")

# Save training checkpoint (full state for resuming)
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_accuracy": best_acc,
    "train_loss": train_loss,
}
torch.save(checkpoint, "checkpoint.pth")

# Load checkpoint and resume training
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_epoch = checkpoint["epoch"] + 1
best_acc = checkpoint["best_accuracy"]

# Export to ONNX
dummy_input = torch.randn(1, 1, 28, 28).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)
```

---

## torch.compile — PyTorch 2.0

`torch.compile` compiles the model using TorchInductor (on top of OpenAI Triton) for significant speedups.

```python
import torch

model = ConvNet(num_classes=10).to("cuda")

# Compile the model — one line!
model = torch.compile(model)

# Different modes
model = torch.compile(model, mode="default")        # Balanced speed/compile time
model = torch.compile(model, mode="reduce-overhead") # Reduce Python overhead
model = torch.compile(model, mode="max-autotune")   # Maximum optimization (slow to compile)

# Training works the same
for data, targets in train_loader:
    data, targets = data.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = model(data)      # Uses compiled graph
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Typical speedup: 10-30% on training, up to 2x on inference
# First few iterations are slower (compilation overhead)
```

---

## Interview Q&A

**Q1: What is the difference between `tensor.view()` and `tensor.reshape()`?** 🟡 Intermediate

`view()` requires the tensor to be contiguous in memory and returns a view (shared memory — no copy). `reshape()` returns a tensor with the desired shape, creating a new memory layout if necessary (copy if non-contiguous). Use `view()` for efficiency when you know the tensor is contiguous; use `reshape()` when you're unsure.

---

**Q2: Why do we call `optimizer.zero_grad()` before each backward pass?** 🟢 Beginner

PyTorch accumulates gradients by default — each `backward()` call adds to existing gradients. If you don't zero them, gradients from previous batches are included in the current update, leading to incorrect weight updates. You zero them at the start of each batch (or iteration) to ensure clean gradient computation.

---

**Q3: What is the difference between `model.train()` and `model.eval()`?** 🟢 Beginner

`model.train()` sets the model to training mode: BatchNorm uses batch statistics, Dropout randomly drops neurons. `model.eval()` sets the model to evaluation mode: BatchNorm uses running mean/variance statistics, Dropout is disabled. Always switch modes correctly — using `model.train()` during evaluation gives incorrect results with Dropout and BatchNorm.

---

**Q4: What is `torch.no_grad()` and when should you use it?** 🟢 Beginner

`torch.no_grad()` is a context manager that disables gradient computation. Use it during inference/evaluation to: (1) save memory (no gradient tensors stored), (2) speed up computation (~30% faster). Never use it during training (gradients needed for backprop). Use with `model.eval()` during validation/testing.

---

**Q5: Explain the gradient flow during backpropagation in PyTorch.** 🔴 Advanced

PyTorch builds a computational graph during the forward pass — each operation creates a node with references to its inputs. During `loss.backward()`, PyTorch traverses this graph in reverse using the chain rule, computing `dL/dw` for each parameter w. Gradients are accumulated in `param.grad`. The optimizer then uses these gradients to update parameters via `optimizer.step()`. `retain_graph=True` keeps the graph for multiple backward passes.

---

**Q6: What is Mixed Precision Training and what are its benefits?** 🔴 Advanced

Mixed Precision Training uses FP16 (half-precision) for most computations and FP32 for numerically sensitive operations (loss, normalization). Benefits: ~2x memory reduction (enabling larger batches), ~2-3x faster on Tensor Core GPUs (A100, V100, RTX 3090+). PyTorch's `torch.cuda.amp.autocast()` handles this automatically, and `GradScaler` prevents gradient underflow (values too small to represent in FP16).

---

**Q7: What is `torch.compile` and how does it improve performance?** 🟡 Intermediate

`torch.compile` (PyTorch 2.0+) applies ahead-of-time graph compilation using TorchInductor, which generates optimized Triton kernels for GPU operations. It fuses operations (eliminating intermediate tensors), uses better memory access patterns, and leverages hardware-specific optimizations. Typical speedup: 10-40% on training, 2x+ on inference. The first iteration has compilation overhead; subsequent iterations use the cached compiled graph.

---

**Q8: How would you implement gradient accumulation for training with large effective batch sizes?** 🔴 Advanced

```python
accumulation_steps = 4  # Effective batch = 4 × mini-batch size

for step, (data, targets) in enumerate(loader):
    outputs = model(data.to(device))
    loss = criterion(outputs, targets.to(device))
    loss = loss / accumulation_steps  # Scale loss

    loss.backward()  # Accumulate gradients

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()   # Update after N steps
        optimizer.zero_grad()  # Clear gradients
```

This achieves an effective batch size of `accumulation_steps × batch_size` without increased memory usage.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch 2.0 Release Notes (torch.compile)](https://pytorch.org/blog/pytorch-2.0-release/)
- [Mixed Precision Training — NVIDIA](https://developer.nvidia.com/automatic-mixed-precision)
- [Deep Learning with PyTorch — Eli Stevens, Luca Antiga (Manning)](https://www.manning.com/books/deep-learning-with-pytorch)
