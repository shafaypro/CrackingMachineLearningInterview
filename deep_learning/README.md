# Deep Learning

A comprehensive reference for deep learning fundamentals, architectures, training techniques, and regularization methods.

---

## Table of Contents

1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Activation Functions](#activation-functions)
3. [Training: Forward Pass, Backpropagation, Optimization](#training)
4. [Regularization Techniques](#regularization-techniques)
5. [Architecture Overview](#architecture-overview)
6. [Transfer Learning and Fine-Tuning](#transfer-learning-and-fine-tuning)
7. [Loss Functions](#loss-functions)
8. [Interview Q&A](#interview-qa)
9. [References](#references)

---

## Neural Network Fundamentals

### Neurons and Layers

A neuron computes a weighted sum of inputs, adds a bias, and applies an activation function:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = W·x + b
output = activation(z)
```

**Layer types:**
- **Input layer:** Receives raw features
- **Hidden layers:** Learn intermediate representations
- **Output layer:** Produces final predictions

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

### Universal Approximation Theorem

A neural network with a single hidden layer containing enough neurons can approximate any continuous function on a compact domain to arbitrary accuracy. This is why neural networks are such powerful function approximators.

---

## Activation Functions

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | Saturates → vanishing gradient |
| Tanh | (eˣ - e^(-x))/(eˣ + e^(-x)) | (-1, 1) | Zero-centered, still saturates |
| ReLU | max(0, x) | [0, ∞) | Fast, no vanishing gradient for x>0 |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | Fixes dying ReLU problem |
| GELU | x·Φ(x) | (-∞, ∞) | Used in Transformers, smooth |
| SiLU/Swish | x·σ(x) | (-∞, ∞) | Smooth, used in modern architectures |
| Softmax | eˣⁱ / Σeˣʲ | (0, 1), sums to 1 | Multi-class output layer |

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(F.sigmoid(x))   # 0.12, 0.27, 0.50, 0.73, 0.88
print(F.tanh(x))      # -0.96, -0.76, 0.0, 0.76, 0.96
print(F.relu(x))      # 0, 0, 0, 1, 2
print(F.leaky_relu(x, 0.01))  # -0.02, -0.01, 0, 1, 2
print(F.gelu(x))      # -0.046, -0.16, 0, 0.84, 1.95
print(F.softmax(x, dim=0))  # Probabilities summing to 1
```

---

## Training

### Forward Pass

Data flows through the network layer by layer, applying weights, biases, and activations.

```
Input x → Layer 1 → Layer 2 → ... → Output ŷ
Loss L = loss_function(ŷ, y)
```

### Backpropagation

Computes gradients of the loss with respect to each parameter using the chain rule.

```
∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁

Chain rule example (2-layer network):
∂L/∂W₁ = (∂L/∂a₂) · (∂a₂/∂z₂) · (∂z₂/∂a₁) · (∂a₁/∂z₁) · (∂z₁/∂W₁)
```

### Gradient Descent Variants

| Optimizer | Update Rule | Strengths | Weaknesses |
|-----------|-------------|-----------|------------|
| SGD | w ← w - η∇L | Simple, memory efficient | Noisy, slow convergence |
| SGD + Momentum | v ← βv + ∇L; w ← w - ηv | Faster convergence | Learning rate tuning |
| AdaGrad | w ← w - η/√(G+ε)·∇L | Adaptive per-param LR | LR decreases monotonically |
| RMSProp | w ← w - η/√(E[g²]+ε)·∇L | Fixes AdaGrad issue | No bias correction |
| Adam | Combines momentum + RMSProp | Fast, robust default | Memory usage, can overfit |
| AdamW | Adam + weight decay | Best default for deep learning | Slightly more memory |

```python
import torch.optim as optim

# Common optimizers
model = NeuralNetwork(784, [256, 128], 10)

sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Learning rate schedulers
scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, patience=5, factor=0.5)
scheduler = optim.lr_scheduler.OneCycleLR(adam, max_lr=0.01, total_steps=1000)
```

---

## Regularization Techniques

### Dropout

Randomly sets a fraction of neurons to zero during training, preventing co-adaptation.

```python
# During training: randomly zero with probability p
# During inference: scale by (1-p) — or use inverted dropout (PyTorch default)

dropout = nn.Dropout(p=0.5)  # 50% dropout
# Standard: 0.1-0.5 for hidden layers, 0.0-0.1 for output
```

**Why it works:** Forces the network to learn redundant representations. Acts like an ensemble of 2^n subnetworks.

### Batch Normalization

Normalizes layer inputs to have zero mean and unit variance within each mini-batch.

```python
# Applied between linear layer and activation
nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),   # Normalize across batch dimension
    nn.ReLU()
)

# For images
nn.Conv2d(64, 128, 3, padding=1)
nn.BatchNorm2d(128)   # Normalize across spatial and batch dimensions
nn.ReLU()
```

**Why it works:** Reduces internal covariate shift, allows higher learning rates, acts as regularizer, reduces sensitivity to initialization.

### Layer Normalization

Normalizes across the feature dimension (not batch). Used in Transformers.

```python
# Preferred in Transformers and RNNs where batch size varies
nn.LayerNorm(normalized_shape=512)
```

### Weight Decay (L2 Regularization)

```python
# Add L2 penalty to weights via optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # L2 regularization coefficient
)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Architecture Overview

### Convolutional Neural Networks (CNNs)

Process spatial data (images, time series) using local filters.

```
Input Image → [Conv → BN → ReLU → Pool] × N → Flatten → FC → Output

Key layers:
- Conv2d: Learn local features using learned filters
- MaxPool2d: Spatial downsampling, translation invariance
- BatchNorm2d: Stabilize training
- Global Average Pooling: Reduce spatial dims without flatten

Famous architectures: AlexNet, VGG, ResNet, EfficientNet, ConvNeXt
```

```python
# ResNet-style block with skip connection
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)
```

### Recurrent Neural Networks (RNNs) and LSTMs

Process sequential data with memory of previous steps.

```
RNN: hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
LSTM adds: forget gate, input gate, output gate, cell state
GRU: Simpler version of LSTM with update and reset gates

Problems with vanilla RNNs:
- Vanishing gradients over long sequences
- Can't capture long-range dependencies
```

```python
# LSTM for sequence classification
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        # Concat final hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))
```

### Transformers

See [Transformers Deep Dive](./intro_transformers.md) for detailed coverage.

- **BERT (encoder-only):** Bidirectional context, great for classification, NER, QA
- **GPT (decoder-only):** Autoregressive generation, text completion
- **T5/BART (encoder-decoder):** Seq2Seq, translation, summarization

### Diffusion Models

Generative models that learn to reverse a gradual noising process.

```
Forward process: x₀ → x₁ → ... → xₜ (add Gaussian noise)
Reverse process: xₜ → xₜ₋₁ → ... → x₀ (learned denoising)
```

Used in: Stable Diffusion (images), DALL-E 3, AudioLDM (audio), protein generation.

---

## Transfer Learning and Fine-Tuning

### Approaches

| Approach | When to Use | Example |
|----------|-------------|---------|
| Feature extraction | Small dataset, similar domain | ResNet features for medical images |
| Fine-tuning last layers | Moderate dataset, similar domain | BERT for text classification |
| Full fine-tuning | Large dataset, sufficient compute | GPT for domain-specific generation |
| Few-shot prompting | Very small dataset | Prompt engineering with LLMs |

```python
import torchvision.models as models

# Load pretrained model
resnet = models.resnet50(pretrained=True)

# Option 1: Feature extraction — freeze all but classifier
for param in resnet.parameters():
    param.requires_grad = False

# Replace classifier head for new task
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)
# Only resnet.fc parameters will be updated

# Option 2: Fine-tune last few layers
for param in resnet.layer4.parameters():
    param.requires_grad = True
for param in resnet.fc.parameters():
    param.requires_grad = True

# Different learning rates for different layers
optimizer = optim.AdamW([
    {"params": resnet.layer4.parameters(), "lr": 1e-4},
    {"params": resnet.fc.parameters(), "lr": 1e-3}
])
```

---

## Loss Functions

| Loss | Use Case | Formula |
|------|----------|---------|
| MSE | Regression | `(1/n)Σ(y - ŷ)²` |
| MAE | Regression (outlier robust) | `(1/n)Σ\|y - ŷ\|` |
| Binary Cross-Entropy | Binary classification | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` |
| Categorical Cross-Entropy | Multi-class classification | `-Σ yᵢ·log(ŷᵢ)` |
| Hinge Loss | SVM, margin-based | `max(0, 1 - y·ŷ)` |
| KL Divergence | Distribution matching | `Σ P·log(P/Q)` |
| Contrastive / Triplet | Embeddings, metric learning | `max(0, d(a,p) - d(a,n) + margin)` |

```python
import torch.nn as nn

criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_bce = nn.BCEWithLogitsLoss()   # More stable than BCELoss
criterion_ce = nn.CrossEntropyLoss()     # Includes softmax
criterion_ce_weighted = nn.CrossEntropyLoss(weight=class_weights)  # For imbalance
```

---

## Interview Q&A

**Q1: What is the vanishing gradient problem and how do we solve it?** 🟡 Intermediate

During backpropagation through deep networks, gradients are multiplied by the derivative of each activation function. Sigmoid/tanh derivatives are < 1, so gradients shrink exponentially through layers, preventing early layers from learning. Solutions: (1) ReLU activations (gradient = 1 for positive inputs), (2) residual/skip connections (gradient flows directly), (3) batch normalization (normalizes inputs, prevents saturation), (4) careful initialization (Xavier/He), (5) gradient clipping.

---

**Q2: Explain backpropagation using the chain rule.** 🔴 Advanced

Backprop computes `∂L/∂w` for all parameters. For a 2-layer network: `L = loss(output, y)`, `output = σ(W₂·h)`, `h = σ(W₁·x)`.

Chain rule: `∂L/∂W₁ = ∂L/∂output · ∂output/∂h · ∂h/∂W₁`

Each term is computed locally: `∂L/∂output` from the loss, `∂output/∂h = W₂ · σ'(W₂·h)`, `∂h/∂W₁ = σ'(W₁·x) · xᵀ`. PyTorch's autograd engine records operations during the forward pass (computational graph) and traverses it in reverse for backprop.

---

**Q3: What is batch normalization and why does it help?** 🟡 Intermediate

Batch normalization normalizes a layer's inputs to zero mean and unit variance within each mini-batch: `x̂ = (x - μB) / √(σB² + ε)`, then scales and shifts with learned parameters γ and β: `y = γx̂ + β`. Benefits: reduces internal covariate shift (distribution of layer inputs changing during training), allows higher learning rates, acts as a regularizer (noise from batch statistics), reduces sensitivity to weight initialization. At inference, uses running mean/variance computed during training.

---

**Q4: What is the difference between BatchNorm and LayerNorm?** 🟡 Intermediate

BatchNorm normalizes across the batch dimension (computes statistics over the batch for each feature). LayerNorm normalizes across the feature dimension (computes statistics over all features for each sample). BatchNorm requires a minimum batch size to compute stable statistics and has issues with very small batches. LayerNorm works per sample, so it's batch-size independent and preferred for Transformers (variable-length sequences), RNNs, and online learning.

---

**Q5: What is dropout and does it help at inference time?** 🟢 Beginner

Dropout randomly sets a fraction p of neurons to zero during training. This prevents neurons from co-adapting and forces learning of redundant representations. At inference, dropout is disabled (model.eval() in PyTorch) and all neurons are active. To compensate for the increased scale (all neurons active vs fraction during training), activations are scaled by 1/(1-p) during training (inverted dropout, PyTorch default).

---

**Q6: What are residual connections and why are they important?** 🟡 Intermediate

Residual connections (skip connections) add the input of a block directly to its output: `output = F(x) + x`. Key benefits: (1) gradients flow directly through the skip connection, addressing vanishing gradients in very deep networks, (2) the block only needs to learn the residual `F(x)` (what to add to x), which is often small and easier to learn, (3) enable training of very deep networks (100+ layers). Introduced in ResNet (2015) and fundamental to Transformers (residual around each attention and FFN block).

---

**Q7: Explain the attention mechanism in neural networks.** 🔴 Advanced

Attention computes a weighted sum of values where weights are based on similarity between a query and keys.

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

1. Q, K, V are linear projections of the input
2. QKᵀ computes similarity scores between queries and keys
3. Scale by √dₖ to prevent vanishing gradients
4. Softmax converts to probabilities
5. Multiply by V to get the attended representation
```

Self-attention: Q, K, V all come from the same sequence (captures relationships within the sequence). Cross-attention: Q from one sequence (decoder), K and V from another (encoder output).

---

**Q8: What is the difference between CNNs and Transformers for image tasks?** 🔴 Advanced

CNNs use local convolutional filters — effective for capturing local patterns and translation equivariance. They process images hierarchically (local → global). Transformers use self-attention — each patch can attend to every other patch from the start, capturing long-range dependencies. ViT (Vision Transformer) splits images into patches and processes them as a sequence. Transformers need more data than CNNs to learn useful representations but scale better. Modern architectures (ConvNeXt, Swin Transformer) blend both approaches.

---

**Q9: What is the exploding gradient problem and how do you prevent it?** 🟡 Intermediate

Exploding gradients occur when gradient values grow exponentially through deep layers, causing extremely large weight updates that destabilize training (NaN losses). Common in RNNs. Solutions: (1) gradient clipping — cap gradients at a maximum norm, (2) weight initialization — use Xavier/He initialization, (3) batch normalization — normalizes activations, preventing extreme values, (4) LSTM/GRU — gating mechanisms limit gradient flow, (5) residual connections — provide gradient highways.

---

**Q10: What is the difference between fine-tuning and training from scratch?** 🟢 Beginner

Training from scratch initializes weights randomly and trains on a dataset from nothing — requires large datasets and significant compute. Fine-tuning starts from pretrained weights (learned on a large dataset) and continues training on a task-specific dataset — requires much less data and compute. Fine-tuning leverages transfer learning: knowledge from the pretraining task generalizes to the new task. For most practical NLP and vision tasks, fine-tuning a pretrained model outperforms training from scratch.

---

**Q11: What is the dying ReLU problem?** 🟡 Intermediate

A ReLU neuron "dies" when its input is always negative — gradient is 0, so the neuron never updates and permanently outputs 0. Causes: high learning rates causing large negative weight updates, poor initialization, gradient flow issues. Solutions: Leaky ReLU (small negative slope for x<0), ELU, GELU/SiLU (smooth, always non-zero gradient), careful learning rate tuning, good initialization.

---

**Q12: Compare the Adam and SGD optimizers.** 🔴 Advanced

**SGD with momentum:** Updates based on gradient with velocity accumulation. Simple, generalizes well in some domains. Requires careful learning rate tuning. Often finds wider minima (better generalization).

**Adam:** Maintains adaptive learning rates per parameter using first moment (mean) and second moment (variance of gradients). Converges faster, less sensitive to learning rate. Can overfit more easily, may converge to sharp minima with worse generalization. Weight decay in Adam should use AdamW (decoupled) for correct regularization. In practice: Adam/AdamW for transformers and NLP, SGD+momentum often used in CV tasks.

---

**Q13: What is the universal approximation theorem and what are its limitations?** 🔴 Advanced

The theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of Rⁿ to arbitrary precision, given a non-polynomial activation function.

Limitations: (1) "sufficient neurons" can be exponential in input dimensions, (2) says nothing about how to find the weights (optimization), (3) doesn't address generalization (approximating training data ≠ generalizing to new data), (4) approximating a function and actually learning it from finite noisy data are different problems.

---

**Q14: What is gradient checkpointing?** 🔴 Advanced

Gradient checkpointing trades compute for memory. Normally, all intermediate activations are stored during the forward pass for use in backpropagation — memory grows linearly with depth. With gradient checkpointing, only a subset of activations are stored (checkpoints); others are recomputed from the nearest checkpoint during backpropagation. This reduces memory from O(n) to O(√n) at the cost of ~30% more compute. Essential for training large models on memory-constrained hardware.

---

**Q15: What are the key differences between encoder-only, decoder-only, and encoder-decoder transformer architectures?** 🟡 Intermediate

| Architecture | Examples | Self-Attention | Use Case |
|---|---|---|---|
| Encoder-only | BERT, RoBERTa | Bidirectional (all tokens attend to all) | Classification, NER, QA |
| Decoder-only | GPT, Llama, Gemma | Causal (tokens only attend to past) | Text generation, chat |
| Encoder-Decoder | T5, BART, Whisper | Encoder: bidirectional; Decoder: causal + cross-attention | Translation, summarization, speech |

---

**Q16: What is knowledge distillation?** 🔴 Advanced

Knowledge distillation trains a small "student" model to mimic a large "teacher" model. The student learns from soft probability distributions (temperature-scaled logits) output by the teacher rather than just hard labels. Soft targets carry richer information about class relationships (e.g., "30% likely class A, 20% likely class B"). This allows small models to achieve performance close to large models, enabling deployment on constrained hardware. Example: DistilBERT (66% of BERT parameters, 97% of BERT performance).

---

**Q17: What is a hyperparameter and how do you tune them?** 🟢 Beginner

Hyperparameters are configuration values set before training that control the learning process (not learned from data). Examples: learning rate, batch size, number of layers, dropout rate, weight decay. Tuning methods: (1) grid search — exhaustive search over specified values, (2) random search — randomly sample combinations, often more efficient, (3) Bayesian optimization — use probabilistic model to select promising hyperparameters, (4) population-based training — evolve hyperparameters during training. Tools: Optuna, W&B Sweeps, Ray Tune.

---

**Q18: What is a variational autoencoder (VAE)?** 🔴 Advanced

A VAE is a generative model that learns a latent distribution rather than discrete encodings. The encoder maps input to a mean μ and variance σ of a Gaussian distribution. The decoder samples from z ~ N(μ, σ²) and reconstructs the input. Loss = reconstruction loss + KL divergence (pushes latent distribution toward N(0, I)). The reparameterization trick (z = μ + σ·ε, ε ~ N(0,1)) enables gradient flow through the sampling step. VAEs enable smooth latent space interpolation and new sample generation.

---

**Q19: What is the difference between discriminative and generative models?** 🟡 Intermediate

**Discriminative models** learn the conditional probability P(Y|X) — the boundary between classes. Examples: logistic regression, SVM, BERT for classification. Generally better at classification when trained on sufficient labeled data.

**Generative models** learn the joint distribution P(X, Y) or just P(X). Can generate new samples from the learned distribution. Examples: Naive Bayes, VAE, GAN, diffusion models. Useful when you need to generate data, with limited labeled data (using P(X) as prior), or for understanding data structure.

---

**Q20: What is gradient accumulation and when would you use it?** 🟡 Intermediate

Gradient accumulation delays optimizer updates for multiple forward/backward passes, effectively increasing the batch size without increasing memory. Instead of updating after each batch, gradients are accumulated over N mini-batches, then an optimizer step is taken. Use when: memory constraints prevent large batch training, but the optimization benefits from large batches (stability, better generalization). Common in LLM fine-tuning: `per_device_batch_size=4, gradient_accumulation_steps=8` → effective batch size 32.

---

## References

- [Deep Learning — Ian Goodfellow, Yoshua Bengio, Aaron Courville (free online)](https://www.deeplearningbook.org/)
- [Dive into Deep Learning — d2l.ai](https://d2l.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning for Image Recognition — He et al. (2015)](https://arxiv.org/abs/1512.03385)
