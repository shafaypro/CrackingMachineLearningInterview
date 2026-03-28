# Unsloth — Fast LoRA Fine-Tuning

A comprehensive guide to Unsloth — the fastest library for fine-tuning LLMs with LoRA and QLoRA.

---

## Table of Contents

1. [What is Unsloth](#what-is-unsloth)
2. [Installation](#installation)
3. [Fine-Tuning Llama 3, Mistral, Gemma](#fine-tuning-llama-3-mistral-gemma)
4. [Full Fine-Tuning Example](#full-fine-tuning-example)
5. [Speed Comparison](#speed-comparison)
6. [Interview Q&A](#interview-qa)
7. [References](#references)

---

## What is Unsloth

Unsloth is an open-source Python library that significantly accelerates LoRA and QLoRA fine-tuning of large language models. It achieves 2-5x faster training and 50-80% less memory usage compared to standard Hugging Face PEFT + Transformers.

**Key features:**
- 2-5x faster training than standard HF approach
- 50-80% memory reduction enabling larger batch sizes
- Supports Llama, Mistral, Gemma, Phi, Qwen, DeepSeek, and more
- Compatible with standard Hugging Face ecosystem (PEFT, TRL, Trainer)
- Manual CUDA kernel implementations for attention and matrix ops
- Free open-source version for single-GPU training

**How it achieves speedups:**
- Rewritten CUDA kernels using OpenAI Triton (avoiding PyTorch overhead)
- Optimized gradient checkpointing
- Memory-efficient QLoRA with NF4 quantization
- Kernel fusion — fewer memory reads/writes

---

## Installation

```bash
# Install for CUDA 12.1 (most common)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Or via pip (stable release)
pip install unsloth

# Verify installation
python -c "import unsloth; print(unsloth.__version__)"

# Install with all dependencies
pip install unsloth trl peft accelerate bitsandbytes datasets transformers
```

### Supported Environments

| Environment | Support |
|-------------|---------|
| NVIDIA GPU (CUDA 12.1+) | Full support |
| Google Colab (free tier T4) | Full support |
| Kaggle (P100, T4) | Full support |
| Apple Silicon (MPS) | CPU fallback |
| AMD GPU (ROCm) | Partial support |

---

## Fine-Tuning Llama 3, Mistral, Gemma

### Supported Models

```python
from unsloth import FastLanguageModel

# Llama 3 family
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.2-3B-Instruct")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.3-70B-Instruct")

# Mistral family
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/mistral-7b-instruct-v0.3")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Mistral-Nemo-Instruct-2407")

# Gemma family
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/gemma-2-9b-it")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/gemma-2-27b-it")

# Phi family
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Phi-3.5-mini-instruct")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/phi-4")

# Qwen family
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-7B-Instruct")

# DeepSeek
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-8B")
```

---

## Full Fine-Tuning Example

A complete working example: load a model, add LoRA, train, save, and push to the Hugging Face Hub.

### Step 1: Load Model

```python
from unsloth import FastLanguageModel
import torch

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048      # Context window
DTYPE = None               # Auto-detect (float16 or bfloat16)
LOAD_IN_4BIT = True        # Use QLoRA (4-bit quantization)

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token="hf_your_token"  # Only needed for gated models
)

print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 2: Add LoRA Adapters

```python
# Add LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank (higher = more parameters, more quality)
    target_modules=[               # Which modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                 # Scaling factor (usually = r)
    lora_dropout=0,                # 0 is optimized by Unsloth
    bias="none",                   # No bias (optimized)
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,              # Rank-stabilized LoRA (optional)
    loftq_config=None,             # LoftQ initialization (optional)
)

model.print_trainable_parameters()
# trainable params: 20,971,520 || all params: 3,233,435,648 || trainable%: 0.6486
```

### Step 3: Prepare Dataset

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Apply the correct chat template for the model
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # or "mistral", "gemma", "phi-3", etc.
)

def format_conversations(examples):
    """Format conversations into the model's chat template."""
    conversations = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in conversations
    ]
    return {"text": texts}

# Load a dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")

# Format the dataset
dataset = dataset.map(format_conversations, batched=True)

# Split into train/val
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training examples: {len(train_dataset)}")
print(f"Sample:\n{train_dataset[0]['text'][:500]}")
```

### Step 4: Train with TRL SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    num_train_epochs=3,

    # Batch size and gradient accumulation
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 2 * 4 = 8

    # Optimizer
    learning_rate=2e-4,
    optim="adamw_8bit",            # Memory-efficient optimizer
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # Mixed precision
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    # Logging and saving
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Seeds for reproducibility
    seed=42,

    # Push to Hub
    push_to_hub=False,  # Set True to auto-push
    report_to="none",   # or "wandb", "mlflow"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,                  # Pack short sequences together for efficiency
    args=training_args,
)

# Show current memory
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB")
print(f"Memory reserved = {start_gpu_memory} GB")

# Train
trainer_stats = trainer.train()

# Print stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak memory: {used_memory} GB")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
```

### Step 5: Test the Fine-Tuned Model

```python
from unsloth import FastLanguageModel

# Switch to inference mode (2x faster)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful ML expert."},
    {"role": "user", "content": "Explain dropout regularization."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    use_cache=True,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
```

### Step 6: Save and Push to Hub

```python
# Save LoRA adapter only (small, ~100MB)
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")

# Save merged model (full model with LoRA baked in)
model.save_pretrained_merged(
    "./merged-model",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit", "lora"
)

# Push LoRA adapter to Hub
model.push_to_hub(
    "username/llama3-finetuned-adapter",
    tokenizer,
    save_method="lora",
    token="hf_your_token"
)

# Push merged model to Hub
model.push_to_hub_merged(
    "username/llama3-finetuned-merged",
    tokenizer,
    save_method="merged_16bit",
    token="hf_your_token"
)

# Save as GGUF for Ollama / llama.cpp
model.save_pretrained_gguf(
    "./gguf-model",
    tokenizer,
    quantization_method="q4_k_m"  # Q4_K_M quantization
)

# Push GGUF to Hub
model.push_to_hub_gguf(
    "username/llama3-finetuned-gguf",
    tokenizer,
    quantization_method="q4_k_m",
    token="hf_your_token"
)
```

### Load the Saved GGUF in Ollama

```bash
# Create an Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./gguf-model/unsloth.Q4_K_M.gguf

SYSTEM "You are a helpful ML expert."
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

# Create and run the model
ollama create my-finetuned-llama -f Modelfile
ollama run my-finetuned-llama
```

---

## Speed Comparison

Benchmarks comparing Unsloth vs standard HuggingFace (HF) training on Llama 3.2 3B with QLoRA:

| Metric | Standard HF | Unsloth Free | Speedup |
|--------|------------|--------------|---------|
| Training speed | 1x | 2.0x | +100% |
| GPU memory | 16 GB | 8 GB | -50% |
| Batch size (16GB GPU) | 2 | 4 | 2x |
| Time for 1000 steps | 120 min | 55 min | 2.2x |

*For Llama 3.2 3B, batch size 2, max_seq_length 2048, on NVIDIA A100 80GB*

**Unsloth Pro** (paid) achieves up to 5x speedup and 80% memory reduction.

---

## Interview Q&A

**Q1: What is Unsloth and what makes it faster than standard HuggingFace training?** 🟡 Intermediate

Unsloth is a library for fast LoRA/QLoRA fine-tuning. It achieves 2-5x speedup through: (1) manually written CUDA kernels using OpenAI Triton that replace PyTorch's generic implementations, (2) kernel fusion — combining multiple operations into one kernel pass to reduce memory bandwidth, (3) optimized gradient checkpointing that is smarter about which activations to recompute, and (4) memory-efficient attention implementations.

---

**Q2: What is the difference between saving a LoRA adapter vs a merged model?** 🟡 Intermediate

A **LoRA adapter** contains only the small additional matrices (r << d). It's compact (~50-200MB) but requires loading the base model + adapter at inference time. A **merged model** combines the LoRA weights back into the base model weights — it's a complete standalone model but at full size. Use adapter format for flexibility (swap adapters), use merged format for production deployment where simplicity matters.

---

**Q3: What is `use_gradient_checkpointing="unsloth"` and when should you use it?** 🔴 Advanced

Gradient checkpointing reduces memory by not storing all intermediate activations during the forward pass — instead it recomputes them during backpropagation. Standard gradient checkpointing has a fixed recomputation strategy. Unsloth's implementation is smarter: it selectively checkpoints the most memory-intensive activations while minimizing recomputation overhead, achieving a better memory-speed tradeoff than the standard approach. Use it whenever training on limited VRAM (< 24GB).

---

**Q4: How would you use a model fine-tuned with Unsloth in Ollama?** 🟡 Intermediate

Unsloth supports exporting to GGUF format via `model.save_pretrained_gguf()`. After export, create an Ollama Modelfile that references the GGUF file, then run `ollama create model-name -f Modelfile`. The model can then be used with `ollama run` or via Ollama's API — making it accessible locally without any Python dependencies.

---

**Q5: What LoRA rank should you choose and how does it affect training?** 🟡 Intermediate

LoRA rank `r` controls the number of trainable parameters. Common values: r=4-8 for small datasets or quick experiments, r=16-32 for general fine-tuning (good quality-efficiency balance), r=64+ for complex task adaptation. Higher `r` = more trainable parameters = better expressiveness but slower training and more memory. Start with r=16, evaluate quality, then adjust. The ratio of trainable to total parameters is roughly `2 * r * L / P` where L is number of target layers and P is total parameters.

---

## References

- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth Notebooks (Google Colab)](https://github.com/unslothai/unsloth#-finetune-for-free)
- [TRL Documentation (SFTTrainer)](https://huggingface.co/docs/trl/sft_trainer)
- [LoRA Paper — Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper — Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314)
