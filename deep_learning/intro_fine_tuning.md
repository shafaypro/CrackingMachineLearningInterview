# Fine-Tuning Large Language Models: LoRA, QLoRA & PEFT

## Why Fine-Tune?

Pre-trained LLMs are general-purpose. Fine-tuning adapts them to specific tasks, domains, or styles without training from scratch. The challenge: LLMs have billions of parameters — full fine-tuning is memory-prohibitive for most practitioners.

**When to fine-tune vs. prompt engineer:**
- **Prompt engineering first** — if you can solve the task with few-shot examples, do that
- **Fine-tune when** — consistent format/style is needed, task is domain-specific (legal, medical), latency requires shorter prompts, behavior that's hard to describe in a prompt

---

## Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT)

| Aspect | Full Fine-Tuning | PEFT (LoRA/QLoRA) |
|--------|-----------------|-------------------|
| Parameters updated | All (~7B for 7B model) | <1% of model (~4-40M) |
| GPU memory (7B) | ~112 GB (bf16) | ~6-10 GB |
| Risk of catastrophic forgetting | High | Low |
| Training speed | Slow | Fast |
| Multiple task adapters | Impractical | Swap adapters cheaply |

---

## LoRA: Low-Rank Adaptation

### Core Idea

Instead of updating weight matrix **W** (d×k), LoRA learns two small matrices **A** (d×r) and **B** (r×k) where r << d.

```
W_new = W_pretrained + ΔW = W_pretrained + B · A
```

During training, W_pretrained is **frozen**. Only A and B are updated.

**Why low rank works:** The updates needed to adapt a pre-trained model to a downstream task tend to have low intrinsic dimensionality — they lie in a small subspace.

### LoRA Math

```
h = W₀x + ΔWx = W₀x + BAx
```

- W₀ ∈ ℝ^(d×k) — frozen pretrained weights
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k) — trainable low-rank matrices
- r = rank (typical: 4, 8, 16, 64)
- A is initialized with random Gaussian, B with zeros → ΔW = 0 at init

**Scaling factor α:** Often ΔW = (α/r) · BA. Setting α=r gives no scaling; α < r downscales the update.

### Parameter Savings Example

```
# 7B model, attention projection: d=4096, k=4096
# Full: 4096 * 4096 = 16.7M parameters
# LoRA r=16: 4096*16 + 16*4096 = 131K parameters → 99.2% savings
```

### Which Modules to Apply LoRA To?

Common choices (from the original paper):
- `q_proj`, `v_proj` — query and value projections in attention
- `k_proj`, `o_proj` — key and output projections
- `gate_proj`, `up_proj`, `down_proj` — MLP layers

Applying LoRA to all linear layers usually outperforms selective application.

### LoRA with HuggingFace PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank
    lora_alpha=32,                 # scaling: alpha/r = 2x
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.62%
```

---

## QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) enables fine-tuning 65B models on a single 48GB GPU by combining:

1. **4-bit NormalFloat (NF4)** quantization of the base model
2. **Double quantization** — quantize the quantization constants themselves
3. **Paged optimizers** — offload optimizer states to CPU RAM when GPU memory spikes
4. **LoRA adapters** trained in bf16

### NF4 Quantization

NF4 is an information-theoretically optimal quantization for normally distributed weights. Standard int4 assumes uniform distribution; NF4 uses quantile bins.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # double quantization
    bnb_4bit_quant_type="nf4",         # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16  # compute in bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Full QLoRA Training Script

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Prepare for k-bit training (enables gradient checkpointing, casts norms to fp32)
model = prepare_model_for_kbit_training(model)

# 3. Add LoRA adapters
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-mistral-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch = 16
    gradient_checkpointing=True,      # save memory at cost of recompute
    optim="paged_adamw_8bit",         # paged optimizer
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 5. Dataset (instruction format)
dataset = load_dataset("your-dataset", split="train")

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

# 6. Train with SFTTrainer (handles packing, formatting)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_instruction,
    max_seq_length=2048,
    packing=True,  # pack short samples into one sequence for efficiency
)

trainer.train()
trainer.save_model()
```

### Memory Comparison (7B Model)

| Method | GPU Memory | Hardware |
|--------|-----------|----------|
| Full fine-tune (bf16) | ~112 GB | 2× A100 80GB |
| LoRA (bf16 base) | ~28 GB | 1× A100 40GB |
| QLoRA (4-bit base) | ~10 GB | 1× RTX 3090 |

---

## Instruction Tuning

Instruction tuning fine-tunes a pretrained LLM to follow natural language instructions by training on (instruction, output) pairs.

### Data Formats

**Alpaca format:**
```json
{
  "instruction": "Classify the sentiment of the following review.",
  "input": "The product broke after one week. Very disappointing.",
  "output": "Negative"
}
```

**ShareGPT / conversational format:**
```json
{
  "conversations": [
    {"from": "human", "value": "Explain gradient descent like I'm five."},
    {"from": "gpt", "value": "Imagine you're lost on a foggy hill..."}
  ]
}
```

**Chat template (model-specific):**
```python
# Llama-3 format
tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is backprop?"}],
    tokenize=False,
    add_generation_prompt=True
)
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is backprop?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
```

### Key Instruction Tuning Tips

1. **Quality > Quantity** — 1K high-quality examples often outperforms 100K noisy ones
2. **Diversity of instructions** — cover many task types (summarization, QA, code, classification)
3. **Response length calibration** — include both short and long answers
4. **System prompts** — include them in training data if you'll use them at inference

---

## RLHF and DPO

### RLHF (Reinforcement Learning from Human Feedback)

Three-stage process:
1. **SFT** — Supervised fine-tuning on demonstrations
2. **Reward modeling** — Train a reward model on human preference pairs (chosen vs rejected)
3. **PPO** — Fine-tune the SFT model with RL using reward model signal

Expensive: requires running 4 models simultaneously (SFT, reward, policy, value function).

### DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) achieves RLHF quality without explicit RL training:

```
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

Where y_w = preferred response, y_l = less preferred response.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,          # KL regularization strength
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)

# Dataset format: {"prompt": ..., "chosen": ..., "rejected": ...}
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,   # frozen reference model (SFT checkpoint)
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**DPO vs RLHF trade-offs:**
- DPO: simpler, stable, no RL infrastructure needed, often competitive quality
- RLHF: more flexible (can use live reward signal), better when reward model is very accurate

---

## Merging Adapters & Model Management

### Merging LoRA Weights Back

After training, you can merge LoRA weights into the base model for faster inference:

```python
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./qlora-output")

# Merge and unload
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-mistral-7b-finetuned")
tokenizer.save_pretrained("./merged-mistral-7b-finetuned")
```

### Multi-Adapter Serving

Keep base model loaded, swap adapters per request:

```python
# Load multiple adapters without merging
model.load_adapter("./adapter-task-a", adapter_name="task_a")
model.load_adapter("./adapter-task-b", adapter_name="task_b")

# Switch at inference time
model.set_adapter("task_a")
output_a = model.generate(...)

model.set_adapter("task_b")
output_b = model.generate(...)
```

---

## Hyperparameter Guidelines

| Hyperparameter | Typical Range | Notes |
|---------------|--------------|-------|
| LoRA rank (r) | 4–64 | Higher r = more capacity, more memory |
| LoRA alpha | r or 2×r | Scales gradient updates |
| Learning rate | 1e-4 to 3e-4 | Higher than full fine-tuning |
| Batch size | 4–32 (effective) | Use gradient accumulation |
| Epochs | 1–5 | Overfit risk increases with more epochs |
| Dropout | 0.05–0.1 | Light regularization |
| Warmup ratio | 0.03–0.05 | Warmup prevents early instability |

---

## Common Interview Questions

**Q: Why does LoRA work well even with rank=8?**
The adaptation needed to steer a pretrained model toward a task has low intrinsic rank — the gradient updates during fine-tuning lie in a low-dimensional subspace of the full parameter space. LoRA captures this efficiently.

**Q: When would you use full fine-tuning over LoRA?**
When compute is available and the task requires fundamental behavioral shifts (e.g., changing the language a model operates in, major domain shift like scientific reasoning). For most task adaptation, LoRA is sufficient and preferred.

**Q: How do you prevent catastrophic forgetting with LoRA?**
LoRA inherently reduces forgetting since the base weights are frozen. Additional strategies: include general-domain samples in your training data (replay), use regularization, keep rank low.

**Q: What's the difference between QLoRA and quantization for inference?**
QLoRA uses quantization during *training* to reduce memory so you can fine-tune on consumer hardware. Inference quantization is for *deployment* speed/size. You can use both: fine-tune with QLoRA, then quantize the merged model again for inference (GPTQ, AWQ, GGUF).

**Q: How would you evaluate if fine-tuning improved the model?**
Automated: task-specific metrics (accuracy, ROUGE, BLEU, code pass@k). LLM-as-judge: use a stronger model (GPT-4) to evaluate response quality on a held-out set. Human eval: for subjective quality. Monitor both capability gain *and* regression on general benchmarks (MT-Bench, MMLU).

**Q: Explain gradient checkpointing trade-off.**
Gradient checkpointing saves GPU memory by not storing all intermediate activations during the forward pass. Instead, they're recomputed during backprop when needed. This trades ~30% training speed for ~60-70% memory reduction — crucial when fine-tuning large models.
