# Hugging Face Guide

A comprehensive guide to the Hugging Face ecosystem — the go-to platform for open-source AI models, datasets, and tools.

---

## Table of Contents

1. [Transformers Library Overview](#transformers-library-overview)
2. [Loading Models and Tokenizers](#loading-models-and-tokenizers)
3. [Pipeline API](#pipeline-api)
4. [Datasets Library](#datasets-library)
5. [PEFT — Parameter-Efficient Fine-Tuning](#peft--parameter-efficient-fine-tuning)
6. [Model Hub](#model-hub)
7. [Spaces and Gradio](#spaces-and-gradio)
8. [Inference Endpoints](#inference-endpoints)
9. [AutoTrain](#autotrain)
10. [Interview Q&A](#interview-qa)
11. [References](#references)

---

## Transformers Library Overview

The `transformers` library provides thousands of pretrained models for NLP, vision, and audio tasks.

```bash
pip install transformers accelerate datasets peft
```

**Core classes:**
- `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`
- `AutoTokenizer`, `AutoProcessor`
- `pipeline` — high-level API for inference
- `Trainer`, `TrainingArguments` — for fine-tuning

---

## Loading Models and Tokenizers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 4-bit quantization (requires bitsandbytes)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs
    trust_remote_code=True
)

# Simple generation
inputs = tokenizer("What is machine learning?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Chat with Chat Template

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful ML tutor."},
    {"role": "user", "content": "Explain gradient descent."}
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][input_ids.shape[1]:],
    skip_special_tokens=True
)
print(response)
```

---

## Pipeline API

The `pipeline` function is the simplest way to use models for inference.

```python
from transformers import pipeline

# Text generation
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0  # GPU 0, or -1 for CPU
)
result = generator("Machine learning is", max_length=100, num_return_sequences=2)
print(result)

# Text classification (sentiment)
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
result = classifier(["I love this product!", "This is terrible."])
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}, {'label': 'NEGATIVE', 'score': 0.98}]

# Named Entity Recognition
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("Elon Musk founded Tesla in California.")
print(result)

# Question answering
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa(
    question="What is gradient descent?",
    context="Gradient descent is an optimization algorithm used to minimize a loss function..."
)
print(result)

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer("Long article text here...", max_length=100, min_length=30)
print(result)

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")
print(result)  # [{'translation_text': 'Bonjour, comment allez-vous?'}]

# Zero-shot classification (no fine-tuning needed)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = zero_shot(
    "I need to debug my Python code",
    candidate_labels=["programming", "cooking", "sports", "music"]
)
print(result)

# Image classification
img_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = img_classifier("image.jpg")
print(result)

# Automatic speech recognition
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = asr("audio.mp3")
print(result["text"])
```

---

## Datasets Library

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Load a dataset from HF Hub
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({'train': Dataset({features: ['text', 'label'], num_rows: 25000}), ...})

# Access splits
train_data = dataset["train"]
test_data = dataset["test"]

# Access examples
print(train_data[0])
print(train_data[:5]["text"])

# Filter
positive_only = train_data.filter(lambda x: x["label"] == 1)

# Map (transform)
def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenized = train_data.map(
    lambda x: tokenize(x, tokenizer),
    batched=True,
    remove_columns=["text"]
)

# Create from Pandas DataFrame
df = pd.DataFrame({"text": ["Sample 1", "Sample 2"], "label": [0, 1]})
custom_dataset = Dataset.from_pandas(df)

# Save and load locally
tokenized.save_to_disk("./tokenized_imdb")
loaded = load_dataset("arrow", data_dir="./tokenized_imdb")

# Push to Hub
custom_dataset.push_to_hub("username/my-dataset", private=True)
```

---

## PEFT — Parameter-Efficient Fine-Tuning

PEFT allows fine-tuning large models with minimal trainable parameters.

```bash
pip install peft
```

### LoRA Fine-Tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Load base model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling factor
    target_modules=[         # Modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 3,793,516,544 || trainable%: 1.1057

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-mistral",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="mlflow"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./lora-adapter")
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA on top of quantized model — this is QLoRA
model = get_peft_model(model, lora_config)
```

### Loading a LoRA Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Optional: merge adapter into base model (faster inference, no PEFT overhead)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

---

## Model Hub

The Hugging Face Hub hosts 700,000+ models, 150,000+ datasets, and 300,000+ Spaces.

```python
from huggingface_hub import HfApi, login, hf_hub_download

# Authenticate
login(token="hf_your_token_here")  # Or set HF_TOKEN environment variable

api = HfApi()

# Search models
models = api.list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,  # Descending
    limit=10
)
for model in models:
    print(model.id, model.downloads)

# Upload a model
api.upload_folder(
    folder_path="./my-model",
    repo_id="username/my-model",
    repo_type="model"
)

# Download specific files
model_path = hf_hub_download(
    repo_id="google/gemma-2-9b-it",
    filename="config.json"
)

# Push model from Transformers
model.push_to_hub("username/my-finetuned-model", private=True)
tokenizer.push_to_hub("username/my-finetuned-model")
```

---

## Spaces and Gradio

Hugging Face Spaces let you deploy ML demos instantly.

```python
# app.py — Deploy this as a HF Space
import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def classify_sentiment(text: str) -> str:
    result = classifier(text)[0]
    return f"{result['label']} (confidence: {result['score']:.2%})"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(label="Enter text", placeholder="Type something here..."),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Classifier",
    description="Classify text sentiment using DistilBERT",
    examples=[
        ["I love this product!"],
        ["This is terrible service."],
        ["The movie was okay, nothing special."]
    ]
)

if __name__ == "__main__":
    demo.launch()
```

---

## Inference Endpoints

Hugging Face Inference Endpoints lets you deploy models on managed infrastructure.

```python
from huggingface_hub import InferenceClient

# Use public Inference API (free tier)
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="hf_your_token"
)

# Text generation
response = client.text_generation(
    "Explain gradient descent in simple terms:",
    max_new_tokens=200,
    temperature=0.7
)
print(response)

# Chat completion
messages = [
    {"role": "user", "content": "What is backpropagation?"}
]
response = client.chat_completion(messages, max_tokens=300)
print(response.choices[0].message.content)

# Embeddings
embedding = client.feature_extraction("Machine learning is fascinating")
print(f"Embedding dimensions: {len(embedding[0])}")

# Image classification
from PIL import Image
result = client.image_classification("cat.jpg")
print(result)
```

---

## AutoTrain

AutoTrain allows fine-tuning models with no-code or minimal code.

```bash
pip install autotrain-advanced

# Fine-tune LLM on a CSV dataset
autotrain llm \
  --train \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --data-path ./data/ \
  --text-column text \
  --train-split train \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --quantization int4 \
  --project-name my-fine-tuned-llm \
  --push-to-hub \
  --hf-token $HF_TOKEN
```

---

## Interview Q&A

**Q1: What is the difference between AutoModel and AutoModelForCausalLM?** 🟢 Beginner

`AutoModel` loads the base model without a task-specific head. `AutoModelForCausalLM` adds a language modeling head for text generation (next-token prediction). Similarly, `AutoModelForSequenceClassification` adds a classification head. The `Auto` prefix means the class automatically detects the correct architecture from the model name.

---

**Q2: What is LoRA and how does it reduce training cost?** 🟡 Intermediate

LoRA (Low-Rank Adaptation) adds small trainable matrices to specific weight matrices in the model. Instead of updating all parameters (e.g., 7 billion), LoRA adds two small matrices A (d×r) and B (r×d) where r << d. During training, only A and B are updated (rank r). Typical trainable parameter reduction: from 100% to 0.1-1%. This allows fine-tuning LLMs on a single GPU that couldn't hold the full model in training mode.

---

**Q3: What is the difference between LoRA and QLoRA?** 🟡 Intermediate

LoRA applies low-rank adapters to the full-precision model. QLoRA (Quantized LoRA) first quantizes the base model to 4-bit precision using NF4 quantization (reducing memory by ~75%), then applies LoRA adapters on top. QLoRA makes it possible to fine-tune 65B+ parameter models on a single consumer GPU (48GB VRAM). The quality is slightly lower than full LoRA but the memory savings are massive.

---

**Q4: What does `device_map="auto"` do?** 🟡 Intermediate

`device_map="auto"` tells the `accelerate` library to automatically distribute model layers across available hardware (multiple GPUs, CPU, disk). It reads the model's architecture, available device memory, and creates an optimal distribution plan. For models too large for a single GPU, this enables inference across multiple GPUs or even offloading some layers to CPU/disk.

---

**Q5: What is the Hugging Face pipeline API and when would you use it?** 🟢 Beginner

The `pipeline` function is a high-level, task-oriented API that handles model loading, preprocessing, inference, and postprocessing in one call. Use it for quick prototyping, demos, and production use cases where the task is standard (classification, NER, summarization, generation). For custom preprocessing, batching control, or performance optimization, drop down to the model/tokenizer level directly.

---

**Q6: How do you prevent gradient issues when fine-tuning with PEFT?** 🔴 Advanced

1. Call `prepare_model_for_kbit_training(model)` when using 4-bit/8-bit quantization — it handles gradient checkpointing, casting, and disabling quantized layer grads
2. Ensure LoRA target modules include all attention projection matrices
3. Set `lora_dropout` to regularize training
4. Use gradient clipping (`max_grad_norm=1.0` in `TrainingArguments`)
5. Enable `gradient_checkpointing=True` to trade compute for memory
6. Use `torch.cuda.amp` (mixed precision) via `fp16=True` or `bf16=True`

---

## References

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers GitHub](https://github.com/huggingface/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [QLoRA Paper — Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314)
- [LoRA Paper — Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
