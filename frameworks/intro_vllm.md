# vLLM — High-Throughput LLM Serving

A comprehensive guide to vLLM — the high-performance inference engine for large language models.

---

## Table of Contents

1. [What is vLLM](#what-is-vllm)
2. [Installation](#installation)
3. [Starting the Server](#starting-the-server)
4. [Python Client Examples](#python-client-examples)
5. [Multi-GPU with Tensor Parallelism](#multi-gpu-with-tensor-parallelism)
6. [Quantization](#quantization)
7. [Streaming Responses](#streaming-responses)
8. [Supported Models](#supported-models)
9. [Interview Q&A](#interview-qa)
10. [References](#references)

---

## What is vLLM

vLLM (Virtual LLM) is an open-source LLM inference and serving library developed by UC Berkeley. It achieves state-of-the-art serving throughput through its key innovation: **PagedAttention**.

**Key features:**
- OpenAI-compatible REST API
- High throughput via continuous batching
- Memory-efficient KV cache with PagedAttention
- Multi-GPU support via tensor and pipeline parallelism
- Quantization: AWQ, GPTQ, FP8, GGUF
- Streaming token generation
- Supports 100+ model architectures

### PagedAttention

The core innovation of vLLM. Traditional inference engines pre-allocate a contiguous memory block for the KV (key-value) cache based on the maximum sequence length. This wastes memory for shorter sequences.

PagedAttention manages the KV cache in fixed-size **pages** (like OS virtual memory):
- Pages are allocated on demand — no wasted memory
- Pages can be shared across requests (e.g., shared system prompts)
- Enables much larger batch sizes → higher throughput

**Result:** 2-24x higher throughput compared to HuggingFace Transformers.

---

## Installation

```bash
# Basic installation (CUDA 12.1+)
pip install vllm

# Specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# CPU-only (limited performance)
pip install vllm-cpu
```

### Docker

```bash
docker pull vllm/vllm-openai:latest

docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct
```

---

## Starting the Server

### Basic Server

```bash
# Start an OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Or using the CLI shorthand
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### With Options

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --dtype float16
```

### Test the Server

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Generate completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is gradient descent?"}],
    "max_tokens": 500
  }'
```

---

## Python Client Examples

### Using OpenAI SDK (Recommended)

```python
from openai import OpenAI

# Point to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require authentication by default
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful ML engineer."},
        {"role": "user", "content": "Explain the difference between batch and online learning."}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Using vLLM Offline API (Direct Python)

For batch processing without starting a server:

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
    stop=["</s>", "<|eot_id|>"]
)

# Batch generation (efficient!)
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is gradient descent?",
    "How does backpropagation work?"
]

# Generate all at once — much faster than one-by-one
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("---")
```

### Chat Template

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# Format messages with chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the transformer architecture?"}
]

# Apply chat template
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Multi-GPU with Tensor Parallelism

Tensor parallelism splits model layers across multiple GPUs, enabling models that don't fit on a single GPU.

```bash
# 2-GPU tensor parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2

# 4-GPU tensor parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4

# 8-GPU for very large models
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2  # Combine with pipeline parallelism
```

```python
from vllm import LLM

# 4-GPU setup via Python API
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9
)
```

---

## Quantization

Quantization reduces model memory footprint for faster inference.

### AWQ (Activation-aware Weight Quantization)

```bash
# Load a pre-quantized AWQ model
vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --quantization awq \
  --dtype half
```

```python
from vllm import LLM

llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    dtype="half"
)
```

### GPTQ

```bash
vllm serve TheBloke/Llama-2-13B-GPTQ \
  --quantization gptq
```

### FP8 (NVIDIA H100/A100)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --dtype float16
```

### Quantization Comparison

| Method | Memory Reduction | Quality | GPU Required |
|--------|-----------------|---------|-------------|
| FP16 (baseline) | — | Best | A100, H100 |
| GPTQ (4-bit) | ~75% | Very good | Most NVIDIA |
| AWQ (4-bit) | ~75% | Good | Most NVIDIA |
| FP8 | ~50% | Excellent | H100, A100 |

---

## Streaming Responses

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Streaming chat completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a Python quicksort implementation."}],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # New line at end
```

### Async Streaming

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

async def stream_response(prompt: str):
    stream = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=500
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(stream_response("Explain transformers in ML"))
```

---

## Supported Models

vLLM supports 100+ model architectures from Hugging Face. Key families:

| Family | Examples |
|--------|---------|
| Llama | Llama 3.1, 3.2, 3.3 (1B–405B) |
| Mistral | Mistral 7B, Mixtral 8x7B, Mistral NeMo |
| Gemma | Gemma 2 (9B, 27B) |
| Qwen | Qwen2.5 (0.5B–72B) |
| Phi | Phi-3, Phi-4 |
| DeepSeek | DeepSeek-R1, DeepSeek-V2 |
| Command R | Cohere Command R+ |
| Falcon | Falcon (7B, 40B) |
| Vision | LLaVA, InternVL2, Qwen2-VL |

---

## Interview Q&A

**Q1: What is PagedAttention and why does it improve throughput?** 🔴 Advanced

PagedAttention manages the KV (key-value) cache in fixed-size pages similar to OS virtual memory. Traditional inference pre-allocates contiguous memory blocks based on max sequence length, wasting memory for shorter sequences and limiting batch size. PagedAttention allocates pages on demand, eliminates internal fragmentation, and allows KV cache sharing across requests (e.g., shared system prompts). This enables larger batch sizes, reducing latency and increasing throughput by 2-24x.

---

**Q2: What is the difference between vLLM and Ollama?** 🟡 Intermediate

vLLM is designed for high-throughput production serving with continuous batching, multi-GPU tensor parallelism, and PagedAttention. It handles many concurrent users efficiently. Ollama prioritizes ease of use for local development — single command to run models, minimal setup. Ollama is for development/single-user; vLLM is for production serving with multiple concurrent users.

---

**Q3: What is continuous batching in vLLM?** 🔴 Advanced

Continuous batching (also called iteration-level scheduling) processes new requests without waiting for existing requests to finish. In traditional static batching, you wait for all requests in a batch to complete before processing the next batch. With continuous batching, as soon as a sequence finishes generating, a new request takes its slot in the batch. This maximizes GPU utilization and reduces average latency.

---

**Q4: When would you use tensor parallelism vs pipeline parallelism?** 🔴 Advanced

**Tensor parallelism** splits individual weight matrices across GPUs. Requires high-bandwidth NVLink interconnects between GPUs (ideally on the same node). Reduces latency since all GPUs work on each token together.

**Pipeline parallelism** splits model layers across GPUs (or nodes). GPUs run different layers sequentially. Works across nodes with slower interconnects. Increases throughput but adds pipeline bubble latency.

Use tensor parallelism within a single node (same server), and combine with pipeline parallelism for multi-node deployments.

---

**Q5: What quantization method would you recommend for production?** 🟡 Intermediate

For NVIDIA H100/A100: FP8 gives the best balance of quality and speed with native hardware support. For older NVIDIA (A10, V100): AWQ is preferred over GPTQ as it has lower quality degradation and similar speed. AWQ uses activation-aware quantization that finds better weight quantization points. GPTQ is older and slightly lower quality at the same bit width.

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [PagedAttention Paper — Efficient Memory Management for Large Language Model Serving (2023)](https://arxiv.org/abs/2309.06180)
- [vLLM Blog — UC Berkeley Sky Computing Lab](https://blog.vllm.ai/)
