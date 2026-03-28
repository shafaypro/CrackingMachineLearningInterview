# Ollama — Run LLMs Locally

A comprehensive guide to Ollama — the simplest way to run large language models on your local machine.

---

## Table of Contents

1. [What is Ollama](#what-is-ollama)
2. [Installation](#installation)
3. [CLI Commands](#cli-commands)
4. [Popular Models](#popular-models)
5. [Modelfile — Custom Models](#modelfile--custom-models)
6. [REST API](#rest-api)
7. [Python Integration](#python-integration)
8. [LangChain + Ollama](#langchain--ollama)
9. [LlamaIndex + Ollama](#llamaindex--ollama)
10. [Multi-Modal Vision with LLaVA](#multi-modal-vision-with-llava)
11. [Embedding Models](#embedding-models)
12. [Performance Tuning](#performance-tuning)
13. [Docker](#docker)
14. [Interview Q&A](#interview-qa)
15. [References](#references)

---

## What is Ollama

Ollama is an open-source tool that allows you to run large language models (LLMs) locally on your machine. It handles model downloading, quantization, hardware detection (CPU/GPU), and provides a simple CLI and REST API — all with a single command.

**Why Ollama?**
- Zero-cost inference after download
- Privacy — data never leaves your machine
- Works offline
- OpenAI-compatible REST API
- Supports macOS (Apple Silicon), Linux, and Windows

---

## Installation

### macOS

```bash
# Option 1: Download the installer from ollama.com
# Option 2: Homebrew
brew install ollama

# Start the Ollama server
ollama serve
```

### Linux

```bash
# Official install script
curl -fsSL https://ollama.com/install.sh | sh

# Ollama runs as a systemd service automatically
# Check status
systemctl status ollama
```

### Windows

Download the installer from [ollama.com/download](https://ollama.com/download) and run it. Ollama will start as a background service.

### Verify Installation

```bash
ollama --version
# ollama version 0.3.x
```

---

## CLI Commands

### Core Commands

```bash
# Run a model interactively (pulls if not present)
ollama run llama3.2

# Run with a specific prompt (non-interactive)
ollama run llama3.2 "Explain quantum computing in simple terms"

# Pull a model without running it
ollama pull mistral

# List downloaded models
ollama list

# Show model information and metadata
ollama show llama3.2

# Show running models and their memory usage
ollama ps

# Remove a model
ollama rm mistral

# Copy a model
ollama cp llama3.2 my-custom-llama

# Push a model to Ollama Hub (requires account)
ollama push username/mymodel

# Start the API server (default: http://localhost:11434)
ollama serve
```

### Running Models

```bash
# Basic chat
ollama run llama3.2

# With system prompt
ollama run llama3.2 --system "You are a senior Python developer"

# One-shot completion
echo "Write a Python fibonacci function" | ollama run codellama

# Pipe input from file
cat code.py | ollama run codellama "Review this code for bugs"

# Specify model variant (parameter count)
ollama run llama3.2:3b        # 3 billion parameter model
ollama run llama3.2:latest    # Default (latest)
ollama run mistral:7b-instruct-q4_0  # Specific quantization
```

---

## Popular Models

| Model | Parameters | Size (4-bit) | Strengths | Command |
|-------|-----------|--------------|-----------|---------|
| **llama3.2** | 3B / 1B | ~2GB / ~0.8GB | Fast, great for low-end hardware | `ollama run llama3.2` |
| **llama3.1** | 8B / 70B | ~4.7GB / ~40GB | Strong general reasoning | `ollama run llama3.1` |
| **mistral** | 7B | ~4GB | Fast, great instruction following | `ollama run mistral` |
| **mistral-nemo** | 12B | ~7GB | Multilingual, long context | `ollama run mistral-nemo` |
| **gemma2** | 9B / 27B | ~5.5GB / ~15GB | Google's model, strong benchmarks | `ollama run gemma2` |
| **phi3** | 3.8B / 14B | ~2.2GB / ~8GB | Microsoft, small but capable | `ollama run phi3` |
| **phi4** | 14B | ~8.5GB | Microsoft's latest, top benchmarks | `ollama run phi4` |
| **qwen2.5** | 7B / 72B | ~4.4GB / ~44GB | Alibaba, strong in code and math | `ollama run qwen2.5` |
| **codellama** | 7B / 13B | ~3.8GB / ~7.4GB | Meta's code-specialized model | `ollama run codellama` |
| **deepseek-r1** | 8B / 70B | ~5GB / ~42GB | Reasoning model, chain-of-thought | `ollama run deepseek-r1` |
| **deepseek-coder** | 6.7B / 33B | ~3.8GB / ~19GB | Strong code generation | `ollama run deepseek-coder` |
| **llava** | 7B / 13B | ~4.5GB | Vision+language (multimodal) | `ollama run llava` |
| **moondream** | 1.8B | ~1.2GB | Lightweight vision model | `ollama run moondream` |
| **nomic-embed-text** | — | ~270MB | Text embeddings | `ollama run nomic-embed-text` |
| **mxbai-embed-large** | — | ~670MB | High-quality embeddings | `ollama run mxbai-embed-large` |

---

## Modelfile — Custom Models

A Modelfile is a configuration file that defines how to build a custom Ollama model — similar to a Dockerfile.

### Basic Modelfile

```dockerfile
# Modelfile
FROM llama3.2

# Set system prompt
SYSTEM """
You are an expert Python developer and code reviewer.
You provide concise, accurate code reviews and follow PEP 8 standards.
Always explain your reasoning.
"""

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
```

### Build and Run Custom Model

```bash
ollama create python-reviewer -f Modelfile
ollama run python-reviewer
```

### Advanced Modelfile Options

```dockerfile
FROM llama3.1:8b

# Quantization level
# FROM llama3.1:8b-instruct-q4_K_M

# System prompt
SYSTEM """You are a helpful AI assistant specializing in data science."""

# Temperature (creativity): 0.0 = deterministic, 1.0 = very creative
PARAMETER temperature 0.8

# Top-p nucleus sampling
PARAMETER top_p 0.95

# Top-k: number of tokens to consider at each step
PARAMETER top_k 50

# Context window size (tokens)
PARAMETER num_ctx 8192

# Max tokens to generate
PARAMETER num_predict 2048

# Stop sequences
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

# Number of GPU layers to load (-1 = all)
PARAMETER num_gpu 35

# Number of threads for CPU
PARAMETER num_thread 8
```

### Modelfile with GGUF

```dockerfile
# Use your own GGUF model file
FROM /path/to/model.gguf

SYSTEM "You are a specialized assistant."
PARAMETER temperature 0.7
```

---

## REST API

Ollama exposes an HTTP API at `http://localhost:11434`.

### Generate (Non-Streaming)

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.2",
    "prompt": "Why is the sky blue?",
    "stream": false
  }'
```

### Generate (Streaming)

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.2",
    "prompt": "Explain backpropagation step by step",
    "stream": true
  }'
```

### Chat Completion (OpenAI-Compatible)

```bash
# Ollama supports the OpenAI chat completions format
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is gradient descent?"}
    ],
    "stream": false
  }'
```

### OpenAI-Compatible API Endpoint

```bash
# Ollama also exposes an OpenAI-compatible endpoint
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Embeddings

```bash
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "nomic-embed-text",
    "prompt": "Machine learning is fascinating"
  }'
```

### List Models

```bash
curl http://localhost:11434/api/tags
```

---

## Python Integration

### Using the `ollama` Python Package

```bash
pip install ollama
```

```python
import ollama

# Simple generation
response = ollama.generate(
    model='llama3.2',
    prompt='What is the capital of France?'
)
print(response['response'])

# Chat interface
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain transformers in ML.'}
    ]
)
print(response['message']['content'])

# Streaming response
for chunk in ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Write a Python web scraper'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### Using OpenAI Python Client

Point the OpenAI client at Ollama's API — perfect for code that already uses the OpenAI SDK.

```python
from openai import OpenAI

# Point to local Ollama server
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required but not used
)

# Use any Ollama model as if it were OpenAI
response = client.chat.completions.create(
    model='llama3.2',
    messages=[
        {'role': 'system', 'content': 'You are a Python expert.'},
        {'role': 'user', 'content': 'Write a binary search function.'}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response.choices[0].message.content)
```

### Async with httpx

```python
import httpx
import asyncio
import json

async def generate_async(prompt: str, model: str = "llama3.2"):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

# Batch processing
async def batch_generate(prompts: list[str]):
    tasks = [generate_async(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(batch_generate([
    "What is machine learning?",
    "What is deep learning?",
    "What is reinforcement learning?"
]))
```

### Embeddings with Python

```python
import ollama
import numpy as np

def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare semantic similarity
emb1 = get_embedding("machine learning is about algorithms")
emb2 = get_embedding("artificial intelligence uses computational methods")
emb3 = get_embedding("I love eating pizza")

print(f"ML vs AI similarity: {cosine_similarity(emb1, emb2):.4f}")
print(f"ML vs pizza similarity: {cosine_similarity(emb1, emb3):.4f}")
```

---

## LangChain + Ollama

```bash
pip install langchain langchain-ollama langchain-community
```

### Basic LangChain with Ollama

```python
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Basic LLM
llm = OllamaLLM(model="llama3.2")
response = llm.invoke("What is the capital of Japan?")
print(response)

# Chat model
chat = ChatOllama(model="llama3.2", temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful ML tutor."),
    HumanMessage(content="Explain gradient descent.")
]
response = chat.invoke(messages)
print(response.content)
```

### LangChain RAG Pipeline with Ollama

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Load documents
loader = TextLoader("my_document.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Create embeddings with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Store in vector database
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create RAG chain
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
answer = rag_chain.invoke("What is the main topic of the document?")
print(answer)
```

### LangChain Streaming

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", streaming=True)

for chunk in llm.stream("Write a short poem about machine learning"):
    print(chunk.content, end="", flush=True)
```

---

## LlamaIndex + Ollama

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
```

```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Configure Ollama as the LLM and embedding model
Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load documents from a directory
documents = SimpleDirectoryReader("./docs").load_data()

# Build index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What is the main topic?")
print(response)
```

---

## Multi-Modal Vision with LLaVA

LLaVA (Large Language and Vision Assistant) can understand images and answer questions about them.

```bash
# Pull the vision model
ollama pull llava
```

### CLI Vision

```bash
# Pass an image to the model
ollama run llava "What is in this image?" --image /path/to/image.jpg
```

### Python Vision

```python
import ollama
import base64
from pathlib import Path

def encode_image(image_path: str) -> str:
    """Convert image to base64 for API submission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Method 1: Using the ollama package
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe what you see in this image in detail.',
        'images': ['/path/to/image.jpg']  # File path
    }]
)
print(response['message']['content'])

# Method 2: Using base64 encoded image via API
import requests

image_data = encode_image("chart.png")

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llava",
        "prompt": "What does this chart show? What are the key trends?",
        "images": [image_data],
        "stream": False
    }
)
print(response.json()["response"])
```

---

## Embedding Models

Ollama supports dedicated embedding models for semantic search and RAG.

| Model | Dimensions | Size | Strengths |
|-------|-----------|------|-----------|
| `nomic-embed-text` | 768 | ~270MB | General purpose, fast |
| `mxbai-embed-large` | 1024 | ~670MB | High quality, MTEB top |
| `all-minilm` | 384 | ~45MB | Very lightweight |
| `bge-m3` | 1024 | ~1.2GB | Multilingual, best quality |

```python
import ollama
import numpy as np

# Generate embeddings
def embed_texts(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response['embedding'])
    return np.array(embeddings)

# Simple semantic search
def semantic_search(query: str, documents: list[str], top_k: int = 3):
    query_emb = embed_texts([query])[0]
    doc_embs = embed_texts(documents)

    # Cosine similarity
    scores = doc_embs @ query_emb / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], scores[i]) for i in top_indices]

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language used widely in data science.",
    "Neural networks are inspired by the human brain.",
    "I enjoy hiking in the mountains on weekends."
]

results = semantic_search("What is AI?", documents)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc}")
```

---

## Performance Tuning

### GPU Layers

```bash
# Load all layers on GPU (fastest)
OLLAMA_GPU_LAYERS=-1 ollama run llama3.2

# Specify number of layers (tune for partial GPU offload)
OLLAMA_GPU_LAYERS=20 ollama run llama3.2
```

In Modelfile:

```dockerfile
PARAMETER num_gpu -1   # -1 = all layers on GPU
```

### Context Size

```dockerfile
# Larger context = more memory, allows longer conversations
PARAMETER num_ctx 8192   # Default is 2048
```

### Threads (CPU Inference)

```dockerfile
# Match to physical CPU cores (not hyperthreaded)
PARAMETER num_thread 8
```

### Quantization Levels

```bash
# Q4_K_M: Good balance of speed and quality (recommended)
ollama pull llama3.2:latest      # Uses Q4_K_M by default

# Q8_0: Better quality, more memory
ollama pull llama3.1:8b-instruct-q8_0

# F16: Full precision (needs lots of RAM/VRAM)
ollama pull llama3.1:8b-instruct-fp16
```

### Environment Variables

```bash
# Set number of parallel requests Ollama can handle
OLLAMA_NUM_PARALLEL=4 ollama serve

# Set maximum loaded models in memory
OLLAMA_MAX_LOADED_MODELS=2 ollama serve

# Custom host
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Keep models loaded in memory (prevent cold starts)
OLLAMA_KEEP_ALIVE=10m ollama serve

# Flash attention (when supported)
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

---

## Docker

### Run Ollama in Docker (CPU)

```bash
docker run -d \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama

# Pull and run a model
docker exec -it ollama ollama run llama3.2
```

### Run Ollama in Docker (NVIDIA GPU)

```bash
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  # Optional: Open WebUI for browser-based chat
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
```

### Pre-pull Models in Docker

```bash
# Pull models into the container at startup
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text
```

---

## Interview Q&A

**Q1: What is Ollama and what problem does it solve?** 🟢 Beginner

Ollama is an open-source tool that simplifies running large language models locally. It handles model downloading, hardware detection (CPU/GPU), quantization management, and exposes a simple CLI and REST API. It solves the problem of complex LLM setup by abstracting away CUDA configuration, model format handling, and server management into a single `ollama run <model>` command.

---

**Q2: How does Ollama differ from using the OpenAI API?** 🟢 Beginner

Ollama runs models locally on your own hardware (no cloud costs, no data sent externally, works offline), while OpenAI API sends requests to OpenAI's servers. Trade-offs: Ollama is free after initial setup but limited to hardware you own; OpenAI API requires payment but provides access to GPT-4 class models without hardware requirements. Ollama is best for development, privacy-sensitive data, and offline use.

---

**Q3: What is a Modelfile in Ollama?** 🟡 Intermediate

A Modelfile is analogous to a Dockerfile — it defines how to build a custom Ollama model. It specifies the base model (`FROM`), system prompt (`SYSTEM`), inference parameters (`PARAMETER temperature`, `PARAMETER num_ctx`), and stop sequences. You build it with `ollama create model-name -f Modelfile` and run it like any other model.

---

**Q4: How do you use Ollama with existing OpenAI-based code?** 🟡 Intermediate

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`. Simply change the `base_url` in the OpenAI client and set `api_key='ollama'` (any non-empty string). All OpenAI chat completion calls will then route to your local Ollama model with no other code changes needed.

---

**Q5: What is the difference between Ollama and vLLM for serving LLMs?** 🟡 Intermediate

Ollama prioritizes simplicity and local development. It works great for single-user workloads, quick experimentation, and desktop use. vLLM uses PagedAttention for KV cache management, enabling high-throughput multi-user inference with continuous batching. vLLM is 3-10x more efficient for concurrent requests. For production APIs serving many users: vLLM. For local development and single-user tools: Ollama.

---

**Q6: What is quantization in the context of Ollama models?** 🟡 Intermediate

Quantization reduces the precision of model weights (e.g., from float32 to 4-bit integers) to decrease memory usage and increase inference speed. Ollama uses GGUF models with quantization levels like Q4_K_M (4-bit with K-means quantization, mixed precision), Q8_0 (8-bit), or F16 (half precision). Q4_K_M is the sweet spot — ~75% memory reduction vs F32 with minimal quality loss.

---

**Q7: How would you set up a private RAG system using Ollama?** 🔴 Advanced

```
1. Install Ollama and pull models:
   - LLM: ollama pull llama3.2
   - Embeddings: ollama pull nomic-embed-text

2. Load and chunk documents (LangChain or LlamaIndex)

3. Generate embeddings using Ollama's embedding API

4. Store vectors in local Chroma or Qdrant

5. At query time:
   - Embed the query with nomic-embed-text
   - Retrieve top-k relevant chunks from vector DB
   - Build a prompt with context + question
   - Send to llama3.2 via Ollama API

6. All computation runs locally — 100% private
```

---

**Q8: What are the hardware requirements for running LLMs with Ollama?** 🟡 Intermediate

| Model Size | VRAM Needed (4-bit) | CPU RAM (CPU-only) |
|-----------|--------------------|--------------------|
| 1-3B params | ~1-2 GB | ~2-4 GB RAM |
| 7-8B params | ~4-5 GB | ~8-10 GB RAM |
| 13B params | ~7-8 GB | ~16 GB RAM |
| 30-34B params | ~18-20 GB | ~32 GB RAM |
| 70B params | ~40+ GB | ~64+ GB RAM |

Apple Silicon (M1/M2/M3/M4) uses unified memory — RAM and VRAM are shared, making Macs excellent for local LLM inference.

---

**Q9: How do you run a vision model with Ollama?** 🟡 Intermediate

Pull a multimodal model like LLaVA (`ollama pull llava`) and pass images via the CLI (`ollama run llava --image photo.jpg`) or via the API/Python SDK. In the Python `ollama` library, include an `images` field in the message with file paths or base64-encoded image data. The model processes both the image and text prompt together.

---

**Q10: What is `OLLAMA_KEEP_ALIVE` and when would you change it?** 🔴 Advanced

`OLLAMA_KEEP_ALIVE` controls how long Ollama keeps a model loaded in memory after the last request. Default is 5 minutes. If it expires, the model is unloaded and must be reloaded on the next request (cold start latency of several seconds).

Set it higher (e.g., `OLLAMA_KEEP_ALIVE=1h`) for interactive development to avoid cold starts. Set to `0` in resource-constrained environments to free memory immediately after each request. Set to `-1` to keep models loaded indefinitely.

---

## References

- [Ollama Official Website](https://ollama.com)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Ollama REST API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Python Library](https://github.com/ollama/ollama-python)
- [Open WebUI — Browser UI for Ollama](https://github.com/open-webui/open-webui)
- [LangChain + Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)
- [LlamaIndex + Ollama Integration](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)
