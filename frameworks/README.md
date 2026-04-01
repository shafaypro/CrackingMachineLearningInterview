# Frameworks & Tools Overview

A comprehensive reference for all major ML/AI frameworks, local LLM runners, fine-tuning tools, and infrastructure platforms.

---

## Table of Contents

1. [Quick Comparison Table](#quick-comparison-table)
2. [Local LLM Tools](#local-llm-tools)
3. [Fine-Tuning Tools](#fine-tuning-tools)
4. [Training Frameworks](#training-frameworks)
5. [Experiment Tracking](#experiment-tracking)
6. [Orchestration Frameworks](#orchestration-frameworks)
7. [Vector Databases](#vector-databases)
8. [Interview Q&A](#interview-qa)
9. [References](#references)

---

## Quick Comparison Table

| Framework | Category | Primary Use Case | When to Use |
|-----------|----------|-----------------|-------------|
| [Ollama](./intro_ollama.md) | Local LLM | Run LLMs locally | Dev/testing, privacy-first, offline |
| [vLLM](./intro_vllm.md) | LLM Serving | High-throughput inference | Production serving at scale |
| [Hugging Face](./intro_huggingface.md) | Model Hub + Library | Load, fine-tune, deploy any model | Research, fine-tuning, prototyping |
| [PyTorch](./intro_pytorch.md) | Training Framework | Build and train neural networks | Custom model development |
| TensorFlow/Keras | Training Framework | Build and train neural networks | Production deployment, mobile |
| JAX | Training Framework | High-performance ML research | Google infra, research labs |
| [LangChain](./intro_langchain.md) | LLM Orchestration | Chain LLM calls, agents, RAG | LLM application development |
| [n8n](../ai_genai/intro_n8n.md) | Workflow Automation | Connect apps, webhooks, approvals, and AI steps | Operational AI automation and business workflows |
| LlamaIndex | RAG Framework | Document indexing and retrieval | Document Q&A, knowledge bases |
| Haystack | NLP Pipeline | Search and NLP pipelines | Enterprise search, NLP products |
| [Unsloth](./intro_unsloth.md) | Fine-tuning | Fast LoRA/QLoRA fine-tuning | Efficient custom model training |
| Axolotl | Fine-tuning | Configurable fine-tuning | Multi-GPU training, YAML configs |
| TRL | Fine-tuning | RLHF, DPO, SFT training | Alignment and instruction tuning |

---

## Local LLM Tools

Run large language models locally on your own hardware without sending data to external APIs.

| Tool | Description | Strengths | Weaknesses | Best For |
|------|-------------|-----------|------------|----------|
| **[Ollama](./intro_ollama.md)** | Simplified local LLM runner | Easiest to use, great CLI/API | Single-GPU, no batch inference | Developers, local chat, prototyping |
| **vLLM** | PagedAttention-based server | Highest throughput, OpenAI API compatible | More setup required | Production serving, multi-user |
| **llama.cpp** | C++ inference engine | Runs on CPU, very lightweight | Manual model management | Edge devices, CPU-only environments |
| **LM Studio** | Desktop GUI for local LLMs | No-code, drag-and-drop models | GUI only, less automatable | Non-technical users |
| **Jan** | Open source alternative to LM Studio | Privacy-focused, offline first | Smaller community | Privacy-conscious users |
| **GPT4All** | Local chat application | Easy installation, many models | Limited API | Beginners, desktop chat |

---

## Fine-Tuning Tools

| Tool | Description | Strengths | Weaknesses | Best For |
|------|-------------|-----------|------------|----------|
| **[Unsloth](./intro_unsloth.md)** | Fast LoRA/QLoRA fine-tuning | 2-5x faster than HF, memory efficient | Fewer model architectures supported | Quick LoRA fine-tuning |
| **Axolotl** | Configurable fine-tuning framework | YAML-based config, multi-GPU, many methods | Steeper config learning curve | Production fine-tuning pipelines |
| **TRL (Hugging Face)** | RLHF, DPO, SFT | Official HF library, well-documented | Slower than Unsloth | Alignment, instruction tuning |
| **LLaMA-Factory** | Web UI + CLI for fine-tuning | Beginner-friendly, many models | Less flexible than code-based | Quick experiments |

---

## Training Frameworks

| Framework | Language | Strengths | Weaknesses | Best For |
|-----------|----------|-----------|------------|----------|
| **[PyTorch](./intro_pytorch.md)** | Python | Dynamic graphs, research-friendly, huge community | Slower deployment tools historically | Research, custom models |
| **TensorFlow / Keras** | Python | Production tooling (TFServing, TFLite), mobile | More verbose, eager/graph confusion | Production, mobile deployment |
| **JAX** | Python | XLA compilation, auto-differentiation, functional | Steep learning curve, less tooling | Google research, TPU training |
| **MXNet** | Python / Scala | AWS native, distributed | Declining community | AWS-specific workloads |

### PyTorch vs TensorFlow Quick Comparison

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| Graph type | Dynamic (eager by default) | Static (graph) + eager mode |
| Research popularity | Dominant in research | Declining in research |
| Production tooling | TorchServe, ONNX | TFServing, TFLite, TF.js |
| Debugging | Easy — standard Python debugging | Harder with graph mode |
| Community | Very large | Large |

---

## Experiment Tracking

| Tool | Hosting | Strengths | Weaknesses | Best For |
|------|---------|-----------|------------|----------|
| **MLflow** | Self-hosted / Databricks | Open source, model registry, simple API | Basic UI | Teams wanting self-hosted |
| **Weights & Biases (W&B)** | SaaS (free tier) | Rich UI, collaboration, sweeps, reports | Paid at scale | Research teams |
| **Neptune.ai** | SaaS | Enterprise features, metadata store | Less known than W&B | Enterprise ML teams |
| **Comet ML** | SaaS | Experiment comparison, dashboards | Paid at scale | Enterprise |
| **DVC** | Self-hosted | Git-native, data versioning | Pipeline complexity | Data versioning focus |

---

## Orchestration Frameworks

For building LLM applications, RAG pipelines, and AI agents.

| Framework | Description | Strengths | Weaknesses | Best For |
|-----------|-------------|-----------|------------|----------|
| **[LangChain](./intro_langchain.md)** | LLM application framework | Huge community, many integrations | Abstraction overhead, breaking changes | Rapid LLM app prototyping |
| **[n8n](../ai_genai/intro_n8n.md)** | Visual workflow automation | Fast integrations, approvals, business workflow visibility | Less suited for deep agent state management | AI-enabled operations and internal tooling |
| **LlamaIndex** | Data indexing + RAG | Best-in-class RAG, document handling | More limited for general agents | Document Q&A, knowledge bases |
| **Haystack** | NLP pipeline framework | Enterprise-ready, production-focused | Smaller community | Enterprise search pipelines |
| **DSPy** | LLM programming | Automatic prompt optimization | New, smaller community | Research, optimized pipelines |
| **Instructor** | Structured LLM output | Simple, reliable structured output | Single-purpose | Structured data extraction |

---

## Vector Databases

| Database | Type | Strengths | Weaknesses | Best For |
|----------|------|-----------|------------|----------|
| **Chroma** | Open source, embedded | Easy setup, local dev | Not for very large scale | Prototyping, local RAG |
| **Pinecone** | Managed SaaS | Scalable, managed, fast | Cost, vendor lock-in | Production RAG at scale |
| **Qdrant** | Open source / managed | High performance, rich filtering | Newer community | Production self-hosted |
| **Weaviate** | Open source / managed | Multimodal, GraphQL API | Complex setup | Enterprise, multimodal search |
| **pgvector** | PostgreSQL extension | No new infra, SQL interface | Slower for large scale | Teams already using PostgreSQL |
| **Milvus** | Open source | Massive scale, distributed | Complex infrastructure | Billion-scale vector search |
| **FAISS** | Library (Meta) | Extremely fast, CPU/GPU | Not a server, no persistence | Research, batch similarity search |

---

## Interview Q&A

**Q1: When would you choose Ollama over vLLM?** 🟢 Beginner

**Ollama** is best for local development, experimentation, and privacy-sensitive use cases where you need simplicity. It's a single command to run models, has an OpenAI-compatible API, and works on a single GPU or CPU.

**vLLM** is for production serving where you need high throughput, low latency, and multi-user concurrency. vLLM's PagedAttention enables efficient KV cache management and batching of requests.

Rule of thumb: Ollama for development, vLLM for production.

---

**Q2: What is PagedAttention in vLLM?** 🔴 Advanced

PagedAttention is vLLM's key innovation. The KV (key-value) cache — which stores attention states for each token — is typically pre-allocated as a contiguous block. This leads to memory fragmentation and waste when requests have different lengths.

PagedAttention manages the KV cache in fixed-size pages (like OS virtual memory paging). Pages are allocated dynamically and can be shared across requests (e.g., for system prompts). This dramatically reduces memory waste, enables larger batch sizes, and increases throughput by 2-4x compared to standard implementations.

---

**Q3: What is the difference between LangChain and LlamaIndex?** 🟡 Intermediate

**LangChain** is a general-purpose LLM application framework with support for chains, agents, tools, memory, and many LLM/API integrations. Good for building complex LLM workflows and agents.

**LlamaIndex** is specialized for data indexing, retrieval, and RAG (Retrieval-Augmented Generation). It has superior document loaders, node parsers, index types, and retrieval strategies. Better at handling large document collections for Q&A.

In practice, many teams use LlamaIndex for the RAG layer and LangChain for agent orchestration.

---

**Q4: When would you choose n8n over LangChain or LangGraph?** 🟡 Intermediate

Choose **n8n** when the problem is mostly workflow automation: webhooks, SaaS integrations, approvals, routing, and AI calls embedded inside business processes.

Choose **LangChain** for general LLM application development and **LangGraph** when you need deeper control over state, loops, and agent execution.

In many production systems, n8n handles the outer operational workflow while LangChain or LangGraph powers the reasoning-heavy backend.

---

**Q5: Compare PyTorch and TensorFlow for production ML.** 🟡 Intermediate

**PyTorch** is dominant in research and has strong production tools (TorchServe, ONNX export, TorchScript). `torch.compile` in PyTorch 2.0 makes it competitive on performance.

**TensorFlow** has mature production tooling: TF Serving (high-performance model serving), TFLite (mobile/edge), TF.js (browser), and tight integration with Google Cloud (Vertex AI).

For new projects in 2025: choose PyTorch — it has the largest research community, best framework support (Hugging Face, Lightning), and strong enough production tooling.

---

**Q6: What is the difference between Chroma and Pinecone for vector search?** 🟡 Intermediate

**Chroma** is an open-source, embedded vector database ideal for local development and prototyping. Zero infrastructure setup — runs in-process or as a local server.

**Pinecone** is a fully managed vector database SaaS. Handles indexing, scaling, and serving automatically. Better for production use cases with millions of vectors and high query throughput. Has costs and vendor dependency.

For development/prototyping: Chroma. For production at scale: Pinecone, Qdrant (self-hosted), or pgvector (if already using PostgreSQL).

---

**Q7: What is Unsloth and how is it different from standard LoRA training?** 🟡 Intermediate

Unsloth is a Python library that provides optimized LoRA/QLoRA fine-tuning for LLMs. It achieves 2-5x speed improvements and 50-60% memory reduction compared to standard Hugging Face PEFT+Transformers training.

Key optimizations:
- Rewritten CUDA kernels for attention and matrix operations
- Optimized gradient checkpointing
- Memory-efficient 4-bit quantization integration
- Unified API for Llama, Mistral, Gemma, and other popular models

Standard LoRA uses the Hugging Face PEFT library which works well but is not optimized at the kernel level.

---

**Q8: What is the difference between pgvector and dedicated vector databases?** 🔴 Advanced

**pgvector** adds vector similarity search to PostgreSQL using an extension. Strengths: no new infrastructure, SQL interface, ACID transactions, join with relational data, familiar tooling. Weaknesses: slower for large-scale (100M+ vectors), limited indexing options (IVFFlat, HNSW).

**Dedicated vector databases** (Pinecone, Qdrant, Weaviate) are purpose-built for vector search: advanced indexing (HNSW, IVF, etc.), horizontal scaling, filtering, and high-throughput queries. They outperform pgvector at large scale.

Choose pgvector if you already use PostgreSQL and have < 10M vectors. Choose a dedicated database for larger scale or advanced filtering needs.

---

## References

- [Ollama Documentation](https://ollama.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [n8n Documentation](https://docs.n8n.io/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
