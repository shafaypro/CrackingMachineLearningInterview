# The Agentic AI Engineer Roadmap (2026 Edition)

A complete, layered curriculum for becoming a production-grade **Agentic AI
Engineer** — from Python foundations to frontier topics like agent platforms and
self-improving agents. Each topic gives you the concept, why it matters, the key
things to know, and a **→ Deep dive** link to the in-repo guide that covers it in
full.

> **How to use this:** Work top-to-bottom. The layers are ordered by dependency —
> you genuinely need the Foundation layer before the Core Agentic layer, and so on.
> The final section, [What Actually Gets You Hired](#what-actually-gets-you-hired),
> is what senior interviews really probe.

---

## Table of Contents

**Foundation Layer**
1. [Python for AI Engineering](#1-python-for-ai-engineering)
2. [LLM Fundamentals](#2-llm-fundamentals)
3. [Prompt Engineering](#3-prompt-engineering)
4. [LLM APIs](#4-llm-apis)

**Core Agentic Layer**
5. [Tool Use / Function Calling](#5-tool-use--function-calling)
6. [Agent Loops](#6-agent-loops)
7. [Memory Systems](#7-memory-systems)
8. [RAG](#8-rag-retrieval-augmented-generation)
9. [Frameworks](#9-frameworks-langchain--llamaindex--crewai--autogen)

**Intermediate Agentic Layer**
10. [Multi-Agent Systems](#10-multi-agent-systems)
11. [MCP](#11-mcp-model-context-protocol)
12. [Planning & Task Decomposition](#12-planning--task-decomposition)
13. [Structured Outputs & Validation](#13-structured-outputs--data-validation)
14. [Long-Context Management](#14-long-context-management)
15. [Agent State Management](#15-agent-state-management)

**Advanced Agentic Layer**
16. [LangGraph / Workflows as Graphs](#16-langgraph--agent-workflows-as-graphs)
17. [Human-in-the-Loop](#17-human-in-the-loop-hitl)
18. [Tool Design & Ecosystems](#18-tool-design--tool-ecosystems)
19. [Evaluation & Testing](#19-evaluation--testing-for-agents)
20. [Observability & Tracing](#20-observability--tracing)
21. [Code Execution Agents](#21-code-execution-agents)
22. [Browser & Computer Use](#22-browser--computer-use-agents)

**Production & Senior Layer**
23. [Reliability & Failure Modes](#23-agent-reliability--failure-modes)
24. [Latency Optimization](#24-latency-optimization)
25. [Cost Management at Scale](#25-cost-management-at-scale)
26. [Security for Agentic Systems](#26-security-for-agentic-systems)
27. [Deployment Patterns](#27-deployment-patterns)
28. [Agentic Architecture Patterns](#28-agentic-architecture-patterns)

**Expert / Frontier Layer**
29. [Fine-tuning for Agentic Behavior](#29-fine-tuning-for-agentic-behavior)
30. [Reasoning Models](#30-reasoning-models)
31. [Multi-Modal Agents](#31-multi-modal-agents)
32. [Agent Communication Standards](#32-agent-communication-standards)
33. [Building Agent Platforms](#33-building-agent-platforms)
34. [Self-Improving Agents / Meta-Agents](#34-self-improving-agents--meta-agents)

- [What Actually Gets You Hired](#what-actually-gets-you-hired)

---

# Foundation Layer

## 1. Python for AI Engineering

The non-negotiable base. You need this before everything else.

- **Core Python** — data structures, comprehensions, generators, context managers, decorators.
- **`async`/`await`** — concurrent API calls (agents make *many*); `asyncio.gather` to parallelize tool calls and model requests.
- **Type hints** — `list[str]`, `dict[str, Any]`, `Optional`, `Literal`, generics; they make agent code maintainable and power schema generation.
- **Pydantic models** — typed, validated data; the backbone of structured outputs and tool schemas.
- **Environment management** — `venv`/`uv`/`poetry`, `.env` files, never hardcoding keys.
- **Working with APIs** — `requests`/`httpx`, retries, timeouts, pagination, streaming.
- **Error handling & logging** — typed exceptions, retry/backoff, structured logging with correlation IDs (essential for debugging non-deterministic agents).

→ **Deep dive:** [Python for AI Engineering](../frameworks/intro_python_for_ai.md) ·
[Pydantic](../frameworks/intro_pydantic.md) · [FastAPI](../frameworks/intro_fastapi.md)

## 2. LLM Fundamentals

How the thing you're orchestrating actually behaves.

- **Transformers (conceptual)** — self-attention lets every token attend to every other; decoder-only models predict the next token autoregressively.
- **Tokens & context windows** — text → tokens (~4 chars each); the window caps input+output. Budget it.
- **Sampling** — `temperature` (randomness) and `top_p` (nucleus) trade determinism for creativity; low for extraction/code, higher for ideation.
- **System vs user prompts** — system sets persona/rules/constraints (higher authority); user carries the task.
- **Few-shot prompting** — examples in the prompt steer format and behavior without fine-tuning.
- **Prompt injection (basics)** — untrusted text can hijack instructions; never trust retrieved/tool content as commands.

→ **Deep dive:** [LLM Fundamentals](./intro_llm_fundamentals.md) · [Transformers](../deep_learning/intro_transformers.md)

## 3. Prompt Engineering

The cheapest, fastest lever on agent quality.

- **Chain-of-Thought (CoT)** — "think step by step" for multi-step reasoning.
- **Zero-shot vs few-shot** — no examples vs a handful of worked examples.
- **Structured-output prompting** — ask for JSON/XML; pair with schema enforcement.
- **Role prompting** — assign an expert persona to shape tone and rigor.
- **ReAct (Reason + Act)** — interleave reasoning traces with tool actions — the canonical agent prompting pattern.
- **XML/JSON coercion** — wrap fields in tags/keys the model reliably emits; parse defensively.
- **Hallucination control** — ground with retrieval, demand citations, lower temperature, allow "I don't know."

→ **Deep dive:** [Prompt Engineering](./intro_prompt_engineering.md)

## 4. LLM APIs

OpenAI, Anthropic, and Google — calling them well.

- **Calling them** — all expose a chat/messages endpoint; you send messages + params and get content blocks back.
- **Streaming** — Server-Sent Events deliver tokens incrementally; essential for UX and to avoid request timeouts on long outputs.
- **Rate limits** — RPM/TPM caps; handle `429` with exponential backoff (SDKs retry automatically).
- **Token counting** — use the **provider's own** counter (not `tiktoken` for Claude — it undercounts).
- **Error handling** — typed exceptions; retry 429/5xx, don't retry 4xx.
- **Model tiers (cost vs capability)** — frontier (hardest reasoning), balanced workhorse, fast/cheap. Route by task.

| Provider | Frontier / reasoning | Balanced | Fast / cheap |
|----------|----------------------|----------|--------------|
| Anthropic | `claude-fable-5`, `claude-opus-4-8` | `claude-sonnet-4-6` | `claude-haiku-4-5` |
| OpenAI | GPT frontier tier | GPT mid tier | GPT mini/nano tier |
| Google | Gemini Pro/Ultra tier | Gemini Flash | Gemini Flash-Lite |

> Always confirm current model IDs and pricing from each provider's docs — they change frequently.

→ **Deep dive:** [Anthropic & Claude API](./intro_anthropic.md) · [Multi-Model Orchestration](./intro_multi_model_orchestration.md)

---

# Core Agentic Layer

## 5. Tool Use / Function Calling

The heartbeat of agentic systems.

- **How it works** — you pass tool definitions (name, description, JSON schema); the model returns a *tool-use request* with arguments; you execute it and feed the **tool result** back into context; repeat.
- **JSON schemas** — `type: object`, typed properties, `required`, `enum`; use strict mode (`additionalProperties: false`) for guaranteed-valid args.
- **Parsing responses** — read tool-use blocks, match each by `tool_use_id`.
- **Multi-tool calls in one turn** — execute independent calls in parallel; return **all** results in **one** message.
- **Result injection** — append the assistant turn (with tool-use blocks) *then* the tool results, so the model can continue.

→ **Deep dive:** [Agent Systems & Tool Use](./intro_agent_tool_use.md)

## 6. Agent Loops

The engine: **observe → think → act → observe**.

- **Minimal ReAct loop** — call model → if it requests a tool, run it, append result, loop; else stop.
- **Stopping conditions** — `end_turn`, goal reached, or a max-iteration / token budget.
- **Handling stalls** — detect repeated states/loops; add a "if stuck, ask for help or stop" instruction; cap retries.
- **Manual vs SDK-driven** — write the loop yourself for approval gates and custom logging; use an SDK tool runner for the common case.

→ **Deep dive:** [Agentic AI](./intro_agentic_ai.md) (§ Agent Patterns, § Production Agent Engineering)

## 7. Memory Systems

What the agent remembers, and where.

| Memory type | What it holds | Backed by |
|-------------|---------------|-----------|
| **In-context (working)** | Current window contents | The prompt itself |
| **Episodic** | Conversation/run history | Message log, DB |
| **Semantic** | Facts & knowledge | Vector store (RAG) |
| **Procedural** | How-to / skills | Tools, prompts, skill files |
| **External / long-term** | Anything beyond the window | Vector DB, SQL, files |

- **When to use each** — keep hot context in-window; offload facts to a vector store; persist run history for resumption; encode procedures as tools/skills.
- **Trade-offs** — more in-context memory = higher cost/latency; retrieve selectively.

→ **Deep dive:** [Agentic AI § Agent Memory](./intro_agentic_ai.md) · [Vector Databases](./intro_vector_databases.md)

## 8. RAG (Retrieval-Augmented Generation)

Ground answers in your data instead of model memory.

- **Chunking** — size + overlap; semantic/recursive splitting beats fixed windows; respect document structure.
- **Embedding models** — turn text into vectors; pick by domain, dimension, cost.
- **Vector databases** — Pinecone, Weaviate, Chroma, pgvector, FAISS.
- **Similarity search** — cosine/dot-product top-k retrieval.
- **Hybrid search** — combine BM25 (lexical) + dense (semantic) for recall on rare terms.
- **Reranking** — a cross-encoder reorders candidates for precision before they hit the prompt.
- **Context stuffing vs retrieval** — stuff when the corpus is tiny; retrieve when it's large.

→ **Deep dive:** [RAG](./intro_rag.md) · [RAG Engineering](./intro_rag_engineering.md) · [Vector Databases — Advanced](./intro_vector_databases_advanced.md)

## 9. Frameworks: LangChain / LlamaIndex / CrewAI / AutoGen

Deeply understand **at least one** — and know its limits.

| Framework | Sweet spot |
|-----------|-----------|
| **LangChain / LangGraph** | General LLM apps; LangGraph for stateful graph agents |
| **LlamaIndex** | Data-centric RAG and indexing |
| **CrewAI** | Role-based multi-agent "crews" |
| **AutoGen** | Conversational multi-agent (Microsoft) |

- **What they provide** — prompt/chain abstractions, tool integrations, memory, retrievers, orchestration.
- **What they hide** — the raw API loop, exact prompts, token usage; debugging through layers is harder.
- **When they help vs hurt** — great for fast prototypes and standard patterns; raw SDK often wins for fine-grained control, performance, and cost transparency. Senior engineers can articulate this trade-off.

→ **Deep dive:** [LangChain](./intro_langchain.md) · [LangGraph](./intro_langgraph.md) · [CrewAI](./intro_crewai.md) · [LangChain LCEL/Advanced](../frameworks/intro_langchain.md)

---

# Intermediate Agentic Layer

## 10. Multi-Agent Systems

Multiple specialized agents beat one generalist.

- **Orchestrator–worker** — a coordinator delegates subtasks to specialized workers.
- **Supervisor pattern** — a supervisor routes, monitors, and aggregates.
- **Delegation & shared state** — pass context explicitly; agents don't share memory unless you wire it.
- **Communication protocols** — message passing, structured handoffs, blackboard.
- **Why specialize** — focused prompts/tools per role → higher reliability than one overloaded agent.

→ **Deep dive:** [Multi-Agent Systems](./intro_multi_agent_systems.md)

## 11. MCP (Model Context Protocol)

Anthropic's open standard for agent↔tool communication.

- **MCP servers** — expose tools/resources/prompts over a standard protocol.
- **Building tools** — implement a server once; any MCP-aware agent can use it.
- **Connecting agents** — declare the server; the agent gains its tools without bespoke integration.
- **Tool discovery** — clients enumerate a server's capabilities at runtime.

→ **Deep dive:** [MCP](./intro_mcp.md) · [Agent Communication Standards](#32-agent-communication-standards)

## 12. Planning & Task Decomposition

Break a goal into executable steps.

- **Plan-and-execute** — generate a plan, then execute steps (vs reactive ReAct).
- **Hierarchical planning** — high-level plan → sub-plans → actions.
- **Replanning** — when a step fails, revise the plan rather than crashing.
- **DAG task graphs** — model dependencies; run independent branches in parallel.

→ **Deep dive:** [Agentic AI § Agent Patterns](./intro_agentic_ai.md) · [Multi-Agent Systems](./intro_multi_agent_systems.md)

## 13. Structured Outputs & Data Validation

Force valid, typed data out of an LLM.

- **Schema enforcement** — JSON-schema / strict tool mode guarantees parseable output.
- **Pydantic + `instructor`** — define a model, get a validated object back.
- **Output parsers** — coerce text → structured; handle partials.
- **Retry on parse failure** — re-ask with the validation error so the model self-corrects.

→ **Deep dive:** [Structured Outputs](./intro_structured_outputs.md) · [Pydantic](../frameworks/intro_pydantic.md)

## 14. Long-Context Management

Stay within the window on long tasks.

- **Sliding window** — keep the most recent N turns.
- **Summarization** — compress old history into a running summary (a "summarizer agent").
- **Context compression** — drop low-value content; keep goals, decisions, open threads.
- **Token budgeting** — allocate the window across system / history / retrieval / output; reserve headroom.
- **Keep vs drop** — keep the task, constraints, and recent state; drop stale tool dumps.

→ **Deep dive:** [Agentic AI § Context Management](./intro_agentic_ai.md) · [LLM Fundamentals § Token Budgeting](./intro_llm_fundamentals.md)

## 15. Agent State Management

Persist and resume agents.

- **Persisting state** — serialize messages, scratchpad, plan, and tool results.
- **Checkpointing** — snapshot at safe boundaries so a crash doesn't lose progress.
- **Resuming** — rehydrate state and continue an interrupted task.
- **Stateful vs stateless** — stateless scales easily but reloads context each call; stateful is cheaper per step but needs a store.

→ **Deep dive:** [Agentic AI § Production Agent Engineering](./intro_agentic_ai.md) · [LangGraph](./intro_langgraph.md)

---

# Advanced Agentic Layer

## 16. LangGraph / Agent Workflows as Graphs

Model agents as explicit state machines.

- **Nodes & edges** — nodes do work; edges define transitions.
- **Conditional branching** — route based on state.
- **Cycles** — loops enable iterate-until-done behavior.
- **Human-in-the-loop nodes** — pause for approval/input.
- **Parallel branches** — fan out independent work.
- **Streaming state** — emit intermediate state for UX and debugging.

→ **Deep dive:** [LangGraph](./intro_langgraph.md)

## 17. Human-in-the-Loop (HITL)

Fully autonomous agents often fail in production — keep a human in the loop where it matters.

- **When to pause** — irreversible/destructive actions, low confidence, high stakes.
- **Interrupt mechanisms** — checkpoints where the agent yields control.
- **Approval workflows** — explicit allow/deny on sensitive tool calls.
- **Confidence thresholds** — auto-proceed above a bar, escalate below it.
- **Async review** — queue for later human sign-off without blocking everything.

→ **Deep dive:** [LangGraph (HITL nodes)](./intro_langgraph.md) · [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)

## 18. Tool Design & Tool Ecosystems

Good tools make good agents.

- **Design** — clear names, strong "call this when…" descriptions, predictable typed outputs.
- **Tool libraries** — group by domain; a registry for discovery.
- **Versioning** — evolve tools without breaking running agents.
- **Composability** — small, single-purpose tools the agent can chain.
- **Avoid tool overload** — too many tools hurts selection accuracy; use tool search / dynamic loading.

→ **Deep dive:** [Agent Tool Use § Advanced Tool-Use Patterns](./intro_agent_tool_use.md)

## 19. Evaluation & Testing for Agents

Standard unit tests don't cover non-determinism.

- **Why unit tests fall short** — same input ≠ same output; you test behavior distributions, not exact strings.
- **Trajectory evaluation** — judge the *path* (tool choices, steps), not just the final answer.
- **LLM-as-judge** — a model grades outputs against a rubric.
- **Eval datasets** — curate task suites with success criteria.
- **Regression testing** — re-run the suite on every change and on model upgrades.
- **Cross-version benchmarking** — quantify quality/cost when you switch models.

→ **Deep dive:** [LLM Evaluation](../mlops/intro_llm_evaluation.md) · [Testing AI Systems](../devops/intro_testing_ai.md)

## 20. Observability & Tracing

You can't debug what you can't see.

- **Tools** — LangSmith, Langfuse, Arize Phoenix, OpenTelemetry (GenAI semantic conventions).
- **End-to-end traces** — every model call, tool call, and decision as spans.
- **Logging tool calls** — inputs, outputs, latency, errors.
- **Debugging failures** — replay a trace to find where the agent went wrong.
- **Cost & latency** — track tokens/cost/latency per run and per step.

→ **Deep dive:** [LangSmith](./intro_langsmith.md) · [LLMOps](./intro_llmops.md) · [Model Monitoring](../mlops/intro_model_monitoring.md)

## 21. Code Execution Agents

Agents that write and run code.

- **Sandboxing** — E2B, Docker, gVisor, or provider-hosted code execution; never run model code on the host.
- **Code interpreters** — model writes code → sandbox runs it → results return.
- **Safety** — resource limits, network egress control, no host filesystem access, timeouts.
- **Dangerous code** — allowlist/denylist, review gates for destructive ops.

→ **Deep dive:** [Agent Tool Use § Server-Side Tools](./intro_agent_tool_use.md) · [Docker](../devops/intro_docker.md)

## 22. Browser & Computer Use Agents

Agents that operate UIs.

- **Web automation** — Playwright/Puppeteer drive real browsers.
- **DOM vs screenshot** — parse the DOM (precise, brittle) vs vision-based screenshot interaction (general, costlier).
- **Computer use** — vision models that take screenshots and emit mouse/keyboard actions.
- **Edge cases** — CAPTCHAs, dynamic pages, auth, rate limits, flakiness.

→ **Deep dive:** [Testing AI Systems (Playwright/Puppeteer)](../devops/intro_testing_ai.md) · [Multimodal AI](./intro_multimodal_ai.md)

---

# Production & Senior Layer

## 23. Agent Reliability & Failure Modes

Why agents fail — and how to contain it.

| Failure mode | Mitigation |
|--------------|-----------|
| Prompt brittleness | Robust prompts, evals, version prompts |
| Tool failure cascades | Graceful tool errors (`is_error`), retries, fallbacks |
| Infinite loops | Max-iteration + token budgets; loop detection |
| Hallucinated tool calls | Strict schemas; validate before executing |
| Context poisoning | Treat tool/retrieved content as untrusted data |

→ **Deep dive:** [Agentic AI § Failure Modes & Guardrails](./intro_agentic_ai.md) · [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)

## 24. Latency Optimization

Make agents feel fast.

- **Streaming** — show tokens/progress during the run.
- **Parallel tool calls** — execute independent tools concurrently.
- **Smaller models for sub-tasks** — route routing/extraction to a fast tier.
- **Caching** — prompt caching of the stable prefix; cache deterministic tool results.
- **Speculative execution / prefetching** — start likely-needed work early.

→ **Deep dive:** [Agentic AI § Cost & Latency](./intro_agentic_ai.md) · [LLM Fundamentals § Prompt Caching](./intro_llm_fundamentals.md)

## 25. Cost Management at Scale

Unit economics decide whether the feature survives.

- **Token budgeting** — cap input/output per task.
- **Model routing** — cheap model by default, escalate to a frontier model only when needed.
- **Prompt compression & caching** — shrink and reuse the stable prefix (~10× cheaper cache reads).
- **Cost per task** — monitor it; alert on outliers.
- **Unit economics** — know the $/task and whether it's sustainable at your usage.

→ **Deep dive:** [LLM Fundamentals § Estimating Cost](./intro_llm_fundamentals.md) · [Multi-Model Orchestration](./intro_multi_model_orchestration.md)

## 26. Security for Agentic Systems

Agents with tools are an attack surface.

- **Prompt injection** — direct (user) and **indirect** (malicious content inside retrieved docs/tool outputs).
- **Privilege escalation** — agents should have least-privilege tool/credential access.
- **Sandboxing** — isolate code/tool execution.
- **Permissions** — what an agent should *never* be able to do (delete prod, move money) without HITL.
- **Data exfiltration** — guard against an injected instruction leaking secrets via a tool call.

→ **Deep dive:** [LLM Security](./intro_llm_security.md)

## 27. Deployment Patterns

Get the agent off your laptop.

- **Serverless vs long-running** — serverless functions for short tasks; long-running services for stateful/streaming agents.
- **Queue-based execution** — Celery / SQS / Cloud Tasks for durable, retryable agent jobs.
- **Webhook-driven** — trigger agents on events.
- **Containerization** — package the agent + deps in Docker; scale horizontally on Kubernetes.

→ **Deep dive:** [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md) · [Docker](../devops/intro_docker.md) · [Kubernetes](../devops/intro_kubernetes.md) · [FastAPI](../frameworks/intro_fastapi.md)

## 28. Agentic Architecture Patterns

Knowing which pattern fits which problem separates seniors from juniors.

| Pattern | Use when |
|---------|----------|
| **Orchestrator–worker** | A coordinator delegates to specialists |
| **Pipeline** | Fixed sequential stages |
| **Blackboard** | Agents share a common workspace/state |
| **Event-driven** | React to external events |
| **Hub-and-spoke** | Central router to many tools/agents |
| **Map-reduce** | Fan out over items, aggregate results |

→ **Deep dive:** [ML System Design Patterns](../system_design/ml_system_design_patterns.md) · [Multi-Agent Systems](./intro_multi_agent_systems.md)

---

# Expert / Frontier Layer

## 29. Fine-tuning for Agentic Behavior

When prompting isn't enough.

- **When fine-tuning beats prompting** — consistent niche behavior, latency/cost wins from a smaller tuned model, hard-to-prompt formats.
- **SFT on trajectories** — supervised fine-tuning on successful agent runs.
- **DPO for tool-use preferences** — preference-optimize toward good tool choices.
- **Datasets from runs** — mine successful trajectories as training data.
- **LoRA / QLoRA** — parameter-efficient tuning for fast, cheap iteration.

→ **Deep dive:** [Fine-Tuning (LoRA, QLoRA, RLHF/DPO)](../deep_learning/intro_fine_tuning.md) · [Unsloth](../frameworks/intro_unsloth.md)

## 30. Reasoning Models

Models that "think" at inference time.

- **How they work** — trained (often with RL) to produce long internal chains of thought before answering; spend more compute on hard problems.
- **When to use** — genuinely hard reasoning/planning; not for simple lookups (slower, costlier).
- **Cost/latency** — reasoning tokens add latency and cost; gate them behind difficulty.
- **In agent pipelines** — use a reasoning model for the planner/orchestrator, a fast model for routine sub-steps. Anthropic's **adaptive thinking + effort** controls are this idea exposed as parameters.

→ **Deep dive:** [Anthropic & Claude (extended/adaptive thinking)](./intro_anthropic.md) · [LLM Fundamentals](./intro_llm_fundamentals.md)

## 31. Multi-Modal Agents

Beyond text.

- **Vision + text** — reason over images alongside text.
- **Document understanding** — PDFs, tables, charts, scans.
- **Audio** — speech-to-text input (and TTS output).
- **Image generation as a tool** — call a generator from the agent.
- **Multi-modal memory** — store and retrieve across modalities.

→ **Deep dive:** [Multimodal AI](./intro_multimodal_ai.md) · [Computer Vision](../deep_learning/intro_computer_vision.md)

## 32. Agent Communication Standards

The emerging interoperability layer.

- **MCP** — standard for agent↔tool/resource communication (tool ecosystem).
- **A2A (Agent-to-Agent)** — Google's protocol for agent↔agent interoperability.
- **Interoperable agents** — agents from different vendors cooperating via shared protocols.
- **Registries & discovery** — find and describe available agents/tools.
- **Why it matters** — avoids N×M bespoke integrations as the ecosystem grows.

→ **Deep dive:** [MCP](./intro_mcp.md)

## 33. Building Agent Platforms

Infrastructure so *other* teams can ship agents.

- **Agent SDKs** — give teams a paved path to define agents.
- **Multi-tenant isolation** — separate state, secrets, and compute per tenant.
- **Permissioning** — scoped tool/credential access per agent/tenant.
- **Marketplaces & registries** — discover and reuse agents/tools.
- **Audit logs** — every action traceable for compliance and debugging.

→ **Deep dive:** [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md) · [LLMOps / MLOps Engineering](../mlops/intro_llmops_mlops_engineering.md)

## 34. Self-Improving Agents / Meta-Agents

Agents that improve agents.

- **Prompt self-optimization** — agents that rewrite and test their own prompts against evals.
- **Automated red-teaming** — agents that attack agents to find failures.
- **Agents that build agents** — generate, test, and refine other agents.
- **Reflection loops** — critique-and-revise to improve outputs.
- **Constitutional AI** — train/steer behavior against a set of principles rather than only human labels.

→ **Deep dive:** [LLM Evaluation](../mlops/intro_llm_evaluation.md) · [Anthropic (Constitutional AI)](./intro_anthropic.md)

---

## What Actually Gets You Hired

Beyond the stack, senior agentic-AI interviews probe judgment. Be ready to answer:

1. **Can you debug a non-deterministic agent failure with incomplete logs?** →
   Reproduce with fixed seeds where possible, replay traces, isolate the failing
   step, add targeted logging, and reason about distributions, not single runs.
2. **Can you design for graceful (not catastrophic) failure?** → Budgets/caps,
   tool-error handling, fallbacks, HITL on irreversible actions, circuit breakers.
3. **Do you understand the autonomy ↔ reliability trade-off?** → More autonomy =
   more capability and more failure surface; add human checkpoints where cost-of-
   error is high.
4. **Have you shipped an agent to real users and handled the edge cases?** →
   Concrete war stories: rate limits, injection attempts, cost spikes, weird inputs.
5. **Can you say when NOT to use an agent?** → If the task is fully specifiable,
   a deterministic pipeline or single LLM call is cheaper, faster, and more reliable.
6. **Do you have opinions on eval *methodology*, not just tools?** → Trajectory vs
   outcome evals, judge calibration, dataset curation, regression gating.
7. **Can you estimate cost and latency *before* building?** → Tokens × price per
   step × expected steps; identify the latency-critical path; plan caching/routing.

---

## Related Guides
- [Agentic AI](./intro_agentic_ai.md) · [Agent Tool Use](./intro_agent_tool_use.md) · [Multi-Agent Systems](./intro_multi_agent_systems.md)
- [LLM Fundamentals](./intro_llm_fundamentals.md) · [Prompt Engineering](./intro_prompt_engineering.md) · [RAG](./intro_rag.md)
- [Project Setup & Engineering](../project_setup/README.md) · [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
