# 2026 Interview Roadmap

A plan, not a reading list. This page tells you **what is being tested in 2026**, **which topics matter for your specific role**, and **what to study in what order given how much time you have**.

> **Not sure which role you're aiming for yet?** Start with [Choose Your Track](./choose-your-track.md),
> which picks a track from what you want to build and gives you a staged path through the repo. Come back
> here once you know the role, or when you're within a few weeks of interviewing.

---

## Table of Contents
1. [What Changed for 2026](#what-changed-for-2026)
2. [Core Evaluation Areas](#core-evaluation-areas)
3. [Role-by-Role Focus Matrix](#role-by-role-focus-matrix)
4. [The Typical Interview Loop](#the-typical-interview-loop)
5. [Study Plans by Time Available](#study-plans-by-time-available)
6. [Depth Checklist](#depth-checklist)
7. [The Week Before](#the-week-before)
8. [Related Topics](#related-topics)

---

## What Changed for 2026

The classical ML fundamentals have not moved — bias/variance, metrics, regularization, and validation are still asked in nearly every loop. What changed is the layer on top.

| Shift | What it means for interviews |
|---|---|
| **From "can you train a model" to "can you ship a system"** | Expect production, monitoring, and cost questions even in "modeling" rounds |
| **Agents are the default architecture** | Tool use, planning loops, failure handling, and cost control are standard questions |
| **RAG is assumed, not impressive** | You'll be asked why your retrieval *failed*, not what RAG is |
| **Evaluation is the differentiator** | "How do you know it works?" is now the hardest question in most loops |
| **Inference economics are interview material** | KV cache, batching, quantization, and cost-per-request appear in AI Engineer loops |
| **Context engineering replaced prompt tricks** | Structuring what the model sees, not clever phrasing |
| **Security is a real round** | Prompt injection, tool permissions, data exfiltration through agents |

The single most common reason strong candidates fail an AI Engineer loop in 2026: they can build the happy path and cannot answer how they'd *evaluate* it, *monitor* it, or *bound its cost*.

---

## Core Evaluation Areas

Every loop draws from these eight. The weight shifts by role, but none is safe to skip entirely.

| # | Area | Sample of what's tested | Start here |
|---|---|---|---|
| 1 | **Classical ML fundamentals** | Bias/variance, regularization, cross-validation, imbalanced data, leakage | [Classical ML](../classical_ml/README.md), [Model Evaluation](../classical_ml/intro_model_evaluation.md), [Ensembles](../classical_ml/intro_ensemble_methods.md) |
| 2 | **Deep learning fundamentals** | Optimizers, normalization, attention, transfer learning, training failures | [Transformers](../deep_learning/intro_transformers.md), [NN Training](../deep_learning/intro_neural_network_training.md) |
| 3 | **LLM application engineering** | Prompting, structured outputs, tool use, RAG, context management | [LLM Fundamentals](../ai_genai/intro_llm_fundamentals.md), [RAG](../ai_genai/intro_rag.md), [Embeddings](../ai_genai/intro_embeddings.md) |
| 4 | **Agents and orchestration** | Planning loops, tool schemas, multi-agent patterns, failure recovery, when *not* to use an agent | [Agentic AI](../ai_genai/intro_agentic_ai.md), [Multi-Agent Systems](../ai_genai/intro_multi_agent_systems.md), [MCP](../ai_genai/intro_mcp.md) |
| 5 | **Evaluation** | Offline vs online metrics, LLM-as-judge, regression suites, hallucination detection | [Model Evaluation](../classical_ml/intro_model_evaluation.md), [LLM Evaluation](../mlops/intro_llm_evaluation.md), [Guardrails](../mlops/intro_evaluation_guardrails.md) |
| 6 | **MLOps and production** | Deployment, drift, experiment tracking, CI/CD, rollback, monitoring | [MLOps](../mlops/README.md), [Model Monitoring](../mlops/intro_model_monitoring.md), [Model Serving](../mlops/intro_model_serving.md) |
| 7 | **System design** | Latency, throughput, cost, caching, batching, failure modes, capacity | [System Design](../system_design/README.md), [Design Patterns](../system_design/ml_system_design_patterns.md), [Inference Optimization](../ai_genai/intro_llm_inference_optimization.md) |
| 8 | **Responsible AI and security** | Prompt injection, tool permissions, PII handling, bias, governance | [LLM Security](../ai_genai/intro_llm_security.md), [Explainability](../mlops/intro_model_explainability.md) |

---

## Role-by-Role Focus Matrix

Priority: **●●● critical** · **●● important** · **● useful**

| Area | ML Engineer | AI / GenAI Engineer | Data Scientist | Data Engineer | MLOps / Platform |
|---|---|---|---|---|---|
| Classical ML | ●●● | ● | ●●● | ● | ●● |
| Statistics / experiment design | ●● | ● | ●●● | ● | ● |
| Deep learning | ●●● | ●● | ● | — | ● |
| LLM app engineering | ●● | ●●● | ●● | ● | ●● |
| RAG and embeddings | ●● | ●●● | ● | ●● | ●● |
| Agents and tool use | ● | ●●● | ● | — | ●● |
| Inference optimization | ●● | ●●● | — | — | ●●● |
| Evaluation | ●●● | ●●● | ●●● | ● | ●●● |
| MLOps / deployment | ●●● | ●● | ● | ●● | ●●● |
| Data pipelines / modeling | ●● | ●● | ●● | ●●● | ●● |
| System design | ●●● | ●●● | ● | ●●● | ●●● |
| SQL | ●● | ● | ●●● | ●●● | ●● |
| Python / coding round | ●●● | ●●● | ●● | ●●● | ●● |
| Docker / K8s / IaC | ●● | ●● | — | ●● | ●●● |
| Security and governance | ● | ●●● | ● | ●● | ●● |

**How to read this**: your loop will test the ●●● rows properly and spot-check the ●● rows. Preparing the ● rows is what you do with leftover time, not first.

---

## The Typical Interview Loop

| Stage | Format | What determines the outcome |
|---|---|---|
| **Recruiter screen** | 20–30 min | Can you describe your work in plain language? Level calibration. |
| **Technical screen** | 45–60 min | Coding (Python/SQL) or an ML fundamentals rapid-fire round |
| **Coding round** | 45–60 min | Data manipulation, occasionally [ML from scratch](../coding_challenges/ml_coding_challenges.md) |
| **ML/AI depth round** | 60 min | Fundamentals with follow-ups going 3–4 layers deep |
| **System design** | 60 min | An open-ended "design X" — the highest-variance round |
| **Project deep-dive** | 45–60 min | Your own past work, interrogated in detail |
| **Behavioral / hiring manager** | 45 min | Ownership, judgment, collaboration |

Two rounds are under-prepared by almost everyone: the **project deep-dive** and **system design**. They are also the two with the widest score spread, which means they decide most loops. See [Behavioral and Project Deep-Dive](./behavioral-interview-guide.md) and the [System Design framework](../system_design/README.md).

---

## Study Plans by Time Available

### 1 week — triage mode

Assume you cannot learn anything new; you can only surface what you already know.

| Day | Focus |
|---|---|
| 1 | [Model Evaluation](../classical_ml/intro_model_evaluation.md) + your target role's ●●● rows, skimmed for gaps |
| 2 | Rehearse your [project deep-dive](./behavioral-interview-guide.md#the-project-deep-dive) out loud, twice. Look up your real numbers. |
| 3 | [System design framework](../system_design/README.md) + one full case study end to end |
| 4 | Coding: [Python](../coding_challenges/python_coding_challenges.md) or [SQL](../coding_challenges/sql_coding_challenges.md), whichever your loop tests |
| 5 | Role-specific depth: [RAG](../ai_genai/intro_rag.md) + [Agents](../ai_genai/intro_agentic_ai.md), or [MLOps](../mlops/README.md) |
| 6 | [2026 Additional Questions](./2026-additional-questions.md) — answer out loud, don't read |
| 7 | [Behavioral stories](./behavioral-interview-guide.md), rest, logistics |

### 1 month — the realistic plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Fundamentals: [classical ML](../classical_ml/README.md), [evaluation](../classical_ml/intro_model_evaluation.md), [ensembles](../classical_ml/intro_ensemble_methods.md), [statistics](../classical_ml/intro_statistics_probability.md) | Can explain bias/variance, leakage, and metric choice without notes |
| 2 | Deep learning + LLM: [transformers](../deep_learning/intro_transformers.md), [training](../deep_learning/intro_neural_network_training.md), [LLM fundamentals](../ai_genai/intro_llm_fundamentals.md), [embeddings](../ai_genai/intro_embeddings.md), [RAG](../ai_genai/intro_rag.md) | A small working RAG app you can discuss |
| 3 | Production: [MLOps](../mlops/README.md), [monitoring](../mlops/intro_model_monitoring.md), [serving](../mlops/intro_model_serving.md), [inference optimization](../ai_genai/intro_llm_inference_optimization.md), [Docker](../devops/intro_docker.md) | Your project, containerized and monitored |
| 4 | Interview mechanics: [system design](../system_design/README.md) cases, [coding](../coding_challenges/README.md), [behavioral](./behavioral-interview-guide.md), mock loops | 3 rehearsed system designs, 8 STAR stories |

### 3 months — build depth and a portfolio

| Month | Focus |
|---|---|
| 1 | Fundamentals with implementation: work through [ML coding challenges](../coding_challenges/ml_coding_challenges.md), rebuild the classics from scratch, and read [classical ML](../classical_ml/README.md) + [deep learning](../deep_learning/README.md) properly |
| 2 | Build one real project end to end — ingestion, model or LLM pipeline, evals, API, container, monitoring. See [Project Setup](../project_setup/README.md) and [Highlighted Projects](../README.md#highlighted-projects). This project becomes your deep-dive story. |
| 3 | Interview preparation: [system design](../system_design/README.md) breadth, [2026 questions](./interview_questions_2026.md), weekly mock interviews, [behavioral](./behavioral-interview-guide.md) story bank |

The three-month plan works because month 2 gives you something real to talk about. A candidate with one genuinely deep project outperforms one with five tutorials, every time.

---

## Depth Checklist

Use this to find gaps. If you cannot answer a line in two minutes without notes, that's your next study session.

**Fundamentals**
- [ ] Explain bias/variance and name three ways to reduce each
- [ ] Choose a metric for an imbalanced problem and defend it against accuracy and ROC-AUC
- [ ] Detect and explain data leakage; name four ways it enters a pipeline
- [ ] Explain why a random train/test split is wrong for temporal or grouped data
- [ ] Compare bagging and boosting, including which base learner each needs and why

**Deep learning**
- [ ] Explain attention, including why scores are divided by `√d_k`
- [ ] Diagnose a NaN loss, in order of likelihood
- [ ] Explain why transformers use LayerNorm rather than BatchNorm
- [ ] Explain what to do when a model doesn't fit in GPU memory, in escalating order

**LLM and AI engineering**
- [ ] Explain why RAG retrieval fails and how you'd tell retrieval from generation problems
- [ ] Explain why LLM inference is memory-bandwidth-bound and what follows from it
- [ ] Design an evaluation suite for a non-deterministic LLM feature
- [ ] Explain prompt injection and how tool permissions bound the damage
- [ ] Say when you would *not* use an agent

**Production**
- [ ] Describe how you'd detect that a deployed model has degraded
- [ ] Explain training/serving skew and how a feature store prevents it
- [ ] Describe a rollback plan for a model deploy
- [ ] Estimate the cost per request of an LLM feature and name the three biggest levers

**Communication**
- [ ] Tell your best project story in 5 minutes, with real numbers
- [ ] Tell a production failure story with root cause and systemic fix
- [ ] Explain a model's performance to a non-technical stakeholder in their vocabulary

---

## The Week Before

- **Re-read your own résumé and projects.** Interviewers ask about the line you wrote two years ago and forgot.
- **Recover your numbers.** Dataset sizes, latency, cost, metric before/after. Approximate is fine; blank is not.
- **Do one mock loop out loud**, ideally with another person. Reading is not rehearsal — the gap between knowing something and saying it fluently is larger than it feels.
- **Prepare four questions per interviewer** and know who you're meeting.
- **Stop learning new topics 48 hours out.** Consolidate what you have; new material at that point mostly adds noise.

---

## Related Topics

- [2026 Additional Questions and Answers](./2026-additional-questions.md)
- [2026 Common Interview Questions](./interview_questions_2026.md)
- [Behavioral and Project Deep-Dive Guide](./behavioral-interview-guide.md)
- [Study Pattern](./study-pattern.md)
- [Resources and References](./resources-and-references.md)
- [Repository Home](../README.md)
