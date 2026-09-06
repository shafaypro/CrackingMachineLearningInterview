# Choose Your Track

This repository has over 100 guides. You do not need all of them — you need one track, in order.

Pick a track below, follow its stages top to bottom, and skip everything else until you finish. Coming back for a second track later is much easier than trying to learn two at once.

---

## Table of Contents
1. [The 30-Second Chooser](#the-30-second-chooser)
2. [Pick by Where You're Coming From](#pick-by-where-youre-coming-from)
3. [The Common Core](#the-common-core)
4. [Track 1 — Machine Learning Engineer](#track-1--machine-learning-engineer)
5. [Track 2 — AI / GenAI Engineer](#track-2--ai--genai-engineer)
6. [Track 3 — Data Scientist](#track-3--data-scientist)
7. [Track 4 — Data Engineer](#track-4--data-engineer)
8. [Track 5 — MLOps / Platform Engineer](#track-5--mlops--platform-engineer)
9. [Track 6 — Deep Learning Engineer](#track-6--deep-learning-engineer)
10. [Track Comparison](#track-comparison)
11. [How to Actually Study This](#how-to-actually-study-this)
12. [Frequently Asked](#frequently-asked)
13. [Related Topics](#related-topics)

---

## The 30-Second Chooser

Find the row that sounds most like you.

| If you want to... | Your track |
|---|---|
| Train models and ship them into a product | [ML Engineer](#track-1--machine-learning-engineer) |
| Build apps on top of LLMs — chatbots, RAG, agents | [AI / GenAI Engineer](#track-2--ai--genai-engineer) |
| Answer business questions with data and experiments | [Data Scientist](#track-3--data-scientist) |
| Build the pipelines and warehouses everything else runs on | [Data Engineer](#track-4--data-engineer) |
| Run the infrastructure that serves and monitors models | [MLOps / Platform Engineer](#track-5--mlops--platform-engineer) |
| Work on model architectures, training, and fine-tuning | [Deep Learning Engineer](#track-6--deep-learning-engineer) |

**Still unsure?** Two honest tie-breakers:

- **The most in-demand entry point right now is the AI / GenAI track.** It has the shortest path from zero to something working, and it needs the least mathematics. If you want to be employable fastest, start there.
- **The most durable foundation is the ML Engineer track.** It is slower, but everything else becomes easier afterwards, and it does not go stale when the tooling changes.

You can also just start with the [Common Core](#the-common-core) below — the first two weeks are identical for every track, so you can defer the decision.

---

## Pick by Where You're Coming From

| Your background | Recommended start | Why |
|---|---|---|
| **Software engineer** | [AI / GenAI](#track-2--ai--genai-engineer) or [MLOps](#track-5--mlops--platform-engineer) | Your existing skills transfer almost entirely; you're adding a layer, not starting over |
| **Data analyst** | [Data Scientist](#track-3--data-scientist) | You already have SQL and statistics — the closest jump |
| **Student / new grad** | [ML Engineer](#track-1--machine-learning-engineer) | You have time for the foundation, and it opens the most doors later |
| **Backend / DevOps engineer** | [MLOps](#track-5--mlops--platform-engineer) | Containers, CI/CD, and monitoring are already yours; add the ML-specific parts |
| **Researcher / academic** | [Deep Learning](#track-6--deep-learning-engineer) | You have the maths; you're adding engineering and production practice |
| **Career switcher, non-technical** | [Common Core](#the-common-core), then [AI / GenAI](#track-2--ai--genai-engineer) | Learn Python properly first; GenAI gives visible results soonest |
| **Already an MLE, want to move into AI** | [AI / GenAI](#track-2--ai--genai-engineer), skipping its foundation stage | Your fundamentals hold; you need LLM-specific depth |

---

## The Common Core

Every track needs these. If you are early, do this first and decide afterwards — roughly **1–2 weeks**.

1. [Python for AI Engineering](../frameworks/intro_python_for_ai.md) — async, typing, Pydantic, APIs, logging
2. [Statistics & Probability](../classical_ml/intro_statistics_probability.md) — distributions, hypothesis testing, the CLT
3. [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md) — the single most reusable guide in this repo
4. [Python Coding Challenges](../coding_challenges/python_coding_challenges.md) and [SQL Coding Challenges](../coding_challenges/sql_coding_challenges.md)
5. [GitHub Project Setup](../project_setup/intro_github_project_setup.md) — so your work is shareable from day one

Everything after this is track-specific.

---

## Track 1 — Machine Learning Engineer

**Job titles**: Machine Learning Engineer, Applied Scientist, ML Software Engineer
**Prerequisites**: comfortable Python; basic linear algebra and probability helps but can be learned alongside
**Realistic time to interview-ready**: 3–4 months from the Common Core, part-time

### Stage 1 — Foundations (3–4 weeks)
- [Classical ML Overview](../classical_ml/README.md) — algorithms, when each applies
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [Feature Engineering & Selection](../classical_ml/intro_feature_engineering.md)
- [Ensemble Methods and Gradient Boosting](../classical_ml/intro_ensemble_methods.md) — what actually wins on tabular data
- [Clustering](../classical_ml/intro_clustering.md) and [Dimensionality Reduction](../classical_ml/intro_dimensionality_reduction.md)

### Stage 2 — Deep learning (3–4 weeks)
- [Deep Learning Overview](../deep_learning/README.md)
- [Neural Network Training](../deep_learning/intro_neural_network_training.md) — optimizers, normalization, debugging
- [Sequence Models](../deep_learning/intro_sequence_models.md) → [Transformers](../deep_learning/intro_transformers.md)
- [PyTorch](../frameworks/intro_pytorch.md)

### Stage 3 — Production (3–4 weeks)
- [MLOps Overview](../mlops/README.md)
- [Model Serving](../mlops/intro_model_serving.md) and [Model Monitoring](../mlops/intro_model_monitoring.md)
- [CI/CD for Machine Learning](../mlops/intro_cicd_for_ml.md)
- [MLflow](../mlops/intro_mlflow.md) · [Docker](../devops/intro_docker.md)

### Stage 4 — Interview prep (2–3 weeks)
- [ML System Design Framework](../system_design/README.md) + both case studies
- [ML Coding Challenges](../coding_challenges/ml_coding_challenges.md)
- [2026 Interview Roadmap](./2026-interview-roadmap.md) · [Behavioral Guide](./behavioral-interview-guide.md) · [Take-Home Projects](./take-home-projects.md)

**Build this**: a tabular prediction model served behind a FastAPI endpoint, containerized, with tracked experiments and a monitoring dashboard. End to end matters more than accuracy.

**You're ready when** you can explain bias/variance without notes, choose a metric and defend it against alternatives, spot data leakage in someone else's setup, and describe how you would monitor and roll back a deployed model.

---

## Track 2 — AI / GenAI Engineer

**Job titles**: AI Engineer, GenAI Engineer, LLM Engineer, Applied AI Engineer
**Prerequisites**: solid Python and API work. **You do not need to know how to train a model.**
**Realistic time to interview-ready**: 2–3 months, part-time — the shortest path here

### Stage 1 — LLM foundations (2–3 weeks)
- [LLM & Generative AI Fundamentals](../ai_genai/intro_llm_fundamentals.md)
- [Prompt Engineering](../ai_genai/intro_prompt_engineering.md)
- [Context Engineering](../ai_genai/intro_context_engineering.md) — what actually goes in the window, and what it costs
- [Structured Outputs & Function Calling](../ai_genai/intro_structured_outputs.md)

### Stage 2 — Retrieval (2–3 weeks)
- [Intro to RAG](../ai_genai/intro_rag.md) → [RAG Engineering](../ai_genai/intro_rag_engineering.md)
- [Embeddings](../ai_genai/intro_embeddings.md) — chunking and retrieval quality live here
- [Vector Databases](../ai_genai/intro_vector_databases.md) → [Advanced](../ai_genai/intro_vector_databases_advanced.md)

### Stage 3 — Agents (2–3 weeks)
- [Agentic AI](../ai_genai/intro_agentic_ai.md) → [Agent Systems & Tool Use](../ai_genai/intro_agent_tool_use.md)
- [Multi-Agent Systems](../ai_genai/intro_multi_agent_systems.md)
- [MCP Protocol](../ai_genai/intro_mcp.md) · [LangGraph](../ai_genai/intro_langgraph.md)

### Stage 4 — Production (2–3 weeks)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md) and [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md) — **the most-tested area in these interviews**
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md) — cost and latency
- [LLM Security](../ai_genai/intro_llm_security.md) — prompt injection is a real round
- [LLMOps](../ai_genai/intro_llmops.md) · [FastAPI](../frameworks/intro_fastapi.md) · [Docker](../devops/intro_docker.md)

### Stage 5 — Interview prep (2 weeks)
- [Backend AI System Design](../system_design/intro_backend_ai_system_design.md) and [ML System Design Patterns](../system_design/ml_system_design_patterns.md)
- [2026 Interview Questions](./interview_questions_2026.md) · [Behavioral Guide](./behavioral-interview-guide.md)

**Build this**: a RAG assistant over a real document set with citations, hybrid search, a reranker, **an evaluation harness**, and a cost-per-request number you can quote. The eval harness is what separates candidates.

**You're ready when** you can diagnose whether a bad RAG answer came from retrieval or generation, design an eval suite for a non-deterministic feature, explain prompt injection and how tool permissions bound it, and state your system's cost per request and the three biggest levers on it.

> The deeper curriculum for this track is [The Agentic AI Engineer Roadmap](../ai_genai/intro_agentic_ai_engineering_roadmap.md) — 34 topics from foundation to frontier. Use it after Stage 3.

---

## Track 3 — Data Scientist

**Job titles**: Data Scientist, Product Data Scientist, Quantitative Analyst
**Prerequisites**: SQL, and genuine comfort with statistics
**Realistic time to interview-ready**: 2–3 months, part-time

### Stage 1 — Statistical foundation (3–4 weeks)
- [Statistics & Probability](../classical_ml/intro_statistics_probability.md) — go deep; this is the whole round
- [Model Evaluation and Metrics](../classical_ml/intro_model_evaluation.md)
- [SQL Coding Challenges](../coding_challenges/sql_coding_challenges.md) — window functions until they're automatic

### Stage 2 — Modeling (3 weeks)
- [Classical ML Overview](../classical_ml/README.md)
- [Feature Engineering](../classical_ml/intro_feature_engineering.md) · [Ensemble Methods](../classical_ml/intro_ensemble_methods.md)
- [Time Series](../classical_ml/intro_time_series.md) · [Clustering](../classical_ml/intro_clustering.md)

### Stage 3 — Experimentation and communication (2–3 weeks)
- [A/B Testing](../mlops/intro_ab_testing.md) — **the defining skill of this role**
- [Model Explainability](../mlops/intro_model_explainability.md) — SHAP for stakeholder conversations
- [Data Quality](../mlops/intro_data_quality.md)

### Stage 4 — Interview prep (2 weeks)
- [Take-Home Projects](./take-home-projects.md) — data science loops lean on these heavily
- [Behavioral Guide](./behavioral-interview-guide.md) · [Python Coding Challenges](../coding_challenges/python_coding_challenges.md)

**Build this**: an end-to-end analysis answering a real business question — framing, EDA, a model, an experiment design, and a written recommendation. The write-up is the deliverable, not the notebook.

**You're ready when** you can design an A/B test including sample size and guardrails, explain a model's output to a non-technical stakeholder in their vocabulary, write window-function SQL fluently, and recognize Simpson's paradox and the peeking problem in a described scenario.

---

## Track 4 — Data Engineer

**Job titles**: Data Engineer, Analytics Engineer, Data Platform Engineer
**Prerequisites**: strong SQL, comfortable Python
**Realistic time to interview-ready**: 2–3 months, part-time

### Stage 1 — Modeling and architecture (2–3 weeks)
- [Data Modeling](../data_engineering/data-modeling.md) — star schemas, slowly changing dimensions
- [Data Architecture](../data_engineering/data-architecture.md) — warehouse, lake, lakehouse
- [SQL Coding Challenges](../coding_challenges/sql_coding_challenges.md)

### Stage 2 — Pipelines (3 weeks)
- [Data Processing Pipelines](../data_engineering/intro_data_processing_pipelines.md) — batch vs streaming, and when each is right
- [Apache Spark](../data_engineering/intro_apache_spark.md) · [Apache Airflow](../data_engineering/intro_apache_airflow.md)
- [dbt](../data_engineering/intro_dbt.md) → [dbt Interview Q&A](../data_engineering/interview_dbt.md)

### Stage 3 — Modern stack (2–3 weeks)
- [Apache Kafka](../data_engineering/intro_apache_kafka.md) — streaming
- [Delta Lake](../data_engineering/intro_delta_lake.md) · [Apache Iceberg](../data_engineering/intro_apache_iceberg.md) · [DuckDB](../data_engineering/intro_duckdb.md)
- [Data Quality](../mlops/intro_data_quality.md)

### Stage 4 — Serving ML (1–2 weeks)
- [Data Engineering for AI](../data_engineering/intro_data_engineering_for_ai.md) — point-in-time correctness, training/serving skew
- [Feature Stores](../mlops/intro_feature_stores.md)

### Stage 5 — Interview prep (2 weeks)
- [Backend System Design Interview Guide](../system_design/backend_system_design_interview_guide.md)
- [Docker](../devops/intro_docker.md) · [Behavioral Guide](./behavioral-interview-guide.md)

**Build this**: an ingestion → transform → warehouse pipeline orchestrated in Airflow, modeled in dbt, with data quality tests that block on failure and a documented lineage graph.

**You're ready when** you can design a star schema from requirements, explain idempotency and why partial writes must never be visible, argue batch versus streaming from the latency the business actually needs, and describe how late-arriving data is handled.

---

## Track 5 — MLOps / Platform Engineer

**Job titles**: MLOps Engineer, ML Platform Engineer, ML Infrastructure Engineer
**Prerequisites**: real software engineering experience; Linux and containers
**Realistic time to interview-ready**: 2–3 months if you already do DevOps

### Stage 1 — ML literacy (2 weeks)
You need enough to reason about models, not to build them.
- [Classical ML Overview](../classical_ml/README.md) · [Model Evaluation](../classical_ml/intro_model_evaluation.md)

### Stage 2 — The MLOps core (3–4 weeks)
- [MLOps Overview](../mlops/README.md)
- [CI/CD for Machine Learning](../mlops/intro_cicd_for_ml.md) — registry, gates, canary, rollback
- [Model Serving](../mlops/intro_model_serving.md) · [Model Monitoring](../mlops/intro_model_monitoring.md)
- [MLflow](../mlops/intro_mlflow.md) · [Feature Stores](../mlops/intro_feature_stores.md) · [Data Quality](../mlops/intro_data_quality.md)

### Stage 3 — Infrastructure (2–3 weeks)
- [Docker](../devops/intro_docker.md) → [Kubernetes](../devops/intro_kubernetes.md) → [Helm](../devops/intro_helm.md)
- [Terraform](../devops/intro_terraform.md) · [GitHub Actions](../devops/intro_github_actions.md)
- [Cloud ML Platforms](../cloud_ml/intro_cloud_ml_platforms.md), then your employer's cloud

### Stage 4 — LLM operations (2 weeks)
- [LLMOps](../ai_genai/intro_llmops.md) · [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [LLM Evaluation](../mlops/intro_llm_evaluation.md) · [Testing AI Systems](../devops/intro_testing_ai.md)

### Stage 5 — Interview prep (2 weeks)
- [ML System Design](../system_design/README.md) · [Backend AI Systems](../system_design/intro_backend_ai_system_design.md)
- [A/B Testing](../mlops/intro_ab_testing.md) · [Behavioral Guide](./behavioral-interview-guide.md)

**Build this**: a deployment pipeline that trains, gates on evaluation, registers, deploys to a canary, and rolls back automatically when metrics degrade. The rollback is the interesting part.

**You're ready when** you can explain training/serving skew and how a feature store prevents it, describe a complete rollback (model *and* feature pipeline), design promotion gates that catch a slice regression, and estimate GPU cost per thousand predictions.

---

## Track 6 — Deep Learning Engineer

**Job titles**: Deep Learning Engineer, Research Engineer, Applied Scientist
**Prerequisites**: linear algebra, calculus, probability. This track is the most mathematically demanding.
**Realistic time to interview-ready**: 4–5 months, part-time

### Stage 1 — Fundamentals (4 weeks)
- [Deep Learning Overview](../deep_learning/README.md)
- [Neural Network Training](../deep_learning/intro_neural_network_training.md) — read this twice
- [PyTorch](../frameworks/intro_pytorch.md)
- [ML Coding Challenges](../coding_challenges/ml_coding_challenges.md) — implement backprop and attention from scratch

### Stage 2 — Architectures (4 weeks)
- [Sequence Models](../deep_learning/intro_sequence_models.md) — why gating works, and what attention replaced
- [Transformers](../deep_learning/intro_transformers.md) — the central topic
- [Computer Vision](../deep_learning/intro_computer_vision.md)
- [Applied Deep Learning](../deep_learning/intro_applied_deep_learning.md)

### Stage 3 — Modern practice (3 weeks)
- [Fine-Tuning](../deep_learning/intro_fine_tuning.md) — LoRA, QLoRA, PEFT, DPO
- [HuggingFace](../frameworks/intro_huggingface.md) · [Unsloth](../frameworks/intro_unsloth.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md) · [vLLM](../frameworks/intro_vllm.md)

### Stage 4 — Interview prep (2 weeks)
- [ML System Design](../system_design/README.md) · [2026 Interview Questions](./interview_questions_2026.md)
- [Behavioral Guide](./behavioral-interview-guide.md)

**Build this**: a transformer implemented from scratch, then a fine-tune of an open model on a domain dataset with a proper evaluation and a documented ablation.

**You're ready when** you can derive attention including why scores are scaled by `√d_k`, diagnose a NaN loss in order of likelihood, explain what to do when a model won't fit in GPU memory in escalating order, and describe LoRA's mechanism and memory savings.

---

## Track Comparison

| | ML Engineer | AI / GenAI | Data Scientist | Data Engineer | MLOps | Deep Learning |
|---|---|---|---|---|---|---|
| **Maths required** | Medium | Low | High | Low | Low | **High** |
| **Coding required** | High | High | Medium | High | **High** | High |
| **Time to job-ready** | 3–4 mo | **2–3 mo** | 2–3 mo | 2–3 mo | 2–3 mo | 4–5 mo |
| **Fastest for a SWE** | — | **Yes** | — | — | **Yes** | — |
| **Guides in the track** | ~20 | ~22 | ~13 | ~16 | ~18 | ~14 |
| **Core interview round** | System design | Evaluation | Statistics & experiments | Data modeling | Reliability & rollback | Architecture depth |
| **Least likely to go stale** | **High** | Medium | **High** | High | Medium | High |

---

## How to Actually Study This

**Follow the stages in order.** They are ordered by dependency, not by interest. Skipping ahead to agents before understanding evaluation is the most common way people stall.

**Build after every stage.** Reading a guide creates recognition, not recall. A small project after each stage converts one into the other, and it gives you the material for the [project deep-dive round](./behavioral-interview-guide.md#the-project-deep-dive), which decides more interviews than the technical rounds do.

**Answer out loud.** Every guide ends with an Interview Q&A section. Read the question, answer it aloud before reading on, and notice the gap. Silent reading consistently overestimates how well you know something.

**One track at a time.** Two half-finished tracks interview worse than one finished one.

**Depth beats breadth in interviews.** One project you can discuss five layers deep beats five tutorials. Interviewers probe until they find your limit; the goal is for that limit to be deep, not wide.

When you're within a few weeks of interviewing, switch to the [2026 Interview Roadmap](./2026-interview-roadmap.md) — it has the role-by-role focus matrix and study plans for 1 week, 1 month, and 3 months.

---

## Frequently Asked

**Do I need a PhD?** No, for every track here except some research positions. PhDs matter for research scientist roles at frontier labs. ML Engineer, AI Engineer, Data Engineer, and MLOps roles are engineering roles that hire on demonstrated ability.

**How much maths do I actually need?** For AI/GenAI, Data Engineering, and MLOps: very little beyond comfort with probability and reading a formula. For ML Engineer: linear algebra and probability at an undergraduate level. For Data Science: real statistical depth, since it is the interview. For Deep Learning: linear algebra, calculus, and probability, genuinely.

**Can I switch tracks later?** Yes, and it gets easier each time — the [Common Core](#the-common-core) is shared, and adjacent tracks overlap heavily. ML Engineer ↔ Deep Learning and AI Engineer ↔ MLOps are the easiest moves.

**Do I have to read every guide in my track?** No. The stages are the path; the rest of the directory is reference. Use the "Guides in This Track" index in each directory's README when you need something specific.

**How long does this really take?** The estimates assume 10–15 hours a week and existing programming ability. Full-time roughly halves them. Starting from no programming background, add 2–3 months for Python first.

**Is the AI/GenAI track a shortcut?** It is faster to a first job, and that is real. It is also a shallower foundation — if the tooling shifts, ML Engineer fundamentals transfer and prompt-level knowledge does not. Many people start with AI/GenAI for the job and backfill fundamentals afterwards, which is a perfectly reasonable order.

**I already work in ML. Where do I start?** Skip to the stage covering what you don't do daily. Most working MLEs have gaps in evaluation, monitoring, and system design rather than in modeling.

---

## Related Topics

- [2026 Interview Roadmap](./2026-interview-roadmap.md) — role focus matrix and time-boxed study plans
- [Study Pattern](./study-pattern.md) — topics by difficulty level
- [Behavioral and Project Deep-Dive Guide](./behavioral-interview-guide.md)
- [ML Take-Home Projects](./take-home-projects.md)
- [2026 Additional Questions](./2026-additional-questions.md) · [2026 Interview Questions](./interview_questions_2026.md)
- [Resources and References](./resources-and-references.md)
- [Repository Home](../README.md)
