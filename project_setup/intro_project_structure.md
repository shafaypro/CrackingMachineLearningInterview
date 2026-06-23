# ML / AI Project Folder Structures (2026 Edition)

A reference for how to lay out real machine-learning, deep-learning, LLM/agent,
and data-engineering projects. A clean structure makes a project reproducible,
testable, and reviewable — and it's something interviewers probe with "how would
you organize this?"

> **Principle:** separate **code** (versioned in git), **data** (versioned with
> DVC / object storage), **configuration** (versioned, environment-overridable),
> and **artifacts** (tracked in a registry). Don't mix them.

---

## Table of Contents
1. [Universal Building Blocks](#universal-building-blocks)
2. [Classic ML Project](#classic-ml-project)
3. [Deep Learning / Training Project](#deep-learning--training-project)
4. [LLM / GenAI Application](#llm--genai-application)
5. [AI Agent Project](#ai-agent-project)
6. [Data Engineering / Pipeline Project](#data-engineering--pipeline-project)
7. [Python Packaging Layouts: `src/` vs flat](#python-packaging-layouts-src-vs-flat)
8. [Where Things Go: A Decision Table](#where-things-go-a-decision-table)
9. [Anti-Patterns](#anti-patterns)
10. [Interview Questions](#interview-questions)

---

## Universal Building Blocks

Almost every project has these, regardless of type:

```
my-project/
├── README.md               # what / why / how
├── LICENSE
├── .gitignore
├── .env.example            # template for secrets (real .env is git-ignored)
├── pyproject.toml          # deps + tool config (ruff, pytest, build)
├── requirements.txt        # or a lockfile (uv.lock / poetry.lock)
├── Makefile                # `make test`, `make train`, `make lint`
├── .pre-commit-config.yaml # lint/format before commit
├── .github/
│   ├── workflows/ci.yml    # CI pipeline
│   └── pull_request_template.md
├── src/                    # source code (importable package)
├── tests/                  # unit + integration tests
├── configs/                # YAML/Hydra configs (no hardcoded params)
├── notebooks/              # exploration only — not the source of truth
├── scripts/                # one-off / CLI entry points
└── docs/                   # documentation
```

---

## Classic ML Project

Tabular / scikit-learn style. Heavily inspired by the popular **Cookiecutter Data Science** layout.

```
ml-project/
├── data/
│   ├── raw/            # immutable original data (git-ignored, DVC-tracked)
│   ├── interim/        # intermediate transformations
│   ├── processed/      # final feature sets for modeling
│   └── external/       # third-party / reference data
├── models/             # serialized models (git-ignored → registry)
├── notebooks/
│   └── 01-eda.ipynb    # number-prefixed, ordered
├── reports/
│   └── figures/        # generated plots
├── src/
│   ├── __init__.py
│   ├── data/           # data loading & validation
│   │   ├── make_dataset.py
│   │   └── validate.py
│   ├── features/       # feature engineering
│   │   └── build_features.py
│   ├── models/         # train / predict
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── visualize.py
├── configs/
│   └── config.yaml
├── tests/
└── dvc.yaml            # data/pipeline versioning (optional but recommended)
```

**Key ideas:** `data/raw` is *immutable* — you never edit it; transformations
flow `raw → interim → processed`. Notebooks are for exploration; production logic
lives in `src/` and is imported into notebooks, not copy-pasted.

---

## Deep Learning / Training Project

PyTorch / training-loop style. Adds experiment tracking, checkpoints, and config-driven runs.

```
dl-project/
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset / DataLoader
│   │   └── transforms.py
│   ├── models/
│   │   ├── architecture.py   # nn.Module definitions
│   │   └── layers.py
│   ├── training/
│   │   ├── trainer.py        # train/eval loop
│   │   ├── losses.py
│   │   └── callbacks.py      # early stopping, checkpointing
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       ├── seed.py           # reproducibility
│       └── metrics.py
├── configs/
│   ├── model/resnet50.yaml
│   ├── data/imagenet.yaml
│   └── train.yaml            # composed with Hydra
├── experiments/              # per-run outputs (git-ignored)
│   └── 2026-06-23_resnet/
│       ├── checkpoints/
│       ├── logs/
│       └── config.yaml       # snapshot of the exact config used
├── scripts/
│   ├── train.py              # python scripts/train.py --config configs/train.yaml
│   └── evaluate.py
├── tests/
└── Dockerfile                # reproducible training env
```

**Key ideas:** every run snapshots its config so it's reproducible; checkpoints
and logs go to `experiments/` (tracked by MLflow / Weights & Biases, not git);
seeds are set centrally for reproducibility.

---

## LLM / GenAI Application

RAG / chat / generation app. Built around prompts, retrieval, and an API surface.

```
llm-app/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   └── routes/
│   ├── llm/
│   │   ├── client.py         # provider client wrapper (Anthropic, etc.)
│   │   ├── prompts/          # versioned prompt templates
│   │   │   ├── system.md
│   │   │   └── rag_answer.md
│   │   └── models.py         # model IDs / config in ONE place
│   ├── rag/
│   │   ├── ingest.py         # chunking + embedding
│   │   ├── retriever.py      # vector search + reranking
│   │   └── vectorstore.py
│   ├── eval/
│   │   ├── datasets/         # golden Q&A sets
│   │   ├── run_eval.py       # offline eval harness
│   │   └── judges.py         # LLM-as-judge / metrics
│   └── guardrails/
│       └── filters.py        # input/output validation, PII checks
├── configs/
│   └── app.yaml              # model, temperature, top_k, chunk size
├── tests/
│   ├── test_retriever.py
│   └── test_prompts.py       # prompt regression tests
├── data/
│   └── knowledge_base/       # source docs for RAG
├── .env.example              # ANTHROPIC_API_KEY=, VECTOR_DB_URL=
└── Dockerfile
```

**Key ideas:**
- **Prompts are versioned artifacts** — keep them in files (`prompts/`), not inline
  string literals scattered through code, so you can diff and regression-test them.
- **Centralize model IDs and parameters** in `models.py`/`config` so a model
  upgrade is a one-line change. Use current model IDs (e.g. `claude-opus-4-8`,
  `claude-sonnet-4-6`, `claude-haiku-4-5`) and read keys from env.
- **An `eval/` harness is not optional** — LLM apps degrade silently; offline evals
  + LLM-as-judge catch regressions. See
  [LLM Evaluation](../mlops/intro_llm_evaluation.md) and
  [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md).
- **Guardrails** separate from business logic — input validation, output schema
  enforcement, PII filtering.

---

## AI Agent Project

Tool-using / multi-step agent. Separates the agent loop, tools, memory, and orchestration.

```
agent-project/
├── src/
│   ├── agents/
│   │   ├── base.py            # shared agent scaffolding
│   │   ├── researcher.py      # a specialized agent / role
│   │   └── orchestrator.py    # routing / multi-agent coordination
│   ├── tools/                 # ONE file per tool, with a clear schema
│   │   ├── search.py
│   │   ├── code_exec.py
│   │   └── registry.py        # tool registration
│   ├── memory/
│   │   ├── short_term.py      # working context
│   │   └── long_term.py       # vector / persistent store
│   ├── prompts/
│   │   └── system_agent.md
│   ├── runtime/
│   │   ├── loop.py            # the agent loop (perceive → plan → act)
│   │   └── tracing.py         # observability / spans
│   └── config.py              # model IDs, effort/thinking, max steps
├── configs/
│   └── agent.yaml
├── evals/
│   ├── tasks/                 # task suite the agent must pass
│   └── run.py                 # success-rate / trajectory eval
├── tests/
│   └── test_tools.py          # tools are unit-testable in isolation
└── .env.example
```

**Key ideas:**
- **Tools are first-class, isolated, and testable.** Each tool = a typed schema +
  a handler. You should be able to unit-test a tool without invoking the model.
- **Separate the agent loop from tools and memory** — the loop orchestrates;
  tools execute; memory persists.
- **Evaluate agents on task success rate and trajectory**, not just final-token
  quality. Keep a task suite in `evals/`.
- **Tracing/observability is built in**, not bolted on — you need to see every
  tool call to debug. See [Agentic AI](../ai_genai/intro_agentic_ai.md),
  [Agent Tool Use](../ai_genai/intro_agent_tool_use.md), and
  [Multi-Agent Systems](../ai_genai/intro_multi_agent_systems.md).

---

## Data Engineering / Pipeline Project

ELT / orchestration style (Airflow + dbt + warehouse).

```
data-platform/
├── dags/                     # Airflow DAGs (orchestration)
│   └── daily_ingest.py
├── dbt/
│   ├── models/
│   │   ├── staging/          # 1:1 with sources, light cleaning
│   │   ├── intermediate/     # business logic / joins
│   │   └── marts/            # final, consumer-facing tables
│   ├── tests/                # dbt data tests (not_null, unique, ...)
│   └── dbt_project.yml
├── ingestion/
│   ├── extractors/           # source connectors
│   └── loaders/
├── great_expectations/       # data quality / validation suites
├── sql/
├── configs/
│   └── pipeline.yaml
├── tests/
└── docker-compose.yml        # local stack (Airflow + warehouse)
```

**Key ideas:** the **medallion / staging→intermediate→marts** layering mirrors
the raw→interim→processed flow in ML. Data quality checks (dbt tests / Great
Expectations) gate the pipeline. See
[Data Architecture](../data_engineering/data-architecture.md),
[Apache Airflow](../data_engineering/intro_apache_airflow.md), and
[dbt](../data_engineering/intro_dbt.md).

---

## Python Packaging Layouts: `src/` vs flat

| Layout | Looks like | Pros | Cons |
|--------|-----------|------|------|
| **`src/` layout** | `src/mypkg/...` | Forces installation to import → catches packaging bugs; no accidental imports from cwd | One extra directory level |
| **Flat layout** | `mypkg/...` at root | Slightly simpler | Easy to "accidentally work" without installing the package |

**Recommendation:** use the **`src/` layout** for anything shippable. Install in
editable mode for development:

```bash
pip install -e .          # reads pyproject.toml, installs your package editable
```

---

## Where Things Go: A Decision Table

| Thing | Where | Tracked by |
|-------|-------|-----------|
| Source code | `src/` | git |
| Tests | `tests/` | git |
| Hyperparameters / settings | `configs/*.yaml` | git |
| Secrets / API keys | `.env` (local), CI secrets | **not git** |
| Raw data | `data/raw/` | DVC / object storage |
| Trained models | registry (MLflow/S3) | **not git** |
| Experiment logs/metrics | MLflow / W&B | tracking server |
| Prompts | `prompts/*.md` | git (versioned!) |
| Notebooks | `notebooks/` | git (clear outputs first) |
| Generated reports/figures | `reports/` | usually **not git** |

---

## Anti-Patterns

| Anti-pattern | Why it hurts | Do instead |
|--------------|--------------|-----------|
| All code in one giant notebook | Untestable, unreviewable, not reproducible | Logic in `src/`, import into notebooks |
| Hardcoded paths / params / model IDs | Breaks on other machines; upgrades are find-and-replace | Centralize in `configs/` + `.env` |
| Committing data & model binaries to git | Bloats repo, slow clones, history pollution | `.gitignore` + DVC / registry |
| Secrets in code | Leak risk; permanent in history | `.env` + CI secrets |
| `utils.py` dumping ground | Becomes an unmaintainable junk drawer | Group by responsibility (`data/`, `features/`, ...) |
| No `tests/` | Silent breakage, scary refactors | Even a few smoke tests pay off |
| Prompts as inline strings everywhere | Can't diff, version, or regression-test | `prompts/` files + prompt tests |

---

## Interview Questions

1. **How do you structure an ML repo for reproducibility?** → Separate code/
   data/config/artifacts; immutable raw data; config-driven runs; pinned deps;
   seeds set; experiments tracked.
2. **Where do trained models and datasets belong — and why not git?** → A registry
   / object storage / DVC; git is optimized for diffable text, not large binaries.
3. **`src/` layout vs flat — which and why?** → `src/` for shippable packages: it
   forces proper installation and prevents accidental cwd imports.
4. **How do you version prompts in an LLM app?** → As files under `prompts/`,
   in git, with regression tests — treat them like code.
5. **What makes a tool "well-structured" in an agent project?** → A typed schema +
   an isolated handler that's unit-testable without the model in the loop.
6. **How do notebooks fit into a production project?** → Exploration and reporting
   only; production logic lives in importable modules under `src/`.

---

### Related Guides
- [How to Set Up a Project on GitHub](./intro_github_project_setup.md)
- [Agentic AI](../ai_genai/intro_agentic_ai.md) · [RAG Engineering](../ai_genai/intro_rag_engineering.md)
- [MLflow](../mlops/intro_mlflow.md) · [Feature Stores](../mlops/intro_feature_stores.md)
- [Data Architecture](../data_engineering/data-architecture.md)
