# Study Pattern & Preparation Guide

This guide provides a structured study plan for ML, AI Engineer, and Data Engineer interview preparation, with difficulty levels and time estimates.

---

## Difficulty Legend

| Level | Label | Description |
|-------|-------|-------------|
| 🟢 Beginner | Entry-level | Conceptual understanding; expected from all candidates |
| 🟡 Intermediate | Mid-level | Applied knowledge; expected for 2–5 YOE roles |
| 🔴 Advanced | Senior-level | Deep technical; expected for senior/staff roles |

---

## Track 1: Classic ML Foundations

**Recommended for:** All ML/AI/DS roles
**Estimated prep time:** 2–3 weeks

| Topic | Difficulty | Key Questions to Master |
|-------|-----------|------------------------|
| Supervised vs Unsupervised Learning | 🟢 | Differences, use cases, examples |
| Bias-Variance Tradeoff | 🟢 | What each means, how to fix each |
| Overfitting & Regularization (L1/L2) | 🟡 | When to use Ridge vs Lasso |
| Gradient Descent (GD, SGD, Mini-batch) | 🟡 | Differences, convergence, learning rate |
| Linear & Logistic Regression | 🟢 | Assumptions, cost functions, interpretation |
| Decision Trees & Random Forests | 🟡 | Gini vs Entropy, bagging, feature importance |
| SVM | 🟡 | Kernel trick, margin, C parameter |
| Evaluation Metrics (Precision, Recall, F1, AUC-ROC) | 🟢 | When to use each, business context |
| K-Nearest Neighbors | 🟢 | Distance metrics, curse of dimensionality |
| Naive Bayes | 🟡 | Assumptions, Bayes theorem, applications |
| Boosting (XGBoost, AdaBoost) | 🟡 | Bagging vs Boosting, gradient boosting |
| Clustering (K-Means, DBSCAN) | 🟡 | Choosing k, inertia, elbow method |
| Dimensionality Reduction (PCA, t-SNE, UMAP) | 🟡 | Variance explained, visualization |
| Recommender Systems | 🔴 | Collaborative filtering, matrix factorization |
| Time Series (ARIMA, Prophet) | 🔴 | Stationarity, seasonality, forecasting |

**Prerequisites:** Python basics, linear algebra fundamentals

---

## Track 2: Deep Learning & Neural Networks

**Recommended for:** ML Engineer, DL Engineer, AI Engineer
**Estimated prep time:** 2 weeks

| Topic | Difficulty | Key Questions to Master |
|-------|-----------|------------------------|
| Neural Network Architecture | 🟢 | Layers, activations, forward/backprop |
| Activation Functions (ReLU, Sigmoid, Softmax) | 🟢 | When to use each, vanishing gradient |
| Batch Normalization | 🟡 | Why it works, training vs inference |
| CNNs | 🟡 | Convolution, pooling, receptive field |
| RNNs & LSTMs | 🟡 | Sequence modeling, gating mechanism |
| Attention Mechanism & Transformers | 🔴 | Self-attention, multi-head, positional encoding |
| Transfer Learning & Fine-tuning | 🟡 | When to fine-tune vs train from scratch |
| Regularization (Dropout, Weight Decay) | 🟡 | Preventing overfitting in deep nets |
| Optimization (Adam, AdaGrad, RMSProp) | 🟡 | Adaptive learning rate methods |

**Prerequisites:** Classic ML Track, calculus, linear algebra

---

## Track 3: AI / GenAI Engineering

**Recommended for:** AI Engineer, GenAI Engineer, LLM Engineer
**Estimated prep time:** 3–4 weeks

| Topic | Difficulty | Key Questions to Master | Guide |
|-------|-----------|------------------------|-------|
| RAG Architecture | 🟡 | Indexing, retrieval, generation pipeline | [RAG Guide](../ai_genai/intro_rag.md) |
| RAG Evaluation (RAGAS) | 🔴 | Context precision/recall, faithfulness, answer relevance | [RAG Guide](../ai_genai/intro_rag.md) |
| Vector Databases | 🟡 | ANN algorithms, HNSW, choosing a vector DB | [Vector DBs Guide](../ai_genai/intro_vector_databases.md) |
| Embedding Models | 🟡 | Dense vs sparse, embedding drift | [Vector DBs Guide](../ai_genai/intro_vector_databases.md) |
| LLMOps | 🔴 | Tracing, evals, cost tracking, deployment | [LLMOps Guide](../ai_genai/intro_llmops.md) |
| Agentic AI | 🔴 | ReAct, Plan-Execute, multi-agent systems | [Agentic AI Guide](../ai_genai/intro_agentic_ai.md) |
| MCP (Model Context Protocol) | 🔴 | Tool servers, client-server architecture | [MCP Guide](../ai_genai/intro_mcp.md) |
| LangChain / LangGraph | 🟡 | LCEL, chains, agents, memory | [LangChain Guide](../ai_genai/intro_langchain.md) |
| Anthropic / Claude API | 🟡 | Tool use, caching, extended thinking | [Anthropic Guide](../ai_genai/intro_anthropic.md) |
| Fine-tuning vs RAG | 🔴 | When to use each, tradeoffs | [2026 Questions](./2026-additional-questions.md) |

**Prerequisites:** Python, REST APIs, basic LLM familiarity

---

## Track 4: MLOps & Production ML

**Recommended for:** MLOps Engineer, Senior ML Engineer
**Estimated prep time:** 2–3 weeks

| Topic | Difficulty | Key Questions to Master | Guide |
|-------|-----------|------------------------|-------|
| MLflow | 🟡 | Experiment tracking, model registry | [MLflow Guide](../mlops/intro_mlflow.md) |
| Feature Stores | 🔴 | Online vs offline, point-in-time correctness | [Feature Stores Guide](../mlops/intro_feature_stores.md) |
| Model Serving | 🔴 | Latency vs throughput, batching, scaling | [Model Serving Guide](../mlops/intro_model_serving.md) |
| Model Explainability (SHAP, LIME) | 🟡 | Global vs local explanations | [Explainability Guide](../mlops/intro_model_explainability.md) |
| Data Quality & Validation | 🟡 | Data contracts, drift detection | [Data Quality Guide](../mlops/intro_data_quality.md) |
| Model Monitoring | 🔴 | Data drift, concept drift, alerting | [LLMOps Guide](../ai_genai/intro_llmops.md) |

**Prerequisites:** Classic ML Track, Python, Docker basics

---

## Track 5: Data Engineering

**Recommended for:** Data Engineer, Analytics Engineer, Platform Engineer
**Estimated prep time:** 3–4 weeks

| Topic | Difficulty | Key Questions to Master | Guide |
|-------|-----------|------------------------|-------|
| Apache Spark | 🟡 | RDDs vs DataFrames, partitioning, joins | [Spark Guide](../data_engineering/intro_apache_spark.md) |
| Apache Kafka | 🟡 | Topics, partitions, consumer groups, exactly-once | [Kafka Guide](../data_engineering/intro_apache_kafka.md) |
| Apache Airflow | 🟡 | DAGs, operators, XComs, sensors | [Airflow Guide](../data_engineering/intro_apache_airflow.md) |
| dbt | 🟡 | Models, tests, macros, incremental models | [dbt Guide](../data_engineering/intro_dbt.md) |
| Apache Iceberg | 🔴 | Time travel, schema evolution, merge-on-read | [Iceberg Guide](../data_engineering/intro_apache_iceberg.md) |
| Delta Lake | 🟡 | ACID transactions, Z-ordering, CDC | [Delta Lake Guide](../data_engineering/intro_delta_lake.md) |
| DuckDB | 🟢 | Columnar analytics, use cases vs Spark | [DuckDB Guide](../data_engineering/intro_duckdb.md) |

**Prerequisites:** SQL proficiency, Python, basic cloud knowledge

---

## Track 6: DevOps & Infrastructure

**Recommended for:** MLOps Engineer, Platform Engineer, DevOps Engineer
**Estimated prep time:** 2–3 weeks

| Topic | Difficulty | Key Questions to Master | Guide |
|-------|-----------|------------------------|-------|
| Docker | 🟢 | Images, containers, networking, Dockerfile | [Docker Guide](../devops/intro_docker.md) |
| Kubernetes | 🟡 | Pods, deployments, services, HPA | [K8s Guide](../devops/intro_kubernetes.md) |
| Helm | 🟡 | Charts, values, templating | [Helm Guide](../devops/intro_helm.md) |
| Terraform | 🟡 | State, modules, plan/apply | [Terraform Guide](../devops/intro_terraform.md) |
| GitHub Actions | 🟢 | CI/CD workflows, secrets, matrix builds | [GitHub Actions Guide](../devops/intro_github_actions.md) |

**Prerequisites:** Linux basics, cloud platform familiarity

---

## Recommended Study Sequence

### For ML Engineer (General)
1. Classic ML Foundations (Track 1) → 2 weeks
2. Deep Learning (Track 2) → 2 weeks
3. MLOps & Production (Track 4) → 1 week
4. DevOps basics (Track 6: Docker, K8s) → 1 week

### For AI / LLM Engineer
1. Classic ML Foundations (Track 1) → 1 week (skim)
2. Deep Learning — Transformers/Attention (Track 2) → 1 week
3. GenAI Engineering (Track 3) → 3 weeks
4. LLMOps (overlap with Track 4) → 1 week

### For Data Engineer
1. SQL & Python proficiency (prerequisite)
2. Data Engineering tools (Track 5) → 3 weeks
3. DevOps basics (Track 6) → 1 week
4. Classic ML overview (Track 1) → 1 week (skim)

### For MLOps Engineer
1. Classic ML (Track 1) → 1 week
2. MLOps & Production (Track 4) → 2 weeks
3. Data Engineering (Track 5) → 2 weeks
4. DevOps (Track 6) → 2 weeks

---

## Computer Science Fundamentals

Essential for all roles:

### Data Structures & Algorithms
- Data structures: Lists, stacks, queues, strings, hash maps, vectors, matrices, classes/objects, trees, graphs
- Algorithms: Recursion, searching, sorting, optimization, dynamic programming
- Complexity: P vs. NP, big-O notation, approximate algorithms
- Computer architecture: Memory, cache, bandwidth, threads/processes, deadlocks

### Probability and Statistics
- Basic probability: Conditional probability, Bayes rule, likelihood, independence
- Probabilistic models: Bayes Nets, Markov Decision Processes, Hidden Markov Models
- Statistical measures: Mean, median, mode, variance, population vs. sample statistics
- Proximity and error metrics: Cosine similarity, MSE, Manhattan/Euclidean distance, log-loss
- Distributions: Uniform, normal, binomial, Poisson
- Analysis methods: ANOVA, hypothesis testing, factor analysis

### Software Engineering
- Library calls, REST APIs, data collection endpoints, database queries
- User interface: Capturing inputs, displaying results and visualizations
- Scalability: Map-reduce, distributed processing
- Deployment: Cloud hosting, containers, microservices

---

## Interview Day Tips

1. **Think out loud** — interviewers want to follow your reasoning, not just the answer
2. **Clarify before you code** — ask about constraints, edge cases, scale requirements
3. **Start simple** — give a naive/brute-force answer first, then optimize
4. **Know your tradeoffs** — every algorithm has pros and cons; be ready to discuss them
5. **Bridge theory to practice** — relate concepts to real systems (e.g., "In production, I would...")
6. **Admit uncertainty honestly** — "I'd need to verify this, but I believe..." is better than guessing confidently

---

*See [2026 Interview Roadmap](./2026-interview-roadmap.md) for the latest focus areas.*
*See [Resources and References](./resources-and-references.md) for books and external learning materials.*
