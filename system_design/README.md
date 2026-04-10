# ML System Design

A guide to approaching ML system design interview questions with structured frameworks and full design examples.

---

## Table of Contents

1. [Framework for ML System Design](#framework-for-ml-system-design)
2. [Backend Interview Guide](#backend-interview-guide)
3. [Key Components](#key-components)
4. [Scalability Considerations](#scalability-considerations)
5. [System Design Q&A](#system-design-qa)
6. [References](#references)

---

## Framework for ML System Design

When approaching an ML system design question, follow this structured framework:

### Step 1: Clarify Requirements (5 minutes)

Ask these questions:
- What is the business goal and success metric?
- What is the scale? (users/day, QPS, data volume)
- What is the acceptable latency?
- Is this real-time or batch?
- What data is available?
- Are there regulatory/compliance requirements?

### Step 2: Define the ML Problem (5 minutes)

- Convert business goal → ML objective
- What is the input? What is the output?
- Is this classification, regression, ranking, generation?
- What is the training signal (labels)?
- Offline vs online evaluation

### Step 3: Data Pipeline (10 minutes)

- Data sources and collection
- Data storage and format
- Feature engineering
- Training/validation/test splits
- Handling class imbalance, missing data

### Step 4: Model Selection (5 minutes)

- Start simple (baseline), then go complex
- Justify model choice based on data size, latency, interpretability
- Consider ensemble approaches

### Step 5: Serving Architecture (10 minutes)

- Real-time (API) vs batch inference
- Model serving infrastructure
- Latency optimization (caching, model compression)
- A/B testing and canary deployment

### Step 6: Monitoring and Maintenance (5 minutes)

- What metrics to monitor?
- When to retrain?
- How to handle concept/data drift?

---

## Backend Interview Guide

If you want the broader non-ML backend concepts that often get mixed into ML system design interviews, use this companion page:

- [Backend System Design Interview Guide](./backend_system_design_interview_guide.md)

It adds a structured review of:
- scalability primitives such as load balancing, auto-scaling, and rate limiting
- data-system fundamentals such as indexing, replication, and sharding
- caching, messaging, and distributed transaction patterns
- reliability topics such as circuit breakers, idempotency, timeouts, and quorum

Use this page for the ML-specific framework and design examples, then use the backend guide as the general system design revision sheet.

---

## Key Components

### Data Pipeline

```
Raw Data Sources:
├── Transactional DB (PostgreSQL, MySQL)
├── Event streams (Kafka)
├── Data warehouse (Snowflake, BigQuery)
└── Third-party APIs

ETL Pipeline:
├── Ingestion (Spark, Airflow)
├── Transformation (dbt, Spark)
├── Validation (Great Expectations)
└── Storage (S3/GCS parquet, Delta Lake)

Feature Pipeline:
├── Batch features → Offline Feature Store
├── Streaming features → Online Feature Store (Redis)
└── Feature Registry (Feast/Tecton)
```

### Model Training

```
Experiment Tracking:
├── MLflow / W&B for logging parameters, metrics
├── DVC for data versioning
└── Hyperparameter optimization (Optuna, Ray Tune)

Training Infrastructure:
├── Single GPU / Multi-GPU
├── Distributed training (PyTorch DDP, DeepSpeed)
└── Cloud (AWS SageMaker, GCP Vertex AI, Azure ML)

Model Registry:
├── Versioned artifact storage
├── Staging → Production lifecycle
└── Rollback capability
```

### Model Serving

```
Inference Options:
├── REST API (FastAPI + Docker + Kubernetes)
├── gRPC (lower latency, binary protocol)
├── Batch prediction jobs (Spark, Airflow)
└── Streaming inference (Kafka + model)

Serving Frameworks:
├── Triton Inference Server (NVIDIA, multi-framework)
├── TorchServe (PyTorch)
├── TF Serving (TensorFlow)
├── BentoML / Seldon (framework-agnostic)
└── KServe (Kubernetes-native)

Optimization:
├── Model quantization (INT8, FP16)
├── ONNX export for cross-platform inference
├── TensorRT for NVIDIA GPU optimization
└── Response caching for repeated queries
```

### Monitoring

```
Infrastructure Metrics:
├── Latency (p50, p95, p99)
├── Throughput (RPS)
├── Error rate
└── GPU/CPU utilization

Data/Model Metrics:
├── Feature drift (PSI, KS test)
├── Prediction distribution shift
├── Performance metrics (if labels available)
└── Business KPIs

Tools:
├── Grafana + Prometheus (infrastructure)
├── Evidently AI (drift detection)
├── Arize AI / WhyLogs (model observability)
└── PagerDuty / OpsGenie (alerting)
```

---

## Scalability Considerations

### Read-Heavy Systems

- Cache feature lookups (Redis)
- Replicate model serving behind load balancer
- Pre-compute predictions for common inputs

### Write-Heavy Systems

- Async logging and feedback collection
- Batch writes to storage
- Event-driven architecture (Kafka)

### High-Throughput Inference

- Horizontal scaling (multiple model replicas)
- Request batching (batch multiple requests per inference)
- Model distillation (smaller, faster model)
- GPU inference servers (vLLM, Triton)

### Low-Latency Inference

- Online store for features (Redis, DynamoDB)
- Model warm-up to avoid cold starts
- Edge deployment for on-device inference
- Quantized models (FP16, INT8)

---

## System Design Q&A

**Q1: Design a movie recommendation system.** 🔴 Advanced

See detailed design: [Recommendation System](./recommendation_system.md)

**Summary:**
- Candidate generation (two-tower model): narrow 10M movies to 1000
- Ranking model (wide & deep or gradient boosting): score 1000 candidates
- Re-ranking: apply business rules (diversity, promotions, freshness)
- Real-time features: user's last 5 clicks, current session
- A/B test new algorithms continuously

---

**Q2: Design a fraud detection system.** 🔴 Advanced

See detailed design: [Fraud Detection System](./fraud_detection.md)

**Summary:**
- Real-time scoring (< 50ms) on every transaction
- Features: user history, device fingerprint, merchant risk, velocity
- Model: gradient boosting + rules engine
- Extreme class imbalance: 0.1% fraud rate
- Monitoring: alert rate, false positive rate, fraud amount caught

---

**Q3: Design an ML pipeline for a spam email classifier.** 🟡 Intermediate

**Problem definition:**
- Input: Email (subject, body, sender, metadata)
- Output: Binary classification (spam/not spam)
- Goal: High precision (don't block legitimate email), reasonable recall

**Data:**
- Labeled emails from user feedback (marked as spam)
- Email headers, content, sender reputation

**Features:**
- TF-IDF of subject/body
- Sender domain reputation
- Email header features (SPF, DKIM pass/fail)
- User-sender relationship
- Reply/engagement history

**Models:**
- Baseline: Logistic Regression on TF-IDF
- Production: Gradient Boosting + text embeddings
- Optional: Fine-tuned BERT for semantic spam detection

**Serving:**
- < 100ms latency for real-time email filtering
- Pre-compute sender reputation features

**Monitoring:**
- False positive rate (users marking legitimate email as not spam)
- New spam patterns (concept drift)
- Adversarial adaptation by spammers

---

**Q4: Design a search ranking system.** 🔴 Advanced

**Problem definition:**
- Input: User query, candidate documents, user context
- Output: Ranked list of documents

**Learning to Rank paradigm:**
- Pointwise: Predict relevance score independently per doc
- Pairwise: Predict which of two docs is more relevant
- Listwise: Optimize directly on ranking metrics (NDCG, MAP)

**Features:**
- Query features: query length, query type (informational vs navigational)
- Document features: PageRank, freshness, content quality
- Query-document features: BM25, embedding similarity, click history
- User features: search history, location, device

**Model:**
- Candidate retrieval: dense retrieval (bi-encoder, ANN search)
- Ranking: LambdaMART (gradient boosting for ranking) or LambdaRank
- Re-ranking: BERT cross-encoder for top-100 candidates

**Evaluation:**
- Offline: NDCG@10, MAP
- Online: Click-through rate, session success rate (user found what they needed)

---

**Q5: Design a real-time content moderation system.** 🔴 Advanced

**Requirements:**
- Detect harmful content (hate speech, NSFW, misinformation)
- Scale: 10M posts per day
- Latency: < 200ms for real-time decisions

**Architecture:**
- **Fast path (< 50ms):** Rule-based filters + lightweight ML model (DistilBERT/quantized)
- **Slow path (async):** Heavy model (full BERT fine-tuned) + human review queue

**Models:**
- Multi-label classification (a post can be multiple violation types)
- Binary classifiers per category (hate, NSFW, spam, misinformation)
- Ensemble of text + image models for multimedia content

**Human-in-the-loop:**
- Low-confidence predictions → human review queue
- Human labels → training data for model improvement
- Appeals process for incorrect decisions

**Monitoring:**
- False positive rate (legitimate content blocked)
- False negative rate (harmful content allowed through)
- Model latency percentiles
- Human review queue length

---

**Q6: How would you A/B test a new recommendation model?** 🟡 Intermediate

**Setup:**
1. Define the primary metric (CTR, conversion, session time)
2. Define guardrail metrics (things that must not degrade)
3. Compute minimum detectable effect (MDE) to size the experiment
4. Randomly assign users to control (old model) and treatment (new model)
5. Ensure consistent assignment (same user always gets same variant)

**Statistical rigor:**
- Use t-test or z-test for continuous metrics
- Use Fisher's exact or Chi-squared for binary metrics
- Set significance level α = 0.05 and power 1-β = 0.80
- Run for minimum 1-2 full business cycles (avoid weekday effects)
- Correct for multiple comparisons if testing multiple metrics

**Pitfalls:**
- **Novelty effect:** Users interact more with new things
- **Peeking problem:** Don't stop early because p < 0.05 at one point
- **Network effects:** In social systems, users in control can be influenced by treatment
- **Sample ratio mismatch:** Verify actual split matches intended split

---

**Q7: Design a time-series anomaly detection system.** 🟡 Intermediate

**Use case:** Detect anomalies in server metrics, financial transactions, IoT sensors

**Approaches:**
- **Statistical:** Z-score, IQR, ARIMA residuals — works for stationary data
- **ML-based:** Isolation Forest, LOF for multivariate tabular data
- **Deep learning:** LSTM Autoencoder — learns normal patterns, flags high reconstruction error
- **Forecasting-based:** Predict expected value, flag when actual > threshold×predicted

**Pipeline:**
- Rolling window features: mean, std, min, max over last N periods
- Seasonal decomposition (STL) to separate trend, seasonality, residual
- Threshold on residuals using dynamic baselines

**Production considerations:**
- Online vs batch detection (near-real-time requires streaming pipeline)
- Alert deduplication (correlate alerts from related metrics)
- Seasonality handling (different baselines for weekday vs weekend)
- Adaptive thresholds that update as system behavior evolves

---

**Q8: How would you handle the cold start problem in recommendations?** 🟡 Intermediate

**Cold start types:**
- **New user:** No interaction history available
- **New item:** No engagement data for this content

**Strategies for new users:**
- Onboarding flow to collect explicit preferences (genres, interests)
- Demographic-based recommendations (age, location, device)
- Popularity-based recommendations as fallback
- Collaborative filtering with similar users (based on sparse signals)
- Quick learning from early interactions (contextual bandits)

**Strategies for new items:**
- Content-based filtering using item features (genre, author, metadata)
- Two-tower model where item tower uses only content features
- Explore-exploit (epsilon-greedy or UCB): show new items to some users to gather signals
- Transfer learning from similar existing items

---

**Q9: What is the trade-off between model complexity and latency in production?** 🟡 Intermediate

Latency directly impacts user experience (100ms slowdown = 1% drop in conversion). Model complexity trade-offs:

| Factor | Simple Model | Complex Model |
|--------|-------------|---------------|
| Latency | < 10ms | 100ms-1s |
| Memory | Low | High |
| Accuracy | Lower | Higher |
| Debugging | Easy | Hard |
| Serving cost | Low | High |

**Strategies to manage the trade-off:**
- Model distillation: Train large model, distill into small fast model
- Cascading: Use fast model first, escalate hard cases to slow model
- Quantization: FP32 → INT8 reduces latency 2-4x
- Async enrichment: Return initial result, update asynchronously with better prediction
- Request batching: Batch multiple requests for GPU efficiency

---

**Q10: How do you ensure reproducibility in ML pipelines?** 🟡 Intermediate

1. **Code versioning:** Git tags for every production training run
2. **Data versioning:** DVC or Delta Lake to snapshot training datasets
3. **Environment reproducibility:** Docker containers with pinned dependency versions
4. **Seed management:** Set random seeds for data splitting, model initialization, sampling
5. **Parameter logging:** Log all hyperparameters (MLflow, W&B)
6. **Artifact storage:** Store model files, tokenizers, preprocessors with the run
7. **Pipeline automation:** Airflow/Kubeflow DAGs for reproducible training runs
8. **Testing:** Unit tests for feature computation, regression tests for model outputs

---

## References

- [Designing Machine Learning Systems — Chip Huyen (O'Reilly)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Machine Learning System Design Interview — Alex Xu, Sahn Lam](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127)
- [ML Systems Design at Recsys, KDD, and MLSys conferences](https://proceedings.mlsys.org/)
- [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Meta's AI Infrastructure blog](https://engineering.fb.com/category/ml-applications/)
