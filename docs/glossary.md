# ML and AI Glossary

Every term you'll hit in this repository or an interview, defined in one or two sentences with the practical point attached — not the textbook definition, but what it means when someone says it in a design review.

Terms link to the guide that covers them in depth. Use `Ctrl+F`.

**Jump to:** [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [Z](#z)

---

## A

**A/B test** — Randomized experiment splitting users between variants to measure causal effect on a business metric. See [A/B Testing](../mlops/intro_ab_testing.md).

**Ablation** — Removing one component to measure its contribution. The standard way to prove a part of your system earns its place.

**Activation function** — Nonlinearity applied after a linear layer (ReLU, GELU, SiLU). Without one, stacked layers collapse to a single linear transform.

**AdamW** — Adam with *decoupled* weight decay. The default optimizer for transformers; ordinary Adam applies L2 through the adaptive denominator, which distorts it. See [NN Training](../deep_learning/intro_neural_network_training.md).

**Agent** — An LLM in a loop with tools, deciding actions until a goal is met. See [Agentic AI](../ai_genai/intro_agentic_ai.md).

**ANN (approximate nearest neighbour)** — Sublinear vector search trading exact recall for speed. HNSW, IVF-PQ, ScaNN. See [Vector Databases](../ai_genai/intro_vector_databases.md).

**ATE / ATT / CATE** — Average treatment effect over everyone / over the treated / conditional on features. Naming the right one is half of a causal answer. See [Causal Inference](../classical_ml/intro_causal_inference.md).

**Attention** — Weighted lookup where each position attends to others by learned relevance. Scaled by `√d_k` to keep softmax out of saturation. See [Transformers](../deep_learning/intro_transformers.md).

**Autoencoder** — Encoder-decoder trained to reconstruct its input through a bottleneck. Not generative on its own — the latent space has no known distribution.

**AUC** — Area under a curve; usually ROC-AUC. Prevalence-invariant, which makes it misleading on rare events. See [Model Evaluation](../classical_ml/intro_model_evaluation.md).

---

## B

**Backpropagation** — Chain rule over the computation graph to get gradients. Products of many Jacobians are why gradients vanish or explode.

**Bagging** — Training models on bootstrap resamples and averaging. Reduces **variance**; needs low-bias base learners. See [Ensemble Methods](../classical_ml/intro_ensemble_methods.md).

**Batch normalization** — Normalizes across the batch dimension. Different train/eval behaviour, which is why forgetting `model.eval()` breaks predictions.

**Beam search** — Keeps the `k` best partial sequences during decoding. Needs length normalization or it favours short outputs.

**BERT** — Encoder-only transformer trained with masked language modelling. Good for classification and retrieval, not generation.

**Bias-variance tradeoff** — Error decomposes into bias² + variance + irreducible noise. More capacity lowers bias and raises variance.

**BM25** — Lexical ranking function over term frequency and inverse document frequency. Still the strongest baseline for exact-term matching, and half of every hybrid retrieval system.

**Boosting** — Sequential ensembling where each model fits the previous ensemble's errors. Reduces **bias**; will overfit without early stopping.

**Broadcasting** — NumPy's rule for operating on arrays of different shapes without copying. See [Pandas and NumPy](../coding_challenges/pandas_numpy_challenges.md).

---

## C

**Calibration** — Whether predicted probabilities match observed frequencies. Independent of ranking quality — a model can rank perfectly and be badly calibrated.

**Canary deployment** — Routing a small traffic percentage to a new version with automated rollback. See [CI/CD for ML](../mlops/intro_cicd_for_ml.md).

**Cardinality (metrics)** — Number of distinct label combinations. Series count is the *product* of label value counts, which is how monitoring systems get killed. See [Observability](../devops/intro_observability.md).

**Chunking** — Splitting documents for embedding and retrieval. Chunk size is a retrieval-quality decision, not preprocessing trivia. See [Embeddings](../ai_genai/intro_embeddings.md).

**Classifier-free guidance (CFG)** — Extrapolating away from the unconditional prediction to strengthen prompt adherence in diffusion. Costs two forward passes per step. See [Generative Models](../deep_learning/intro_generative_models.md).

**Collider** — A common *effect* of two variables. Conditioning on one **creates** spurious association — which is why "control for everything" is wrong advice.

**Confounder** — A common *cause* of treatment and outcome. This is the bias you must adjust for.

**Context engineering** — Deciding what enters the context window, in what order, at what cost. The dominant design activity in LLM applications. See [Context Engineering](../ai_genai/intro_context_engineering.md).

**Continuous batching** — Evicting finished sequences and admitting new ones every decode step. The single largest throughput win in LLM serving. See [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md).

**Cross-encoder** — Jointly encodes query and document for a relevance score. Far more accurate than a bi-encoder, far too slow for retrieval — hence reranking.

**CUPED** — Variance reduction using a pre-experiment covariate. Typically 30–50% less variance, meaning shorter experiments for free.

---

## D

**DAG (causal)** — Directed graph of causal relationships; tells you what to adjust for and, crucially, what not to.

**Data drift** — Input distribution shifts over time. Distinct from **concept drift**, where the input-output relationship itself changes.

**Data leakage** — Training on information unavailable at prediction time. The most common cause of "great offline, useless in production". See [Model Evaluation](../classical_ml/intro_model_evaluation.md).

**DDIM** — Deterministic diffusion sampler that skips steps, enabling 20–50 step generation and reproducible outputs.

**Diffusion model** — Generative model that learns to reverse a gradual noising process. Stable to train and excellent at coverage; slow to sample.

**Distillation** — Training a small student to match a large teacher's output distribution. The one compression technique that reliably delivers real latency wins. See [Model Compression](../deep_learning/intro_model_compression.md).

**Doubly robust** — Estimator combining outcome and propensity models; consistent if *either* is correct.

**Dropout** — Randomly zeroing activations during training. Inverted scaling means `model.eval()` is mandatory at inference.

---

## E

**Early stopping** — Halting when validation stops improving. Effectively a capacity constraint; always worth having.

**Embedding** — Dense vector where geometric proximity means semantic similarity. Measures relatedness, *not* truth — "the drug works" and "the drug does not work" embed close together. See [Embeddings](../ai_genai/intro_embeddings.md).

**Error budget** — `1 - SLO`. Converts reliability from an argument into arithmetic. See [Observability](../devops/intro_observability.md).

**Exposure bias** — Mismatch between teacher-forced training and autoregressive inference, where a model consumes its own outputs.

---

## F

**Feature store** — Central system serving consistent features to training and inference. Exists primarily to prevent training/serving skew. See [Feature Stores](../mlops/intro_feature_stores.md).

**FID** — Fréchet Inception Distance; compares real and generated *distributions*, which is why it penalizes mode collapse. Biased by sample count.

**FlashAttention** — Kernel that tiles attention in SRAM and never materializes the `n×n` score matrix. Memory becomes linear in sequence length.

**F1 score** — Harmonic mean of precision and recall. Harmonic because it refuses to reward getting one by destroying the other.

---

## G

**GAN** — Generator versus discriminator in a minimax game. Sharp samples, single-step generation, prone to mode collapse.

**GNN** — Neural network over graph structure via message passing. Each layer extends the receptive field one hop. See [Graph Neural Networks](../deep_learning/intro_graph_neural_networks.md).

**GQA (grouped-query attention)** — Query heads share key/value heads, shrinking the KV cache 4–8×. The highest-impact efficiency change in modern LLMs.

**Gradient clipping** — Capping gradient norm. Optional hygiene for transformers, genuinely mandatory for RNNs.

**Guardrails** — Input and output checks around a model. Measure the **false-positive rate**, not just catch rate — over-blocking is the common failure.

---

## H

**Hallucination** — Confidently stated, unsupported output. Usually a *retrieval* failure in RAG systems rather than a generation failure.

**HNSW** — Hierarchical navigable small world graph; high-recall ANN index, memory-hungry.

**Hybrid search** — Fusing lexical (BM25) and dense retrieval. They fail in complementary ways, which is the whole argument for it.

**Histogram (metrics)** — Bucketed distribution whose counts are additive across instances, so fleet-wide quantiles are computable. Prefer over summaries.

---

## I

**Idempotency** — Repeating an operation gives the same result. What makes retries and backfills safe.

**Instrumental variable** — Affects treatment but the outcome only through treatment. Identifies **LATE** — the effect among compliers, not the whole population.

**Interleaving** — Merging two rankers' results into one list. Roughly 10× more sensitive than A/B testing for ranking changes. See [Search and Ranking](../system_design/search_ranking_system.md).

**IPW (inverse propensity weighting)** — Reweighting by `1/P(treatment)` to simulate randomization. Also the standard debiasing tool for click data.

**Isolation Forest** — Anomaly detector exploiting that anomalies are easy to isolate. Subsampling improves accuracy, not just speed. See [Anomaly Detection](../classical_ml/intro_anomaly_detection.md).

---

## K

**KV cache** — Stored keys and values so each decode step is `O(n)` rather than `O(n²)`. Frequently exceeds the model weights at long context.

**KL divergence** — Asymmetric measure of difference between distributions. The regularizer in a VAE and the objective in distillation.

---

## L

**LambdaMART** — Gradient boosted trees with gradients weighted by NDCG change. Still the workhorse for tabular learning-to-rank.

**LATE** — Local average treatment effect: the effect among compliers, which is what an instrument identifies.

**Latent diffusion** — Running diffusion in a VAE-compressed latent space. ~48× fewer values; the architecture behind Stable Diffusion.

**LayerNorm** — Normalizes across features within one sample. Batch-size independent, which is why transformers use it over BatchNorm.

**LoRA** — Low-rank adapters trained while the base model stays frozen. Works because weight *updates* genuinely are low-rank. See [Fine-Tuning](../deep_learning/intro_fine_tuning.md).

**Lost in the middle** — Models retrieve facts less reliably from the middle of a long context. Put the most relevant chunk last.

**LSTM** — Gated recurrent unit with an additive cell state, which is what fixes vanishing gradients. Initialize the forget-gate bias to 1. See [Sequence Models](../deep_learning/intro_sequence_models.md).

---

## M

**MAD (median absolute deviation)** — Robust spread measure with a 50% breakdown point. Use instead of σ when outliers are what you're hunting.

**Mediator** — Variable on the causal path. Adjusting for it removes part of the effect you're measuring.

**Message passing** — The GNN template: build messages per edge, aggregate permutation-invariantly, update node state.

**MLOps** — Practices for deploying and operating ML: versioning, CI/CD, monitoring, retraining. See [MLOps](../mlops/README.md).

**Mode collapse** — A GAN producing only a few outputs. High precision, low recall in generative terms.

**MoE (mixture of experts)** — Many parameters, few active per token. High capacity at low inference FLOPs, at a memory cost.

---

## N

**NDCG** — Normalized discounted cumulative gain. Handles graded relevance *and* position discount, which is why it's the search default.

**Negative sampling** — Choosing non-matches to train against. Random negatives are too easy; hard negatives are where the gains are.

**Nested loop join** — Per-row lookup into the inner table. Optimal when the outer side is small, catastrophic when the estimate was wrong. See [SQL Optimization](../data_engineering/intro_sql_optimization.md).

---

## O

**Observability** — Ability to answer questions you didn't anticipate, without shipping code. Distinct from monitoring known failure modes.

**Over-smoothing** — Repeated GNN aggregation drives all node representations together. Why GNNs stay shallow.

**Overfitting** — Fitting noise; low training error with high validation error. Confirm it isn't leakage or a distribution mismatch first.

---

## P

**PagedAttention** — Non-contiguous, paged KV-cache allocation eliminating fragmentation. A memory allocator, complementary to FlashAttention's kernel work.

**Parallel trends** — The identifying assumption of difference-in-differences. Untestable post-treatment; support it with pre-period plots.

**Point-in-time correctness** — Joining features as of the event timestamp, not as of now. The core reason feature stores exist.

**Positivity / overlap** — Every unit has non-zero probability of either treatment. No causal method extrapolates into a region with no comparison units.

**PR-AUC** — Area under the precision-recall curve. The honest metric on imbalanced problems, where ROC-AUC flatters.

**Prefill vs decode** — Prompt processing (compute-bound, parallel) versus token generation (memory-bandwidth-bound, sequential). Nearly every LLM serving optimization follows from this split.

**Prompt caching** — Reusing a computed prompt prefix. Requires stable-prefix-first ordering; often the single largest LLM cost reduction.

**Prompt injection** — Malicious instructions in model input, including *retrieved* content. Defended by capability bounding, not prompting.

**Propensity score** — `P(treatment | X)`. Reduces multi-dimensional adjustment to one number.

**PSI (population stability index)** — Distribution shift measure. Above ~0.25 conventionally signals significant drift.

---

## Q

**Quantization** — Reducing numerical precision. Since decode is bandwidth-bound, fewer bytes per weight means faster generation. Usually better spent fitting a *bigger* model than shrinking a fixed one.

**Qini curve** — Cumulative incremental conversions versus fraction targeted. How you evaluate uplift models without per-row ground truth.

---

## R

**RAG** — Retrieving documents and injecting them into the prompt. Debug by measuring retrieval and generation *separately*. See [RAG Engineering](../ai_genai/intro_rag_engineering.md).

**Reparameterization trick** — Writing `z = μ + σ·ε` so gradients flow through a sampling step. The mechanism that makes VAEs trainable.

**Reranking** — Rescoring top candidates with an expensive model. Consistently the highest-value single addition to naive RAG.

**RRF (reciprocal rank fusion)** — Fusing ranked lists by rank rather than score, avoiding cross-system score calibration. `k ≈ 60`.

---

## S

**Sargable** — A predicate that can use an index. Wrapping a column in a function destroys it — the top reason an added index changes nothing.

**Shadow deployment** — New model scores real traffic with output discarded. The only validation step with genuinely zero user risk.

**SHAP** — Additive per-prediction feature attributions. The defensible choice over impurity-based importance, which inflates high-cardinality features.

**Skew (data)** — One key holding a disproportionate share of rows, making one distributed task dominate runtime. Salt the key.

**Sleeping dogs** — Segment that converts only if *not* treated. The reason response models can produce negative incremental revenue.

**SLI / SLO / SLA** — The measurement / the internal target / the external contract. Set the SLA looser than the SLO.

**Speculative decoding** — A draft model proposes tokens the target verifies in one pass. Provably preserves the output distribution — pure latency win.

**SUTVA** — One unit's treatment doesn't affect another's outcome. Broken constantly by social and marketplace products.

---

## T

**Teacher forcing** — Feeding ground-truth previous tokens during training. Fast and stable, but creates exposure bias.

**Temperature** — Softmax scaling. Low is deterministic, high is diverse. In distillation it exposes the small-probability structure that carries the signal.

**Training/serving skew** — Features computed differently in training and production. The most common silent production ML failure.

**Transductive vs inductive** — Whether a model can handle nodes or items unseen at training time. GCN is transductive; GraphSAGE is inductive.

**TTFT (time to first token)** — Latency until the first output token. What users perceive; driven by prefill and queueing, not decode.

---

## U

**Uplift model** — Estimates `P(convert | treated) - P(convert | untreated)`. Targets *incremental* effect, not conversion probability. Requires experimental training data. See [Causal Inference](../classical_ml/intro_causal_inference.md).

**Unconfoundedness** — Conditional on observed covariates, treatment is as good as random. Untestable, and the assumption everything rests on.

---

## V

**VAE** — Autoencoder with a probabilistic latent and a KL term, making the latent space samplable. Blurry samples because pixel-wise MSE optimizes the conditional mean.

**Vectorization** — Replacing Python loops with array operations. Typically 100–1000× faster; `apply(axis=1)` is the usual culprit.

**vLLM** — Serving engine with PagedAttention and continuous batching. The default self-hosted choice. See [vLLM](../frameworks/intro_vllm.md).

---

## W

**Weight decay** — Shrinking weights toward zero. Exclude biases and normalization parameters — decaying LayerNorm gains fights the normalization.

**Window function** — SQL computation over a row set without collapsing rows. Replaces slow correlated subqueries.

**WGAN** — GAN using Wasserstein distance, which stays informative when distributions don't overlap — hence usable gradients and a meaningful loss.

---

## Z

**Zero-shot** — Performing a task with no task-specific examples, relying on pretraining and instructions.

**Z-score** — Standardized deviation from the mean. Use the **modified** z-score (median and MAD) when outliers contaminate the estimate.

---

## Related Topics

- [Choose Your Track](./choose-your-track.md) — pick a learning path
- [2026 Interview Roadmap](./2026-interview-roadmap.md)
- [Study Pattern](./study-pattern.md)
- [Resources and References](./resources-and-references.md)
- [Repository Home](../README.md)
