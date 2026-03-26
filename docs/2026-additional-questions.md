# 2026 Additional Questions and Answers

#### 1) What is the difference between training, fine-tuning, instruction-tuning, and in-context learning?
Training usually means learning model weights from scratch on a very large corpus. Fine-tuning updates an existing pretrained model on task-specific data. Instruction-tuning is a kind of fine-tuning where the model is trained to follow tasks framed as instructions. In-context learning does not update weights at all; it improves behavior by changing the prompt, examples, tools, and retrieved context at inference time.

#### 2) When would you prefer RAG over fine-tuning?
Use RAG when knowledge changes frequently, citations matter, or you need access control over documents at query time. Use fine-tuning when you want to change style, structure, task behavior, or improve performance on a repeated format. In production, many systems use both: fine-tuning for behavior and RAG for fresh knowledge.

#### 3) What are the core components of a RAG pipeline?
A solid RAG pipeline usually includes document ingestion, chunking, embedding generation, indexing, retrieval, reranking, prompt assembly, model generation, and evaluation. In stronger systems, you also add metadata filters, caching, citation generation, and fallback behavior when retrieval confidence is low.

#### 4) What makes chunking important in retrieval systems?
Chunking affects recall, precision, latency, and answer quality. Chunks that are too small lose context, while chunks that are too large retrieve noise and waste tokens. Good chunking preserves semantic boundaries, keeps metadata, and is tuned using retrieval and answer quality evaluation rather than guesswork.

#### 5) What is embedding drift and why does it matter?
Embedding drift happens when the meaning of stored vectors and live query vectors becomes inconsistent because of model changes, preprocessing changes, or distribution shifts in data. It matters because retrieval quality can silently degrade even when the application still appears healthy. A safe migration plan includes dual indexing, offline evaluation, and gradual cutover.

#### 6) How would you evaluate a RAG system?
Evaluate retrieval and generation separately. Retrieval metrics include recall@k, precision@k, MRR, and nDCG. Answer quality can be measured using groundedness, citation accuracy, task success rate, human preference, and business KPIs. Good evaluation also includes adversarial cases, ambiguous queries, and no-answer scenarios.

#### 7) What is the difference between a vector database and a traditional database?
A traditional database is optimized for exact filters, transactions, and structured queries. A vector database is optimized for nearest-neighbor similarity search over embeddings. In practice, modern AI systems often combine both: structured filtering for metadata and vector search for semantic retrieval.

#### 8) What are rerankers and why are they useful?
Retrievers are optimized to fetch a broad candidate set quickly. Rerankers score the candidate documents more precisely against the user query and improve final relevance. They are useful because first-stage retrieval is often recall-oriented, while reranking improves precision before the final prompt is built.

#### 9) What is prompt injection and how do you defend against it?
Prompt injection is when untrusted input attempts to override system behavior or extract hidden instructions, data, or tool permissions. Defenses include strict tool permission boundaries, separating instructions from untrusted content, input and output filtering, retrieval trust labels, sandboxing external actions, and requiring application-side policy checks before executing model-suggested actions.

#### 10) What is the difference between hallucination and retrieval failure?
Hallucination is when the model generates unsupported or fabricated content. Retrieval failure is when the relevant evidence is not found or not passed into the prompt. Retrieval failure often causes hallucination, but they are not identical. Strong debugging isolates whether the problem came from indexing, retrieval, prompt construction, or generation.

#### 11) What metrics matter for a production ML or AI service besides accuracy?
Accuracy alone is not enough. Teams also track latency, p95/p99 response time, throughput, cost per request, cache hit rate, failure rate, calibration, user satisfaction, business conversion metrics, drift indicators, and fairness or safety signals where relevant.

#### 12) How do you design an evaluation set for LLM applications?
Build a dataset from real tasks, edge cases, failure cases, and high-risk queries. Label what a good answer looks like, define rubric-based scoring, and keep slices for domain difficulty, ambiguity, multilingual input, safety, and long-context behavior. A strong evaluation set is versioned and used for regression testing before every model or prompt change.

#### 13) What is model distillation and when is it useful?
Distillation transfers behavior from a larger teacher model to a smaller student model. It is useful when you want lower latency and lower cost while retaining most of the task performance. It is especially common in production settings where large models are too expensive for every request.

#### 14) What is quantization and what tradeoff does it introduce?
Quantization reduces the numerical precision of weights or activations, for example from FP16 to INT8 or lower. The benefit is lower memory use, faster inference, and cheaper deployment. The tradeoff is possible loss in accuracy or stability, especially on sensitive tasks or very aggressive compression settings.

#### 15) What is LoRA or QLoRA in fine-tuning?
LoRA updates a small set of low-rank adapter weights instead of updating all model parameters, which reduces compute and memory cost. QLoRA combines parameter-efficient adaptation with quantized base weights, making large-model fine-tuning more practical on limited hardware.

#### 16) How would you choose between batch inference, real-time inference, and asynchronous inference?
Use batch inference for large scheduled workloads where latency is not critical. Use real-time inference when the user experience depends on immediate responses. Use asynchronous inference when requests are expensive or long-running but still user-triggered, and the system can notify the user later or update the result in the background.

#### 17) What is data leakage in modern ML systems?
Data leakage is when training or feature generation uses information that would not be available at prediction time. In modern systems this can happen through bad joins, future timestamps, label-derived features, or evaluation sets contaminated by prompt logs and previous model outputs. Leakage gives misleadingly high offline performance and usually collapses in production.

#### 18) How would you monitor an ML model after deployment?
Monitor input data quality, feature distributions, prediction distributions, latency, failures, calibration, business impact, and downstream feedback. For generative systems, also monitor refusal rates, hallucination indicators, retrieval quality, toxicity or policy violations, and prompt attack attempts. Monitoring needs alerts, dashboards, and an operational response plan.

#### 19) What is the role of human-in-the-loop systems in AI products?
Human-in-the-loop systems are used when mistakes are expensive, ambiguity is high, or the model should assist rather than fully automate. They improve safety, create high-quality feedback data, support escalation paths, and help teams collect edge cases for future evaluation and model improvement.

#### 20) How would you answer "Design an AI assistant for enterprise knowledge search" in an interview?
Start with requirements: document sources, freshness, access control, latency, citation quality, and acceptable risk. Then describe ingestion, chunking, embeddings, indexing, metadata filters, retrieval, reranking, generation, and evaluation. Finish with operational concerns such as observability, cost controls, prompt-injection defenses, permission checks, caching, and fallback to keyword search or no-answer behavior.

#### 21) What is the difference between offline evaluation and online evaluation?
Offline evaluation uses historical or curated data and is useful for fast iteration, regression testing, and controlled comparison. Online evaluation measures real user impact through experiments such as A/B tests, shadow traffic, or interleaving. Strong teams use offline evaluation for safety and speed, then online evaluation to confirm product value.

#### 22) What is calibration and why can it matter more than accuracy?
Calibration measures whether predicted probabilities reflect true likelihoods. A model can have strong accuracy but poor confidence estimates. Calibration matters in ranking, fraud, medical screening, and decision support systems because downstream actions often depend on the quality of confidence, not just the top prediction.

#### 23) What are common failure modes in agentic systems?
Common failures include poor tool selection, invalid tool arguments, state loss across steps, excessive looping, prompt injection, hidden dependency on brittle prompts, and weak stopping criteria. Good agent design limits tool scope, validates every action outside the model, and logs each step for debugging.

#### 24) What does a good answer sound like for "How do you reduce hallucinations?"
A strong answer is not "just use a better prompt." It should mention grounding with retrieval, narrower task framing, structured outputs, tool use, refusal policies, confidence-aware fallbacks, response verification where feasible, and better evaluation on known hallucination-prone slices.

#### 25) What 2026 topics are most likely to appear in ML engineer interviews?
The highest-probability modern topics are RAG design, LLM evaluation, embeddings, vector search, reranking, prompt injection, tool or function calling, agent orchestration, fine-tuning tradeoffs, inference optimization, monitoring, and AI system design under latency and cost constraints.
