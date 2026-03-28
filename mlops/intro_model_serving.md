# Model Serving

Model serving is the process of deploying a trained ML model to production so it can respond to inference requests. It covers architecture patterns, latency vs throughput tradeoffs, batching, scaling, and deployment strategies.

---

## Table of Contents
1. [Serving Patterns](#serving-patterns)
2. [REST vs gRPC](#rest-vs-grpc)
3. [Batching Strategies](#batching-strategies)
4. [Model Versioning & Deployment Strategies](#model-versioning--deployment-strategies)
5. [Serving Frameworks](#serving-frameworks)
6. [Hardware Considerations](#hardware-considerations)
7. [Scaling & Autoscaling](#scaling--autoscaling)
8. [Latency vs Throughput Tradeoffs](#latency-vs-throughput-tradeoffs)
9. [Interview Q&A](#interview-qa)
10. [Common Pitfalls](#common-pitfalls)
11. [Related Topics](#related-topics)

---

## Serving Patterns

### Real-Time (Synchronous) Inference

```
User Request → Load Balancer → Model Server → Response
    ~100ms          ~1ms           ~50-100ms
```

Best for: interactive applications (recommendation, fraud detection, chatbots). Requires low latency. Scale horizontally with multiple instances.

### Batch Inference

```
Scheduled job → Load batch from DB/S3 → Model Server → Write predictions → DB/S3
   (daily/hourly)   (millions of records)   (parallel)
```

Best for: non-time-sensitive predictions (email scores, ad targeting). High throughput, no latency requirement. Use GPU batching for maximum efficiency.

### Async / Event-Driven Inference

```
Event → Message Queue (Kafka/SQS) → Consumer (Model Server) → Write result → DB
                                        (async workers)
```

Best for: workloads with variable load, long-running predictions (< 30s acceptable latency). Decouples producers from consumers.

### Streaming Inference

```
Kafka topic → Stream processor (Flink/Spark Streaming) → Model → Output Kafka topic
              (real-time windowed features)
```

Best for: IoT, fraud detection, real-time personalization with feature computation.

---

## REST vs gRPC

| | REST/HTTP | gRPC |
|-|-----------|------|
| **Protocol** | HTTP/1.1, JSON | HTTP/2, Protocol Buffers |
| **Serialization** | JSON (human-readable) | Protobuf (binary, ~5-10x smaller) |
| **Latency** | Higher (JSON parsing) | Lower (binary encoding) |
| **Streaming** | Limited | Native bidirectional streaming |
| **Browser support** | Native | Needs gRPC-web proxy |
| **Use case** | External APIs, simple services | Internal microservices, low-latency |

```python
# REST API with FastAPI
from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    pred = model.predict(X)[0]
    conf = model.predict_proba(X).max()
    return PredictResponse(prediction=float(pred), confidence=float(conf))

# Run: uvicorn serve:app --host 0.0.0.0 --port 8080 --workers 4
```

---

## Batching Strategies

### Static Batching

All inputs must arrive before processing starts. Used in offline batch inference.

```python
import numpy as np
import joblib

model = joblib.load("model.pkl")

def batch_predict(records: list[dict]) -> list[float]:
    X = np.array([[r['age'], r['salary'], r['score']] for r in records])
    return model.predict(X).tolist()

# Process in chunks to control memory
def predict_large_dataset(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        preds = batch_predict(chunk)
        results.extend(preds)
    return results
```

### Dynamic Batching (Server-Side)

Accumulate requests over a time window, then process as a batch. Maximizes GPU utilization without increasing individual request latency much.

```python
import asyncio
import time
from collections import deque

class DynamicBatchingServer:
    def __init__(self, model, max_batch_size=64, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.queue = deque()
        asyncio.create_task(self._batch_processor())

    async def predict(self, features):
        future = asyncio.Future()
        self.queue.append((features, future))
        return await future

    async def _batch_processor(self):
        while True:
            await asyncio.sleep(self.max_wait_ms)
            if not self.queue:
                continue

            # Drain queue up to max_batch_size
            batch = []
            futures = []
            while self.queue and len(batch) < self.max_batch_size:
                features, future = self.queue.popleft()
                batch.append(features)
                futures.append(future)

            # Run batch inference
            import numpy as np
            X = np.array(batch)
            predictions = self.model.predict(X)

            # Resolve futures
            for future, pred in zip(futures, predictions):
                future.set_result(float(pred))
```

---

## Model Versioning & Deployment Strategies

### Blue-Green Deployment

```
Traffic (100%) → Blue (v1)    →    switch    →  Traffic (100%) → Green (v2)
                                (instant rollback available)
```

Zero downtime. Full instant rollback. Requires 2x infrastructure temporarily.

### Canary Deployment

```
Traffic → 95% → v1 (stable)
       → 5%  → v2 (canary)
    (gradually shift traffic after validating metrics)
```

Reduces risk. Monitor error rates, latency, and business metrics before full rollout.

### Shadow Mode

```
All traffic → v1 (serves response to user)
           → v2 (receives same request, runs in parallel, result discarded)
```

Validate v2 in production without user impact. Compare outputs between v1 and v2.

```python
# Shadow mode implementation
import threading
import logging

class ShadowModeServer:
    def __init__(self, primary_model, shadow_model):
        self.primary = primary_model
        self.shadow = shadow_model

    def predict(self, X):
        primary_pred = self.primary.predict(X)

        # Run shadow asynchronously — don't block
        def run_shadow():
            try:
                shadow_pred = self.shadow.predict(X)
                # Log discrepancies
                if abs(primary_pred - shadow_pred) > 0.1:
                    logging.info(f"Shadow disagreement: primary={primary_pred}, shadow={shadow_pred}")
            except Exception as e:
                logging.error(f"Shadow model error: {e}")

        threading.Thread(target=run_shadow, daemon=True).start()
        return primary_pred  # Return primary immediately
```

---

## Serving Frameworks

| Framework | Best For | Highlights |
|-----------|---------|-----------|
| **FastAPI** | Custom REST APIs, small models | Simple, async, Pydantic validation |
| **TensorFlow Serving** | TensorFlow/Keras models | High performance, gRPC, batching |
| **TorchServe** | PyTorch models | Managed lifecycle, REST/gRPC |
| **Triton Inference Server** | Multi-framework GPU serving | NVIDIA, dynamic batching, ensemble |
| **BentoML** | Any framework, packaging | Easy deployment, cloud-native |
| **Ray Serve** | Distributed, complex pipelines | Actor-based, Python-first |
| **Seldon Core** | Kubernetes-native | A/B testing, monitoring built-in |

### BentoML Example

```python
import bentoml
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Save model to BentoML
model = RandomForestClassifier()
model.fit(X_train, y_train)
bento_model = bentoml.sklearn.save_model("fraud_classifier", model)

# Create a service
import bentoml
from bentoml.io import NumpyNdarray

svc = bentoml.Service("fraud_detection", runners=[bento_model.to_runner()])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data: np.ndarray) -> np.ndarray:
    return bento_model.to_runner().predict.run(input_data)

# Deploy: bentoml serve service:svc --production
# Containerize: bentoml containerize fraud_detection:latest
```

---

## Hardware Considerations

| Workload | Recommended Hardware | Why |
|----------|---------------------|-----|
| Small tabular models | CPU (multi-core) | GPU overhead not worth it |
| Deep learning inference | GPU (T4, A10, A100) | Matrix ops are massively parallel |
| Large LLMs (70B+) | Multi-GPU with NVLink | Memory bandwidth bottleneck |
| High-throughput batch | Large GPU + dynamic batching | Maximize utilization |
| Low-latency real-time | CPU or T4 + quantization | Minimize cold start |

### Quantization for Faster Inference

```python
# PyTorch model quantization (reduces model size, speeds up CPU inference)
import torch

model = torch.load("model.pt")
model.eval()

# Dynamic quantization (quantize weights; activations quantized at runtime)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Size and speed comparison
original_size = sum(p.numel() for p in model.parameters()) * 4 / 1e6  # MB (float32)
quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / 1e6  # MB (int8)
print(f"Original: {original_size:.1f}MB, Quantized: {quantized_size:.1f}MB")
```

---

## Scaling & Autoscaling

```yaml
# Kubernetes HorizontalPodAutoscaler for model serving
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second  # Custom metric from Prometheus
      target:
        type: AverageValue
        averageValue: 100
```

---

## Latency vs Throughput Tradeoffs

| Optimization | Latency Impact | Throughput Impact | Tradeoff |
|-------------|---------------|------------------|---------|
| Batching | Slight increase (wait time) | Major increase | Accept small latency hit for throughput |
| Quantization | Decrease (faster ops) | Increase | Slight accuracy decrease |
| Model distillation | Decrease | Increase | Accuracy decrease |
| Caching predictions | Major decrease (cache hit) | Increase | Stale predictions |
| Async serving | Decrease for client | Increase | Result not immediately available |
| GPU vs CPU | Decrease (for large batches) | Increase | Cost increase |

---

## Interview Q&A

**Q1: What is the difference between batch and real-time inference?**
Batch inference processes a large dataset offline (e.g., nightly job scoring all customers). Real-time inference responds to individual requests with low latency (< 100ms). Batch: high throughput, no latency constraint, GPU-efficient. Real-time: low latency, auto-scaling required, may need prediction caching.

**Q2: What is dynamic batching and why is it important for GPU serving?**
Dynamic batching accumulates multiple incoming requests over a short time window (e.g., 5-10ms), then processes them as a single batch. GPUs are massively parallel — a single request uses < 1% of a GPU's capacity, while a batch of 64 may use 80%. Dynamic batching dramatically improves GPU utilization and cost efficiency without significantly increasing P99 latency.

**Q3: How do you do a canary deployment for an ML model?**
Route a small percentage (5-10%) of traffic to the new model version. Monitor key metrics: prediction latency, error rates, and most importantly business metrics (CTR, conversion, revenue). Use a load balancer or service mesh (Istio) to split traffic. Gradually increase canary traffic if metrics are good; roll back if they degrade.

**Q4: What is shadow mode and when would you use it?**
Shadow mode routes all production traffic to both the current model (which serves users) and a new model (which runs in parallel but responses are discarded). You collect predictions from both and compare. Use it when: you want to validate a new model on real production traffic before it affects users, or when offline metrics don't translate well to production.

**Q5: What are the tradeoffs between REST and gRPC for model serving?**
REST uses JSON over HTTP/1.1: human-readable, easy to debug, broad tooling support, but slower due to JSON serialization. gRPC uses Protocol Buffers over HTTP/2: 5-10x smaller payloads, faster parsing, native streaming, but requires generated client code and browser support via proxy. Use REST for external APIs; gRPC for internal high-throughput services.

**Q6: How would you reduce serving latency for a large deep learning model?**
1. **Quantization**: INT8/FP16 instead of FP32 — 2-4x speedup with minimal accuracy loss
2. **Distillation**: smaller student model trained on teacher outputs
3. **Pruning**: remove low-importance weights
4. **TensorRT optimization**: compile for specific GPU hardware
5. **Prediction caching**: cache frequent/identical inputs
6. **Model parallelism**: split large models across GPUs

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Single model instance | SPOF, latency spikes under load | Multiple replicas + load balancer |
| No health checks | Dead pods serve traffic | Liveness + readiness probes |
| Loading model on every request | Catastrophic latency | Load model once at startup |
| No versioning | Can't roll back bad models | Tag models with versions; use blue-green |
| Ignoring cold start | First request much slower | Pre-warm instances; use min-replicas > 0 |
| Not monitoring prediction distribution | Silent model drift | Log prediction distributions; alert on shift |
| GPU idle at low traffic | Expensive waste | Scale to zero or use CPU for low-traffic |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [MLflow](./intro_mlflow.md) | Model registry manages versions deployed to serving |
| [Feature Stores](./intro_feature_stores.md) | Online feature store is queried during serving |
| [Kubernetes](../devops/intro_kubernetes.md) | Deployments, HPA, and services for model serving |
| [Docker](../devops/intro_docker.md) | Containerizing model servers |
| [LLMOps](../ai_genai/intro_llmops.md) | LLM-specific serving patterns (streaming, batching) |
| [Study Pattern](../docs/study-pattern.md) | Model Serving is an Advanced (🔴) MLOps topic |
