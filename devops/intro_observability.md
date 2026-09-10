# Observability: Metrics, Logs, and Traces

Monitoring tells you *that* something broke. Observability tells you *why*, for a failure nobody predicted. The distinction matters because ML and AI systems fail in ways dashboards weren't built for — a model degrading silently, one shard adding 800 ms to p99, an agent looping through 40 tool calls before answering.

---

## Table of Contents
1. [Monitoring vs Observability](#monitoring-vs-observability)
2. [The Three Pillars](#the-three-pillars)
3. [Metrics](#metrics)
4. [Prometheus and PromQL](#prometheus-and-promql)
5. [Structured Logging](#structured-logging)
6. [Distributed Tracing](#distributed-tracing)
7. [OpenTelemetry](#opentelemetry)
8. [SLIs, SLOs, and Error Budgets](#slis-slos-and-error-budgets)
9. [Alerting That Works](#alerting-that-works)
10. [Dashboards](#dashboards)
11. [Observability for ML and LLM Systems](#observability-for-ml-and-llm-systems)
12. [Cost Control](#cost-control)
13. [Interview Q&A](#interview-qa)
14. [Common Pitfalls](#common-pitfalls)
15. [Related Topics](#related-topics)

---

## Monitoring vs Observability

**Monitoring** watches known failure modes: CPU above 80%, error rate above 1%. You decided in advance what to measure.

**Observability** is the property of being able to answer questions you didn't anticipate — "why are requests from Android users in Brazil slow, but only for accounts created this month?" — without shipping new code.

The practical difference is **cardinality and context**. A monitoring system aggregates away the dimensions you'd need to answer that question. An observable system keeps enough per-request detail to slice by attributes you didn't think of in advance.

---

## The Three Pillars

| Pillar | Answers | Cost | Cardinality |
|---|---|---|---|
| **Metrics** | *Is* something wrong? How much, how often? | Cheap, constant per series | **Low** — must stay bounded |
| **Logs** | What exactly happened in this one case? | Expensive at volume | Unbounded |
| **Traces** | Where did the time go across services? | Moderate (sampled) | High |

They complement rather than substitute. The normal workflow: a **metric** alerts, a **trace** localizes the slow or failing span, **logs** for that trace ID explain the specific failure.

Wire them together with **exemplars** — attach a trace ID to metric samples — so you can jump from a latency spike on a graph straight to a trace of a request that caused it. That linkage is what makes the three pillars a system rather than three silos.

---

## Metrics

Four types, and choosing correctly matters:

| Type | Semantics | Example |
|---|---|---|
| **Counter** | Monotonically increasing | `http_requests_total` |
| **Gauge** | Goes up and down | `queue_depth`, `gpu_memory_bytes` |
| **Histogram** | Bucketed distribution, aggregatable across instances | `request_duration_seconds` |
| **Summary** | Client-computed quantiles | Rarely the right choice |

**Histogram over summary, almost always.** Summaries compute quantiles on each instance, and **quantiles cannot be averaged** — the p99 across ten servers is not the mean of ten p99s. Histograms ship bucket counts, which *are* additive, so the aggregate quantile is computed correctly at query time.

```python
from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter("http_requests_total", "Total requests", ["method", "endpoint", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),  # tune to your SLO
)
IN_FLIGHT = Gauge("http_requests_in_flight", "Concurrent requests")

@LATENCY.labels(endpoint="/predict").time()
def handle_predict(req):
    with IN_FLIGHT.track_inprogress():
        resp = model_predict(req)
    REQUESTS.labels("POST", "/predict", "200").inc()
    return resp
```

**Bucket boundaries must bracket your SLO.** If the SLO is 200 ms and your buckets jump 0.1 → 0.5, you cannot measure whether you're meeting it — `histogram_quantile` interpolates within a bucket, so a 200 ms threshold sitting mid-bucket is a guess.

### The cardinality rule

**Total series = product of all label value counts.** This is the single most important operational fact about metrics:

```
endpoint (20) × status (5) × region (3)        =    300 series   ✓
endpoint (20) × status (5) × user_id (1000000) = 100M series     ✗ kills Prometheus
```

Never put user IDs, request IDs, email addresses, full URLs with parameters, or timestamps in labels. High-cardinality dimensions belong in **logs and traces**, not metrics. A single unbounded label has taken down more monitoring systems than any other cause.

---

## Prometheus and PromQL

Prometheus **pulls** metrics from HTTP endpoints on a schedule, stores them as time series, and evaluates alert rules.

```promql
# Request rate per second, 5-minute window
rate(http_requests_total[5m])

# Error ratio
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))

# p99 latency from a histogram — note rate() inside
histogram_quantile(0.99,
  sum by (le, endpoint) (rate(http_request_duration_seconds_bucket[5m])))

# Predict disk exhaustion within 4 hours
predict_linear(node_filesystem_avail_bytes[6h], 4*3600) < 0
```

**`rate` vs `increase` vs `irate`**: `rate` gives per-second average over the window and handles counter resets; `increase` is total change over the window (`rate × window`); `irate` uses only the last two samples and is for volatile graphs, not alerts.

**The most common PromQL error** is applying `histogram_quantile` without `rate` inside. The `_bucket` series are counters, so without `rate()` you compute the quantile over all history since process start rather than the recent window — which looks plausible and is wrong.

**Pull vs push**: pull gives Prometheus control over scrape frequency and makes target health observable (a target that's down simply fails to scrape). Push is needed for short-lived batch jobs that finish before any scrape happens — that's what the Pushgateway is for, and it's the *only* thing it's for.

---

## Structured Logging

Log JSON, not prose. Prose is unqueryable at scale.

```python
import structlog

log = structlog.get_logger()

log.info(
    "prediction_served",
    request_id=req_id,
    trace_id=trace_id,          # links this log to its trace
    model_version="v2.3.1",
    latency_ms=42,
    input_tokens=1200,
    confidence=0.87,
    user_segment="enterprise",  # segment, not user_id — PII discipline
)
```

**Rules that hold up in production:**

- **One event per line, structured fields**, so you can filter on `latency_ms > 1000 AND model_version = "v2.3.1"` without regex.
- **Always include the trace ID**, which is what turns logs from isolated records into the detail view of a trace.
- **Never log secrets or raw PII.** Redact at the logging layer, not by convention — a redaction filter in the pipeline, because convention fails eventually.
- **Sample high-volume success logs**, keep all errors. Logging every one of a million successful requests is expensive and tells you nothing that metrics don't.
- **Levels mean something**: ERROR = a human must look; WARN = degraded but handled; INFO = business events; DEBUG = off in production.

The most common log anti-pattern is using logs as metrics — counting log lines to derive a rate. It works, it's 100× more expensive than a counter, and it breaks when sampling is enabled.

---

## Distributed Tracing

A **trace** is one request's journey; a **span** is one operation within it. Spans nest to form a tree, carrying a shared trace ID propagated across service boundaries via HTTP headers.

```
Trace: 7f3a9c...                                    total 1240 ms
├─ api-gateway            [██                    ]    40 ms
├─ auth-service           [ █                    ]    15 ms
├─ retrieval-service      [   ████████           ]   320 ms
│  ├─ embed-query         [   ██                 ]    80 ms
│  ├─ vector-search       [     ███              ]   150 ms
│  └─ rerank              [        ██            ]    90 ms
└─ llm-service            [          ██████████  ]   850 ms   ← the actual cost
```

The value is immediate: this picture tells you optimizing the vector search is pointless when generation is 70% of the time.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def handle_request(query, user):
    with tracer.start_as_current_span("handle_request") as span:
        span.set_attribute("user.segment", user.segment)
        span.set_attribute("query.length", len(query))

        with tracer.start_as_current_span("retrieve") as s:
            docs = retrieve(query)
            s.set_attribute("retrieval.doc_count", len(docs))
            s.set_attribute("retrieval.doc_ids", ",".join(d.id for d in docs))

        with tracer.start_as_current_span("generate") as s:
            answer = generate(query, docs)
            s.set_attribute("llm.model", MODEL)
            s.set_attribute("llm.input_tokens", answer.usage.input_tokens)
            s.set_attribute("llm.output_tokens", answer.usage.output_tokens)
        return answer
```

**Sampling** keeps this affordable:

| Strategy | How | Trade-off |
|---|---|---|
| **Head-based** | Decide at trace start, e.g. 1% | Cheap, simple; misses rare errors |
| **Tail-based** | Buffer the trace, decide after seeing it | Keeps all errors and slow traces; needs a collector with memory |

Tail-based sampling is what you want in practice — keep 100% of errors and traces over the SLO, plus a small random sample of successes for baseline comparison. Head-based at 1% will miss the rare failure you most need to see.

---

## OpenTelemetry

The vendor-neutral standard for producing all three signals. Its value is that instrumentation is decoupled from backend — instrument once, export to Prometheus, Jaeger, Datadog, or anything else, and switch without touching application code.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4317")))
trace.set_tracer_provider(provider)

FastAPIInstrumentor.instrument_app(app)    # auto-instrument the framework
```

**Auto-instrumentation** covers the frameworks and clients (HTTP, database drivers, message queues) for free; add manual spans for domain logic that matters — a retrieval step, a model call, a business decision.

The **Collector** sits between applications and backends, handling batching, retry, redaction, sampling, and fan-out to multiple destinations. Putting it in the path from the start means changing vendors later is a config change rather than a redeploy of every service.

---

## SLIs, SLOs, and Error Budgets

| Term | Definition |
|---|---|
| **SLI** | The measurement: "proportion of requests served under 300 ms" |
| **SLO** | The target: "99.5% of requests under 300 ms over 30 days" |
| **SLA** | The contract, with financial consequences. Set it looser than the SLO. |
| **Error budget** | `1 - SLO`. At 99.5%, you may fail 0.5% — about 3.6 hours a month. |

The **error budget** is the useful concept because it converts reliability from an argument into arithmetic. Budget remaining means you can ship risky changes; budget exhausted means the team freezes features and works on reliability. It also states plainly that 100% is the wrong target — chasing it costs enormously and prevents shipping.

**Pick SLIs the user experiences.** CPU utilization is not an SLI; latency, error rate, and correctness are. For ML systems, add quality: "95% of predictions served with a model no more than 24 hours stale" is a legitimate SLI.

---

## Alerting That Works

**Alert on symptoms, not causes.** "p99 latency exceeds the SLO" is actionable and catches every cause. "CPU > 80%" fires when nothing is wrong and stays silent when the failure is a slow dependency.

**Multi-window, multi-burn-rate alerting** is the technique worth knowing by name. A single threshold either fires on brief blips or reacts too slowly to a real outage. Instead, alert on how fast you're consuming the error budget, across two windows:

```promql
# Fast burn: 14.4× budget rate — pages immediately (exhausts 30-day budget in ~2 days)
(error_ratio_5m > 14.4 * 0.005) and (error_ratio_1h > 14.4 * 0.005)

# Slow burn: 6× budget rate over longer windows — a ticket, not a page
(error_ratio_6h > 6 * 0.005) and (error_ratio_3d > 6 * 0.005)
```

Requiring **both** a short and a long window to breach suppresses one-minute blips while still catching genuine sustained degradation quickly.

**Every page must be actionable.** If the responder's correct action is "acknowledge and go back to sleep", delete the alert. Alert fatigue is the mechanism by which real incidents get missed, and it's caused by exactly these.

Include in every alert: what broke, the user impact, a link to the relevant dashboard, and a link to the runbook.

---

## Dashboards

Structure by audience:

**Service dashboard — the RED method** (for request-driven services): **R**ate, **E**rrors, **D**uration. Three panels answer most questions.

**Resource dashboard — the USE method** (for infrastructure): **U**tilization, **S**aturation, **E**rrors, per resource (CPU, memory, disk, network).

**Business dashboard**: conversion, revenue, active users — the metrics that say whether the system is achieving anything.

Practical rules: put the SLO line on the latency graph so "is this bad?" is answerable at a glance; always show p50, p95, and p99 rather than averages; annotate deploys so correlation is visible; and keep the top-level dashboard to one screen. A dashboard with 40 panels is used during exactly zero incidents.

---

## Observability for ML and LLM Systems

Standard service observability is necessary and insufficient — a model can be 100% available, fast, error-free, and completely wrong.

**Model-specific signals:**

| Signal | Why |
|---|---|
| Prediction distribution | Shifts before accuracy metrics can be computed |
| Input feature drift (PSI, KL) | Earliest available warning |
| Feature null rate and range violations | Upstream pipeline breakage |
| Model version serving traffic | Confirms the rollout state |
| Delayed accuracy (as labels arrive) | Ground truth, weeks late |
| Training/serving skew checks | The most common silent failure |

**LLM-specific signals:**

| Signal | Why |
|---|---|
| **TTFT and inter-token latency** separately | Different causes; averaging hides both |
| Input/output tokens per request | Drives cost directly |
| **Cost per request**, by feature and tenant | Unbounded per-request cost is unique to LLMs |
| Cache hit rate (prefix and semantic) | The main cost lever |
| Tool error rate and step count distribution | Agent loops surface here first |
| Schema validation failure rate | Earliest signal of a model or prompt regression |
| Abstention / refusal rate | Falling rate often means rising hallucination |
| Guardrail trip rate **and false-positive rate** | Over-blocking is the common failure |

Two points worth making in an interview. **Schema validation failure rate is the cheapest high-value alert** for an LLM system — it catches provider model changes, prompt regressions, and malformed tool calls in one signal. And **cost needs SLOs like latency does**: a per-request ceiling and a budget circuit breaker, because an agent stuck in a retry loop can burn a month's budget in an afternoon.

Span-level tracing matters more here than in conventional services, because agent and RAG failures almost never originate in the final step — a wrong answer usually traces to a retrieval five spans earlier.

---

## Cost Control

Observability bills routinely rival compute bills. Levers:

- **Metrics**: enforce cardinality limits; drop unused series; use recording rules to precompute expensive queries; shorten retention for high-resolution data while keeping downsampled history.
- **Logs**: sample successes aggressively, keep all errors; tier storage (hot 7 days, cold 90); stop logging what metrics already cover.
- **Traces**: tail-based sampling — 100% of errors and slow traces, ~1% of successes.

The most effective single action is usually **finding and dropping unused metrics and log fields**. Instrumentation accretes and nobody removes it; auditing what's actually queried typically finds a large fraction of spend on data no one has looked at in a year.

---

## Interview Q&A

#### What's the difference between monitoring and observability?

Monitoring watches predetermined failure modes — you decide in advance what to measure and alert on thresholds. Observability is the ability to answer questions you didn't anticipate, without shipping new code.

The practical difference is cardinality and retained context. Monitoring aggregates: you get error rate per endpoint. Observability keeps per-request detail with rich attributes, so when someone asks "why are Android users in Brazil on accounts created this month seeing errors?", you can slice by dimensions nobody predicted.

In practice you need both, and they play different roles: metrics are cheap and always-on so they're what you alert from; traces and logs carry the high-cardinality context that lets you diagnose. The handoff — alert fires, trace localizes, logs explain — is the actual workflow.

#### Why prefer histograms over summaries in Prometheus?

Because **quantiles cannot be aggregated**. A summary computes quantiles locally on each instance, so if ten servers each report their p99, there's no valid way to combine them into a fleet-wide p99 — averaging ten p99 values is meaningless, and it's the number people accidentally show on dashboards.

A histogram ships bucket counts, which are plain counters and therefore additive. You sum buckets across instances and compute the quantile at query time with `histogram_quantile`, which gives a correct fleet-wide figure. It also lets you compute *any* quantile after the fact, whereas summaries fix the quantiles at instrumentation time.

The cost is that accuracy depends on bucket boundaries, so buckets must bracket your SLO threshold — with buckets jumping 0.1 to 0.5, a 200 ms SLO is unmeasurable because the quantile is interpolated within that gap.

#### Why is metric cardinality dangerous?

Because series count is the **product** of all label value counts, so it explodes multiplicatively. Endpoint × status × region might be 300 series, which is fine. Add `user_id` with a million values and it's 100 million series — each with its own memory, index entry, and storage. That will take down a Prometheus server.

The rule is that metric labels must be **bounded and low-cardinality**: endpoint, status class, region, model version. Anything unbounded — user IDs, request IDs, session IDs, full URLs with query parameters, error messages containing IDs — belongs in logs or trace attributes, where high cardinality is expected and priced accordingly.

The insidious part is that it's usually accidental: someone adds a label that looks small in staging and is unbounded in production. Cardinality limits enforced at the collector are worth having as a guardrail.

#### How would you debug "the API is slow"?

Narrow before deep-diving. First, **is it real and who's affected?** Check the latency graph — p50 versus p99 tells you whether everything is slow or only the tail, which are different problems. Slice by endpoint, region, and customer to see whether it's global or localized.

Then **when did it start**, correlated against deploy annotations, config changes, and traffic volume. A step change at a deploy is a different investigation from gradual degradation.

Then **traces**: pull traces from the slow population and look at the span breakdown. This localizes it immediately — a slow dependency, a database query, a model call, or time spent queuing before work even starts. Compare against traces from the fast population.

Then **logs for those trace IDs** to get the specifics — which query, which parameters, which error.

Common findings in ML systems specifically: a cold cache after deploy, one slow shard dominating a scatter-gather p99, a downstream model provider degrading, queue buildup from insufficient concurrency, or a change that grew the context length and therefore prefill time.

#### What is an error budget and why is it useful?

It's the complement of the SLO — with a 99.5% availability SLO, you're permitted 0.5% failure, roughly 3.6 hours a month. That allowance is the budget.

Its value is converting reliability from an opinion into a shared number. When budget remains, the team ships risky changes freely. When it's exhausted, feature work stops and reliability work starts — a policy agreed in advance, so it isn't renegotiated during an incident.

It also makes explicit that 100% reliability is the wrong target: it costs enormously, delivers nothing users can perceive above a certain point, and prevents shipping. Some deliberate failure budget is the correct engineering position, and the error budget is how you say that without sounding careless.

#### How do you alert without alert fatigue?

Alert on **symptoms users experience**, not causes. "p99 latency exceeds SLO" catches every underlying cause and is always worth investigating. "CPU > 80%" fires constantly when nothing is wrong and misses failures that aren't CPU-bound.

Use **multi-window, multi-burn-rate** alerting: page when error-budget consumption is fast in *both* a short and a long window, which suppresses one-minute blips while still catching sustained degradation quickly. Slower burn rates become tickets rather than pages.

Then be ruthless: **every page must have an action.** If the correct response is "acknowledge and go back to bed", delete the alert — it's actively harmful, because fatigue is the mechanism by which real incidents get ignored. Route non-urgent signals to tickets or dashboards. And include in every alert what broke, the user impact, and a runbook link, so the responder isn't starting from nothing at 3am.

#### What would you monitor for an LLM feature that you wouldn't for a normal API?

Four things that don't exist elsewhere.

**Cost per request**, broken down by feature and tenant, with a hard per-request ceiling and a budget circuit breaker. LLM cost varies per request and is effectively unbounded — an agent in a retry loop can burn a month's budget in hours, which no conventional service can do.

**Latency split into TTFT and inter-token latency**, because they have different causes — TTFT is prefill and queueing, inter-token is decode bandwidth — and a combined average hides both.

**Quality signals**, since the service can be 100% available and completely wrong: schema validation failure rate (the cheapest early warning for provider model changes and prompt regressions), abstention rate (a falling rate often means hallucination is rising), sampled groundedness checks, and user feedback.

**Agent behaviour**: tool error rates and the distribution of steps per request, where loops and degradation show up first.

Plus one operational note: **pin model snapshots** and alert on version changes, because quality can shift with no deploy on your side when a provider updates a floating alias.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Unbounded label values | Cardinality explosion kills the metrics backend | Bounded labels only; IDs go in logs/traces |
| Averaging p99 across instances | Quantiles aren't additive; the number is meaningless | Histograms + `histogram_quantile` at query time |
| `histogram_quantile` without `rate()` | Computes over all history, not the recent window | Always `rate(..._bucket[5m])` inside |
| Histogram buckets that don't bracket the SLO | Can't measure whether you're meeting it | Choose buckets around the threshold |
| Alerting on causes (CPU, memory) | Noisy and incomplete | Alert on user-facing symptoms |
| Single-threshold alerts | Blips page; slow burns go unnoticed | Multi-window multi-burn-rate |
| Pages with no action | Fatigue causes real incidents to be missed | Delete them or downgrade to tickets |
| Head-based sampling at 1% | Misses the rare errors you need most | Tail-based: all errors, sample successes |
| Logging unstructured prose | Unqueryable at scale | Structured JSON with consistent fields |
| Logs without trace IDs | Can't connect a log to its request | Propagate and log the trace ID everywhere |
| Counting log lines instead of using counters | ~100× more expensive; breaks under sampling | Emit a metric |
| Monitoring only infrastructure for an ML service | Model can be available and wrong | Add drift, quality, and version metrics |
| Logging raw PII or secrets | Compliance incident | Redact in the pipeline, not by convention |
| Never auditing instrumentation | Large spend on data nobody queries | Periodically drop unused metrics and fields |

---

## Related Topics

- [Model Monitoring](../mlops/intro_model_monitoring.md)
- [CI/CD for Machine Learning](../mlops/intro_cicd_for_ml.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Kubernetes](./intro_kubernetes.md)
- [Docker](./intro_docker.md)
- [Terraform](./intro_terraform.md)
- [Testing AI Systems](./intro_testing_ai.md)
- [LLMOps](../ai_genai/intro_llmops.md)
- [LLM Inference Optimization](../ai_genai/intro_llm_inference_optimization.md)
- [Backend AI System Design](../system_design/intro_backend_ai_system_design.md)
- [Designing a Production LLM Assistant](../system_design/llm_assistant_system.md)
