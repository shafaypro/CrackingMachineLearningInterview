# Types of Data Processing Pipelines – Complete Guide (2026 Edition)

**Data processing pipelines** sit on a spectrum from **batch** (high latency, low engineering complexity, strong data quality) to **streaming** (sub-second latency, high engineering complexity, hardest to get right). Choosing the correct point on that spectrum is one of the most common — and most revealing — data engineering interview questions.

> The single most important idea: **latency and engineering complexity trade off against each other.** You do not pick "real-time" because it sounds modern; you pick the *highest-latency* option that still meets the business requirement, because it is cheaper, simpler, and easier to make correct.

---

## The Latency ↔ Complexity Spectrum

```
  LOWER LATENCY  ───────────────────────────────────────────►  (faster results)
  HIGHER COMPLEXITY ────────────────────────────────────────►  (harder to build)

  BATCH        MICROBATCH      NEAR REAL-TIME     REAL-TIME        STREAMING
  hours          hours           2–5 min          ~seconds         ms–seconds
  ┌────────┐   ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
  │ once   │   │ assembly │    │ rapid batch│    │ conveyor │    │  river   │
  │ a day  │   │  line    │    │  assembly  │    │  belt    │    │ (always  │
  │        │   │          │    │            │    │          │    │ flowing) │
  └────────┘   └──────────┘    └────────────┘    └──────────┘    └──────────┘
  Strongest DQ ◄──────────────────────────────────────────►  Weakest DQ
                                                              (state is hard)
```

| Pipeline | Typical latency | Processing model | Data quality (DQ) | Eng. complexity | Primary tech |
|----------|----------------|------------------|-------------------|-----------------|--------------|
| **Batch** | 4–6 hours (daily) | Full dataset, partition-based | ⭐⭐⭐⭐⭐ Strongest | ⭐ Lowest | Spark |
| **Microbatch** | 4–6 hours / shorter windows | Larger scheduled batches | ⭐⭐⭐⭐ Strong | ⭐⭐ Low | Spark, Spark Structured Streaming |
| **Near real-time** | 2–5 minutes | Small micro-batches | ⭐⭐⭐ Good | ⭐⭐⭐ Medium | Spark Structured Streaming |
| **Real-time** | seconds (100ms–2s) | Event-driven w/ inherent latency | ⭐⭐ Harder | ⭐⭐⭐⭐ High | Flink, Kafka |
| **Streaming** | ms–seconds | Event-by-event, continuous | ⭐⭐ Hardest | ⭐⭐⭐⭐⭐ Highest | Kafka, Flink |

> *Inspired by the "Types of Data Processing Pipelines" framing from Zach Wilson / Zach Morris' Data Engineering Bootcamp. Expanded here into an interview-prep guide.*

---

## 1. Batch Processing

> **Analogy:** a factory that processes all of the day's work at once.

Data is collected over a window (usually a day) and processed together in a single scheduled run. This is the workhorse of analytics and the default you should reach for unless requirements force you faster.

**Features**
- Processed in scheduled intervals (cron / Airflow DAG)
- Full dataset, partition-based processing
- Schedule-driven execution
- Strong data quality (DQ) capabilities — you have the whole dataset, so you can validate, dedupe, and reconcile

**Latency components**
- Job scheduling: minutes
- Full dataset processing: hours
- Typical total: **4–6 hours daily**
- Usually runs at midnight UTC (after the day closes)

**✅ Use when**
- Daily/periodic refresh is acceptable (dashboards, reports, ML training data)
- Complex data quality checks are required
- You want to optimize resource cost (run big clusters once, then shut down)

**🚫 Avoid when**
- You need intraday or real-time data
- The use case is a customer-facing live analytics surface

**Tech:** Apache Spark, dbt, Airflow-orchestrated SQL.

---

## 2. Microbatch Processing

> **Analogy:** a traditional assembly line with larger batch sizes.

Microbatch shrinks the batch window but keeps the batch *mental model*. The tech stack overlaps heavily with near real-time, which is why teams often graduate from microbatch to near real-time without re-platforming.

**Features**
- Processed in shorter scheduled intervals
- Larger batch windows (vs near real-time)
- Better suited for DQ checks (still operating on a bounded set)
- Similar tech stack to near real-time

**Latency components**
- Job scheduling: minutes
- Full dataset processing: hours
- Typical total: **4–6 hours**
- Usually runs midnight UTC

**✅ Use when**
- Robust DQ checks are needed
- You have complex event correlation
- High data volume processing where bounded batches are easier to reason about

**🚫 Avoid when**
- You have real-time alerting needs
- Transformations are simple (the batch overhead isn't worth it)

**Tech:** Apache Spark, Spark Structured Streaming (in fixed/trigger-interval mode).

---

## 3. Near Real-Time Processing

> **Analogy:** a rapid batch assembly line.

The bridge between batch and streaming. You process small batches every few minutes — fast enough for "fresh" dashboards and operational use cases, but still bounded enough to run quality checks.

**Features**
- Processed in small batches every few minutes
- Small batch collection windows
- Micro-batch processing model
- Better DQ capability vs pure streaming

**Latency components**
- Collection window: 2–5 min
- Processing time: variable
- Writing time: depends on the sink

**✅ Use when**
- You need to compute throughout the day (not just at midnight)
- Transformations are complex but a few minutes of delay is fine

**🚫 Avoid when**
- You have millisecond-latency requirements
- The scenario is a simple pass-through (just use streaming or batch)

**Tech:** Spark Structured Streaming (micro-batch mode).

---

## 4. Real-Time Processing

> **Analogy:** a fast conveyor belt with minimal stops.

True event-driven processing with low — but not zero — latency. Once you cross into real-time, **state management** becomes a first-class engineering problem (windowing, late data, exactly-once semantics).

**Features**
- Instant processing with inherent latencies (network + compute)
- Near-immediate processing
- Event-time processing (not just arrival-time)
- State management requirements

**Latency components**
- Network time: 100–200 ms
- Processing overhead: 1–2 sec
- Total: seconds (end-to-end can stretch to minutes under load)

**✅ Use when**
- Customer-facing applications
- Critical business operations that react to events

**🚫 Avoid when**
- You have complex DQ needs (hard to validate a never-ending stream)
- Limited team expertise (real-time is operationally demanding)
- Heavy transformations that are cheaper in batch

**Tech:** Apache Flink, Kafka (often Kafka Streams or Flink on Kafka).

---

## 5. Streaming Processing

> **Analogy:** a river — water keeps flowing continuously.

The lowest-latency end of the spectrum: each event is processed the moment it is generated. This is the most powerful and the most expensive to operate. Native **watermarking** and continuous state are what separate real streaming engines from microbatch dressed up as streaming.

**Features**
- Immediate processing upon generation
- Event-by-event processing
- Continuous state management
- Native watermarking support (handling out-of-order / late events)

**Latency components**
- Event capture: ~150 ms
- Flink processing: ~100 ms – 2 sec
- Sink writing: variable, based on destination

**✅ Use when**
- Critical real-time decisions
- Fraud / security monitoring
- Customer-facing real-time features (live personalization, live dashboards)

**🚫 Avoid when**
- You have large state and/or complex DQ requirements that are cheaper to satisfy in batch

**Tech:** Apache Kafka, Apache Flink.

---

## When to Use What — Decision Framework

This is the question interviewers actually care about. Walk the requirement *down* the spectrum and stop at the first row that satisfies the business need.

```
Q1: Does a human or system act on this data in < 1 second?
      ├─ YES → STREAMING / REAL-TIME  (fraud, alerting, live UX)
      └─ NO  → continue
Q2: Do you need fresh data multiple times per hour (intraday)?
      ├─ YES → NEAR REAL-TIME  (operational dashboards, ops monitoring)
      └─ NO  → continue
Q3: Is a once-or-twice-a-day refresh acceptable?
      ├─ YES → BATCH / MICROBATCH  (reporting, ML training, reconciliation)
      └─ NO  → re-examine the requirement — "real-time" is often a want, not a need
```

### Decision cheat-sheet

| If the requirement is… | Choose | Why |
|------------------------|--------|-----|
| Daily report / dashboard | **Batch** | Cheapest, strongest DQ, simplest to operate |
| ML feature/training tables | **Batch / Microbatch** | Reproducibility & full-dataset validation matter more than freshness |
| Heavy DQ on high volume | **Microbatch** | Bounded sets are easy to validate; cost-efficient |
| "Fresh-ish" ops dashboard (minutes) | **Near real-time** | Intraday freshness without streaming complexity |
| Customer-facing reaction (seconds) | **Real-time** | Event-driven, low latency, manageable state |
| Fraud / security / live personalization | **Streaming** | Per-event, lowest latency, continuous state |

### The trade-offs you must mention in an interview

1. **Latency vs cost & complexity** — every step toward real-time multiplies operational burden (24/7 on-call, state stores, backpressure, exactly-once).
2. **Data quality** — batch sees the whole dataset and can validate/reconcile; streaming sees one event at a time, so DQ is far harder. Late and out-of-order data become real problems.
3. **Reprocessing / backfills** — trivial in batch (re-run the job), painful in streaming (replay from Kafka offsets, rebuild state).
4. **State management** — bounded in batch, unbounded and continuous in streaming (windows, watermarks, checkpointing).
5. **Team expertise & on-call** — a real-time system is only as good as the team's ability to operate it at 3am.

> **Interview soundbite:** *"Pick the highest-latency pipeline that still meets the SLA. Latency is a cost you pay in engineering complexity, data-quality risk, and on-call burden — so only buy as much of it as the business actually needs."*

---

## Tech Stack Quick Reference

| Technology | Best fit | Processing model | Notes |
|------------|----------|------------------|-------|
| **Apache Spark** | Batch, Microbatch | Distributed batch | Industry standard for large-scale ETL & ML data prep |
| **Spark Structured Streaming** | Microbatch, Near real-time | Micro-batch (trigger intervals); has a low-latency continuous mode | Same Spark API/tech stack — easy graduation path from batch |
| **Apache Flink** | Real-time, Streaming | True event-at-a-time | Best-in-class state mgmt, event-time, watermarks, exactly-once |
| **Apache Kafka** | Real-time, Streaming transport | Distributed log / Kafka Streams | The backbone for moving events; replayable via offsets |

**Mental model:** *Spark for bounded data, Flink for unbounded data, Kafka to move events between them.*

---

## Interview Questions & Answers

**Q: What's the difference between microbatch and streaming?**
Microbatch groups events into small bounded batches processed on a trigger interval (seconds to minutes) — Spark Structured Streaming's default. Streaming processes each event individually, continuously, with no batch boundary (Flink). Microbatch trades a little latency for simpler DQ and a familiar batch mental model; streaming buys lower latency at the cost of harder state management.

**Q: Why not just use real-time/streaming for everything?**
Because latency is expensive. Streaming adds 24/7 operations, continuous state stores, backpressure handling, exactly-once semantics, harder data-quality validation, and painful reprocessing. If a daily batch meets the SLA, streaming is wasted cost and added risk. Choose the highest-latency option that meets the requirement.

**Q: When would you pick batch over streaming even though streaming is "better"?**
When the consumer doesn't need sub-hour freshness (reports, ML training, financial reconciliation), when you need strong data-quality guarantees over the full dataset, when you want cheap reprocessing/backfills, or when the team can't operate a streaming system reliably.

**Q: What is near real-time and when is it the right choice?**
Small micro-batches every 2–5 minutes — the bridge between batch and streaming. Pick it when you need intraday freshness (operational dashboards, monitoring) and complex transformations, but millisecond latency isn't required and you still want decent DQ. It avoids streaming's complexity while beating batch's staleness.

**Q: What makes data quality harder in streaming than in batch?**
Batch sees the entire bounded dataset, so it can dedupe, validate against totals, and reconcile. Streaming sees one event at a time with no end, so you must handle out-of-order and late-arriving events (via watermarks), maintain correct windowed state, and accept that some checks (e.g., "does today's total reconcile?") simply can't be done until the window closes.

**Q: What are watermarks and why do they matter?**
A watermark is the engine's notion of "event-time progress" — a threshold that says *"I don't expect events older than this anymore."* It lets a streaming system decide when a window is complete and it's safe to emit results and drop state, while still tolerating bounded lateness. Without watermarks, windows never close and state grows unbounded.

**Q: Spark vs Flink — when do you choose each?**
Spark (incl. Structured Streaming) excels at batch and micro-batch over bounded data with a unified batch/stream API — the natural choice if you're already a Spark shop and need microbatch/near-real-time. Flink is a true event-at-a-time engine with superior state management, event-time processing, and exactly-once — choose it for genuine low-latency streaming and complex stateful operations.

**Q: How do you handle late-arriving data?**
In **batch**, late data is just picked up in the next scheduled run (or a targeted backfill of the affected partition). In **streaming**, you define an allowed-lateness window with watermarks: events within the watermark update their window; events beyond it are dropped or routed to a side output for separate handling.

**Q: A stakeholder says "we need this in real-time." How do you respond?**
Clarify the *actual* requirement: who/what consumes the data, what decision it drives, and the real freshness SLA. "Real-time" is frequently a want, not a need. Quantify the cost difference, then recommend the highest-latency tier that meets the genuine SLA — often near real-time or microbatch rather than full streaming.

---

## Key Takeaways

- Pipelines form a spectrum: **Batch → Microbatch → Near real-time → Real-time → Streaming**, trading **latency** for **engineering complexity** and **data-quality difficulty**.
- **Default to batch.** Move faster only when the business SLA genuinely requires it.
- **Spark** owns bounded/batch; **Flink** owns unbounded/streaming; **Kafka** moves the events.
- The interview win is articulating the **trade-offs** (cost, DQ, reprocessing, state, on-call) — not naming the lowest-latency tool.

---

### Related Guides
- [Apache Spark](./intro_apache_spark.md)
- [Apache Kafka](./intro_apache_kafka.md)
- [Apache Airflow](./intro_apache_airflow.md)
- [Data Architecture](./data-architecture.md)
- [Data Modeling](./data-modeling.md)
