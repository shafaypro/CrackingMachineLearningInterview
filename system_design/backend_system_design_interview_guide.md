# Backend System Design Interview Guide

This guide adapts the provided standalone HTML system design guide into the repo's existing markdown-driven study flow. It is meant to complement the ML-first design framework with a broader backend interview checklist.

---

## Table of Contents

1. [At a Glance](#at-a-glance)
2. [Top 10 to Memorize First](#top-10-to-memorize-first)
3. [How to Use This Guide](#how-to-use-this-guide)
4. [Reference Backend Architecture](#reference-backend-architecture)
5. [How to Explain Architecture in an Interview](#how-to-explain-architecture-in-an-interview)
6. [Scalability](#1-scalability)
7. [Databases](#2-databases)
8. [Caching](#3-caching)
9. [Distributed Systems](#4-distributed-systems)
10. [Messaging and Events](#5-messaging-and-events)
11. [Networking and APIs](#6-networking-and-apis)
12. [Storage](#7-storage)
13. [Reliability and Fault Tolerance](#8-reliability-and-fault-tolerance)
14. [Search and Real-Time Systems](#9-search-and-real-time-systems)
15. [Worked Architecture Examples](#worked-architecture-examples)
16. [Operational Review Checklist](#operational-review-checklist)
17. [Key Trade-Off Cheat Sheet](#key-trade-off-cheat-sheet)
18. [Suggested Interview Walkthrough](#suggested-interview-walkthrough)
19. [Common Mistakes in Backend Interviews](#common-mistakes-in-backend-interviews)
20. [References](#references)
21. [Recommended Follow-Ups in This Repo](#recommended-follow-ups-in-this-repo)

---

## At a Glance

- Scope: 32 high-yield backend system design concepts
- Priority set: 10 concepts worth memorizing first
- Coverage: scalability, databases, caching, distributed systems, messaging, networking, storage, reliability, and search
- Best use: review this page before system design interviews, then go deeper into the case-study files in this track

---

## Top 10 to Memorize First

1. Load balancing
2. Caching strategies
3. SQL vs NoSQL
4. Database sharding
5. CAP theorem
6. Consistent hashing
7. Message queues and Kafka
8. Horizontal vs vertical scaling
9. Circuit breaker pattern
10. REST vs GraphQL vs gRPC

Why these first:
- They show up repeatedly in product, platform, and infrastructure interviews.
- They give you enough vocabulary to explain most trade-offs in larger system design answers.
- They form the backbone of common discussions around scale, reliability, and data architecture.

---

## How to Use This Guide

For each concept, focus on five things:

1. Definition: what the concept is in plain language
2. Why it matters: why interviewers care
3. Example: where it appears in real systems
4. Common prompts: how it shows up in interview questions
5. Trade-off: when it helps and what it costs

---

## Reference Backend Architecture

Most interview systems can be explained as a variation of this baseline:

```text
Client
  |
  v
DNS / CDN / Edge
  |
  v
Load Balancer / API Gateway
  |
  v
Stateless Application Services
  |                 \
  |                  \--> Cache (Redis / Memcached)
  |
  +--> Primary Database / Search Index / Object Store
  |
  +--> Message Queue / Event Stream
          |
          v
       Async Workers / Consumers
          |
          v
   Secondary Stores / Notifications / Analytics / ML Pipelines
```

### What each layer does

- Client: browser, mobile app, internal service, or batch producer making requests
- DNS / CDN / Edge: routes the request and offloads static assets or geo-distributed traffic
- Load Balancer / API Gateway: terminates TLS, routes traffic, applies auth, rate limits, and observability hooks
- Stateless application services: enforce business logic and stay horizontally scalable
- Cache: reduces read latency and protects the database from repeated hot queries
- Primary datastore: holds the source of truth for transactional or user-facing state
- Queue / event bus: moves slow, high-volume, or decoupled work off the synchronous path
- Workers: process async jobs such as notifications, indexing, retries, or media processing
- Secondary systems: analytics stores, search indexes, object stores, ML pipelines, or third-party integrations

### Hot path versus async path

A strong answer usually distinguishes:

- the latency-sensitive path: what must happen before the user gets a response
- the deferred path: what can happen after the response

Examples:

- In checkout, charging the card may be on the hot path, but sending an email is async.
- In a feed system, writing the post may be synchronous, but fan-out and ranking can be async.
- In media upload, generating thumbnails and indexing metadata are usually async.

### A simple request lifecycle

For a read-heavy endpoint:

1. Client sends request
2. Load balancer routes to an app instance
3. App checks cache
4. On cache hit, return immediately
5. On cache miss, query database
6. Store result in cache
7. Return response

For a write-heavy endpoint:

1. Client sends mutation request
2. App validates auth and business rules
3. App writes durable state to source-of-truth store
4. App emits event to queue or stream
5. Background workers update search, notifications, analytics, or downstream materialized views

---

## How to Explain Architecture in an Interview

Interviewers are not looking for a giant diagram first. They want a sequence of defensible decisions.

### Good explanation order

1. Clarify requirements
2. State system APIs or core entities
3. Draw the happy path
4. Choose storage
5. Add scale controls
6. Add failure handling
7. Add observability and operations

### Questions you should ask first

- What is the scale in users, QPS, write rate, and storage growth?
- What latency matters: p50 only, or p95/p99?
- Is the system read-heavy or write-heavy?
- Is correctness strict or is eventual consistency acceptable?
- Do we need multi-region or is one region enough?
- Are there compliance, privacy, or audit constraints?

### What strong answers sound like

Instead of saying:

- "I'll use microservices and Kafka and Redis."

Say:

- "The API tier stays stateless behind a load balancer so we can scale horizontally."
- "The primary write goes to a transactional store because correctness matters."
- "Search indexing and notifications are moved to Kafka consumers so the user-facing path stays under 100 ms."
- "We'll place Redis in front of the hot read endpoints because traffic is skewed and repeated."

### Capacity estimation you should be able to do quickly

Even rough math earns points if it is directionally correct.

Examples:

- `10M` daily active users with `5` requests per day is about `50M` requests/day.
- `50M / 86400` is roughly `580` requests/sec average.
- If peak is `5x` average, design for about `3K` requests/sec.
- If each write is `2 KB`, and there are `20M` writes/day, raw write volume is about `40 GB/day` before replication and indexing overhead.

You do not need exact precision. You need enough math to justify architecture choices.

---

## 1. Scalability

### 1. Horizontal vs Vertical Scaling

- Definition: Vertical scaling adds more CPU, RAM, or disk to one machine. Horizontal scaling adds more machines and distributes load across them.
- Why it matters: Almost every "design for millions of users" problem starts with a scaling decision.
- Example: Netflix scales service fleets horizontally across many instances instead of relying on a single oversized server.
- Common prompts: "How would you scale this API by 10x?" "Design YouTube, Twitter, or Instagram."
- Trade-off: Vertical scaling is simpler but capped. Horizontal scaling removes the single-box limit but introduces distributed systems complexity.

### 2. Load Balancing

- Definition: A load balancer distributes incoming requests across multiple backends. Common strategies include round robin, least connections, and layer-7 routing.
- Why it matters: It is usually the first scaling component interviewers expect you to mention.
- Example: Ride requests can be routed through regional load balancers before reaching matching services.
- Common prompts: URL shortener, API gateway, "How do you handle traffic spikes?"
- Trade-off: Stateless backends are easier to balance; sticky sessions reduce rehydration costs but can create hot spots.

### 3. Rate Limiting

- Definition: Rate limiting constrains how many requests a client can issue in a time window. Common algorithms include token bucket, leaky bucket, fixed window, and sliding window.
- Why it matters: Protects against abuse, DDoS-style bursts, and noisy neighbors.
- Example: Public APIs often enforce per-user or per-key quotas to keep traffic fair.
- Common prompts: "Design a rate limiter." "How would you protect a shared API platform?"
- Trade-off: Tight limits protect infrastructure but can punish legitimate bursty traffic. Token bucket is a common compromise.

### 4. Auto-Scaling

- Definition: Auto-scaling changes capacity dynamically based on metrics like CPU, request rate, queue depth, or latency.
- Why it matters: Shows you understand cost control as well as performance.
- Example: Order-processing workers can scale with queue backlog during peak traffic.
- Common prompts: "How would the system respond to seasonal spikes?" "How do you size workers for bursty workloads?"
- Trade-off: Aggressive scaling reduces latency but can increase cost and thrash if policies are noisy.

---

## 2. Databases

### 5. SQL vs NoSQL

- Definition: SQL databases favor relational structure, transactions, and rich querying. NoSQL systems favor flexible schemas and scale-oriented access patterns.
- Why it matters: Database choice is foundational in almost every design.
- Example: Payment ledgers fit relational databases; social activity feeds often lean toward NoSQL stores.
- Common prompts: "Which database would you choose and why?" "How do you store user content at scale?"
- Trade-off: SQL is strong on joins, consistency, and transactions. NoSQL is strong on horizontal scale and schema flexibility.

### 6. Database Indexing

- Definition: Indexes create fast access paths for lookups, sorting, and filtering without scanning the full table.
- Why it matters: Interviewers often check whether you understand how queries stay fast as data grows.
- Example: A composite index on `(user_id, created_at)` can speed timeline or order-history queries.
- Common prompts: "Why is the query slow?" "How do you optimize lookup latency?"
- Trade-off: Indexes improve reads but increase storage and write amplification.

### 7. Replication

- Definition: Replication copies data across multiple nodes for availability, durability, and read scaling.
- Why it matters: It is one of the first tools for high availability.
- Example: Primary-replica databases send writes to the primary and spread reads across replicas.
- Common prompts: "How do you improve read throughput?" "How do you survive node failure?"
- Trade-off: Replication improves resilience but introduces lag, failover complexity, and consistency questions.

### 8. Sharding

- Definition: Sharding partitions data across multiple databases so one machine does not hold all data or traffic.
- Why it matters: Eventually many fast-growing systems outgrow a single database instance.
- Example: User data can be distributed by hashed user ID across many shards.
- Common prompts: "One database is too large. What now?" "How do you scale writes?"
- Trade-off: Sharding removes single-node limits but complicates joins, resharding, and hot key management.

---

## 3. Caching

### 9. Cache-Aside

- Definition: The application checks the cache first, falls back to the database on a miss, then writes the result into cache.
- Why it matters: It is the most common caching pattern in production systems.
- Example: Product detail pages are often cached in Redis after the first database read.
- Common prompts: "How do you reduce read load?" "How would you speed up a feed or profile page?"
- Trade-off: Simple to implement, but cache invalidation and staleness management become your responsibility.

### 10. Write-Through and Write-Behind Cache

- Definition: Write-through updates cache and backing store together. Write-behind writes to cache first and persists asynchronously later.
- Why it matters: Interviewers look for whether you understand caching on the write path, not only reads.
- Example: Session state may use write-through; high-volume counters may use write-behind buffering.
- Common prompts: "How do you cache hot writes?" "How would you handle very frequent updates?"
- Trade-off: Write-through improves consistency but increases write latency. Write-behind boosts throughput but risks data loss on failure.

### 11. CDN and Edge Caching

- Definition: A CDN caches content near users at edge locations to reduce latency and origin load.
- Why it matters: Global systems nearly always rely on edge caching for static and semi-static assets.
- Example: Images, JavaScript bundles, and video chunks are served from edge POPs instead of origin servers.
- Common prompts: "How do you improve global latency?" "How would you scale an image or video platform?"
- Trade-off: Excellent for read-heavy content, but invalidation and personalization can be difficult.

### 12. Consistent Hashing

- Definition: Consistent hashing maps keys to cache or storage nodes so that adding or removing nodes remaps only a small fraction of keys.
- Why it matters: It is a standard answer for distributed caches and partitioned key-value systems.
- Example: Memcached clusters commonly use consistent hashing to spread keys.
- Common prompts: "How do you distribute keys across many cache nodes?" "What happens when one node is added or removed?"
- Trade-off: Great for minimizing churn, but poor virtual-node strategy can still create uneven distribution.

---

## 4. Distributed Systems

### 13. CAP Theorem

- Definition: In the presence of a network partition, a distributed system can prioritize consistency or availability, but not fully guarantee both.
- Why it matters: It gives a framework for discussing consistency trade-offs clearly.
- Example: A banking ledger leans toward consistency; a social feed often leans toward availability.
- Common prompts: "Would you choose CP or AP here?" "What happens during a partition?"
- Trade-off: The theorem is not a design recipe, but it forces you to decide what the system sacrifices under failure.

### 14. Strong vs Eventual Consistency

- Definition: Strong consistency means every read reflects the latest successful write. Eventual consistency means replicas converge over time.
- Why it matters: Many interview questions hinge on whether stale reads are acceptable.
- Example: Inventory counts often need strong consistency; like counts can usually tolerate eventual consistency.
- Common prompts: "Can users see stale data?" "How much inconsistency can the product tolerate?"
- Trade-off: Strong consistency improves correctness but usually costs latency and availability.

### 15. Leader Election

- Definition: A distributed system may elect one node as leader to coordinate writes, scheduling, or ownership.
- Why it matters: It appears in consensus systems, schedulers, and service coordination.
- Example: Kubernetes control-plane components rely on leader election for safe coordination.
- Common prompts: "How do you prevent two coordinators from acting at once?" "How does failover happen?"
- Trade-off: Central leadership simplifies coordination but creates failover and split-brain considerations.

### 16. Consensus and Raft

- Definition: Consensus protocols let distributed nodes agree on a sequence of operations even when some nodes fail. Raft is a commonly discussed protocol because it is easier to reason about than Paxos.
- Why it matters: It comes up when discussing metadata stores, configuration management, and distributed locks.
- Example: etcd uses Raft for replicated metadata and leader election.
- Common prompts: "How do you keep replicas in sync?" "How does a distributed lock service work?"
- Trade-off: Consensus gives stronger guarantees but requires quorum and increases write-path coordination cost.

### 17. Two-Phase Commit

- Definition: Two-phase commit coordinates atomic changes across multiple participants by asking each to prepare, then commit or abort.
- Why it matters: Interviewers may ask how to preserve consistency across systems.
- Example: Legacy financial workflows sometimes used 2PC for cross-database updates.
- Common prompts: "How do you commit changes across services atomically?" "How do you avoid partial updates?"
- Trade-off: Strong but blocking, coordinator-dependent, and often avoided in large microservice architectures.

### 18. Saga Pattern

- Definition: A saga models a distributed transaction as a sequence of local transactions with compensating actions if a later step fails.
- Why it matters: It is the modern answer when 2PC is too rigid for microservices.
- Example: Travel booking can reserve a room, authorize payment, and compensate prior steps if later steps fail.
- Common prompts: "How do you handle partial failure in an order flow?" "How do services coordinate without a global transaction?"
- Trade-off: More resilient and scalable than 2PC, but harder to reason about and debug end to end.

---

## 5. Messaging and Events

### 19. Message Queues and Event Streaming

- Definition: Message queues decouple producers and consumers for task processing. Event streams provide durable ordered logs that multiple consumers can replay.
- Why it matters: Async communication is core to modern large systems.
- Example: Kafka powers activity feeds, audit streams, and analytics pipelines in many companies.
- Common prompts: Notification service, analytics pipeline, "How do you decouple services with different speeds?"
- Trade-off: Improves resilience and throughput, but increases operational complexity and requires idempotent consumers.

### 20. Event Sourcing

- Definition: Instead of storing only current state, the system stores the sequence of events that produced that state.
- Why it matters: Useful for auditability, replay, and debugging in domains with strict history requirements.
- Example: Ledger-like systems can reconstruct balances by replaying transaction events.
- Common prompts: "How do you support audit trails?" "How would you implement time travel or rollback?"
- Trade-off: Great for history and recovery, but current-state queries usually require snapshots or derived views.

### 21. CQRS

- Definition: Command Query Responsibility Segregation separates the write model from the read model so each can be optimized independently.
- Why it matters: Often paired with event-driven architectures when reads and writes have very different shapes.
- Example: Feed systems may precompute read-optimized timelines while writes flow through a separate command path.
- Common prompts: "How do you optimize for heavy reads and different write semantics?" "How do you support many query patterns?"
- Trade-off: Strong read optimization, but eventual consistency and extra infrastructure become part of the design.

---

## 6. Networking and APIs

### 22. REST vs GraphQL vs gRPC

- Definition: REST is resource-oriented over HTTP. GraphQL lets clients request exactly the fields they need. gRPC uses strongly typed schemas and efficient binary transport, usually over HTTP/2.
- Why it matters: API protocol choice affects latency, evolution, client ergonomics, and caching.
- Example: Internal service-to-service calls may use gRPC; public developer APIs may use REST or GraphQL.
- Common prompts: "How do you design the API?" "What protocol should mobile clients use?"
- Trade-off: REST is simple and cache-friendly, GraphQL reduces over-fetching, and gRPC is fast and typed but less browser-native.

### 23. WebSockets and Long Polling

- Definition: WebSockets keep a persistent bidirectional connection open. Long polling keeps an HTTP request open until data is available and then reconnects.
- Why it matters: Real-time systems need a push mechanism.
- Example: Chat apps use persistent connections to deliver messages and presence updates.
- Common prompts: Chat system, live sports, collaborative editing, "How do you push updates in real time?"
- Trade-off: WebSockets are efficient at scale for bidirectional traffic but require connection state management.

### 24. DNS and Anycast Routing

- Definition: DNS maps names to addresses. Anycast advertises the same IP from many locations so traffic naturally reaches the nearest healthy edge.
- Why it matters: Global systems start with traffic steering before application logic even runs.
- Example: Edge providers route users to the closest data center to reduce latency and absorb attacks.
- Common prompts: "How does a request reach your system?" "How do you design a global service?"
- Trade-off: Powerful for geo-routing and resilience, but routing behavior can be difficult to debug.

---

## 7. Storage

### 25. Object vs Block vs File Storage

- Definition: Object storage handles blobs at large scale, block storage acts like attached disks, and file storage exposes shared hierarchical file systems.
- Why it matters: Choosing the wrong storage type can hurt latency, cost, and operational simplicity.
- Example: Media files often live in object storage while transactional metadata lives in a database.
- Common prompts: "How would you store uploads?" "Where should user-generated media live?"
- Trade-off: Object storage is cheap and durable but higher latency; block storage is fast but less elastic; file storage is convenient but not always the most scalable.

### 26. Data Partitioning Strategies

- Definition: Range, hash, and list partitioning distribute data based on different keying schemes.
- Why it matters: Partitioning strategy strongly affects balance, query efficiency, and hot spot risk.
- Example: Time-series systems often partition by time range, while general key-value systems often partition by hashed key.
- Common prompts: "Why is one shard 10x hotter?" "How do you support efficient range scans?"
- Trade-off: Hashing balances load but hurts range locality; range partitioning helps range queries but risks hot partitions.

---

## 8. Reliability and Fault Tolerance

### 27. Circuit Breaker Pattern

- Definition: A circuit breaker stops repeated calls to a failing dependency after an error threshold and retries only after a cooldown window.
- Why it matters: Prevents cascading failures in service meshes and microservice graphs.
- Example: A checkout service can fast-fail calls to a degraded recommendation service instead of exhausting threads.
- Common prompts: "What happens if a downstream service is failing?" "How do you contain blast radius?"
- Trade-off: Improves system stability, but threshold tuning matters or you risk false opens or slow protection.

### 28. Idempotency

- Definition: An operation is idempotent if repeating it does not change the final result beyond the first successful application.
- Why it matters: Safe retries depend on idempotency.
- Example: Payment APIs use idempotency keys so retrying a timed-out request does not double charge the user.
- Common prompts: "How do you handle duplicate requests?" "How do retries avoid creating duplicate orders?"
- Trade-off: Usually requires request tracking, deduplication state, and careful API contract design.

### 29. Health Checks, Timeouts, and Retries

- Definition: Health checks remove unhealthy instances, timeouts bound waiting time, and retries recover from transient faults.
- Why it matters: These are baseline reliability controls for any production system.
- Example: A load balancer stops routing to a slow or unhealthy instance while clients retry with backoff.
- Common prompts: "How do you handle slow dependencies?" "What do you do when a service becomes flaky?"
- Trade-off: Retries improve transient-failure handling but can create retry storms without backoff, jitter, and circuit breakers.

### 30. Replication Factor and Quorum

- Definition: Replication factor decides how many copies of data exist. Quorum reads and writes require a majority or another chosen threshold.
- Why it matters: It lets you tune durability, availability, and consistency explicitly.
- Example: Distributed databases can offer consistency levels like `ONE`, `QUORUM`, or `ALL`.
- Common prompts: "How do you guarantee durability?" "How do you balance consistency and latency?"
- Trade-off: Higher replication improves durability but increases storage cost and often write latency.

---

## 9. Search and Real-Time Systems

### 31. Search Indexes

- Definition: Inverted indexes map terms to documents so full-text search can run efficiently at scale.
- Why it matters: Traditional database indexes are not enough for large search products.
- Example: Restaurant or marketplace search often uses Elasticsearch for text, filters, and ranking.
- Common prompts: "How would you build search?" "How do you implement autocomplete or faceted discovery?"
- Trade-off: Search engines are powerful but create sync and consistency work between the primary database and the index.

### 32. Time-Series Data and Storage

- Definition: Time-series systems are optimized for append-heavy writes, retention policies, and range queries over timestamps.
- Why it matters: Monitoring, IoT, analytics, and location streams all produce time-based data.
- Example: Metrics platforms store CPU, memory, latency, and service health as time-series records.
- Common prompts: "Design a metrics platform." "How do you store and query high-volume telemetry?"
- Trade-off: Time-series stores excel on time-based access patterns, but they are not general-purpose transactional databases.

---

## Worked Architecture Examples

These are compact examples of how the concepts in this guide fit together in actual interview answers.

### Example 1: URL Shortener

#### Requirements

- Create short links quickly
- Redirect with low latency
- Handle read traffic much higher than write traffic
- Avoid collisions in short-code generation

#### High-level design

```text
Client -> LB/API -> Shortener Service -> SQL/NoSQL Store
                               |
                               +-> Cache for hot short codes
                               +-> Async analytics event stream
```

#### Key decisions

- Use a durable primary store for mapping `short_code -> long_url`
- Cache the hottest short codes because redirect traffic is skewed
- Keep analytics off the hot redirect path by sending click events asynchronously
- Pre-generate IDs or use a central ID service to avoid short-code collisions

#### Trade-offs to discuss

- SQL is fine early because the entity is simple and correctness matters
- NoSQL becomes more attractive if partitioned key lookups dominate at very large scale
- Redirect latency should take priority over perfect real-time analytics

### Example 2: Notification System

#### Requirements

- Accept notification requests from many upstream services
- Support email, push, SMS, and in-app notifications
- Handle spikes without losing requests
- Support retries and provider fallback

#### High-level design

```text
Producer Services
    |
    v
Notification API -> Queue / Topic -> Channel Workers -> External Providers
                         |
                         +-> Status Store / Retry DLQ / Audit Log
```

#### Key decisions

- Use a queue because notifications are naturally asynchronous
- Separate workers by channel to isolate provider issues
- Store delivery attempts and status transitions for debugging and audit
- Use retry with backoff and dead-letter queues for poison messages

#### Failure cases to discuss

- external SMS provider times out
- push provider rate-limits requests
- duplicate requests arrive from upstream callers
- user preference service is temporarily unavailable

Strong interview point:
- mention idempotency early because notification systems are retry-heavy and duplicate sends are a common production failure

### Example 3: Chat or Real-Time Messaging

#### Requirements

- Low-latency send and receive
- Message persistence
- Online presence
- Ordering within a conversation

#### High-level design

```text
Client <-> WebSocket Gateway <-> Chat Service
                               |      \
                               |       \-> Presence Store (Redis)
                               |
                               +-> Message Store
                               +-> Stream / Queue for fan-out, push, analytics
```

#### Key decisions

- Use WebSockets for bidirectional real-time communication
- Store message history durably in a database
- Use Redis or similar for ephemeral presence state
- Use async fan-out to notify other connected users or devices

#### Trade-offs to discuss

- strict global ordering is expensive; per-conversation ordering is often enough
- presence is ephemeral and can tolerate some inaccuracy
- multi-device sync may require sequence numbers or message offsets

### Example 4: Search and Discovery Service

#### Requirements

- Full-text search with filters
- Relevance ranking
- Near-real-time updates
- Support typo tolerance or stemming

#### High-level design

```text
Writers -> Primary DB -> Change Events -> Indexer -> Search Engine
Clients -> API -> Search Service -> Search Engine
```

#### Key decisions

- Keep source-of-truth writes in the primary database
- Build the search index asynchronously
- Accept eventual consistency between DB and index
- Use specialized search infrastructure instead of stretching an OLTP database

#### Common follow-up

Interviewers often ask how stale the index can be. Your answer should tie staleness budget to product expectations.

---

## Operational Review Checklist

Before finishing a backend design answer, quickly review these dimensions.

### Functional shape

- What are the core APIs?
- What are the main entities and their relationships?
- What is the read path?
- What is the write path?

### Scale

- expected QPS and peak QPS
- storage growth over time
- read/write ratio
- request and payload size

### Reliability

- timeouts
- retries with backoff and jitter
- circuit breakers
- dead-letter queues
- graceful degradation path

### Data correctness

- consistency model
- idempotency strategy
- deduplication strategy
- backup and restore plan
- replication and failover

### Operations

- metrics: QPS, latency, error rate, queue depth
- logs: structured request and failure logs
- tracing: cross-service call visibility
- alerts: SLO-based rather than noise-based

### Security

- auth and authz model
- rate limiting and abuse controls
- encryption in transit and at rest
- audit logging where needed

---

## Key Trade-Off Cheat Sheet

| Trade-Off | Choose A When... | Choose B When... |
|---|---|---|
| Consistency vs Availability | Bank transfers, inventory, strict correctness | Social feeds, shopping carts, graceful degradation matters more |
| SQL vs NoSQL | Complex joins, transactions, strong schema discipline | Massive write scale, flexible schema, denormalized access |
| Sync vs Async | User needs immediate result on the hot path | Throughput, decoupling, and resilience matter more |
| Cache-Aside vs Write-Through | Read-heavy workloads can tolerate some staleness | Writes must keep cache close to source of truth |
| Monolith vs Microservices | Early stage product, small team, simpler deployment | Independent scaling, team autonomy, bounded contexts |
| Strong vs Eventual Consistency | Correctness must dominate latency | Slightly stale reads are acceptable |
| Normalization vs Denormalization | Write-heavy data with strong integrity constraints | Read-heavy systems that benefit from precomputed views |

---

## Suggested Interview Walkthrough

When answering a system design problem, move in this order:

1. Clarify users, traffic, latency, and correctness requirements.
2. Draw the happy path end to end.
3. Choose storage and explain read and write patterns.
4. Add caches, queues, and background workers where they remove pressure from the hot path.
5. Explain scaling: load balancers, sharding, replication, and auto-scaling.
6. Explain failure handling: retries, timeouts, circuit breakers, idempotency, and fallback behavior.
7. Close with observability, cost, and the key trade-offs you deliberately chose.

---

## Common Mistakes in Backend Interviews

- jumping into microservices before establishing the core data model and traffic shape
- naming tools without explaining why they are needed
- forgetting the difference between the synchronous user path and the background path
- ignoring failure modes and only describing the happy path
- choosing strong consistency everywhere without discussing latency and availability cost
- choosing eventual consistency everywhere without checking whether correctness permits it
- omitting idempotency in retry-heavy workflows
- skipping observability entirely
- never mentioning backpressure, queue depth, or hot keys in high-scale designs

---

## References

- Designing Data-Intensive Applications, Martin Kleppmann
- System Design Interview, Alex Xu
- Web Scalability for Startup Engineers, Artur Ejsmont
- Release It!, Michael T. Nygard
- Site Reliability Engineering, Google
- The Architecture of Open Source Applications
- Kafka, Redis, PostgreSQL, and Elasticsearch official documentation

---

## Recommended Follow-Ups in This Repo

- [ML System Design Framework](./README.md)
- [Backend and System Design for AI](./intro_backend_ai_system_design.md)
- [Recommendation System](./recommendation_system.md)
- [Fraud Detection](./fraud_detection.md)
- [ML System Design Patterns](./ml_system_design_patterns.md)
