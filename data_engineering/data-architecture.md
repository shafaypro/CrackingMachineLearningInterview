# Data Architecture

Data architecture is the operating design of how data moves, changes, is governed, and creates value across the enterprise. It is broader than a storage choice and deeper than a reference diagram. A good architecture defines responsibilities, interfaces, failure modes, quality controls, security boundaries, and ownership models that still work when the platform is under real production pressure.

For principal-level work, the goal is not to chase fashionable patterns. The goal is to choose an architecture that matches business volatility, latency needs, team topology, governance obligations, and cost limits.

---

## What Data Architecture Is and Why It Matters

Data architecture answers questions such as:

- where data originates
- how data enters the platform
- how raw data becomes trusted data
- which systems perform transformation and serving
- how data is governed, secured, and observed
- who owns which data products and contracts

It matters because architecture determines:

- delivery speed for new use cases
- platform reliability
- data quality and trust
- recovery from failures and bad releases
- total platform cost
- the number of teams that can work independently without breaking each other

Weak architecture often looks fine in a diagram and fails in operations. Strong architecture reduces ambiguity in how the platform should behave under change, scale, and incident conditions.

---

## Core Principles of Modern Data Architecture

### 1. Treat Data as a Product

Important data assets need:

- clear ownership
- quality expectations
- discoverability
- contracts
- lifecycle management

### 2. Separate Storage from Serving Semantics

Raw landing zones, integration layers, curated marts, and low-latency serving layers should not be conflated. They serve different purposes and should be optimized differently.

### 3. Design for Change, Not Just Current Volume

Most enterprise pain comes from schema drift, source changes, new domains, and evolving ownership, not from the first version of the pipeline.

### 4. Make Lineage and Observability First-Class

If you cannot explain where a metric came from, who changed a schema, or which downstream assets are affected by a failed job, the architecture is incomplete.

### 5. Prefer Explicit Contracts Over Tribal Knowledge

Use schema contracts, data product definitions, testable SLAs, and versioned interfaces. Verbal agreements do not scale.

### 6. Optimize for Business Value, Not Maximal Technical Purity

The most elegant architecture is often the wrong one if the team cannot operate it.

---

## Layers of a Data Platform

The exact implementation differs by organization, but the logical layers are broadly consistent.

| Layer | Purpose | Typical Technologies | Key Design Questions |
|---|---|---|---|
| Source systems | Systems that generate data | SaaS apps, OLTP DBs, event producers, logs, IoT | Who owns semantics at the source? |
| Ingestion | Move data into the platform | Fivetran, Kafka, Debezium, Airbyte, custom CDC | Batch, CDC, or streaming? |
| Storage | Durable raw and curated persistence | Data warehouse, object storage, lakehouse tables | What is the system of record? |
| Processing | Transform, enrich, reconcile | Spark, dbt, Flink, Beam, SQL engines | Where do business rules live? |
| Serving | Expose usable data for workloads | marts, feature stores, serving DBs, APIs | What latency and freshness are required? |
| Consumption | End-user or system usage | BI tools, notebooks, ML pipelines, apps | Who are the consumers and how self-service is it? |
| Governance | Control policy and trust | catalog, lineage, RBAC, contracts, quality tools | How are ownership and policy enforced? |
| Observability | Monitor system health and data health | logging, metrics, tracing, anomaly detection | How do we detect and contain failures? |

### Practical Platform Flow

```text
Sources -> Ingestion -> Raw Storage -> Processing -> Curated Storage -> Serving -> Consumption
                           |                |             |
                      Governance       Observability   Security
```

This is intentionally simple. Real platforms add feedback loops, replay paths, quarantine zones, and lineage controls.

---

## Batch vs Streaming Architectures

This is one of the most over-simplified choices in data engineering. The correct decision depends on freshness, correctness, operational overhead, and downstream actionability.

| Topic | Batch | Streaming |
|---|---|---|
| Freshness | Minutes to days | Seconds to sub-second |
| Complexity | Lower | Higher |
| Cost predictability | Higher | Lower |
| Operational burden | Lower | Higher |
| Typical strengths | Simplicity, reprocessing, finance-grade reporting | Event-driven use cases, personalization, monitoring |

### When to Use Batch

- daily reporting
- finance reconciliation
- warehouse transformations
- ML training data generation
- workloads where a 15-minute or 24-hour delay is acceptable

### When Not to Use Batch

- fraud detection
- low-latency customer notifications
- operational metrics and alerting
- real-time personalization

### When to Use Streaming

- event-driven product features
- near-real-time operational analytics
- CDC replication with low freshness tolerance
- online features or risk signals

### When Not to Use Streaming

- when no consumer acts on fresh data
- when the team lacks capability to operate stateful streaming safely
- when batch satisfies the business requirement at far lower cost

### Common Mistakes

- choosing streaming because it feels more advanced
- building streaming pipelines that still land data for hourly reporting only
- underestimating replay, state management, and late data handling

---

## Batch Layer, Speed Layer, and Serving Layer

These layers matter most in Lambda-style thinking, but the concepts are useful even outside Lambda architecture.

### Batch Layer

The batch layer computes complete, authoritative views from the full historical dataset.

Responsibilities:

- reprocessing from durable history
- large-scale reconciliations
- high-correctness aggregations
- rebuilding trusted outputs after logic changes

When the business cares about correctness more than immediacy, the batch layer is the safety net.

### Speed Layer

The speed layer computes incremental results from recent events to provide freshness before the batch layer catches up.

Responsibilities:

- reduce staleness
- maintain real-time or near-real-time views
- provide provisional outputs until full recomputation occurs

The speed layer is useful when:

- stale data harms the product
- recent events matter materially
- consumers can tolerate eventual correction

### Serving Layer

The serving layer exposes data in a form optimized for consumption. It is not just storage. It is a contract boundary.

Examples:

- BI marts and semantic models
- aggregate tables
- feature store online tables
- low-latency APIs backed by materialized views

Serving design questions:

- what latency must queries meet
- what freshness guarantees are required
- are consumers human, analytical, or machine-driven
- do we serve raw entities, aggregates, or features

Common mistake: treating the warehouse itself as the serving layer for every workload, including low-latency application features.

---

## Lambda Architecture

Lambda architecture combines:

- a batch layer for correctness
- a speed layer for freshness
- a serving layer that merges the outputs

```text
Incoming events
   | \
   |  \-> Speed layer -> Real-time views
   |
   \----> Batch layer -> Recomputed views
                \          /
                 \-> Serving layer
```

### Pros

- strong correctness path through batch recomputation
- low-latency visibility through speed processing
- resilient when streaming results need later correction

### Cons

- duplicate logic across batch and speed paths
- higher engineering and testing cost
- difficult operational model for smaller teams

### When to Use Lambda Architecture

- strict correctness is required
- recent data must be visible quickly
- reprocessing full history is materially valuable
- streaming outputs may need reconciliation later

Typical examples:

- fraud scoring plus daily correction
- financial event monitoring plus audited end-of-day settlement
- large telemetry platforms with immediate alerting and later canonical recompute

### When Not to Use Lambda Architecture

- small teams
- simple warehouse reporting
- platforms that can tolerate micro-batch instead
- cases where one incremental model can satisfy both freshness and correctness

### Common Mistakes

- implementing Lambda before proving the business need
- letting batch and streaming semantics drift apart
- forgetting that the serving layer must reconcile duplicates and corrections

---

## Kappa Architecture

Kappa architecture removes the separate batch path and treats the streaming log as the primary source for both current processing and replay.

```text
Event log -> Stream processing -> Serving outputs
        \-> Replay same log when logic changes
```

### Pros

- one processing model instead of two
- cleaner mental model for event-native systems
- strong fit for log-centric architectures

### Cons

- replay and backfill can still be operationally hard
- not every analytical workload fits cleanly into streaming-first processing
- historical recomputation at scale may be less straightforward than a dedicated batch layer

### When to Use Kappa Architecture

- event-first systems with durable logs
- operational analytics and real-time data products
- teams mature in stream processing
- use cases where replaying the log is realistic and sufficient

### When Not to Use Kappa Architecture

- heavy warehouse-centric analytics programs
- environments without durable, replayable event logs
- teams that primarily operate SQL batch transformations

### Common Mistakes

- assuming Kappa eliminates all backfill complexity
- forcing every domain into streaming when many are batch-native
- underinvesting in event schemas and replay tooling

---

## Delta Architecture

The term "Delta architecture" is used inconsistently in the industry. In practice, teams usually mean an incremental architecture centered on transactional lakehouse tables, CDC, and change propagation rather than separate Lambda-style paths.

In this model:

- raw data lands in append-friendly tables
- changes are merged incrementally into curated tables
- downstream consumers read transactional snapshots or change feeds
- batch and streaming often converge on the same table abstraction

### How It Differs from Lambda and Kappa

| Topic | Lambda | Kappa | Delta Architecture |
|---|---|---|---|
| Primary abstraction | Two computation paths | Event log | Transactional change tables |
| Batch vs stream split | Explicit | Mostly stream-first | Often unified on table format |
| Replay model | Batch recompute + stream | Replay log | Rebuild from raw plus change tables |
| Common technology fit | Hadoop-era hybrid stacks | Kafka/Flink-heavy stacks | Delta Lake, Iceberg, Hudi, streaming upserts |

### When to Use Delta Architecture

- lakehouse platforms
- CDC-heavy ingestion
- mixed batch and near-real-time requirements
- teams that want simpler convergence than full Lambda

### When Not to Use It

- purely operational event systems better served by log-native Kappa patterns
- small organizations where a managed warehouse is simpler

### Pros

- reduced duplication of logic
- strong compatibility with medallion design
- good fit for replay, time travel, and incremental processing

### Cons

- requires disciplined table maintenance
- performance can degrade without compaction and clustering
- teams may overestimate how "real-time" table-centric pipelines are

### Common Mistakes

- assuming table ACID features eliminate the need for contracts and observability
- mixing raw and curated semantics in the same table tier
- ignoring file layout and compaction economics

---

## Lakehouse Architecture

Lakehouse architecture combines:

- low-cost object storage
- table semantics such as ACID transactions and schema evolution
- support for analytics, data science, and ML on shared storage

### Pros

- one platform for multiple workloads
- open storage format options
- better alignment between analytics and ML data access

### Cons

- governance can be weaker than in tightly managed warehouses
- performance engineering becomes the platform team’s responsibility
- easy to accumulate raw-data sprawl

### When to Use Lakehouse

- mixed BI, ML, and data science workloads
- large historical storage needs
- multi-engine access patterns
- organizations that want open table formats and reduced warehouse lock-in

### When Not to Use Lakehouse

- small analytics teams with no platform engineering capacity
- organizations needing a simpler managed operating model

### Lakehouse vs Data Warehouse

| Topic | Data Warehouse | Lakehouse |
|---|---|---|
| Operational model | More managed | More flexible, more responsibility |
| Workload breadth | Analytics-first | Analytics plus DS and ML |
| Open storage | Usually less open | Usually more open |
| Governance defaults | Often stronger out of the box | Must be implemented carefully |

Opinionated guidance: if the organization mainly needs governed analytics and standard BI, a warehouse is often the better operating choice. If the organization needs a shared data platform for analytics, ML, and large-scale raw retention, a lakehouse is often the better architectural choice.

---

## Medallion Architecture

Medallion architecture structures data into quality tiers, commonly:

- Bronze: raw landed data
- Silver: cleaned, standardized, conformed data
- Gold: serving-ready business models and aggregates

### When to Use

- lakehouse programs
- teams needing clear separation of raw, conformed, and serving layers
- environments with multiple consumers and replay needs

### When Not to Use

- if teams interpret medallion as a substitute for ownership and semantics
- if every dataset is copied through every layer with no reason

### Common Mistakes

- defining tiers physically but not semantically
- creating unnecessary copies at every stage
- turning gold into an uncontrolled dumping ground

---

## Data Mesh

Data mesh is an organizational and architectural model where domains own their data products and publish them through shared platform standards.

### Core Ideas

- domain ownership
- data as a product
- self-serve platform capabilities
- federated computational governance

### Pros

- aligns ownership with domain knowledge
- scales better across many business units
- reduces central bottlenecks when implemented well

### Cons

- requires strong platform enablement
- governance is harder than most teams expect
- uneven domain maturity creates uneven data quality

### When to Use

- large organizations with many domains
- central data team is a delivery bottleneck
- business units have genuine engineering capability

### When Not to Use

- small organizations
- weak engineering standards
- no shared metadata, identity, or governance platform

### Common Mistakes

- declaring "mesh" while keeping no product contracts
- decentralizing ownership without funding platform capabilities
- confusing federation with lack of standards

---

## Data Fabric

Data fabric focuses on connecting and governing data across distributed systems through metadata, automation, policy, and integration tooling.

It is more about interoperability and intelligent control across a fragmented estate than about domain ownership itself.

### Pros

- helps unify discovery, lineage, policy, and integration
- useful in heterogeneous enterprise environments
- can improve governance across existing fragmented platforms

### Cons

- sometimes marketed too vaguely
- can become tool-led rather than architecture-led
- does not solve ownership problems by itself

### When to Use

- complex enterprise landscapes with many platforms
- strong need for cross-system lineage, metadata, and policy automation
- modernization programs where full platform consolidation is unrealistic

### When Not to Use

- as a substitute for clear operating ownership
- if the real problem is weak domain accountability rather than metadata connectivity

---

## Data Mesh vs Data Fabric

| Topic | Data Mesh | Data Fabric |
|---|---|---|
| Primary focus | Ownership model and domain-aligned products | Cross-platform integration, metadata, and automation |
| Main problem solved | Central team bottlenecks and domain scalability | Fragmented systems and inconsistent governance |
| Dependency | Requires strong platform standards | Requires strong metadata and policy tooling |
| Risk | Inconsistent domain execution | Tool sprawl without operating model clarity |

They are not direct opposites. Large enterprises may use both:

- mesh for ownership
- fabric capabilities for metadata, discovery, policy, and integration across the estate

---

## Centralized vs Decentralized Data Platforms

### Centralized Platform

A central platform team owns ingestion, transformation, governance, and often modeling standards.

#### Pros

- strong consistency
- lower governance fragmentation
- easier platform standardization

#### Cons

- can become a delivery bottleneck
- central team may lack deep domain context

### Decentralized or Federated Platform

Domain teams own more of their pipelines and data products while a platform team provides shared capabilities and guardrails.

#### Pros

- closer alignment to business context
- more scalable ownership model in larger organizations

#### Cons

- higher risk of divergence
- requires mature governance and enablement

### Tradeoffs Between Centralized and Federated Ownership

| Topic | Centralized | Federated |
|---|---|---|
| Standardization | Stronger | Harder |
| Domain responsiveness | Lower | Higher |
| Governance overhead | Lower initially | Higher by design |
| Platform leverage | High | High only with good enablement |

Practical guidance: centralize foundational platform capabilities, security, metadata, and standards. Federate domain semantics and product ownership only when domain teams are capable of sustaining them.

---

## Monolith vs Domain-Oriented Data Platforms

A monolithic platform often means:

- one team
- one repo or transformation project
- one release flow
- weak separation of ownership boundaries

A domain-oriented platform usually means:

- bounded domain ownership
- domain-level contracts
- shared platform services
- explicit interfaces between domains

Monoliths are not always wrong. Small teams often benefit from one coherent platform until scale and coordination costs justify separation.

### When to Use a Monolith

- early-stage teams
- low domain complexity
- few engineers

### When to Move Toward Domain Orientation

- conflicting priorities between business domains
- release coordination overhead
- repeated ownership ambiguity
- metrics or entities with domain-specific semantics that keep colliding

---

## Event-Driven Architectures in Data Engineering

Event-driven architecture treats business events as first-class integration contracts.

Examples:

- `order_created`
- `payment_authorized`
- `shipment_delivered`
- `user_subscription_cancelled`

### Benefits

- strong decoupling
- better support for streaming use cases
- auditable change history
- easier propagation of business events across systems

### Risks

- low-quality event definitions create platform-wide confusion
- ordering and idempotency must be handled explicitly
- schema evolution becomes critical

### Best Practices

- define event ownership clearly
- separate event name from payload version
- include event time, idempotency key, and producer identity
- distinguish business events from technical events

---

## Real-Time and Near-Real-Time Processing Tradeoffs

Real-time has a cost. The business should earn that cost.

| Requirement | Recommended Pattern |
|---|---|
| Seconds matter and action is immediate | Streaming or event-driven serving |
| Minutes are acceptable | Micro-batch or incremental table updates |
| Daily or hourly is fine | Batch |

### Decision Questions

- what decision depends on freshness
- what is the economic value of lower latency
- can downstream systems act on fresher data
- what is the cost of wrong early results versus later corrected results

Common mistake: demanding "real-time" for dashboards nobody checks more than twice a day.

---

## Data Orchestration and Workflow Design

Orchestration manages dependency, retries, scheduling, and recoverability. It should not become the home of every transformation detail.

### Good Workflow Design

- tasks are idempotent
- retries are safe
- dependencies are explicit
- backfills are designed, not improvised
- failure domains are contained

### Common Orchestration Mistakes

- embedding large business logic directly in the orchestrator
- creating DAGs that are impossible to backfill safely
- mixing infrastructure concerns with transformation semantics
- using cron complexity where event triggers would be cleaner

### When to Use Central Orchestration

- batch dependencies are material
- SLAs need explicit tracking
- cross-system coordination is required

### When Not to Over-Orchestrate

- purely declarative warehouse transformations that can run within a simpler build graph
- streaming systems where event-time processing should be native to the stream engine

---

## Metadata, Lineage, Governance, and Security

These are not add-ons. They are architectural controls.

### Metadata

You need technical metadata and business metadata:

- schema
- ownership
- freshness
- tags
- lineage
- business definitions

### Lineage

Lineage should answer:

- where did this dataset come from
- which transformations produced it
- which dashboards, models, or APIs consume it
- what breaks if this table changes

### Governance

Governance should be implemented as operating controls, not just review boards:

- data classification
- retention policies
- access approval flows
- quality SLAs
- contract versioning

### Security

Security design should include:

- least privilege access
- row and column controls where required
- encryption in transit and at rest
- secret management
- audit logging

Common mistake: centralizing sensitive data in one lake without sufficiently granular access boundaries.

---

## Reliability, Scalability, and Cost Optimization

Architecture choices should be evaluated on all three together. Optimizing only one usually backfires.

### Reliability

Design for:

- replayability
- idempotency
- failure isolation
- schema rollback or compatibility
- observability and alerting

### Scalability

Design for:

- partition-aware processing
- horizontal scaling of stateless components
- bounded state in streaming jobs
- controlled concurrency in orchestration

### Cost Optimization

Design for:

- storage lifecycle tiers
- query pruning
- right-sized compute
- materialization only where reuse justifies it
- avoiding duplicate copies of low-value data

### Practical Guidance

- compute near data where possible
- compact small files in lakehouse platforms
- do not materialize every intermediate table permanently
- align retention policy with legal and analytical need

---

## Common Anti-Patterns

- one central platform with no documented ownership boundaries
- "real-time" everywhere, regardless of business value
- lakehouse raw zones treated as trusted data
- every domain inventing its own event conventions
- duplicated logic across streaming, batch, BI, and ML pipelines
- governance committees with no technical enforcement
- architecture diagrams that ignore failure handling and replay
- adopting mesh without platform capabilities
- adopting fabric tooling without solving ownership ambiguity

---

## Best Practices and Decision Framework

Use this sequence when choosing or revising a data architecture:

1. Identify the primary workloads.
   - BI
   - regulatory reporting
   - operational analytics
   - machine learning
   - event-driven product features
2. Identify freshness requirements.
3. Identify ownership model and team maturity.
4. Decide where canonical truth lives.
5. Choose the minimum architecture pattern that satisfies the need.
6. Define serving contracts explicitly.
7. Attach metadata, lineage, and quality controls at design time.
8. Revisit cost, replay, and failure recovery before production rollout.

### Opinionated Guidance

- Start simpler than you think.
- Add streaming only when the business benefits materially.
- Use lakehouse patterns when workload diversity justifies them.
- Use domain ownership only when domain teams can sustain data products.
- Invest early in metadata, lineage, contracts, and security boundaries.

---

## Final Summary

Modern data architecture is not a choice between buzzwords. It is the disciplined design of:

- how data enters the platform
- how truth is refined
- how products are served
- how change is governed
- how teams collaborate without chaos

Lambda architecture remains useful when correctness and low latency both matter and the organization can afford the complexity. Kappa architecture works well in event-native systems with durable logs and strong stream-processing maturity. Delta-style lakehouse architectures are often the most practical middle ground for modern platforms that need incremental processing, historical traceability, and shared analytics and ML access. Data mesh addresses ownership scaling; data fabric addresses interoperability and metadata-driven control. Lakehouses broaden platform flexibility, while warehouses often remain the better operating model for analytics-first teams that value managed simplicity.

The best architecture is the one that makes data trustworthy, usable, governable, and evolvable under real business conditions.
