# Data Modeling

Data modeling is the discipline of translating business processes, operational constraints, and analytical needs into durable data structures. For a senior data engineer, it is not a documentation exercise. It is a control surface for correctness, performance, governance, cost, and long-term change velocity.

Well-modeled systems make downstream work easier: analytics becomes interpretable, machine learning features become reproducible, and operational services can evolve without every schema change turning into a breaking event. Poor modeling creates the opposite outcome: brittle joins, duplicate logic, silent metric drift, and expensive rework.

---

## What Data Modeling Is and Why It Matters

Data modeling defines:

- what entities exist
- how those entities relate to each other
- what business events should be recorded
- what grain and history should be preserved
- how the model should behave under growth, change, and audit requirements

At enterprise scale, modeling decisions directly affect:

- query performance
- storage footprint
- semantic consistency across teams
- data quality and testability
- compliance and traceability
- feature reproducibility for ML
- the effort required to onboard new consumers

The correct question is rarely "What is the ideal schema?" The better question is "What model best serves the operating and analytical reality of this business with acceptable complexity?"

---

## Conceptual, Logical, and Physical Data Models

These are separate layers. Teams that skip that separation usually mix business semantics with engine-specific implementation too early.

| Model Type | Purpose | Audience | Typical Content | Common Mistake |
|---|---|---|---|---|
| Conceptual | Define core business concepts and boundaries | Business stakeholders, architects | Customer, Order, Product, Subscription | Treating conceptual entities as tables too early |
| Logical | Define attributes, relationships, keys, and business rules | Data engineers, analytics engineers, architects | Entity relationships, cardinality, optionality, domains | Ignoring grain and temporal rules |
| Physical | Implement for a specific engine and workload | Data engineers, DBAs, platform engineers | Partitioning, clustering, indexes, file layout, compression | Copying logical design without tuning for access patterns |

### Practical Example

For a commerce platform:

- Conceptual model: `Customer`, `Order`, `Payment`, `Shipment`
- Logical model: `Order` has many `OrderLine`; `Payment` belongs to one `Order`; an `Order` may have multiple status changes over time
- Physical model:
  - OLTP database with normalized `orders`, `order_lines`, `payments`
  - warehouse fact table `fact_orders`
  - dimensions `dim_customer`, `dim_product`, `dim_date`

If you jump directly into physical tables, you usually encode the first use case instead of the stable business model.

---

## OLTP vs OLAP Modeling

Operational and analytical systems should not be modeled the same way by default.

| Characteristic | OLTP Modeling | OLAP Modeling |
|---|---|---|
| Primary goal | Correct transactional updates | Fast reads and aggregations |
| Typical workload | Point lookups, inserts, updates, deletes | Scans, joins, aggregates, time-series analysis |
| Data shape | Normalized | Denormalized or dimensional |
| Latency target | Milliseconds for transactions | Seconds to minutes for broad queries |
| History handling | Often current-state oriented | Usually historical and audit-friendly |
| Typical engines | Postgres, MySQL, SQL Server | Snowflake, BigQuery, Redshift, Databricks, DuckDB |

### When to Use OLTP Modeling

- user-facing applications
- order management
- payment processing
- inventory reservation
- systems that require transactional integrity and row-level updates

### When Not to Use OLTP Modeling

- executive reporting
- dashboard workloads with repeated multi-table joins
- feature generation over large historical windows
- self-service analytics

### When to Use OLAP Modeling

- recurring reporting
- BI tools
- experimentation analysis
- ML training data creation
- historical trend analysis

### When Not to Use OLAP Modeling

- write-heavy transactional services
- systems requiring complex per-row updates with low latency

Common mistake: forcing a normalized application schema into the warehouse and asking analysts to rebuild business semantics in every query.

---

## Keys and Relationships

Keys determine both integrity and usability. Weak key design creates duplicate facts, broken joins, and unstable lineage.

### Primary Keys

Primary keys uniquely identify a row within a table. In analytical platforms, they are often logical rather than physically enforced, so teams must validate them with tests.

### Foreign Keys

Foreign keys express relationships between entities. Even if the warehouse does not enforce them, the model should declare them semantically and validate them in tests or contracts.

### Surrogate Keys

Surrogate keys are synthetic identifiers, typically integers, hashes, or UUIDs, created independently of business meaning.

Use surrogate keys when:

- natural keys are large or unstable
- multiple source systems collide on identifiers
- dimensions need versioned rows
- you need warehouse-friendly joins

### Natural Keys

Natural keys come from the business domain, such as `email`, `order_number`, or `country_code`.

Use natural keys when:

- the identifier is truly stable
- the business already reasons in that identifier
- cross-system portability matters

Do not confuse familiarity with stability. Email addresses, SKU names, and human-readable codes often change more than teams expect.

### Practical Guidance

| Key Type | Pros | Cons | Typical Use |
|---|---|---|---|
| Natural key | Business-readable, easy for debugging | May change, may not be globally unique | Stable reference data |
| Surrogate integer | Efficient joins, compact storage | Requires mapping logic | Warehouse dimensions |
| UUID | Easy distributed generation | Large, less efficient for some workloads | Event IDs, distributed systems |
| Hash key | Useful for dedupe and integration | Collision risk, opaque to users | Data Vault hubs, lakehouse merge keys |

### Common Mistakes

- using mutable business fields as primary keys
- joining facts to dimensions on descriptive text
- treating composite natural keys casually and then dropping part of the grain later
- assuming uniqueness without validating it continuously

---

## Normalization vs Denormalization

Normalization reduces redundancy and update anomalies. Denormalization reduces join cost and improves read performance. Neither is universally better.

| Topic | Normalized Model | Denormalized Model |
|---|---|---|
| Read performance | Lower for wide analytical queries | Higher for common analytics queries |
| Write integrity | Stronger | Weaker unless carefully managed |
| Redundancy | Lower | Higher |
| Ease of change | Better for transactional domains | Better for stable reporting domains |
| Ease for analysts | Lower | Higher |
| Typical use | OLTP systems | Data marts, serving layers |

### When to Use Normalized Models

- transactional systems
- master data domains
- domains with frequent updates
- integration layers where business rules are still evolving

### When Not to Use Normalized Models

- BI-serving layers with repeated joins
- high-concurrency dashboard queries
- teams with many self-service analytics users

### When to Use Denormalized Models

- star schemas
- aggregate marts
- feature tables
- reporting tables tuned for repeated access patterns

### When Not to Use Denormalized Models

- volatile operational data with frequent corrections
- domains with weak source governance
- early integration stages where semantics are not stabilized

### Common Mistakes

- denormalizing too early before the grain is understood
- keeping every historical attribute in one giant "gold" table
- normalizing a warehouse so aggressively that every dashboard requires ten joins

---

## Dimensional Modeling

Dimensional modeling is still the default recommendation for business analytics and BI because it optimizes for understandable queries, stable metrics, and predictable performance.

### Facts

Facts represent measurable business events at a specific grain.

Examples:

- one order line per row
- one click event per row
- one daily account balance per row

Good fact tables define:

- exact grain
- additive or semi-additive measures
- event timestamp
- foreign keys to dimensions

### Dimensions

Dimensions describe the context around facts.

Examples:

- customer
- product
- merchant
- campaign
- date

Dimensions should carry descriptive attributes used for filtering, grouping, and slicing metrics.

### Star Schema

A star schema has a central fact table with directly connected denormalized dimensions.

```text
dim_customer   dim_product   dim_date
      \             |          /
           fact_order_line
                |
           dim_channel
```

#### Pros

- simple for analysts
- good BI performance
- clear metric ownership
- works well with semantic layers

#### Cons

- some redundancy in dimensions
- less ideal for highly volatile source integration

#### When to Use

- dashboarding
- recurring KPI reporting
- experimentation analysis
- finance and commercial reporting

#### Common Mistakes

- mixing multiple grains into one fact table
- using dimensions as dumping grounds for unrelated attributes
- embedding non-conformed business logic inconsistently across marts

### Snowflake Schema

A snowflake schema normalizes one or more dimensions into additional related tables.

Use it when:

- dimensions are very large
- attribute hierarchies are reused heavily
- you have governance reasons to centralize dimension substructures

Do not choose snowflake by default for BI. It usually increases query complexity faster than it increases value.

### Slowly Changing Dimensions (SCD)

Historical behavior is a modeling choice, not an accident.

| SCD Type | Behavior | Use Case | Main Risk |
|---|---|---|---|
| Type 0 | Keep original value forever | Immutable attributes | Limited correction flexibility |
| Type 1 | Overwrite current value | Typos, non-historical cleanup | Loses history |
| Type 2 | New row per change with effective dates | Customer tier, product category changes | Larger dimensions and more complex joins |
| Type 3 | Add previous value columns | Limited before/after analysis | Poor scalability for many changes |
| Type 4 | Current table plus history table | Large history management separation | Dual-table complexity |
| Type 6 | Hybrid of 1, 2, and 3 | Complex reporting with current and historical views | High implementation complexity |

### Practical Example: Fact and Dimension Grain

Bad model:

- `fact_orders` contains one row per order
- revenue, product category, coupon code, and shipment carrier are all mixed together
- analysts later try to answer product-level and shipment-level questions from the same table

Better model:

- `fact_order_line` at one row per purchased item
- `dim_customer`
- `dim_product`
- `dim_order`
- `dim_date`

This allows revenue by product, by channel, by customer cohort, and by fulfillment path without re-deriving grain later.

---

## Data Vault Basics

Data Vault is an integration-oriented modeling approach optimized for auditability, historical traceability, and parallel ingestion from many source systems.

Core structures:

- `Hub`: unique business keys
- `Link`: relationships between business keys
- `Satellite`: descriptive context and change history

### Simplified Example

```text
Hub_Customer(customer_bk, customer_hk, load_ts, record_source)
Hub_Order(order_bk, order_hk, load_ts, record_source)
Link_Customer_Order(customer_hk, order_hk, link_hk, load_ts)
Sat_Customer_Profile(customer_hk, customer_name, segment, effective_ts)
Sat_Order_Status(order_hk, status, amount, effective_ts)
```

### When to Use Data Vault

- many upstream systems with changing schemas
- strict lineage and audit requirements
- enterprise integration programs
- long-lived platforms where source systems will continue changing

### When Not to Use Data Vault

- small teams that need dashboards quickly
- simple single-source domains
- self-service analytics as the primary outcome

### Pros

- resilient to source change
- good historical traceability
- supports incremental delivery across domains
- separates business keys from descriptive context cleanly

### Cons

- harder for analysts to consume directly
- more tables and more joins
- requires disciplined automation
- can create architectural ceremony if applied to small problems

### Common Mistakes

- exposing raw vault structures directly to BI users
- implementing Data Vault without automation standards
- assuming Data Vault replaces dimensional marts

In practice, many enterprise teams use Data Vault for integration and lineage, then publish dimensional or domain-serving marts for consumption.

---

## Modeling for Analytics, BI, Machine Learning, and Operational Systems

Different consumers need different shapes of truth.

| Consumer | Model Preference | Why |
|---|---|---|
| Operational application | Normalized OLTP | Transaction correctness and efficient updates |
| BI dashboards | Dimensional marts | Predictable joins and business-friendly semantics |
| Ad hoc analytics | Wide marts plus reusable conformed dimensions | Speed with enough semantic control |
| Machine learning training | Event-level tables, snapshot tables, feature views | Point-in-time correctness and reproducibility |
| Online inference | Narrow, low-latency feature serving model | Fast lookup and freshness guarantees |

### Modeling for Analytics and BI

Prefer:

- clear grain
- conformed dimensions
- explicit metric definitions
- marts aligned to business processes, not source applications

### Modeling for Machine Learning

Prefer:

- event timestamps preserved at source grain
- point-in-time safe joins
- leakage prevention
- stable identifiers across training and serving
- features separated from labels and outcomes

Common ML mistake: training on a denormalized "latest state" table that accidentally includes future information.

### Modeling for Operational Systems

Prefer:

- explicit transactional boundaries
- normalized structures
- referential integrity
- status transitions modeled as events where audit matters

---

## Modeling for Lakehouse Environments

Lakehouse environments change the physical implementation, not the need for good modeling.

What changes in a lakehouse:

- storage is typically file-based object storage
- tables may be backed by Delta Lake, Iceberg, or Hudi
- batch and streaming often converge onto the same table abstractions
- schema evolution is easier technically, which can tempt teams into weak governance

### Recommended Patterns

- bronze for landed raw data
- silver for cleaned, conformed, business-stable entities
- gold for marts, aggregates, and high-value serving tables

### Warehouse Modeling vs Lakehouse Modeling

| Topic | Warehouse-Centric Modeling | Lakehouse-Centric Modeling |
|---|---|---|
| Storage abstraction | Managed warehouse tables | Open table formats on object storage |
| Change handling | Often warehouse-native DML | Merge/upsert plus incremental file compaction |
| Performance tuning | Distribution, sort keys, clustering | Partitioning, clustering, file size, compaction |
| Governance risk | More centralized controls | Easier sprawl if contracts are weak |
| Typical advantage | Simplicity for BI | Flexibility across BI, DS, and ML workloads |

### When to Use Lakehouse-Oriented Modeling

- mixed analytics, data science, and ML workloads
- open storage format strategy
- large-scale raw and curated data retention
- multi-engine access requirements

### When Not to Use It

- small analytics footprint where a managed warehouse is simpler
- organizations without strong metadata and governance discipline

---

## Schema Evolution and Schema Design Best Practices

Schema evolution should be intentional, versioned, and observable.

### Best Practices

- define canonical entity grain before adding columns
- separate ingestion schemas from serving schemas
- version contracts for event payloads and external tables
- prefer additive changes over breaking changes
- make nullability a deliberate decision, not a default
- capture effective dates and ingestion timestamps
- document business definitions alongside physical schema

### Common Evolution Strategies

| Change Type | Recommendation |
|---|---|
| Add nullable column | Usually safe if semantics are clear |
| Rename column | Prefer additive migration with deprecation window |
| Change meaning of an existing field | Create a new field; do not silently repurpose |
| Split one table into many | Publish compatibility views during migration |
| Change event key or grain | Treat as a contract change and version it explicitly |

### Common Mistakes

- allowing raw event producers to redefine field semantics without warning
- merging two source attributes into one warehouse column because they "look similar"
- overusing generic `metadata` JSON instead of modeling governed columns

---

## Partitioning and Clustering Considerations

Partitioning and clustering are physical design choices. They should follow workload evidence, not vendor marketing.

### Partitioning

Use partitioning when:

- queries prune large date or domain ranges
- ingestion naturally arrives in partitionable windows
- maintenance operations benefit from scoped rewrites

Avoid over-partitioning. Tiny partitions create metadata overhead and poor scan efficiency.

Good partition candidates:

- event date
- ingestion date
- region for very large datasets

Bad partition candidates:

- high-cardinality user IDs
- volatile fields
- fields rarely used for pruning

### Clustering

Clustering improves locality for frequently filtered or joined columns.

Use clustering when:

- partitions are still large
- query predicates are consistent
- engine supports effective clustering maintenance

### Common Mistakes

- partitioning by timestamp at hour-level when query volumes do not justify it
- never compacting files in lakehouse tables
- choosing partition columns before understanding consumption patterns

---

## Data Quality Implications of Modeling Choices

Modeling decisions either surface or hide quality problems.

Examples:

- no declared grain leads to duplicate facts
- weak keys cause join amplification
- overloaded dimension tables create inconsistent definitions
- denormalized wide tables hide source-of-truth ownership
- late-arriving data without effective timestamps breaks historical accuracy

### Quality Controls That Should Map to the Model

| Modeling Concern | Quality Check |
|---|---|
| Primary key uniqueness | Unique and not-null tests |
| Foreign key integrity | Relationship tests |
| Fact grain | Duplicate detection on grain-defining columns |
| SCD behavior | Non-overlapping effective date tests |
| Event ordering | Late-arrival and watermark monitoring |
| Schema stability | Contract validation and drift alerts |

Good models make quality checks obvious. If the testing strategy is hard to define, the model is usually underspecified.

---

## Comparison: Dimensional Modeling vs Data Vault

| Topic | Dimensional Modeling | Data Vault |
|---|---|---|
| Primary goal | Business consumption and analytics | Enterprise integration and history |
| Consumer friendliness | High | Low to medium |
| Change tolerance | Moderate | High |
| Time to first dashboard | Fast | Slower |
| Auditability | Good if designed well | Excellent |
| Number of tables | Lower | Higher |
| Best fit | BI, semantic layers, reporting marts | Multi-source ingestion, lineage-heavy programs |

### Decision Guidance

Choose dimensional modeling when the main problem is consistent business reporting.

Choose Data Vault when the main problem is integrating many changing source systems with traceable history.

In many enterprises, the right answer is both:

- Data Vault or equivalent in the integration layer
- dimensional marts in the presentation layer

---

## Comparison: Normalized vs Denormalized Models

| Topic | Normalized | Denormalized |
|---|---|---|
| Best for | Operational correctness | Analytical usability |
| Redundancy | Low | Higher |
| Write complexity | Lower anomaly risk | Higher anomaly risk |
| Read complexity | Higher | Lower |
| Change management | Stronger in transactional domains | Stronger in reporting domains |

### Decision Guidance

Choose normalized structures for transactional domains and canonical integration entities.

Choose denormalized structures when the same joins are executed repeatedly by analysts, dashboards, or serving layers.

---

## Comparison: Warehouse Modeling vs Lakehouse Modeling

| Topic | Warehouse Modeling | Lakehouse Modeling |
|---|---|---|
| Main bias | Managed analytical simplicity | Flexible multi-engine platform |
| Governance posture | Often centralized | Must be designed explicitly |
| Change handling | More warehouse-managed | More platform-engineering responsibility |
| ML alignment | Good, but sometimes secondary | Usually better aligned with DS and ML workloads |

### Decision Guidance

If the organization is analytics-first and wants a tightly managed platform, warehouse-centric modeling is often the better operating model.

If the organization needs shared access across analytics, data science, ML, and streaming use cases, lakehouse modeling often provides more leverage, but only if metadata and table-management discipline are strong.

---

## Common Anti-Patterns

- one giant "gold" table containing every attribute anyone ever asked for
- fact tables with mixed grains
- dimensions built from whichever source system was easiest to access
- silently changing field meaning without versioning
- exposing raw bronze data as if it were curated truth
- overusing JSON blobs instead of governing important columns
- using SCD Type 2 everywhere without a business reason
- optimizing storage layout before defining semantic ownership

---

## Best Practices and Decision Framework

Use this sequence when deciding how to model a domain:

1. Define the business process or entity boundary.
2. Declare the grain explicitly.
3. Identify stable keys and change behavior.
4. Separate canonical integration models from serving models.
5. Choose the consumer shape:
   - normalized for operations
   - dimensional for BI
   - event or feature oriented for ML
6. Choose the physical design:
   - partitioning
   - clustering
   - compaction
   - incremental strategy
7. Attach quality tests and ownership to the model.
8. Document the contract and the reasons behind the design.

### Opinionated Guidance

- Model business processes, not source tables.
- Preserve event time and business grain early.
- Favor simple marts over overly abstract "enterprise" schemas that no one uses.
- Do not ask analysts to reverse-engineer operational semantics from application tables.
- Do not confuse lakehouse flexibility with permission to skip modeling discipline.

---

## Final Summary

Good data modeling is less about formal diagrams and more about durable decisions:

- what happened
- at what grain
- for which business entity
- with what history
- under which constraints

Dimensional modeling remains the most practical default for analytics and BI. Normalized models remain the right default for operational systems. Data Vault is valuable for integration-heavy, high-change enterprises when used with discipline and automation. Lakehouse platforms expand implementation options, but they do not remove the need for clear keys, grain, historical strategy, and governed schemas.

If a model is easy to query, hard to misuse, resilient to source change, and explicit about time and grain, it is usually on the right track.
