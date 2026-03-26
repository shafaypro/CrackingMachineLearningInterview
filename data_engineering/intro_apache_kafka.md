# Apache Kafka – Complete Guide (2026 Edition)

**Apache Kafka** is the leading distributed event streaming platform. It handles trillions of events per day at companies like LinkedIn, Uber, and Netflix — powering real-time data pipelines, event-driven architectures, and streaming analytics.

---

## What is Kafka?

Kafka is a **distributed commit log** that allows producers to publish events and consumers to read them, at any scale, in real-time.

```
Microservice A  →
                  → Kafka (durable log) → Analytics Pipeline
Microservice B  →                       → Another Microservice
                                        → Data Warehouse (Snowflake/Redshift)
```

### Kafka vs Message Queues (RabbitMQ, SQS)

| Feature | Kafka | Traditional MQ |
|---------|-------|---------------|
| **Message retention** | Days/weeks (configurable) | Deleted after consumption |
| **Consumer model** | Multiple consumers read independently | Typically point-to-point |
| **Throughput** | Millions of msgs/sec | Thousands |
| **Ordering** | Guaranteed per partition | Varies |
| **Replay** | Yes — rewind consumer offset | No |
| **Use case** | Streaming, event sourcing, large scale | Task queues, RPC |

---

## Core Concepts

```
Producer → Topic → Partition → Consumer Group

┌──────────────── Topic: "orders" ───────────────────┐
│  Partition 0:  [msg0] [msg3] [msg6] ...            │
│  Partition 1:  [msg1] [msg4] [msg7] ...            │
│  Partition 2:  [msg2] [msg5] [msg8] ...            │
└────────────────────────────────────────────────────┘
```

| Concept | Description |
|---------|-------------|
| **Event/Message** | A record with key, value, timestamp, headers |
| **Topic** | Named stream of events. Like a table or folder. |
| **Partition** | Ordered, immutable log. Topics are split into partitions for parallelism. |
| **Offset** | Position of a message within a partition (monotonically increasing integer) |
| **Producer** | Writes events to topics |
| **Consumer** | Reads events from topics |
| **Consumer Group** | Group of consumers sharing load. Each partition → one consumer per group. |
| **Broker** | A Kafka server. Cluster = multiple brokers. |
| **ZooKeeper/KRaft** | Cluster metadata management. KRaft replaces ZooKeeper in modern Kafka. |
| **Replication Factor** | How many copies of each partition (for fault tolerance) |

### Consumer Groups

```
Topic: "orders" (3 partitions)

Consumer Group "analytics":
  Consumer A → Partition 0
  Consumer B → Partition 1
  Consumer C → Partition 2

Consumer Group "warehouse":
  Consumer X → Partition 0, 1
  Consumer Y → Partition 2

Both groups read independently — no coordination.
```

---

## Running Kafka

### Docker (local development)

```yaml
# docker-compose.yml
version: "3"
services:
  kafka:
    image: apache/kafka:3.8.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
```

```bash
docker compose up -d
```

### Managed Kafka in 2026

| Service | Provider | Notes |
|---------|----------|-------|
| **Confluent Cloud** | Confluent | Feature-rich, expensive |
| **Amazon MSK** | AWS | Managed Kafka on AWS |
| **Aiven for Kafka** | Aiven | Multi-cloud managed |
| **Upstash Kafka** | Upstash | Serverless, pay-per-use |
| **Redpanda Cloud** | Redpanda | Kafka-compatible, lower latency |

---

## Kafka CLI

```bash
# Topic management
kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --topic orders --partitions 3 --replication-factor 1

kafka-topics.sh --bootstrap-server localhost:9092 --list

kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe --topic orders

kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete --topic orders

# Produce messages (console producer)
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic orders
> {"order_id": "1", "amount": 99.99}
> {"order_id": "2", "amount": 149.00}

# Consume messages
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic orders --from-beginning

# Consumer group inspection
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group --describe

# Reset offsets
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group --topic orders --reset-offsets --to-earliest --execute
```

---

## Python Producer

```python
from confluent_kafka import Producer
import json

producer = Producer({
    "bootstrap.servers": "localhost:9092",
    "acks": "all",                    # wait for all replicas to confirm
    "enable.idempotence": True,       # exactly-once delivery
    "compression.type": "snappy",     # compress messages
    "linger.ms": 5,                   # batch messages for 5ms
    "batch.size": 65536,              # 64KB batch size
})

def delivery_callback(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} partition {msg.partition()} offset {msg.offset()}")

# Produce with key (key determines partition)
order = {"order_id": "ORD-001", "customer_id": "C-123", "amount": 99.99}

producer.produce(
    topic="orders",
    key="C-123",              # same key → same partition (ordering per customer)
    value=json.dumps(order),
    callback=delivery_callback
)

# Flush (wait for all messages to be delivered)
producer.flush()

# Context manager pattern
from contextlib import contextmanager

@contextmanager
def kafka_producer(config):
    p = Producer(config)
    try:
        yield p
    finally:
        p.flush(timeout=30)

with kafka_producer({"bootstrap.servers": "localhost:9092"}) as p:
    for i in range(1000):
        p.produce("events", value=json.dumps({"id": i}))
        p.poll(0)  # serve delivery callbacks
```

---

## Python Consumer

```python
from confluent_kafka import Consumer, KafkaError
import json

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "order-processor",
    "auto.offset.reset": "earliest",         # start from beginning if no committed offset
    "enable.auto.commit": False,             # manual commit for at-least-once
    "max.poll.interval.ms": 300000,          # 5 min processing time before rebalance
})

consumer.subscribe(["orders", "returns"])

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"Reached end of partition {msg.partition()}")
            else:
                raise Exception(msg.error())
            continue

        # Process message
        order = json.loads(msg.value().decode("utf-8"))
        print(f"Processing order {order['order_id']} from partition {msg.partition()} offset {msg.offset()}")

        try:
            process_order(order)
            # Manual commit after successful processing
            consumer.commit(asynchronous=False)
        except Exception as e:
            print(f"Processing failed: {e}")
            # Don't commit — message will be re-delivered

finally:
    consumer.close()

def process_order(order: dict):
    # Your business logic here
    pass
```

### Batch Consumer

```python
# Process in batches for efficiency
from confluent_kafka import Consumer

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "batch-processor",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
})
consumer.subscribe(["events"])

BATCH_SIZE = 1000
BATCH_TIMEOUT = 5.0  # seconds

batch = []
while True:
    msg = consumer.poll(timeout=0.1)

    if msg and not msg.error():
        batch.append(json.loads(msg.value()))

    if len(batch) >= BATCH_SIZE or (batch and consumer.poll(timeout=BATCH_TIMEOUT) is None):
        # Process batch
        process_batch(batch)
        consumer.commit(asynchronous=False)
        batch = []
```

---

## Kafka Schemas with Schema Registry

In production, always use a **Schema Registry** to enforce message schemas.

```python
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField

# Define Avro schema
order_schema = """
{
  "type": "record",
  "name": "Order",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "created_at", "type": {"type": "long", "logicalType": "timestamp-millis"}}
  ]
}
"""

schema_registry_client = SchemaRegistryClient({"url": "http://localhost:8081"})
avro_serializer = AvroSerializer(schema_registry_client, order_schema)

producer = Producer({"bootstrap.servers": "localhost:9092"})

order = {"order_id": "ORD-001", "customer_id": "C-123", "amount": 99.99, "created_at": 1700000000000}

producer.produce(
    topic="orders",
    value=avro_serializer(order, SerializationContext("orders", MessageField.VALUE))
)
```

---

## Kafka Streams & Kafka Connect

### Kafka Connect (no-code integrations)

```json
// Source connector: PostgreSQL → Kafka
{
  "name": "postgres-source",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "localhost",
    "database.port": "5432",
    "database.user": "postgres",
    "database.password": "secret",
    "database.dbname": "mydb",
    "table.include.list": "public.orders,public.customers",
    "plugin.name": "pgoutput"
  }
}

// Sink connector: Kafka → Snowflake
{
  "name": "snowflake-sink",
  "config": {
    "connector.class": "com.snowflake.kafka.connector.SnowflakeSinkConnector",
    "topics": "orders",
    "snowflake.url.name": "myaccount.snowflakecomputing.com",
    "snowflake.database.name": "MYDB",
    "snowflake.schema.name": "PUBLIC",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "com.snowflake.kafka.connector.records.SnowflakeAvroConverter"
  }
}
```

### Debezium (CDC — Change Data Capture)

Debezium captures every database change and publishes it to Kafka:

```
PostgreSQL WAL → Debezium Connector → Kafka Topic → Consumers
                                      ┌─────────────────────────────┐
                                      │ Event: {                    │
                                      │   "op": "u",  // update     │
                                      │   "before": {"amount": 100},│
                                      │   "after":  {"amount": 150} │
                                      │ }                           │
                                      └─────────────────────────────┘
```

---

## Kafka in Data Engineering Pipelines

```
PostgreSQL (OLTP)
      ↓ Debezium (CDC)
Kafka Topics
      ↓ Kafka Connect Sink / Spark Streaming / Flink
Snowflake / Delta Lake / S3
      ↓ dbt
Analytics / BI
```

```python
# Spark Structured Streaming reading from Kafka
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("KafkaStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "orders") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON
orders = df.select(
    F.from_json(
        F.col("value").cast("string"),
        "order_id STRING, amount DOUBLE, customer_id STRING"
    ).alias("data")
).select("data.*")

# Write to Delta Lake
query = orders.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoints/orders") \
    .table("orders_streaming")
```

---

## Key Configuration Reference

### Producer

| Config | Recommended | Purpose |
|--------|-------------|---------|
| `acks=all` | Production | Wait for all replicas |
| `enable.idempotence=true` | Production | Exactly-once |
| `compression.type=snappy` | Production | Compress messages |
| `linger.ms=5-50` | Production | Batch messages |
| `retries=INT_MAX` | Production | Retry on transient errors |

### Consumer

| Config | Value | Purpose |
|--------|-------|---------|
| `auto.offset.reset` | `earliest`/`latest` | Where to start reading |
| `enable.auto.commit` | `false` | Manual commit for reliability |
| `group.id` | Your app name | Consumer group identity |
| `max.poll.records` | `500` | Records per poll |
| `session.timeout.ms` | `30000` | Heartbeat timeout |

### Topic

| Config | Value | Purpose |
|--------|-------|---------|
| `retention.ms` | `604800000` (7 days) | How long to keep messages |
| `replication.factor` | `3` (production) | Fault tolerance |
| `num.partitions` | `throughput / single_partition_throughput` | Parallelism |
| `min.insync.replicas` | `2` | Required replicas for acks=all |

---

## Kafka in 2026

| Trend | Description |
|-------|-------------|
| **KRaft mode** | ZooKeeper fully removed in Kafka 4.0 |
| **Kafka 4.0** | KRaft-only, tiered storage GA, improved quota system |
| **Tiered Storage** | Offload old segments to S3/GCS — cheaper long retention |
| **Redpanda** | Kafka-compatible alternative in C++ — 10x lower latency |
| **WarpStream** | Kafka-compatible, disaggregated architecture, zero inter-zone costs |
| **Apache Flink** | Primary stream processor alongside Kafka (replaced Kafka Streams at scale) |
| **Confluent Tableflow** | Auto-sync Kafka topics to Iceberg tables |
