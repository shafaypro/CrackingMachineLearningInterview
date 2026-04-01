# n8n - AI Workflow Automation Guide

## What is n8n?

**n8n** is a workflow automation platform that lets you connect APIs, databases, webhooks, queues, and AI services using a visual node-based editor.

It is especially useful when you need to:

- automate repetitive business workflows,
- connect LLMs to external systems,
- build human approval flows,
- trigger AI jobs from forms, Slack, email, or webhooks,
- ship internal AI tooling without writing a full orchestration backend first.

In interviews, n8n usually comes up as a **low-code workflow orchestrator** that sits between product automation tools and code-first agent frameworks.

---

## Core Mental Model

Think of n8n as:

```text
Trigger -> Workflow -> Nodes -> External Systems -> Output / Next Action
```

Each workflow is a graph of nodes:

- **Trigger nodes** start execution.
- **Action nodes** call APIs or perform logic.
- **AI nodes** invoke models, prompts, tools, or memory-backed flows.
- **Logic nodes** branch, merge, filter, retry, and transform data.
- **Human review steps** pause or redirect execution when needed.

---

## Core Concepts

| Concept | What it means |
|---------|----------------|
| **Workflow** | The end-to-end automation graph |
| **Node** | One unit of work such as HTTP request, database query, AI call, or condition |
| **Trigger** | Event that starts the workflow: webhook, cron, Slack message, form submit, queue event |
| **Execution** | A single run of a workflow |
| **Credential** | Stored auth for external services |
| **Expression** | Dynamic values computed from previous nodes |
| **Webhook** | HTTP endpoint that can start a flow or receive callbacks |
| **Queue mode** | Scaled execution model using workers and a queue backend |

---

## Where n8n Fits in an AI Stack

| Layer | Typical Tools | n8n's role |
|------|---------------|------------|
| UI / event source | Slack, forms, email, CRM, webhook | Receives triggers and routes work |
| AI orchestration | LangGraph, custom Python, MCP tools, prompt pipelines | Can call these systems or host lightweight orchestration directly |
| Data / storage | Postgres, Redis, S3, vector DBs | Reads and writes operational context |
| Human review | Slack approvals, dashboards, ticketing systems | Inserts approval and escalation steps |
| Delivery | Email, CRM updates, issue trackers, chatops | Pushes outputs into business systems |

The practical takeaway:

- Use **n8n** when workflow automation and integration breadth matter.
- Use **LangGraph** or code-first orchestration when complex state machines, custom control flow, or deep engineering control matter.
- Use both together when you want visual business automation around a code-first agent core.

---

## Common AI Use Cases

### 1. Lead enrichment workflow

```text
Webhook -> Validate payload -> Search CRM -> Call LLM for summary
        -> Enrich from external APIs -> Score lead -> Route to sales rep
        -> Notify Slack
```

### 2. Support ticket triage

```text
Zendesk trigger -> Classify urgency -> Retrieve KB context
                 -> Draft response -> Human approval for high-risk tickets
                 -> Post reply -> Log metrics
```

### 3. Content operations

```text
Notion page created -> Generate summary -> Create social draft
                     -> Human review -> Publish -> Update tracking sheet
```

### 4. Internal AI assistant back office

```text
User request -> n8n webhook -> auth check -> call agent backend
             -> save transcript -> create follow-up tasks -> notify channel
```

---

## n8n vs Other Workflow Tools

| Tool | Best For | Strength | Weakness |
|------|----------|----------|----------|
| **n8n** | AI-enabled business workflows and app integrations | Visual builder plus custom logic flexibility | Very complex stateful reasoning is still easier in code |
| **LangGraph** | Stateful AI agents and multi-step reasoning | Explicit control over loops, branches, and memory | More engineering effort |
| **Apache Airflow** | Scheduled data and batch pipelines | Strong DAG-based scheduling for data workloads | Less natural for app automation and human-facing triggers |
| **Zapier** | Simple SaaS automation | Fastest for non-technical teams | Less engineering control |
| **Temporal** | Long-running durable workflows | Strong execution guarantees and recovery | Heavier engineering and ops overhead |

---

## Example: AI Ticket Triage in n8n

### Flow

```text
Zendesk Trigger
  -> Normalize ticket text
  -> Classify issue type with LLM
  -> Retrieve relevant FAQ / docs
  -> Draft response
  -> IF priority = high
       -> Slack approval
     ELSE
       -> Send draft automatically
  -> Write audit log to database
```

### What interviewers care about

- where the trigger comes from,
- how credentials are managed,
- how you prevent duplicate processing,
- where human approval is required,
- how you monitor failures and retries,
- whether the LLM is allowed to act directly or only suggest actions.

---

## Production Design Considerations

### 1. Idempotency

If the same webhook arrives twice, the workflow should not create duplicate tickets, emails, or downstream actions.

Use:

- event IDs,
- database dedupe keys,
- "upsert" style writes,
- replay-safe workflow design.

### 2. Human-in-the-loop

Do not fully automate high-risk actions such as:

- deleting records,
- issuing refunds,
- sending legal or compliance responses,
- changing production infrastructure.

Insert approval checkpoints through Slack, email, or an internal review queue.

### 3. Error handling

Workflows should define:

- retry policy,
- timeout behavior,
- dead-letter or failure queue pattern,
- notification path for operators.

### 4. Secrets and credentials

Keep credentials in the platform credential store, not inline in nodes or prompts.

### 5. Auditability

For AI-assisted business flows, log:

- input payload,
- prompt or task version,
- model output,
- human approvals,
- downstream actions taken.

### 6. Separation of concerns

n8n should often coordinate the workflow while heavier reasoning or proprietary logic runs in:

- a Python backend,
- a FastAPI service,
- a LangGraph agent service,
- or internal tools exposed via API.

---

## When to Use n8n

**Use n8n when:**

- you need many SaaS integrations quickly,
- business teams need visibility into the automation,
- workflows are event-driven and operational,
- you want fast iteration on AI-enhanced internal tools,
- human approval and routing are part of the process.

**Avoid using n8n alone when:**

- the workflow needs advanced graph-state control,
- you need long-running agent memory with custom recovery logic,
- you require deep custom testing around reasoning loops,
- the critical path depends on complex code-first orchestration.

In those cases, use n8n as the outer automation layer and call a code-first backend.

---

## Interview Questions

**Q1: What is n8n and how is it different from LangGraph?**

> n8n is a node-based workflow automation platform focused on integrating business systems, triggers, approvals, and API calls. LangGraph is a code-first framework for stateful AI agent orchestration. n8n is better for operational automation; LangGraph is better for complex reasoning workflows.

**Q2: When would you choose n8n over Airflow?**

> Choose n8n for event-driven application workflows, SaaS integrations, webhooks, approvals, and AI-powered back-office automation. Choose Airflow for scheduled data pipelines, batch jobs, and data platform orchestration.

**Q3: What are the risks of using n8n in AI workflows?**

> The main risks are poor idempotency, hidden credential sprawl, weak approval boundaries, limited testing of complex logic, and letting the LLM trigger unsafe downstream actions. Mitigate with dedupe keys, credential isolation, approval gates, and service-layer validation.

**Q4: How would you productionize an n8n workflow?**

> Add queue-based execution, structured logging, retries, alerting, API-level validation, approval checkpoints, external state storage, and clear ownership for secrets and deployments.

**Q5: Can n8n be part of an agent architecture?**

> Yes. It often works well as the outer orchestration layer that handles triggers, routing, notifications, approvals, and integration steps, while a code-first agent backend handles planning, tool selection, and reasoning.

**Q6: What is the main tradeoff of low-code AI orchestration?**

> Speed and integration breadth improve, but very complex control flow, fine-grained testing, and custom runtime behavior are easier to manage in code.

---

## Related Topics

| Topic | Why it matters |
|------|----------------|
| [Agentic AI](./intro_agentic_ai.md) | n8n can orchestrate AI agents and approval flows |
| [Agent Systems & Tool Use](./intro_agent_tool_use.md) | n8n is often used to wrap tool-enabled AI workflows |
| [LangGraph](./intro_langgraph.md) | Strong comparison point for code-first orchestration |
| [MCP](./intro_mcp.md) | MCP tools can sit behind APIs that n8n workflows call |
| [Apache Airflow](../data_engineering/intro_apache_airflow.md) | Useful contrast between app automation and data orchestration |

---

## References

- [n8n official site](https://n8n.io/)
- [n8n documentation](https://docs.n8n.io/)
