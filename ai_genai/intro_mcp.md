# Model Context Protocol (MCP) – Complete Guide (2026 Edition)

**Model Context Protocol (MCP)** is an open standard created by Anthropic that defines how LLMs connect to external tools, data sources, and services. Launched in November 2024, MCP has become the **USB-C of AI integrations** — one protocol to connect any LLM to any tool.

---

## Table of Contents
1. [What is MCP?](#what-is-mcp)
2. [Why MCP Matters](#why-mcp-matters)
3. [Architecture](#architecture)
4. [Core Primitives](#core-primitives)
5. [MCP in Claude Code](#mcp-in-claude-code)
6. [Popular MCP Servers](#popular-mcp-servers)
7. [Building an MCP Server (Python)](#building-an-mcp-server-python)
8. [Building an MCP Server (TypeScript)](#building-an-mcp-server-typescript)
9. [MCP Client (Using a Server)](#mcp-client-using-a-server)
10. [MCP in 2026 — Ecosystem](#mcp-in-2026--ecosystem)

---

## What is MCP?

Before MCP, connecting an LLM to external tools was bespoke:
- Every LLM had its own tool-use format
- Every integration had to be written from scratch per-LLM
- Context/memory was per-conversation

MCP standardizes this with a **client-server protocol**:

```
┌─────────────────────────────────────────────────────┐
│                   MCP Client (LLM Host)              │
│  Claude Desktop / Claude Code / Cursor / Zed / etc. │
└─────────────────────────┬───────────────────────────┘
                          │ MCP Protocol (JSON-RPC 2.0)
          ┌───────────────┼──────────────────┐
          ▼               ▼                  ▼
   ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
   │ GitHub MCP  │ │Postgres MCP │ │ Slack MCP    │
   │   Server   │ │   Server   │ │   Server    │
   └─────────────┘ └─────────────┘ └──────────────┘
```

**One MCP server** can be used by **any MCP-compatible client** (Claude, GPT, Gemini, local LLMs, etc.).

---

## Why MCP Matters

### Before MCP
```python
# Every integration was custom per LLM
# OpenAI format:
{"type": "function", "function": {"name": "get_data", ...}}

# Anthropic format:
{"name": "get_data", "input_schema": {...}}

# Every developer had to rewrite integrations
```

### After MCP
```
One server definition → Works with any MCP client
```

Benefits:
- **Standardized** — build once, use everywhere
- **Secure** — servers run locally or in controlled environments
- **Composable** — mix and match servers
- **Discoverable** — clients can query server capabilities at runtime
- **Stateful** — maintain context across a session

---

## Architecture

### Transport Layers

MCP supports two transport mechanisms:

| Transport | Use Case |
|-----------|----------|
| **stdio** | Local servers. Client spawns server as subprocess. Most common. |
| **HTTP + SSE** | Remote servers. Client connects over network. |

### Protocol Flow

```
Client                          Server
  │                               │
  │──── initialize ──────────────>│  (announce capabilities)
  │<─── initialized ──────────────│
  │                               │
  │──── tools/list ──────────────>│  (discover available tools)
  │<─── tools list ───────────────│
  │                               │
  │──── tools/call ──────────────>│  (invoke tool)
  │<─── tool result ──────────────│
  │                               │
  │──── resources/list ──────────>│  (discover resources)
  │<─── resources list ───────────│
  │                               │
  │──── resources/read ──────────>│  (read resource)
  │<─── resource content ─────────│
```

---

## Core Primitives

MCP servers expose three types of capabilities:

### 1. Tools (LLM calls these)
Functions the LLM can invoke to take actions or retrieve data.

```json
{
  "name": "execute_sql",
  "description": "Run a SQL query against the data warehouse",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "SQL query"},
      "database": {"type": "string", "enum": ["prod", "staging"]}
    },
    "required": ["query"]
  }
}
```

### 2. Resources (LLM reads these)
Read-only data that can be embedded in the LLM's context. Like files or database views.

```json
{
  "uri": "postgres://localhost/mydb/users",
  "name": "Users table schema",
  "mimeType": "application/json"
}
```

### 3. Prompts (pre-built prompt templates)
Reusable prompt templates with arguments.

```json
{
  "name": "analyze_dataset",
  "description": "Analyze a dataset for quality issues",
  "arguments": [
    {"name": "table_name", "required": true},
    {"name": "focus_area", "required": false}
  ]
}
```

---

## MCP in Claude Code

Claude Code has first-class MCP support.

```bash
# Add an MCP server
claude mcp add server-name command [args]

# Examples:
claude mcp add github npx @modelcontextprotocol/server-github
claude mcp add postgres npx @modelcontextprotocol/server-postgres postgresql://localhost/mydb
claude mcp add filesystem npx @modelcontextprotocol/server-filesystem /path/to/dir

# List configured servers
claude mcp list

# Remove a server
claude mcp remove server-name

# Test server connectivity
claude mcp get server-name
```

### Claude Desktop Configuration

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://localhost/mydb"
      ]
    },
    "my-custom-server": {
      "command": "python",
      "args": ["/path/to/my_mcp_server.py"]
    }
  }
}
```

---

## Popular MCP Servers

### Official (by Anthropic)

```bash
# Filesystem — read/write local files
npx @modelcontextprotocol/server-filesystem /allowed/path

# GitHub — repos, PRs, issues
npx @modelcontextprotocol/server-github

# PostgreSQL — query databases
npx @modelcontextprotocol/server-postgres postgresql://localhost/db

# Brave Search — web search
npx @modelcontextprotocol/server-brave-search

# Slack — send messages, read channels
npx @modelcontextprotocol/server-slack

# Google Drive — access documents
npx @modelcontextprotocol/server-gdrive

# Memory — persistent key-value memory
npx @modelcontextprotocol/server-memory

# Sequential Thinking — structured reasoning tool
npx @modelcontextprotocol/server-sequential-thinking
```

### Community Servers (2026 Ecosystem)

| Server | Description |
|--------|-------------|
| `mcp-server-snowflake` | Query Snowflake DWH |
| `mcp-server-dbt` | Run dbt models, tests, docs |
| `mcp-server-kubernetes` | Manage K8s clusters |
| `mcp-server-terraform` | Plan/apply Terraform |
| `mcp-server-jira` | Create/update Jira tickets |
| `mcp-server-linear` | Linear issue tracking |
| `mcp-server-notion` | Read/write Notion pages |
| `mcp-server-playwright` | Browser automation |
| `mcp-server-puppeteer` | Browser automation (Node) |
| `mcp-server-sentry` | Query Sentry errors |
| `mcp-server-datadog` | Query Datadog metrics |
| `mcp-server-aws` | AWS operations |
| `mcp-server-gcp` | GCP operations |

---

## Building an MCP Server (Python)

```bash
pip install mcp
```

### Simple Data Engineering MCP Server

```python
# my_data_mcp_server.py
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Initialize the MCP server
app = Server("data-engineering-tools")

# ─── Tools ────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_sql",
            description="Execute a SQL query and return results as JSON",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 100)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_tables",
            description="List all available tables and their schemas",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "Database schema to inspect (default: public)"
                    }
                }
            }
        ),
        types.Tool(
            name="get_table_stats",
            description="Get row count, size, and last updated time for a table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"}
                },
                "required": ["table_name"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "run_sql":
        result = await execute_sql(
            arguments["query"],
            limit=arguments.get("limit", 100)
        )
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    elif name == "list_tables":
        schema = arguments.get("schema", "public")
        tables = await get_tables(schema)
        return [types.TextContent(type="text", text=json.dumps(tables, indent=2))]

    elif name == "get_table_stats":
        stats = await get_stats(arguments["table_name"])
        return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

    raise ValueError(f"Unknown tool: {name}")

# ─── Resources ────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="data://schemas/all",
            name="All Database Schemas",
            description="Complete schema documentation for all tables",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "data://schemas/all":
        schemas = await get_all_schemas()
        return json.dumps(schemas, indent=2)
    raise ValueError(f"Unknown resource: {uri}")

# ─── Actual Implementation ────────────────────────────

import asyncpg

DB_URL = "postgresql://localhost/mydb"

async def execute_sql(query: str, limit: int = 100) -> list[dict]:
    conn = await asyncpg.connect(DB_URL)
    try:
        # Add LIMIT to SELECT queries for safety
        if query.strip().upper().startswith("SELECT") and "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]
    finally:
        await conn.close()

async def get_tables(schema: str = "public") -> list[dict]:
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = $1
            ORDER BY table_name, ordinal_position
        """, schema)
        # Group by table
        tables = {}
        for row in rows:
            t = row["table_name"]
            if t not in tables:
                tables[t] = []
            tables[t].append({"column": row["column_name"], "type": row["data_type"]})
        return [{"table": k, "columns": v} for k, v in tables.items()]
    finally:
        await conn.close()

async def get_stats(table_name: str) -> dict:
    conn = await asyncpg.connect(DB_URL)
    try:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
        return {"table": table_name, "row_count": count}
    finally:
        await conn.close()

async def get_all_schemas():
    return await get_tables()

# ─── Entry Point ──────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Register in Claude Desktop

```json
{
  "mcpServers": {
    "my-data-server": {
      "command": "python",
      "args": ["/path/to/my_data_mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/mydb"
      }
    }
  }
}
```

---

## Building an MCP Server (TypeScript)

```bash
npm init -y
npm install @modelcontextprotocol/sdk zod
npm install -D typescript ts-node @types/node
```

```typescript
// src/server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const server = new Server(
  { name: "my-tools-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// Define tool schemas with Zod
const SearchSchema = z.object({
  query: z.string().describe("Search query"),
  limit: z.number().optional().default(10),
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "search",
      description: "Search for information",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", description: "Max results", default: 10 },
        },
        required: ["query"],
      },
    },
  ],
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "search") {
    const { query, limit } = SearchSchema.parse(request.params.arguments);

    // Perform actual search...
    const results = await performSearch(query, limit);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(results, null, 2),
        },
      ],
    };
  }

  throw new Error(`Unknown tool: ${request.params.name}`);
});

async function performSearch(query: string, limit: number) {
  // Your implementation here
  return [{ title: "Result 1", url: "https://example.com" }];
}

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## MCP Client (Using a Server)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["my_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # Call a tool
            result = await session.call_tool(
                "run_sql",
                {"query": "SELECT COUNT(*) FROM orders"}
            )
            print("Result:", result.content[0].text)

            # List resources
            resources = await session.list_resources()

            # Read a resource
            content = await session.read_resource("data://schemas/all")
            print("Schema:", content.contents[0].text)

import asyncio
asyncio.run(use_mcp_server())
```

---

## MCP in 2026 — Ecosystem

### Adoption

MCP has been adopted by:
- **Anthropic** — Claude Desktop, Claude Code, Claude API
- **OpenAI** — ChatGPT supports MCP servers
- **Google** — Gemini MCP support
- **Microsoft** — GitHub Copilot, VS Code
- **Cursor, Windsurf, Zed** — AI code editors
- **LangChain, LlamaIndex** — frameworks integrate MCP

### MCP Registry

The community maintains an official registry at [mcp.so](https://mcp.so) with 2,000+ servers.

### Key Patterns in 2026

| Pattern | Description |
|---------|-------------|
| **MCP + RAG** | MCP server exposes search tools backed by vector DB |
| **MCP + Data Catalog** | LLMs discover and query datasets via MCP |
| **MCP for Agents** | Multi-agent systems coordinate via shared MCP servers |
| **Secure MCP** | OAuth 2.0 authentication for remote MCP servers |
| **MCP Proxies** | Aggregate multiple MCP servers behind a single endpoint |

### Remote MCP (HTTP Transport)

```python
# Remote MCP server with authentication
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Remote Data Server")

@mcp.tool()
async def query_warehouse(sql: str) -> str:
    """Execute SQL against the data warehouse"""
    results = await run_query(sql)
    return json.dumps(results)

# Run as HTTP server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8080)
```

```bash
# Client connects to remote server
claude mcp add remote-warehouse --url https://data-server.mycompany.com/mcp
```

---

## Cheat Sheet

```bash
# Claude Code MCP commands
claude mcp add <name> <command> [args]
claude mcp list
claude mcp remove <name>
claude mcp get <name>

# Common servers
claude mcp add github npx @modelcontextprotocol/server-github
claude mcp add postgres npx @modelcontextprotocol/server-postgres <DB_URL>
claude mcp add filesystem npx @modelcontextprotocol/server-filesystem /path
claude mcp add memory npx @modelcontextprotocol/server-memory
claude mcp add brave-search npx @modelcontextprotocol/server-brave-search

# Python MCP server template
pip install mcp
# → implement list_tools() + call_tool() + main()

# TypeScript MCP server template
npm install @modelcontextprotocol/sdk
# → implement ListToolsRequestSchema + CallToolRequestSchema handlers
```
