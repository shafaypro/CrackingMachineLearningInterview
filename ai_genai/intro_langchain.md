# LangChain & LangGraph – Guide (2026 Edition)

**LangChain** is the most widely used framework for building LLM-powered applications. **LangGraph** is its stateful, graph-based extension for building complex agents and multi-agent workflows.

---

## LangChain Overview

LangChain provides:
- **Integrations** — 200+ LLM providers, vector DBs, document loaders, tools
- **LCEL** (LangChain Expression Language) — compose chains declaratively
- **Agents** — ReAct, tool-calling, and custom agent loops
- **Memory** — conversation history management
- **LangSmith** — observability and evaluation platform

```bash
pip install langchain langchain-anthropic langchain-community
```

---

## LCEL – LangChain Expression Language

LCEL uses the `|` pipe operator to compose chains:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Simple chain: prompt → LLM → parse
chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "vector databases"})
print(result)

# Streaming
for chunk in chain.stream({"topic": "Kubernetes"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([
    {"topic": "Docker"},
    {"topic": "Terraform"},
    {"topic": "dbt"},
])
```

---

## RAG Chain

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatAnthropic(model="claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is dbt used for?")
print(answer)
```

---

## Document Loaders & Text Splitters

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, DirectoryLoader,
    GitLoader, NotionDirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("report.pdf")
docs = loader.load()

# Load all .md files in a directory
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()

# Load webpage
loader = WebBaseLoader("https://docs.dbt.dev/docs/introduction")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Add to vector store
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
```

---

## Agents in LangChain

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Define tools with @tool decorator
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is 22°C and sunny."

@tool
def run_sql(query: str) -> str:
    """Execute a SQL query and return results."""
    # Real implementation would query your DB
    return '[{"user_count": 1234}]'

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Search results for: {query}"

tools = [get_weather, run_sql, search_web]

# Create agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data analyst assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "How many users do we have and what's the weather in NYC?"})
print(result["output"])
```

---

## LangGraph – Stateful Agent Graphs

LangGraph is the evolution beyond simple chains. It models workflows as **graphs** with nodes (LLM calls, tools) and edges (conditional routing, loops).

```bash
pip install langgraph
```

### Basic Graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, ToolMessage
import json

llm = ChatAnthropic(model="claude-sonnet-4-6")

# State schema
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, run_sql])

# Node: call the LLM
def call_llm(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Node: execute tool calls
def execute_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        # Find and execute the tool
        tool_map = {"get_weather": get_weather, "run_sql": run_sql}
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])

        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_results}

# Routing: should we call tools or end?
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return END

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", execute_tools)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")   # loop back after tool execution

app = graph.compile()

# Run
result = app.invoke({"messages": [("user", "What's our user count and NYC weather?")]})
print(result["messages"][-1].content)
```

---

### Multi-Agent Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ResearchState(TypedDict):
    task: str
    research: str
    draft: str
    review: str
    final: str

# Specialized agents
def researcher(state: ResearchState) -> ResearchState:
    """Searches and collects information."""
    response = llm.invoke(f"Research: {state['task']}")
    return {"research": response.content}

def writer(state: ResearchState) -> ResearchState:
    """Writes a draft based on research."""
    response = llm.invoke(f"Write a report based on:\n{state['research']}")
    return {"draft": response.content}

def reviewer(state: ResearchState) -> ResearchState:
    """Reviews and improves the draft."""
    response = llm.invoke(f"Review and improve:\n{state['draft']}")
    return {"review": response.content, "final": response.content}

# Linear pipeline
workflow = StateGraph(ResearchState)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)

pipeline = workflow.compile()
result = pipeline.invoke({"task": "Current state of vector databases"})
print(result["final"])
```

---

## LangSmith (Observability)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

# All LangChain calls are now traced automatically in LangSmith
# → View traces, latency, token usage, errors at smith.langchain.com
```

---

## LangChain vs LlamaIndex vs Raw API

| Feature | LangChain | LlamaIndex | Raw Anthropic API |
|---------|-----------|------------|-------------------|
| RAG pipelines | ✓ | ✓ (more specialized) | Manual |
| Agent loops | ✓ (LangGraph) | ✓ | Manual |
| Integrations | 200+ | 100+ | Manual |
| Learning curve | Medium | Medium | Low |
| Flexibility | High | Medium | Highest |
| Observability | LangSmith | LlamaTrace | Manual |
| Best for | Full-stack LLM apps | RAG/knowledge apps | Simple, custom apps |
