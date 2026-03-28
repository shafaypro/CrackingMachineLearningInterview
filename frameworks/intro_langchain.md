# LangChain Guide

> **Note:** A comprehensive LangChain introduction is available at [`ai_genai/intro_langchain.md`](../ai_genai/intro_langchain.md). This file covers additional advanced topics: LangChain Expression Language (LCEL), LangSmith, LangGraph, and chains vs agents vs tools.

---

## Table of Contents

1. [LangChain Expression Language (LCEL)](#langchain-expression-language-lcel)
2. [LangSmith — Observability](#langsmith--observability)
3. [LangGraph — Stateful Agents](#langgraph--stateful-agents)
4. [Chains vs Agents vs Tools](#chains-vs-agents-vs-tools)
5. [Interview Q&A](#interview-qa)
6. [References](#references)

---

## LangChain Expression Language (LCEL)

LCEL is a declarative way to compose LangChain components using the pipe (`|`) operator. It provides streaming, async, batching, and retries out of the box.

```bash
pip install langchain langchain-openai langchain-community
```

### Basic LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
parser = StrOutputParser()

# Compose with | operator
chain = prompt | model | parser

# Invoke
result = chain.invoke({"topic": "gradient descent"})
print(result)

# Stream
for chunk in chain.stream({"topic": "backpropagation"}):
    print(chunk, end="", flush=True)

# Batch (process multiple inputs efficiently)
results = chain.batch([
    {"topic": "gradient descent"},
    {"topic": "backpropagation"},
    {"topic": "regularization"}
])
```

### LCEL with Retriever (RAG)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Setup retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=["LLMs are trained on large datasets", "RLHF aligns models with human preferences"],
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:""")

model = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain with LCEL
rag_chain = (
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("How are LLMs aligned?")
print(answer)
```

### LCEL with Structured Output

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief summary")
    sentiment: str = Field(description="positive, negative, or neutral")

model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(MovieReview)

prompt = ChatPromptTemplate.from_template("Review this movie: {movie}")
chain = prompt | structured_model

review = chain.invoke({"movie": "Inception"})
print(type(review))       # <class 'MovieReview'>
print(review.rating)      # 9
print(review.sentiment)   # positive
```

### Custom Runnables

```python
from langchain_core.runnables import RunnableLambda

# Wrap any Python function as a runnable
def preprocess(text: str) -> str:
    return text.strip().lower()

def postprocess(text: str) -> dict:
    return {"result": text, "length": len(text)}

chain = (
    RunnableLambda(preprocess)
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(postprocess)
)
```

---

## LangSmith — Observability

LangSmith is the observability platform for LangChain applications — trace all LLM calls, inputs, outputs, and costs.

### Setup

```bash
pip install langsmith
```

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "my-project-name"

# Now all LangChain calls are automatically traced
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke("What is RAG?")  # Auto-traced to LangSmith
```

### Manual Tracing

```python
from langsmith import traceable

@traceable(name="my-custom-function")
def process_query(query: str) -> str:
    # LangSmith traces this automatically
    result = my_chain.invoke(query)
    return result

# Evaluate with LangSmith
from langsmith.evaluation import evaluate

def correctness_evaluator(run, example):
    prediction = run.outputs["output"]
    expected = example.outputs["answer"]
    score = 1.0 if expected.lower() in prediction.lower() else 0.0
    return {"score": score}

results = evaluate(
    target=lambda x: chain.invoke(x),
    data="my-dataset-name",
    evaluators=[correctness_evaluator],
    experiment_prefix="gpt4o-mini-test"
)
```

---

## LangGraph — Stateful Agents

LangGraph is a framework for building stateful, multi-step AI workflows as directed graphs.

```bash
pip install langgraph
```

### Simple Agent Graph

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

model = ChatOpenAI(model="gpt-4o-mini")

# Define nodes
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Check if we should call a tool or end
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"
    return END

def call_tool(state: AgentState) -> AgentState:
    # Execute the tool call
    tool_result = "Tool result here"
    return {"messages": [HumanMessage(content=tool_result)]}

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tool", "agent")

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="What is the weather in London?")]
})
```

### ReAct Agent with Tools

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implement actual search here
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

model = ChatOpenAI(model="gpt-4o-mini")
tools = [search_web, calculate]

agent = create_react_agent(model, tools)

result = agent.invoke({
    "messages": [HumanMessage(content="What is 25 * 37 + 100?")]
})

for message in result["messages"]:
    print(f"{type(message).__name__}: {message.content}")
```

---

## Chains vs Agents vs Tools

| Concept | Description | When to Use |
|---------|-------------|-------------|
| **Chain** | Fixed sequence of steps, deterministic | Known, repeatable workflows (RAG, summarization) |
| **Agent** | LLM decides what to do next, dynamic | Open-ended tasks requiring reasoning and tool use |
| **Tool** | A callable function the LLM can invoke | Extending LLM capabilities (search, code execution, API calls) |

### When to Use Chains

```python
# Chain: fixed pipeline, always does the same steps
chain = prompt | model | parser | postprocess
result = chain.invoke(input)  # Predictable, fast, no surprises
```

Best for: summarization, translation, extraction, classification — tasks with a known input-output pattern.

### When to Use Agents

```python
# Agent: LLM decides which tools to call and in what order
agent = create_react_agent(model, [search, calculator, database_lookup])
result = agent.invoke({"messages": [HumanMessage("Research X and compute Y")]})
```

Best for: research tasks, multi-step problem solving, tasks where you don't know in advance what operations are needed.

### Tool Design Best Practices

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.
    Use this when you need up-to-date information or facts you don't know.
    """
    # Implementation here
    return f"Results for {query}"
```

---

## Interview Q&A

**Q1: What is LCEL and what are its advantages over traditional LangChain chains?** 🟡 Intermediate

LCEL (LangChain Expression Language) is a declarative composition syntax using the `|` pipe operator. Advantages: automatic streaming support, async execution, batch processing, built-in retry logic, and easy serialization. Traditional chains (like `LLMChain`) were imperative and verbose. LCEL makes composition more Pythonic and provides these features with no additional code.

---

**Q2: What is the difference between a LangChain chain and a LangGraph agent?** 🟡 Intermediate

A LangChain chain is a fixed, linear (or branching but predetermined) sequence of operations — the flow is defined at build time. A LangGraph agent is a directed graph where the LLM decides at runtime which node to execute next based on its state. LangGraph supports cycles (retry loops, multi-step reasoning), persistent state, and conditional branching driven by LLM reasoning. Use chains for predictable workflows, LangGraph for dynamic multi-step tasks.

---

**Q3: What is LangSmith and why would you use it in production?** 🟡 Intermediate

LangSmith is an observability platform for LangChain applications. It traces every LLM call (inputs, outputs, tokens, latency, cost), enables debugging of complex chains and agents, supports dataset creation for evaluation, and allows A/B testing of prompts. In production, it's essential for: debugging unexpected model outputs, monitoring LLM costs and latency, and running regression tests on prompt changes.

---

**Q4: When would you choose LangChain over LlamaIndex?** 🟡 Intermediate

Choose **LangChain** when building general LLM applications, agents with multiple tools, complex chains combining different AI services, or workflows beyond document Q&A.

Choose **LlamaIndex** when the primary use case is document indexing, retrieval, and Q&A over large knowledge bases. LlamaIndex has superior document loaders, chunking strategies, index types, and retrieval algorithms for RAG.

Many production systems use both: LlamaIndex for the retrieval layer and LangChain for orchestration.

---

**Q5: What are the key design considerations when building a LangGraph multi-agent system?** 🔴 Advanced

1. **State management:** Define a clear TypedDict state schema that all agents read from and write to
2. **Message passing:** Use message queues in state to pass context between agents
3. **Routing logic:** Implement clear conditional edges using deterministic routing functions
4. **Tool isolation:** Each agent should have a specific set of tools; avoid overlap
5. **Error handling:** Add retry nodes and error state transitions
6. **Human-in-the-loop:** Add interrupt_before/interrupt_after for human approval steps
7. **Observability:** Use LangSmith tracing to debug multi-agent flows

---

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Also see: ai_genai/intro_langchain.md](../ai_genai/intro_langchain.md)
