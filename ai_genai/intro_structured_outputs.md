# Structured Outputs & Function Calling

## Why Structured Outputs Matter

Raw LLM text is unreliable for programmatic consumption. Production systems need outputs they can:
- Parse deterministically
- Validate against a schema
- Route to downstream systems
- Store in databases

Structured outputs bridge the gap between probabilistic language models and deterministic software systems.

---

## Function Calling / Tool Use

Function calling lets the model decide *when* to call a function and *what arguments* to pass, rather than generating free text. The model returns a structured call specification; your code executes it.

### OpenAI Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Define tools (JSON Schema format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Paris, France'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

# 2. Send message with tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo and Berlin right now?"}],
    tools=tools,
    tool_choice="auto"  # "auto" | "required" | {"type": "function", "function": {"name": "..."}}
)

# 3. Handle tool calls
message = response.choices[0].message
if message.tool_calls:
    tool_results = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Execute the actual function
        if func_name == "get_weather":
            result = get_weather(**args)  # your implementation
        elif func_name == "search_web":
            result = search_web(**args)
        
        tool_results.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": json.dumps(result)
        })
    
    # 4. Send results back to model for final response
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo and Berlin?"},
            message,  # assistant's tool call message
            *tool_results  # tool results
        ],
        tools=tools
    )
    print(final_response.choices[0].message.content)
```

### Anthropic Tool Use

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    }
]

def run_tool_loop(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # Stop if no tool calls
        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute tool calls and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Execute the tool
                    if block.name == "get_stock_price":
                        result = get_stock_price(block.input["ticker"])
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

result = run_tool_loop("What is the current price of Apple stock?")
```

---

## JSON Mode and Structured Outputs

### JSON Mode (Soft Guarantee)

Instructs the model to produce valid JSON, but doesn't enforce a schema:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Extract entities. Return JSON with keys: persons (list), organizations (list), locations (list)."},
        {"role": "user", "content": "Apple CEO Tim Cook announced at Stanford University..."}
    ]
)
data = json.loads(response.choices[0].message.content)
```

### Structured Outputs (Hard Guarantee)

Enforces a specific JSON Schema — the model is **constrained** to produce only schema-valid output via constrained decoding:

```python
from pydantic import BaseModel
from typing import Literal

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float  # 0.0-1.0
    key_phrases: list[str]
    requires_escalation: bool

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # structured outputs requires this model or newer
    messages=[
        {"role": "system", "content": "Analyze the sentiment of customer reviews."},
        {"role": "user", "content": "This product is absolutely terrible, I want a refund immediately!"}
    ],
    response_format=SentimentResult,  # Pydantic model → JSON Schema
)

result: SentimentResult = response.choices[0].message.parsed
print(result.sentiment)           # "negative"
print(result.requires_escalation) # True
```

### Anthropic Structured Outputs with Pydantic

```python
from anthropic import Anthropic
from pydantic import BaseModel
import json

client = Anthropic()

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool
    features: list[str]

def extract_product_info(text: str) -> ProductInfo:
    schema = ProductInfo.model_json_schema()
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Extract product information from this text and return ONLY valid JSON
matching this schema:
{json.dumps(schema, indent=2)}

Text: {text}"""
        }]
    )
    
    raw = response.content[0].text.strip()
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return ProductInfo.model_validate_json(raw)
```

---

## Pydantic Integration Patterns

### Output Parsing with Validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import json

class AnalysisResult(BaseModel):
    summary: str = Field(..., max_length=500)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    topics: list[str] = Field(..., min_length=1, max_length=10)
    language: str = Field(default="en")
    is_spam: bool

    @field_validator("topics")
    @classmethod
    def topics_must_be_lowercase(cls, v):
        return [topic.lower().strip() for topic in v]
    
    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v):
        valid_codes = {"en", "fr", "de", "es", "ja", "zh"}
        if v not in valid_codes:
            raise ValueError(f"Language must be one of {valid_codes}")
        return v

def parse_with_retry(
    llm_output: str,
    model: type[BaseModel],
    max_retries: int = 2,
    llm_client=None
) -> BaseModel:
    """Parse LLM output into a Pydantic model with LLM-assisted retry on failure."""
    for attempt in range(max_retries + 1):
        try:
            # Try to extract JSON from the response
            raw = extract_json_from_text(llm_output)
            return model.model_validate_json(raw)
        except Exception as e:
            if attempt == max_retries:
                raise
            if llm_client:
                # Ask the LLM to fix its own output
                fix_prompt = f"""
Your previous output failed validation with this error: {e}

Original output:
{llm_output}

Return only the corrected JSON matching this schema:
{json.dumps(model.model_json_schema(), indent=2)}
"""
                llm_output = llm_client.generate(fix_prompt)

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain markdown or other content."""
    import re
    # Try to find JSON in code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    # Try to find raw JSON object or array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return match.group(1)
    return text.strip()
```

### Hierarchical Structured Extraction

```python
from pydantic import BaseModel
from typing import Optional

class Address(BaseModel):
    street: Optional[str] = None
    city: str
    country: str
    postal_code: Optional[str] = None

class Person(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None

class ContractParties(BaseModel):
    seller: Person
    buyer: Person
    notary: Optional[Person] = None
    contract_date: str
    total_value: float
    currency: str

# The schema is automatically inferred including nested models
schema = ContractParties.model_json_schema()
```

---

## Instructor Library

[Instructor](https://github.com/jxnl/instructor) patches LLM clients to return Pydantic models directly with automatic retry:

```python
import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

# Patch OpenAI
client = instructor.from_openai(OpenAI())

# Patch Anthropic  
client = instructor.from_anthropic(Anthropic())

class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str
    skills: list[str]

# Extract structured data directly — instructor handles retries automatically
profile = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserProfile,  # <- Instructor magic
    messages=[{
        "role": "user",
        "content": "John is a 32-year-old software engineer who loves Python, Kubernetes, and distributed systems."
    }]
)

print(profile.name)       # John
print(profile.age)        # 32
print(profile.skills)     # ["Python", "Kubernetes", "distributed systems"]
```

Instructor automatically:
- Converts Pydantic models to the correct schema format per provider
- Retries on validation failures with the error message fed back to the model
- Handles streaming partial objects

---

## Tool Schema Design Best Practices

### Good Tool Design

```python
# GOOD: Specific, well-described, constrained parameters
{
    "name": "create_calendar_event",
    "description": "Create a new calendar event for the user. Use this when the user wants to schedule a meeting, appointment, or reminder.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the event (max 100 characters)"
            },
            "start_datetime": {
                "type": "string",
                "format": "date-time",
                "description": "Event start in ISO 8601 format, e.g. '2024-04-15T14:00:00Z'"
            },
            "duration_minutes": {
                "type": "integer",
                "minimum": 15,
                "maximum": 480,
                "description": "Event duration in minutes (15-480)"
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string", "format": "email"},
                "description": "Email addresses of attendees"
            },
            "is_video_call": {
                "type": "boolean",
                "description": "Whether to include a video conferencing link"
            }
        },
        "required": ["title", "start_datetime", "duration_minutes"]
    }
}

# BAD: Vague, no constraints, poor descriptions
{
    "name": "do_calendar_thing",
    "description": "Calendar stuff",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string"}  # What goes here? Model has to guess
        }
    }
}
```

### Tool Description Principles

1. **Describe when to use** — "Use this when the user wants to X, Y, or Z"
2. **Describe when NOT to use** — prevents the model from calling it in wrong situations
3. **Be specific about data formats** — ISO 8601 for dates, not "a date string"
4. **Use enums for constrained choices** — prevents hallucinated values
5. **Keep tool count under ~20** — too many tools degrades selection accuracy

---

## Parallel Tool Calls

Modern models can call multiple tools in one turn when they can be executed concurrently:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Compare the weather in NYC, London, and Tokyo right now"}],
    tools=tools,
    parallel_tool_calls=True  # Default: True
)

# Model may return multiple tool calls in one response
# message.tool_calls = [
#   ToolCall(function=get_weather(location="New York")),
#   ToolCall(function=get_weather(location="London")),
#   ToolCall(function=get_weather(location="Tokyo"))
# ]

import asyncio

async def execute_parallel_tools(tool_calls):
    tasks = []
    for call in tool_calls:
        args = json.loads(call.function.arguments)
        tasks.append(execute_tool(call.function.name, args))
    
    results = await asyncio.gather(*tasks)
    return [
        {"tool_call_id": call.id, "role": "tool", "content": json.dumps(r)}
        for call, r in zip(tool_calls, results)
    ]
```

---

## Streaming Structured Outputs

For large structured outputs, stream partial JSON to reduce time-to-first-token:

```python
# With Instructor and streaming
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)

class LongReport(BaseModel):
    title: str
    executive_summary: str
    sections: list[str]
    recommendations: list[str]
    conclusion: str

# Stream partial objects as they are generated
for partial_report in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=LongReport,
    messages=[{"role": "user", "content": "Write a report on renewable energy trends"}],
    stream=True
):
    # partial_report is populated progressively
    if partial_report.title:
        print(f"Title: {partial_report.title}")
    if partial_report.executive_summary:
        print(f"Summary: {partial_report.executive_summary[:100]}...")
```

---

## Common Interview Questions

**Q: What is the difference between JSON mode and Structured Outputs in OpenAI's API?**
JSON mode (`response_format: {"type": "json_object"}`) guarantees the output is valid JSON but doesn't enforce any schema — the model chooses the keys and structure. Structured Outputs (`response_format: SomePydanticModel`) uses constrained decoding to guarantee the output matches a specific schema exactly, including required fields, types, and allowed values. Structured Outputs is strictly stronger: every valid structured output is also valid JSON, but not vice versa.

**Q: How do you handle cases where the model's tool call has incorrect arguments?**
Three layers of defense: (1) Schema validation — use strict JSON Schema with enums, min/max, format constraints to limit what the model can produce; (2) Application validation — validate tool args with Pydantic before executing, return a tool error result rather than crashing; (3) Error feedback — if validation fails, return the error as a tool result and let the model retry with the error as context. This is the "retry loop" pattern — the model usually corrects itself when told what was wrong.

**Q: When should you use function calling vs. prompting for structured data extraction?**
Function calling is preferred when: (1) you need reliable schema enforcement across many calls, (2) the model needs to decide whether to extract data or do something else (tool choice logic), (3) you want to take action based on extracted data (tool execution). Plain prompting with JSON instructions works when: (1) you always need output (no conditional logic), (2) you need to support models without function calling, (3) the schema is simple and prompt-based extraction is already reliable enough.

**Q: How does constrained decoding work in structured outputs?**
The model's token probabilities are masked at each generation step to only allow tokens that could lead to a valid continuation of the target schema. Before generation, the JSON Schema is compiled into a finite automaton (or trie) of valid token sequences. At each step, only tokens that advance along a valid path through this automaton are allowed. This guarantees schema-valid output with zero post-processing — no need for retry logic, but the model loses some expressiveness (it can't generate schema-invalid content even if it would be more accurate).

**Q: What is the "tool poisoning" attack and how do you defend against it?**
Tool poisoning is when a malicious document or user input contains hidden instructions that trick the model into calling a destructive tool (e.g., "delete all files" embedded invisibly in a retrieved document). Defenses: (1) Principle of least privilege — only expose tools needed for the current task; (2) Confirmation gates — require human approval before executing irreversible tools; (3) Input isolation — wrap user/retrieved content in delimiters and instruct the model that instructions inside delimiters are data, not instructions; (4) Tool result validation — validate tool inputs against allowlists before execution.
