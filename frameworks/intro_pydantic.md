# Pydantic — Data Validation for AI Systems

## What is Pydantic?

Pydantic is a Python data validation library that uses **type annotations** to define schemas and validate data at runtime. In AI/ML systems, it's used for:

- **API request/response validation** (FastAPI integration)
- **LLM structured output** enforcement
- **Configuration management**
- **Data pipeline schema validation**

---

## Core Concepts (Pydantic v2)

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from datetime import datetime
import json

class MLModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Test accuracy 0-1")
    model_type: Literal["classification", "regression", "generation"]
    tags: list[str] = Field(default_factory=list, max_length=10)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    is_production: bool = False

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, v: str) -> str:
        if " " in v:
            raise ValueError("Model name must not contain spaces")
        return v.lower()

    @model_validator(mode="after")
    def check_production_requirements(self) -> "MLModel":
        if self.is_production and self.accuracy < 0.85:
            raise ValueError("Production models require accuracy >= 0.85")
        return self

# Usage
try:
    model = MLModel(
        name="bert_classifier",
        accuracy=0.94,
        model_type="classification",
        is_production=True
    )
    print(model.model_dump())
    print(model.model_dump_json(indent=2))
except ValidationError as e:
    print(e.errors())  # Detailed error list
```

---

## Pydantic for LLM Structured Output

The most critical use case in 2026: forcing LLMs to return validated, typed outputs.

### With LangChain

```python
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

class InterviewAnswer(BaseModel):
    """Structured answer to an ML interview question"""
    main_answer: str = Field(description="Core answer in 2-3 sentences")
    key_concepts: list[str] = Field(description="Top 3-5 concepts to mention")
    code_example: Optional[str] = Field(default=None, description="Python code example if relevant")
    follow_up_questions: list[str] = Field(description="2 likely follow-up questions")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Question difficulty")

llm = ChatAnthropic(model="claude-sonnet-4-6")
structured_llm = llm.with_structured_output(InterviewAnswer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ML interview coach. Answer the question in the exact format requested."),
    ("human", "Question: {question}")
])

chain = prompt | structured_llm

result: InterviewAnswer = chain.invoke({
    "question": "Explain the difference between bagging and boosting"
})

# result is a fully typed, validated Python object
print(result.main_answer)
print(result.key_concepts)
print(result.difficulty)
```

### Direct with Anthropic SDK

```python
from anthropic import Anthropic
from pydantic import BaseModel
import json

class RAGEvaluation(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    issues: list[str]
    recommendation: Literal["approve", "review", "reject"]

client = Anthropic()

def evaluate_rag_response(question: str, context: str, answer: str) -> RAGEvaluation:
    schema = RAGEvaluation.model_json_schema()

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=[{
            "name": "submit_evaluation",
            "description": "Submit the RAG evaluation result",
            "input_schema": schema
        }],
        tool_choice={"type": "tool", "name": "submit_evaluation"},
        messages=[{
            "role": "user",
            "content": f"Evaluate this RAG response:\nQuestion: {question}\nContext: {context}\nAnswer: {answer}"
        }]
    )

    tool_input = response.content[0].input
    return RAGEvaluation.model_validate(tool_input)

result = evaluate_rag_response(
    question="What is RLHF?",
    context="RLHF stands for Reinforcement Learning from Human Feedback...",
    answer="RLHF is a technique to align LLMs using human preferences."
)
print(result)
```

---

## Nested Models for Complex AI Outputs

```python
from pydantic import BaseModel, Field
from typing import Optional

class CodeExample(BaseModel):
    language: Literal["python", "sql", "bash", "yaml"]
    code: str
    explanation: str

class SystemDesignAnswer(BaseModel):
    problem_statement: str
    requirements: list[str] = Field(min_length=2, max_length=10)
    architecture_components: list[str]
    data_flow: str = Field(description="Step-by-step data flow description")
    scalability_considerations: list[str]
    tradeoffs: dict[str, str] = Field(description="Key: tradeoff name, Value: explanation")
    code_examples: list[CodeExample] = Field(default_factory=list, max_length=3)
    interview_tips: list[str]

# LLM outputs complex nested structure, Pydantic validates it
llm = ChatAnthropic(model="claude-opus-4-6")
structured_llm = llm.with_structured_output(SystemDesignAnswer)
result = structured_llm.invoke("Design a real-time fraud detection system for ML interview")
```

---

## Configuration Management

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, AnyUrl
from functools import lru_cache

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",     # APP_ANTHROPIC_API_KEY, etc.
        case_sensitive=False
    )

    # LLM Settings
    anthropic_api_key: SecretStr
    default_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.7

    # Vector DB
    pinecone_api_key: SecretStr
    pinecone_index: str = "production-index"
    pinecone_environment: str = "us-east-1"

    # Database
    database_url: AnyUrl
    db_pool_size: int = 10

    # Rate Limiting
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    # Feature Flags
    enable_streaming: bool = True
    enable_cache: bool = True

@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()

# Use in FastAPI
from fastapi import Depends

@app.get("/config-check")
async def config_check(settings: AppSettings = Depends(get_settings)):
    return {
        "model": settings.default_model,
        "streaming": settings.enable_streaming
    }
```

---

## Data Pipeline Validation

```python
from pydantic import BaseModel, field_validator
from typing import Any
import numpy as np

class TrainingExample(BaseModel):
    features: list[float] = Field(..., min_length=1)
    label: int = Field(..., ge=0)
    weight: float = Field(default=1.0, gt=0)
    source: str

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features cannot contain NaN or Inf")
        return v

class DataBatch(BaseModel):
    examples: list[TrainingExample] = Field(..., min_length=1)
    batch_id: str
    created_at: datetime

    @model_validator(mode="after")
    def check_label_distribution(self) -> "DataBatch":
        labels = [e.label for e in self.examples]
        unique_labels = set(labels)
        if len(unique_labels) == 1 and len(self.examples) > 100:
            import warnings
            warnings.warn("Batch contains only one class label — possible data issue")
        return self

# Validate incoming training data
def process_batch(raw_data: list[dict]) -> DataBatch:
    try:
        batch = DataBatch(
            examples=[TrainingExample(**ex) for ex in raw_data],
            batch_id=str(uuid.uuid4()),
            created_at=datetime.now()
        )
        return batch
    except ValidationError as e:
        logger.error(f"Data validation failed: {e.errors()}")
        raise
```

---

## Pydantic v1 vs v2 Differences

| Feature | v1 | v2 (current) |
|---------|----|--------------|
| **Validators** | `@validator` | `@field_validator`, `@model_validator` |
| **Export** | `.dict()`, `.json()` | `.model_dump()`, `.model_dump_json()` |
| **JSON schema** | `.schema()` | `.model_json_schema()` |
| **Performance** | Python | Rust core (10-50x faster) |
| **Config** | `class Config:` | `model_config = SettingsConfigDict(...)` |
| **Strict mode** | Manual | Built-in `model_validate(data, strict=True)` |

---

## Common Patterns in AI Systems

### Output Parser with Fallback

```python
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

parser = PydanticOutputParser(pydantic_object=InterviewAnswer)

# Try structured output, fall back to raw string
def safe_parse(llm_output: str) -> InterviewAnswer | str:
    try:
        return parser.parse(llm_output)
    except (ValidationError, ValueError) as e:
        logger.warning(f"Failed to parse structured output: {e}")
        return llm_output  # Return raw string as fallback
```

### Discriminated Unions (for Multi-Agent Routing)

```python
from pydantic import BaseModel
from typing import Union, Annotated
from pydantic import Discriminator, Tag

class ResearchTask(BaseModel):
    task_type: Literal["research"] = "research"
    query: str
    max_sources: int = 10

class CodeTask(BaseModel):
    task_type: Literal["code"] = "code"
    description: str
    language: str = "python"

class WritingTask(BaseModel):
    task_type: Literal["write"] = "write"
    topic: str
    word_count: int = 500

# Pydantic auto-selects the right type based on task_type discriminator
AgentTask = Annotated[
    Union[ResearchTask, CodeTask, WritingTask],
    Discriminator("task_type")
]

task_data = {"task_type": "code", "description": "Write a binary search", "language": "python"}
task = TypeAdapter(AgentTask).validate_python(task_data)
# task is a CodeTask instance
```

---

## Interview Questions

**Q: Why is Pydantic important in LLM applications?**
> LLMs return unstructured text. Pydantic enforces that outputs conform to a schema — catching missing fields, wrong types, invalid values before they cause downstream failures. With `with_structured_output()`, the LLM is guided via JSON Schema / tool calling to produce valid structured data.

**Q: How does Pydantic v2 differ from v1 in performance?**
> Pydantic v2's core is written in Rust (via the `pydantic-core` library), making validation 10-50x faster than v1. API changes: `@validator` → `@field_validator`, `.dict()` → `.model_dump()`, `class Config` → `model_config = SettingsConfigDict(...)`.

**Q: How do you handle LLM output validation failures?**
> Retry with a clearer prompt that includes the schema and error message. Use fallback to a raw string parser if structured parsing fails after N retries. Log all failures for dataset curation — these become training examples for improving prompts.

**Q: What's the difference between `model_validate` and direct instantiation?**
> Direct instantiation (`MyModel(field=value)`) validates kwargs. `model_validate(dict_or_object)` validates from a dict/object — useful when data comes from JSON. `model_validate(data, strict=True)` enforces strict type matching (no coercion). Use strict mode when accepting external data to avoid silent type coercions.
