# Testing AI Systems — Playwright, Puppeteer & Beyond

## Testing AI Applications is Different

Traditional software testing: deterministic outputs, binary pass/fail.
AI application testing: probabilistic outputs, quality scores, behavioral properties.

```
Traditional Testing:          AI System Testing:
assert output == expected      assert quality(output) >= threshold
                               assert output_is_safe(output)
                               assert output_is_grounded(output, context)
                               assert latency < 3000ms
                               assert cost_per_call < $0.01
```

---

## Playwright — End-to-End Testing for AI UIs

Playwright is Microsoft's browser automation library for E2E testing of AI-powered web applications.

### Setup

```bash
pip install playwright pytest-playwright
playwright install chromium  # Download browser
```

### Basic AI Chatbot UI Test

```python
import pytest
from playwright.sync_api import Page, expect, sync_playwright
import re

@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()

@pytest.fixture
def page(browser):
    page = browser.new_page()
    yield page
    page.close()

class TestMLChatbotUI:

    def test_basic_question_gets_response(self, page: Page):
        """Test that the chatbot responds to a basic ML question"""
        page.goto("http://localhost:3000")

        # Type question
        page.fill('[data-testid="chat-input"]', "What is overfitting?")
        page.click('[data-testid="send-button"]')

        # Wait for response (AI can take up to 30s)
        response_element = page.wait_for_selector(
            '[data-testid="assistant-message"]',
            timeout=30000
        )

        response_text = response_element.inner_text()

        # Validate response quality (not empty, relevant)
        assert len(response_text) > 50, "Response too short"
        assert any(word in response_text.lower() for word in
                  ["training", "generalization", "validation", "data"]), \
            "Response doesn't mention relevant ML concepts"

    def test_streaming_response_appears_incrementally(self, page: Page):
        """Test that streaming tokens appear progressively, not all at once"""
        page.goto("http://localhost:3000")
        page.fill('[data-testid="chat-input"]', "Explain transformers")
        page.click('[data-testid="send-button"]')

        # Capture text at different time intervals
        texts = []
        for _ in range(5):
            page.wait_for_timeout(500)
            el = page.query_selector('[data-testid="assistant-message"]')
            if el:
                texts.append(el.inner_text())

        # Text should grow over time (streaming)
        non_empty = [t for t in texts if t]
        assert len(non_empty) >= 2, "No incremental content detected"
        assert len(non_empty[-1]) > len(non_empty[0]), "Text didn't grow (streaming broken?)"

    def test_error_handling_displayed_to_user(self, page: Page):
        """Test that LLM errors are shown gracefully"""
        # Intercept API calls and return error
        page.route("**/api/chat", lambda route: route.fulfill(
            status=502, body='{"error": "LLM service unavailable"}'
        ))

        page.goto("http://localhost:3000")
        page.fill('[data-testid="chat-input"]', "Test question")
        page.click('[data-testid="send-button"]')

        # Verify error is shown gracefully (not raw exception)
        error_element = page.wait_for_selector('[data-testid="error-message"]', timeout=5000)
        assert "try again" in error_element.inner_text().lower()

    def test_chat_history_persists(self, page: Page):
        """Test that conversation history is maintained"""
        page.goto("http://localhost:3000")

        # First message
        page.fill('[data-testid="chat-input"]', "My name is Alice")
        page.click('[data-testid="send-button"]')
        page.wait_for_selector('[data-testid="assistant-message"]', timeout=20000)

        # Second message — should remember context
        page.fill('[data-testid="chat-input"]', "What is my name?")
        page.click('[data-testid="send-button"]')
        page.wait_for_selector(
            '[data-testid="assistant-message"]:nth-child(2)', timeout=20000
        )

        messages = page.query_selector_all('[data-testid="assistant-message"]')
        last_response = messages[-1].inner_text()
        assert "alice" in last_response.lower(), "Chatbot forgot user's name"

    def test_response_latency_within_sla(self, page: Page):
        """Test that responses arrive within SLA"""
        import time
        page.goto("http://localhost:3000")
        page.fill('[data-testid="chat-input"]', "What is 2+2?")

        start = time.time()
        page.click('[data-testid="send-button"]')
        page.wait_for_selector('[data-testid="assistant-message"]', timeout=15000)
        latency = time.time() - start

        assert latency < 10.0, f"Response took {latency:.1f}s — exceeds 10s SLA"
```

### Async Playwright for Parallel Tests

```python
import asyncio
import pytest
from playwright.async_api import async_playwright, expect

@pytest.mark.asyncio
async def test_concurrent_users():
    """Test that the system handles concurrent users"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        async def simulate_user(user_id: int) -> bool:
            page = await browser.new_page()
            await page.goto("http://localhost:3000")
            await page.fill('[data-testid="chat-input"]', f"User {user_id}: What is ML?")
            await page.click('[data-testid="send-button"]')

            response = await page.wait_for_selector(
                '[data-testid="assistant-message"]',
                timeout=30000
            )
            text = await response.inner_text()
            await page.close()
            return len(text) > 20

        # Simulate 10 concurrent users
        results = await asyncio.gather(*[
            simulate_user(i) for i in range(10)
        ])

        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Only {success_rate:.0%} of concurrent requests succeeded"

        await browser.close()
```

---

## Puppeteer (Node.js)

Puppeteer is Chrome's official Node.js browser automation library. Used when your test infra is JavaScript-based.

```javascript
// test/chatbot.test.js
const puppeteer = require('puppeteer');

describe('ML Chatbot E2E', () => {
    let browser, page;

    beforeAll(async () => {
        browser = await puppeteer.launch({ headless: true });
        page = await browser.newPage();
    });

    afterAll(async () => {
        await browser.close();
    });

    test('responds to ML questions', async () => {
        await page.goto('http://localhost:3000');
        await page.type('[data-testid="chat-input"]', 'Explain gradient descent');
        await page.click('[data-testid="send-button"]');

        // Wait for response with timeout
        await page.waitForSelector('[data-testid="assistant-message"]', {
            timeout: 30000
        });

        const responseText = await page.$eval(
            '[data-testid="assistant-message"]',
            el => el.textContent
        );

        expect(responseText.length).toBeGreaterThan(50);
        expect(responseText.toLowerCase()).toMatch(/gradient|loss|learning rate/);
    });

    test('captures network calls for debugging', async () => {
        const requests = [];

        page.on('request', req => {
            if (req.url().includes('/api/chat')) {
                requests.push({
                    url: req.url(),
                    method: req.method(),
                    body: req.postData()
                });
            }
        });

        await page.type('[data-testid="chat-input"]', 'What is RAG?');
        await page.click('[data-testid="send-button"]');
        await page.waitForSelector('[data-testid="assistant-message"]', { timeout: 20000 });

        // Verify request was made with correct payload
        expect(requests.length).toBe(1);
        const body = JSON.parse(requests[0].body);
        expect(body).toHaveProperty('question');
    });
});
```

---

## API-Level Testing for LLM Backends

### pytest with httpx for FastAPI

```python
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app  # Your FastAPI app

client = TestClient(app)

class TestChatAPI:

    def test_valid_question_returns_200(self):
        response = client.post("/chat", json={
            "question": "What is machine learning?",
            "max_tokens": 256
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "latency_ms" in data

    def test_empty_question_returns_422(self):
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_question_too_long_returns_422(self):
        response = client.post("/chat", json={"question": "x" * 5000})
        assert response.status_code == 422

    @patch("app.main.call_anthropic")
    def test_llm_failure_returns_502(self, mock_llm):
        mock_llm.side_effect = Exception("API Error")
        response = client.post("/chat", json={"question": "test"})
        assert response.status_code == 502

    def test_rate_limit_enforced(self):
        """Test that rate limiting kicks in"""
        responses = []
        for _ in range(15):  # Exceed 10 req/sec limit
            r = client.post("/chat", json={"question": "test"})
            responses.append(r.status_code)

        assert 429 in responses, "Rate limiting not enforced"

    def test_streaming_endpoint_returns_sse(self):
        with client.stream("POST", "/chat/stream", json={"question": "Explain AI"}):
            # Check response headers
            pass  # Streaming test is complex — use async client
```

### Mock LLM for Fast Tests

```python
from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.fixture
def mock_anthropic_response():
    mock = MagicMock()
    mock.content = [MagicMock(text="This is a mock LLM response about machine learning.")]
    mock.usage.input_tokens = 50
    mock.usage.output_tokens = 100
    mock.model = "claude-sonnet-4-6"
    return mock

@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    client = AsyncMock()
    client.messages.create.return_value = mock_anthropic_response
    return client

@pytest.fixture
def test_app(mock_anthropic_client):
    app.state.anthropic_client = mock_anthropic_client
    return TestClient(app)

def test_chat_with_mock_llm(test_app, mock_anthropic_client):
    """Fast test that doesn't call real LLM"""
    response = test_app.post("/chat", json={"question": "What is ML?"})
    assert response.status_code == 200
    assert "mock LLM response" in response.json()["answer"]

    # Verify LLM was called with correct params
    mock_anthropic_client.messages.create.assert_called_once()
    call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 1024  # Default
```

---

## Tracing and Debugging AI Applications

### OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Setup (once at app startup)
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ml-app")

@app.post("/chat")
async def chat(request: ChatRequest):
    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("question.length", len(request.question))
        span.set_attribute("model", request.model)

        with tracer.start_as_current_span("vector_retrieval"):
            docs = retrieve_context(request.question)
            span.set_attribute("docs.count", len(docs))

        with tracer.start_as_current_span("llm_generation"):
            answer = await generate_answer(request.question, docs)
            span.set_attribute("answer.tokens", len(answer.split()))

        return {"answer": answer}
```

---

## CI/CD Integration

```yaml
# .github/workflows/ai-tests.yml
name: AI Application Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests (mocked LLM)
        env:
          ANTHROPIC_API_KEY: "test-key"
        run: pytest tests/unit/ -v --mock-llm

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - name: Start app
        run: docker-compose up -d
      - name: Install Playwright
        run: pip install playwright && playwright install chromium
      - name: Run E2E tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: pytest tests/e2e/ -v --timeout=60
      - name: Run eval suite
        run: python scripts/run_eval.py --threshold 0.85
```

---

## Playwright vs Puppeteer

| Dimension | Playwright (Python/JS) | Puppeteer (JS) |
|-----------|----------------------|----------------|
| **Language** | Python, JS, Java, C# | JavaScript only |
| **Browsers** | Chrome, Firefox, Safari | Chrome/Edge only |
| **Auto-wait** | Built-in smart waits | Manual waits |
| **Network intercept** | Full request/response mock | Request intercept only |
| **Parallel tests** | Built-in via `pytest-xdist` | External tooling |
| **Screenshots/video** | Built-in | Built-in |
| **Best for** | Cross-browser AI UI testing | Chrome-specific JS apps |

---

## Interview Questions

**Q: How do you test an LLM chatbot UI end-to-end?**
> Use Playwright or Puppeteer to automate browser interactions. Key tests: (1) basic response received, (2) streaming works incrementally, (3) errors shown gracefully, (4) conversation history maintained, (5) latency within SLA. For AI-specific: assert that responses contain relevant domain terms, not just that they're non-empty.

**Q: How do you write fast unit tests for LLM applications?**
> Mock the LLM client. Use `unittest.mock.AsyncMock` to return fixed responses without calling the real API. This makes tests 100x faster, free, and deterministic. Only call real LLMs in integration/eval tests that run less frequently.

**Q: What's the biggest challenge in E2E testing for AI apps?**
> Non-determinism — the same test can pass one run and fail the next. Solutions: (1) test behavioral properties ("mentions relevant terms") not exact strings, (2) use fixed seeds/temperature=0 in tests, (3) set adequate timeouts (AI is slow), (4) mock LLM for most tests, only use real LLM for integration tests.

**Q: How do you test streaming responses?**
> Capture response text at intervals (e.g., every 500ms). Assert that the text grows over time. Check that the first token arrives within acceptable time (e.g., < 2s). Verify the final response is complete. Use network interception to validate SSE format.
