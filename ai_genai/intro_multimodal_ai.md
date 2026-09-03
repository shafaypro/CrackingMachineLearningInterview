# Multimodal AI

This guide covers systems that reason across multiple data types such as text, images, audio, and documents.

---

## Overview

Multimodal AI combines inputs and outputs across different modalities. It matters because many real-world products are not text-only:

- document understanding needs vision plus OCR plus reasoning
- voice assistants need speech-to-text, text reasoning, and text-to-speech
- image assistants need visual grounding and language generation

---

## Core Concepts

### Vision models

Vision models process images or video to detect objects, read charts, classify scenes, or answer questions about visual content.

### Audio models

Audio pipelines usually involve automatic speech recognition, speaker segmentation, text reasoning over transcripts, and text-to-speech output when needed.

### Multimodal reasoning

This is the ability to combine evidence across modalities. For example, a document assistant may read a chart, inspect surrounding text, and answer a question that depends on both.

### Cross-modal representations

Many systems project text, image, and audio signals into learned shared spaces so related content can be matched or reasoned over together.

---

## Key Skills

### Image understanding

In practice, this means framing the task correctly: classification, detection, OCR, chart understanding, or visual Q&A.

### Speech-to-text and text-to-speech

You should know how latency, transcription quality, chunking, and streaming affect user experience in voice systems.

### Cross-modal reasoning

A strong engineer can design systems where one modality grounds or validates another instead of treating each independently.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| GPT multimodal models | Unified text-image reasoning | Rapid prototyping for image and document Q&A |
| Gemini | Strong multimodal reasoning APIs | Multimodal product prototyping and evaluation |
| Whisper | Open-source speech-to-text | Transcript generation and voice pipelines |
| Open-source vision models | Detection, OCR, and image feature extraction | Self-hosted or specialized vision tasks |
| TTS engines | Convert text output into speech | Voice assistants and accessibility products |

---

## Projects

### Image Q&A system

- Goal: Let users ask questions about uploaded images.
- Key components: image upload, prompt design, structured answer formatting, optional OCR.
- Suggested tech stack: multimodal LLM API, FastAPI, object storage.
- Difficulty: Intermediate.

### Voice assistant

- Goal: Accept voice input, transcribe it, reason over it, and speak a response.
- Key components: streaming audio ingestion, speech-to-text, LLM reasoning, text-to-speech, conversation state.
- Suggested tech stack: Whisper, OpenAI or Gemini API, FastAPI, WebSocket frontend.
- Difficulty: Advanced.

### Document understanding system

- Goal: Extract and answer questions from complex PDFs with images, tables, and text.
- Key components: OCR, layout parsing, chunking, retrieval, citation mapping.
- Suggested tech stack: OCR library, multimodal model, vector store, FastAPI.
- Difficulty: Advanced.

### Multimodal chatbot

- Goal: Support conversations over text, screenshots, and documents in one interface.
- Key components: multimodal input adapter, session memory, model routing, safety filters.
- Suggested tech stack: multimodal API, LiteLLM, Redis, React.
- Difficulty: Advanced.

---

## Example Code

```python
def build_image_qa_prompt(question: str) -> str:
    return (
        "You are an assistant answering questions about an uploaded image. "
        "Use only visible evidence. If the answer is uncertain, say so. "
        f"Question: {question}"
    )
```

---

## Suggested Project Structure

```text
document-understanding/
├── src/
│   ├── ocr.py
│   ├── parser.py
│   ├── retriever.py
│   ├── qa.py
│   └── api.py
├── sample_docs/
├── evals/
└── README.md
```

---

## Interview Q&A

#### How do vision-language models actually connect the two modalities?

Most follow the same pattern: a vision encoder (usually a ViT) turns the image into a grid of patch embeddings, a projection module maps those into the language model's token embedding space, and the LLM then attends over image tokens and text tokens together in one sequence. The projector is small — a linear layer in LLaVA, a resampler with learned queries in Flamingo and BLIP-2 — and is often the only part trained initially, with both encoders frozen.

CLIP is the other core idea and works differently: it trains an image encoder and a text encoder contrastively so that matching image-text pairs land close in a *shared* space. That gives zero-shot classification and image-text retrieval, but not generation.

#### How many tokens does an image cost, and why does it matter?

A great deal — typically hundreds to a couple of thousand tokens per image depending on resolution and the model's tiling scheme. High-resolution handling usually splits the image into tiles, each costing a full grid of patch tokens, so a single detailed screenshot can cost more than a page of text.

This drives real design decisions: downscale images to the smallest resolution that preserves the detail the task needs, crop to the region of interest rather than sending full pages, cache image analysis results rather than re-sending the same image across turns, and batch questions about one image into a single call instead of one call per question.

#### How would you build a document understanding system?

I'd resist making it a single VLM call. The pipeline that actually works: classify the document type, then route — native-text PDFs go through a text extractor (cheap, exact), scanned pages go through OCR, and pages with complex layout, charts, or handwriting go to a VLM. Extract to a strict schema with structured outputs so downstream code can rely on it, and attach a confidence signal per field.

Then the parts candidates forget: validation rules on the extracted fields (dates parse, totals sum, IDs match a format), a human review queue for low-confidence extractions, and an eval set of real documents with ground-truth fields so you can measure field-level accuracy rather than eyeballing outputs.

#### How do you evaluate a multimodal system?

Per task, and never by looking at a few examples. For extraction, field-level precision and recall against labeled documents. For captioning or VQA, task-specific accuracy on a held-out set plus an LLM judge validated against human labels. For retrieval, recall@k on image-text pairs.

The multimodal-specific failure to test for is **hallucinated visual detail** — the model describing objects, text, or values that aren't in the image. Build an adversarial slice: images that are blurry, blank, rotated, or contain text the model would expect but that isn't there, and measure how often it invents content.

#### When is a multimodal model the wrong choice?

When a deterministic extractor exists: a native-text PDF should be parsed, not described by a VLM that may paraphrase or hallucinate. When the task is high-volume, narrow, and stable — a fine-tuned small classifier is far cheaper than a VLM per image. When exactness is required and errors are costly, unless you add validation and a human review path. And when latency matters, since image prefill is expensive and dominates time-to-first-token.

#### How do you handle audio in a production pipeline?

Decide between cascaded and end-to-end. **Cascaded** (STT → LLM → TTS) is the default: each stage is separately testable, replaceable, and cheap, and you get a text transcript for logging, evaluation, and compliance. **End-to-end speech models** cut latency substantially and preserve tone and interruption handling, which matters for natural conversation, but are harder to debug and evaluate.

For cascaded pipelines the practical issues are streaming (start transcribing before the speaker stops), endpointing (deciding when a turn ended), and domain vocabulary — proper nouns and jargon are where word error rate concentrates, so a custom vocabulary or biasing list is usually the highest-leverage fix.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Sending full-resolution images by default | Token cost and latency explode for no accuracy gain | Downscale to the minimum useful resolution; crop to the region of interest |
| One API call per question about the same image | Re-pays the image token cost every time | Batch all questions into one call |
| Using a VLM on native-text PDFs | Slower, costlier, and can paraphrase exact values | Route by document type; extract text directly when possible |
| Free-text output for extraction tasks | Downstream parsing is brittle | Structured outputs with a strict schema |
| No validation of extracted fields | Hallucinated values flow silently into systems | Format and consistency checks + low-confidence review queue |
| Evaluating on clean examples only | Real inputs are blurry, rotated, and partially blank | Adversarial eval slice for visual hallucination |
| Ignoring image preprocessing consistency | Different resize/orientation between eval and production shifts results | Pin one preprocessing path used by both |
| Assuming OCR accuracy transfers across domains | Handwriting, tables, and jargon are where WER concentrates | Measure on your own documents; add a vocabulary bias list |

---

## Related Topics

- [LLM Fundamentals](./intro_llm_fundamentals.md)
- [RAG](./intro_rag.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
