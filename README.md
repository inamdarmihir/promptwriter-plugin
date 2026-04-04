<div align="center">

# ✨ Prompt Guidance Tool

**AI-powered prompt rewriting CLI that turns weak prompts into high-performing ones.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOSA1IDktNXYtNWwtOSA1LTktNXoiLz48L3N2Zz4=&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97757?style=flat-square&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Docker](https://img.shields.io/badge/Docker-Qdrant-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)

<br/>

*Powered by **Qdrant RAG**, a **RLHF feedback loop**, and **Context Engineering** — works with any local or cloud LLM.*

</div>

---

## How It Works

```
Your prompt
  ↓
[Classify]        Detect type, domain, weaknesses, complexity
  ↓
[Multi-query RAG] 3 targeted Qdrant searches from different angles
  ↓
[RLHF Lookup]     Inject high-rated past rewrites as few-shot examples
  ↓
[Budget Allocate] Pick the right amount of context for the complexity
  ↓
[LLM Rewrite]     Structured rewriting policy → 7-section output
  ↓
Enhanced prompt + analysis + what changed + confidence score
```

The tool **learns over time** — every rewrite you rate gets stored and used as a few-shot example the next time a similar prompt comes in.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **Multi-provider LLMs** | Ollama (local), OpenAI, Anthropic/Claude, Mistral |
| 🔢 **Multi-provider embeddings** | Ollama, OpenAI, Sentence Transformers (fully local) |
| 🗄️ **Qdrant vector store** | Knowledge base + RLHF preference store, both in Qdrant |
| 🧠 **Context Engineering** | Prompt classification + multi-query RAG fusion + budget-aware context assembly |
| 🔁 **RLHF feedback loop** | Rate rewrites → stored as few-shot examples → future rewrites improve automatically |
| 📄 **Multi-format ingestion** | Excel, CSV, PDF, TXT, Markdown, JSON, JSONL, DOCX |
| 🛡️ **Strict rewriting policy** | Never changes intent, always explains every change, output is always a complete ready-to-use prompt |
| 🎨 **Rich CLI** | Colour-coded side-by-side diff, source attribution, confidence score, alternative versions |

---

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Pull Ollama models (if using local LLMs)

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Install

```bash
pip install -e .
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER, API keys, etc.
```

### 5. Seed the knowledge base

```bash
python data/generate_sample.py        # creates data/prompt_techniques.xlsx
pgt ingest data/prompt_techniques.xlsx
```

### 6. Rewrite your first prompt

```bash
pgt enhance "explain transformers"
```

---

## CLI Commands

### `pgt enhance` — rewrite a prompt

```bash
pgt enhance "summarize this document"

# With options
pgt enhance "fix my code" \
  --provider openai \
  --model gpt-4o \
  --framework "Claude API" \
  --context "This is a Python Flask app" \
  --top-k 6 \
  --sources          # show retrieved knowledge excerpts

# Pipe from stdin
cat my_prompt.txt | pgt enhance
```

### `pgt ingest` — add knowledge to the vector store

```bash
pgt ingest data/prompt_techniques.xlsx
pgt ingest docs/guidelines.pdf
pgt ingest notes.md --chunk-size 300 --overlap 30
```

Supported formats: `.xlsx` `.xls` `.csv` `.pdf` `.txt` `.md` `.rst` `.json` `.jsonl` `.docx`

### `pgt feedback` — rate a rewrite (RLHF)

```bash
pgt feedback \
  --original "explain quantum computing" \
  --enhanced "You are a quantum physicist... [full rewrite]" \
  --rating 5 \
  --comment "Role + format spec made the answer much more focused"
```

High-rated rewrites are embedded and stored in Qdrant. The next time a similar prompt arrives, they are automatically injected as few-shot examples — no model fine-tuning required.

### `pgt feedback-stats` — view the RLHF learning curve

```bash
pgt feedback-stats
```

### `pgt search` — explore the knowledge base

```bash
pgt search "few-shot examples"
pgt search "chain of thought reasoning" --top-k 8
```

### `pgt status` — health check

```bash
pgt status
```

Shows Qdrant connectivity, total chunks, ingested sources, and available Ollama models.

### `pgt interactive` — REPL mode

```bash
pgt interactive
pgt interactive --provider anthropic --model claude-3-5-haiku-20241022
```

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` · `openai` · `anthropic` · `mistral` |
| `EMBED_PROVIDER` | `ollama` | `ollama` · `openai` · `sentence-transformers` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Model for generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `EMBED_DIMENSIONS` | `768` | Must match your embedding model |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `MISTRAL_API_KEY` | — | Required for Mistral provider |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | — | For Qdrant Cloud |
| `QDRANT_COLLECTION` | `prompt_knowledge` | Collection name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Default retrieval count |

### Embedding dimensions by model

| Model | Provider | Dimensions |
|---|---|---|
| `nomic-embed-text` | Ollama | 768 |
| `mxbai-embed-large` | Ollama | 1024 |
| `text-embedding-3-small` | OpenAI | 1536 |
| `text-embedding-3-large` | OpenAI | 3072 |
| `mistral-embed` | Mistral | 1024 |
| `all-MiniLM-L6-v2` | Sentence Transformers | 384 |

---

## Architecture

```
src/prompt_guidance/
├── config.py              Pydantic-settings configuration
├── llm.py                 BaseLLM + Ollama / OpenAI / Anthropic / Mistral + factory
├── embeddings.py          BaseEmbeddings + Ollama / OpenAI / SentenceTransformers + factory
├── vectorstore.py         QdrantStore — upsert, search, delete, list sources
├── ingestors.py           Registry + Excel / CSV / PDF / Text / JSON / DOCX loaders
├── pipeline.py            TextChunker · IngestPipeline · RAGPipeline
├── context_engineer.py    Classify → multi-query RAG → RLHF → budget → assemble
├── feedback.py            FeedbackStore — save ratings, retrieve few-shot examples
├── rewriter.py            PromptRewriter — full pipeline + structured output parser
└── cli.py                 Typer CLI (enhance / ingest / search / status / feedback / …)
```

### Context Engineering pipeline

```python
# 1. Classify the prompt
classification = context_engineer.classify(prompt)
# → {type: "code_generation", domain: "technical",
#    complexity: "medium", weaknesses: ["no_role", "no_format"]}

# 2. Multi-query retrieval (3 targeted searches, fused + deduped)
chunks = context_engineer.multi_retrieve(prompt, classification, budget)

# 3. RLHF: fetch similar high-rated past rewrites
rlhf_examples = context_engineer.get_rlhf_examples(query_vector, budget)

# 4. Assemble with budget allocation
context = context_engineer.assemble(chunks, rlhf_examples, classification, budget)
```

### RLHF loop

```python
# After a rewrite
result = rewriter.rewrite("explain machine learning")

# User rates it
rewriter.save_feedback(result, rating=5, comment="Role + format worked perfectly")

# Next similar prompt automatically gets this as a few-shot example
result2 = rewriter.rewrite("explain neural networks")
# → context now includes the 5-star example above
```

---

## Extending the Tool

### Add a new LLM provider

```python
# src/prompt_guidance/llm.py
class GroqLLM(BaseLLM):
    def generate(self, system: str, user: str) -> str:
        ...

_PROVIDERS["groq"] = GroqLLM
```

### Add a new file type

```python
# src/prompt_guidance/ingestors.py
@register(".html")
class HTMLIngestor:
    def load(self, file_path: Path) -> list[Document]:
        ...
```

### Add a new vector store

Implement the four methods used by `RAGPipeline` and `IngestPipeline`:

- `initialize()` · `upsert(chunks)` · `search(vector, top_k)` · `delete_by_source(source)`

---

## Rewriting Policy

The LLM is bound by a strict policy on every rewrite:

1. **Preserve intent** — never change what the user is asking for
2. **Complete output** — the enhanced prompt must be self-contained and ready to paste
3. **Explain every change** — all modifications appear in the "What Changed" section
4. **Additive only** — strengthen existing phrasing, do not replace meaning
5. **Keep domain terms** — do not substitute the user's technical vocabulary
6. **Evidence-based** — only apply techniques present in the retrieved knowledge or RLHF examples

---

## Sample Knowledge Base

`data/generate_sample.py` creates `prompt_techniques.xlsx` with four sheets:

| Sheet | Contents |
|---|---|
| Techniques | 10 core techniques with bad/good examples and impact ratings |
| Anti-Patterns | 7 common mistakes and how to fix them |
| Framework Tips | OpenAI, Claude, Ollama, Mistral, LangChain, LlamaIndex, RAG systems |
| Prompt Templates | Ready-to-use templates for code review, summarization, data analysis, RAG, creative writing |

Ingest your own documents to build a domain-specific knowledge base.

---

## Requirements

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud%20or%20Local-DC244C?style=flat-square)](https://qdrant.tech/)

- **Python ≥ 3.11**
- **Docker** (for Qdrant) or [Qdrant Cloud](https://cloud.qdrant.io)
- At least one of: Ollama running locally, or an API key for OpenAI / Anthropic / Mistral

---

<div align="center">

Built with ❤️ using [Qdrant](https://qdrant.tech/) · [Ollama](https://ollama.com/) · [Anthropic Claude](https://www.anthropic.com/) · [Pydantic](https://docs.pydantic.dev/)

</div>
