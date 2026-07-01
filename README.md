# 🧠 Advanced RAG Techniques — Playbook & Toolkit

A hands-on collection of **Retrieval-Augmented Generation (RAG)** patterns that solve real-world failure modes of naive RAG systems. Each module is self-contained, well-documented, and ready to run.

---

## Problem

Naive RAG (embed → retrieve → generate) breaks in production:

| Failure Mode | Symptom |
|---|---|
| **Pure vector search misses exact matches** | Product codes (`SKU-7742X`), error codes (`E_CONN_REFUSED`), acronyms (`WCAG`), names → zero results |
| **Poorly phrased queries** | Users ask one way, documents are written another way → relevant chunks are never retrieved |
| **Irrelevant content dilutes context** | Retrieved chunks are 90% filler; LLM gets "lost in the middle" and hallucinates |
| **Chunk size conflict** | Small chunks = better search; large chunks = better LLM context. You can't have both — or can you? |
| **Uncontrolled API costs** | No pre-flight budget → every oversized prompt costs money. No tracking → can't attribute costs per user |

---

## Solution — Three Modules

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Playbook                             │
├─────────────────┬───────────────────┬───────────────────────┤
│  hybridSearch   │ optimizeRagTech   │ tokenTracking&Limit   │
│                 │                   │                       │
│  Vector + BM25  │ Multi-Query       │ Pre-flight budget     │
│  (keywords)     │ Contextual Comp.  │ Per-user tracking     │
│                 │ Parent Doc        │ Cost ceiling          │
│                 │ Hybrid Ensemble   │                       │
└─────────────────┴───────────────────┴───────────────────────┘
```

### 1. [`hybridSearch/`](hybridSearch/README.md) — Hybrid Search (Vector + BM25)

Combines **OpenAI embeddings** (semantic meaning) with **BM25** (keyword/term-frequency) via weighted Reciprocal Rank Fusion.

```
Query ──┬─► Vector Retriever ──┐
         │                      ├──► EnsembleRetriever (RRF) ──► Results
         └─► BM25 Retriever ────┘
```

**[→ Go to module](hybridSearch/README.md)**

### 2. [`optimizeRagTechnique/`](optimizeRagTechnique/README.md) — Advanced RAG Pipeline

Four techniques composed into an end-to-end pipeline:

| # | Technique | Benefit |
|---|-----------|---------|
| 1 | **Multi-Query Retriever** | LLM rewrites query into 3 variants → catches poor phrasing |
| 2 | **Hybrid Ensemble (BM25 + Vector)** | Handles both semantics and exact keywords |
| 3 | **Parent Document Retriever** | Small child chunks for search, large parent chunks for LLM context |
| 4 | **Contextual Compression** | Strips irrelevant filler from chunks → saves tokens, reduces hallucination |

**Pipeline flow:**
```
Query → MultiQuery → Hybrid Ensemble (BM25 + ParentDoc Vector) → Compress → LLM
```

**[→ Go to module](optimizeRagTechnique/README.md)**

### 3. [`tokenTrackingandLimiting/`](tokenTrackingandLimiting/README.md) — Token Budgeting

Two classes for production cost control:

| Class | What it does |
|-------|-------------|
| `TokenBudget` | Local token counting (tiktoken), max-token enforcement, aggregate stats |
| `BudgetedLLM` | Pre-flight rejection before any API call + exact usage recording from response metadata |

**[→ Go to module](tokenTrackingandLimiting/README.md)**

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd RAG

# 2. Install dependencies
pip install langchain langchain-core langchain-openai langchain-chroma langchain-community langchain-text-splitters tiktoken python-dotenv

# 3. Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Run any module
python hybridSearch/hybridSearch.py
python optimizeRagTechnique/completePipeline.py
python optimizeRagTechnique/parentDocumentRetriever.py
python tokenTrackingandLimiting/TokenTracking
```

---

## Technologies

`Python 3` · `LangChain` · `OpenAI` (`gpt-4o-mini`, `text-embedding-3-small`) · `ChromaDB` · `BM25` · `tiktoken` · `python-dotenv`
