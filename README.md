# Hybrid Search for RAG

A hands-on demonstration of **Hybrid Search** (Vector + BM25) for Retrieval-Augmented Generation (RAG) — and why pure semantic search fails on things like product codes, error codes, and acronyms.

---

## The Problem

Vector search (embeddings) understands **meaning**, but struggles with:

| Problem | Example | Why it fails |
| :--- | :--- | :--- |
| **Product Codes** | `SKU-7742X` | No semantic meaning — embedding scatters across unrelated docs |
| **Error Codes** | `E_CONN_REFUSED` | The model doesn't know what this string means |
| **Acronyms** | `WCAG` | Retrieves docs about "compliance" but misses WCAG-specific results |
| **Exact Names** | `John Smith` | Vector search finds "accounting" and "Smith" but skips "John" |

> 📄 Full code: [`hybridSearch.py`](hybridSearch.py)

---

## The Solution: Hybrid Search

Combine **two retrievers** so one catches what the other misses:

```
User Query
    │
    ├──► Vector Retriever (semantic meaning)
    │
    └──► BM25 Retriever (keyword matching)
    │
    └──► Ensemble Retriever (merges & re-ranks)
              │
              ▼
         Final Results
```

---

## Code Walkthrough

### 1. Sample Documents ([line 16](hybridSearch.py#L16))

The dataset includes products, error codes, troubleshooting guides, and compliance standards:

```python
documents = [
    Document(page_content="Product SKU-7742X is our flagship dual-band router..."),
    Document(page_content="Error code E_CONN_REFUSED indicates the server rejected the connection..."),
    Document(page_content="WCAG 2.1 compliance requires all images to have alt text..."),
]
```

### 2. Vector Retriever ([line 126](hybridSearch.py#L126))

Embeds documents into a vector store and searches by **semantic similarity**:

```python
vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

**Example:** Query `"network hardware specs"` → returns documents about routers and modems.

### 3. BM25 Retriever ([line 139](hybridSearch.py#L139))

Searches by **exact keyword matching** (no AI, just math):

```python
bm25_retriever = BM25Retriever.from_documents(documents, k=3)
```

**Example:** Query `"SKU-7742X"` → returns the exact document containing that product code.

### 4. Ensemble Retriever ([line 146](hybridSearch.py#L146))

Merges both results with weighted scoring:

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
```

You can tune the `weights` to favor meaning (`[0.7, 0.3]`) or keywords (`[0.3, 0.7]`).

---

## When to Use Hybrid Search

- **Enterprise data** — product SKUs, employee IDs, error codes
- **Technical documentation** — exact config keys, command names, log messages
- **Legal documents** — clause numbers, statute references, exact phrases
- Any scenario where **accuracy is critical**

---

## Running the Code

```bash
# Install dependencies
pip install langchain langchain-chroma langchain-openai python-dotenv

# Set your OpenAI key
echo "OPENAI_API_KEY=..." > .env

# Run
python hybridSearch.py
```

---

## Chunking Strategies

| Strategy | How it works | Best for |
| :--- | :--- | :--- |
| **Fixed** | Split every N characters | Simple, fast prototyping |
| **Recursive** | Split by `\n\n` → `\n` → `" "` (hierarchical) | General text, code |
| **Semantic** | Split at topic boundaries | Long articles, reports |
| **Late** | Keep full doc, retrieve at query time | Question-answering |

### Separators Matter

The `RecursiveCharacterTextSplitter` uses a hierarchy:

```
["\n\n", "\n", " ", ""]
```

This ensures chunks break at paragraphs first, then lines, then words — preserving context.

| Data Type | Recommended Splitter |
| :--- | :--- |
| **Markdown** | `MarkdownHeaderTextSplitter` |
| **Code** | `RecursiveCharacterTextSplitter.from_language` |
| **JSON** | `RecursiveJsonSplitter` |

---

## Hybrid Search Tips

### ⚠️ BM25 Rebuild
BM25 **does not** support incremental updates — you must rebuild the retriever every time you add documents. Keep a reference to your document list and re-create `BM25Retriever.from_documents(...)` on each change.

### 🔧 Tune Weights
- Start **50/50** (`[0.5, 0.5]`)
- Adjust based on query patterns (favor BM25 for codes/IDs, vector for concepts)
- Log which retriever contributes to each result to guide tuning

### 📏 K Value
- Retrieve **more** than you think you need — let the ensemble's RRF sort it out
- Use `k=4` or higher for better recall

### ⏱️ Latency
- Hybrid adds roughly **20–50ms** per query (two searches instead of one)
- Well worth it for the accuracy gain

---

> **Low effort, HIGH IMPACT upgrade.** Hybrid search won on every query type in testing — no reason not to use it.

---

## Key Takeaways

1. **Vector search alone** misses exact matches (codes, IDs, names)
2. **BM25 alone** misses semantic meaning (synonyms, concepts)
3. **Hybrid search** combines both — one catches what the other misses
4. **Chunking quality** directly affects retrieval accuracy
