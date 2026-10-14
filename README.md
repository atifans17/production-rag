# Production RAG Chatbot

An end-to-end, production-oriented Retrieval-Augmented Generation (RAG) system built to answer technical questions over machine learning literature and documentation (including AI textbooks, Python scientific computing guides, and FastAPI specifications). Designed to move beyond naïve single-vector retrieval, this pipeline implements structure-aware chunking, hybrid search (dense embeddings + BM25) fused with Reciprocal Rank Fusion (RRF), cross-encoder re-ranking, and a calibrated relevance threshold. The entire architecture is evaluated systematically against a 33-question benchmark tracking Hit@K, MRR, latency, and cost to guarantee robust, hallucination-resistant retrieval before generation.

---

## Architecture

```
User Query
    │
    ├─── Exact Match Cache ───► [Cache Hit: Return Immediate Response]
    │
    ▼ (Cache Miss)
Hybrid Retrieval (struct_500_50)
    │
    ├─── Dense Retrieval (ChromaDB + sentence-transformers/all-MiniLM-L6-v2)
    └─── Sparse Retrieval (BM25 keyword matching)
    │
    ▼
Reciprocal Rank Fusion (RRF) ──► Candidate Fusion Pool (Top-K)
    │
    ▼
Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)
    │
    ▼
Relevance Threshold Gate (Threshold = 1.0)
    │
    ├─── Score < 1.0 ──────────► [Direct Refusal: Out-of-Domain / Unanswerable]
    │
    ▼ (Score >= 1.0)
Strict Grounded Prompt + Top Chunks
    │
    ▼
LLM Generation (Ollama Llama 3.2 / Groq Llama 3.1) ──► Cited Answer + Cache Write
```

---

## Results

### Retrieval (30+ Labelled Questions)

Evaluation across retrieval strategies on the benchmark dataset using the selected `struct_500_50` collection:

| Stage | Mode / Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Baseline | `dense` | 71% | 82% | 82% | 0.768 |
| Keyword | `bm25` | 75% | 89% | 96% | 0.838 |
| Fusion | `hybrid` (Dense + BM25 via RRF) | 71% | 96% | 96% | 0.839 |
| **Final Pipeline** | **`hybrid_rerank` (Hybrid + Cross-Encoder)** | **89%** | **100%** | **100%** | **0.946** |

The multi-stage retrieval pipeline demonstrated steady, compounding improvements:

* Moving from pure `dense` to `hybrid` widened candidate coverage, jumping Hit@3 from 82% to 96% and MRR from 0.768 to 0.839.
* Adding a cross-encoder re-ranker (`hybrid_rerank`) produced the largest leap in precision, driving **Hit@1 to 89%**, **Hit@3 and Hit@5 to 100%**, and **MRR to 0.946**.

---

### Chunking Experiments

Evaluations performed across multiple chunking configurations using dense vector retrieval:

| Collection | Chunk Strategy | Overlap | Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_200_20` | Fixed Token (200 tokens) | 20 tokens | dense | 57% | 79% | 86% | 0.675 |
| `fixed_500_0` | Fixed Token (500 tokens) | 0 tokens | dense | 64% | 86% | 86% | 0.738 |
| `fixed_500_50` | Fixed Token (500 tokens) | 50 tokens | dense | 64% | 82% | 86% | 0.729 |
| **`struct_500_50`** | **Structure-Aware (500 tokens)** | **50 tokens** | **dense** | **71%** | **82%** | **82%** | **0.768** |

**What was learned:**

1. **Window Size:** Small chunk windows (`fixed_200_20`) fragmented key technical concepts across boundaries, yielding a poor 57% Hit@1 and 0.675 MRR. Expanding chunk sizes to ~500 tokens preserved the contextual cohesion required for technical documentation.
2. **Contextual Headers Win:** `struct_500_50` outperformed all naive fixed chunking strategies on primary rank precision, achieving a **71% Hit@1** and **0.768 MRR**. Prepending document titles and section metadata directly into the chunk text provides semantic grounding that helps bi-encoders retrieve the exact target chunk at rank 1.

---

### Cost and Latency

* **Exact-Match Cache Hit:** ~2–5 ms total response time, $0.000000 compute cost (100% savings on downstream LLM and re-ranker operations).
* **Unanswerable / Out-of-Domain Filter:** Short-circuited at the retrieval gate when top candidate scores fall below `1.0`. Eliminates unnecessary LLM inference calls entirely on out-of-domain queries.
* **Full Retrieval & Generation Pass (`hybrid_rerank`):**
  * Average Retrieval + Re-ranking Latency: ~120–180 ms.
  * LLM Generation Latency: ~800–1,600 ms (dependent on backend model and local compute/cloud provider).
  * Average Cost per Answered Query: $0.0000 (when using local Ollama `llama3.2` or free-tier Groq `llama-3.1-8b-instant`).

---

## How to Run

### 1. Clone & Configure Environment

```bash
git clone https://github.com/<your-username>/production-rag.git
cd production-rag
cp .env.example .env
```

Ensure your `.env` specifies your preferred LLM provider (`ollama`, `groq`, or OpenAI compatible) and sets `COLLECTION=struct_500_50`.

### 2. Build Container

```bash
docker compose build
```

### 3. Ingest and Build Vector Index

```bash
docker compose run --rm api python -m src.index struct_500_50
```

### 4. Start the Service

```bash
docker compose up -d
```

The FastAPI application will be live at `http://localhost:8000`, with interactive OpenAPI documentation available at `http://localhost:8000/docs`.

---

## Design Decisions

* **Chunking (`struct_500_50`):** 500-token chunks with 50-token overlap retain complete algorithmic explanations and code snippets without arbitrary mid-sentence cuts. Prepending parent section headers to every chunk disambiguates generic terms across different chapters.
* **Hybrid Search (Dense + BM25):** Pure dense search can miss exact code tokens, specific method signatures, or technical identifiers. BM25 provides lexical precision, while dense vectors provide semantic generalization.
* **Reciprocal Rank Fusion (RRF):** Dense distance scores and BM25 scores operate on entirely different mathematical scales. Rather than attempting fragile score normalization, RRF combines them purely based on ranking positions ($RRF\_Score = \sum \frac{1}{60 + rank}$).
* **Cross-Encoder Re-ranking:** Bi-encoders compress entire chunks into single vectors independently, losing granular token-to-token interactions. Re-ranking the top 15 fused candidates with a cross-encoder evaluates the exact question-context pair jointly, boosting Hit@1 from 71% to 89%.
* **Relevance Threshold (`1.0`):** Empirical score distribution analysis showed unanswerable query scores peak significantly lower than answerable queries (median 0.47 vs 4.88). Setting a 1.0 cutoff enables deterministic rejection of unanswerable queries before calling the generative LLM.
* **Embedding Model (`all-MiniLM-L6-v2`):** Delivers a strong balance of latency, memory footprint, and retrieval accuracy for CPU-based local and containerized deployments without requiring dedicated GPU infrastructure.

---

## Limitations and Next Steps

* **Scanned PDFs & OCR:** Current ingestion expects extractable text streams; image-only or scanned PDF pages require integrating an OCR extraction stage (e.g., Tesseract / Surya).
* **Table Parsing:** Complex multi-column Markdown/HTML tables are flattened into raw text chunks, which can disrupt tabular structural alignment during semantic search.
* **Exact-Match Cache:** The caching layer currently operates on exact query string hashes. Implementing a semantic cache (e.g., vector similarity over queries with cosine threshold > 0.96) will increase cache hit rates for paraphrased inputs.
* **LLM-as-a-Judge Bias:** Automated faithfulness and correctness evaluations rely on LLM judges, which carry inherent position and leniency biases.
* **Evaluation Scale:** The current evaluation set comprises 33 hand-curated questions. Future iterations will expand the test harness to 200+ samples with multi-turn conversational edge cases.
* **Language Support:** Ingestion tokenizers, BM25 stopword structures, and prompts are currently optimized strictly for English documentation.
