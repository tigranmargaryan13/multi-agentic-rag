# Multi-Agentic Multimodal RAG System

A financial knowledge assistant for audit, finance, and strategy teams. User queries are routed to specialist agents backed by domain-specific document collections (ESG, IFRS, competitor reports). Agents run in parallel and their answers are synthesised into a single coherent, cited response.

## Architecture

```
User Query
    │
    ▼
Source Classification Agent
    │
    ├── ambiguous?   → ask clarification question
    ├── no sources?  → ask web-search permission → DuckDuckGo ReAct Agent
    └── sources found ──────────────────────────────────────────────────┐
            ├── ESG Agent        (ReAct + FAISS)  ╮                     │
            ├── IFRS Agent       (ReAct + FAISS)  ├── parallel          │
            └── Competitor Agent (ReAct + FAISS)  ╯                     │
                                        ▼                               │
                              Synthesis Node (LLM)  ◄───────────────────┘
                                        │
                                        ▼
                              Final Answer + Follow-up prompt
```

## Data Sources

| Source | Content |
|---|---|
| **ESG** | GHG Protocol standards — scope 1/2/3 emissions, project accounting, value chain |
| **IFRS** | IFRS 18 — Presentation and Disclosure in Financial Statements |
| **Competitor** | Pfizer Annual Report — market intelligence and financial performance |
| **Web** | DuckDuckGo real-time search (fallback, requires user approval) |

## Tech Stack

| Component | Choice |
|---|---|
| LLM | OpenAI `gpt-4.1-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS (CPU) |
| Orchestration | LangGraph |
| Agents | LangChain ReAct |
| UI | Streamlit |

## Setup

### 1. Prerequisites

- Python 3.12+
- OpenAI API key

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Optional overrides:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `gpt-4.1-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RETRIEVE_DOCS_NUMBER` | `4` | Chunks retrieved per agent per query |
| `CHUNK_SIZE` | `1024` | Ingestion chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Chunk overlap between splits (tokens) |

### 4. Add documents

Place PDFs in the matching sub-folders under `documents/`:

```
documents/
├── ESG_documents/                    # GHG Protocol PDFs
├── IFRS_documents/                   # IFRS 18 PDF
└── annual_reports_of_comperitors/    # Competitor annual reports
```

### 5. Ingest

```bash
python build_index.py
```

Builds FAISS indexes under `faiss_index/`. To re-ingest a single source, delete its folder and re-run:

```bash
rm -rf faiss_index/ifrs
python build_index.py
```

## Running

### Streamlit (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### CLI

```bash
python main.py
```

Pass `--ingest` to run ingestion before starting the chat loop:

```bash
python main.py --ingest
```

### Docker

```bash
docker compose up --build
```

Requires a `.env` file with `OPENAI_API_KEY`. The `faiss_index/` and `documents/` folders are mounted as volumes — indexes built on the host are used directly without being baked into the image.

## How it works

1. **Source Classification** — A structured LLM call analyses the query and returns the relevant source keys (`esg`, `ifrs`, `competitor`). If the query is ambiguous it returns a clarification question instead.

2. **Parallel RAG Agents** — For each selected source a ReAct agent is invoked concurrently. Each agent uses a `search_documents` tool backed by its FAISS index to retrieve the most relevant chunks, then generates a source-specific answer with citations.

3. **Synthesis** — A synthesis node merges all agent answers into one cohesive response, attributes facts to their sources, and ends with a relevant follow-up question.

4. **Web Fallback** — When no internal source matches, the user is asked for permission before a DuckDuckGo web search is performed.

5. **Feedback Loop** — After every answer the user is prompted to ask a follow-up or request information from a specific source.

## Multimodal ingestion

Each PDF is processed with three extraction layers:

| Layer | Method |
|---|---|
| **Text** | Raw text via PyMuPDF |
| **Tables** | Markdown tables via PyMuPDF `find_tables` — tries both `lines` (bordered) and `text` (borderless/column) strategies |
| **Visuals** | GPT-4 Vision descriptions of charts, diagrams, and figures on image-bearing pages |

Table and visual content are embedded alongside regular text so agents can answer questions that require data from figures or structured tables.

## Improvements

Potential next steps to harden and scale the system.

### Vector store

| Current | Improvement |
|---|---|
| FAISS (in-process, file-backed) | Replace with a managed vector DB — **Pinecone**, **Weaviate**, or **pgvector** — to gain persistence, horizontal scaling, metadata filtering, and real-time index updates without re-ingesting the full collection |

A managed store also removes the need to mount `faiss_index/` as a Docker volume and makes multi-replica deployments trivial.

### Testing

| Type | What to cover |
|---|---|
| **Unit tests** | `DocumentIndexer._table_to_markdown` edge cases (empty tables, single-row, special characters); `AgentFactory` classifier prompt formatting; `GraphState` reducer `_merge_dicts` |
| **Integration tests** | Full graph invocation with a mocked LLM to verify routing decisions, clarification paths, web-fallback gating, and `chat_history` threading across turns without hitting the OpenAI API |
| **RAG evaluation** | Measure retrieval quality (precision@k, recall@k) and answer faithfulness / relevance using a framework like **RAGAS** or **TruLens** against a golden Q&A dataset |

### Agents & retrieval

- **Hybrid search** — combine dense vector search with BM25 keyword search (sparse + dense) and re-rank results with a cross-encoder to improve precision on exact terminology (e.g. specific IFRS paragraph numbers or GHG values).
- **Multi-provider LLM support** — abstract the LLM behind a configurable interface so the system can switch between OpenAI, Azure OpenAI, or Anthropic models via a single settings change.
- **Streaming responses** — pipe the synthesis node's token stream directly to the Streamlit UI instead of waiting for the full answer, reducing perceived latency.
