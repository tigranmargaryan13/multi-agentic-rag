# Multi-Agentic Multimodal RAG System

An internal knowledge assistant built for audit, finance, and strategy teams. You ask a question, it figures out which documents are relevant, queries them in parallel, and hands you back one clean answer with citations. If nothing in the internal library covers it, it asks before hitting the web.

## How it works

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
            ├── IFRS Agent       (ReAct + FAISS)  ├── run in parallel   │
            └── Competitor Agent (ReAct + FAISS)  ╯                     │
                                        ▼                               │
                              Synthesis Node (LLM)  ◄───────────────────┘
                                        │
                                        ▼
                              Final Answer + Follow-up prompt
```

Each query goes through four stages:

1. **Classification** — A structured LLM call reads the query and returns which sources are relevant (`esg`, `ifrs`, `competitor`). If the question is too vague, it asks for clarification instead.

2. **Parallel RAG agents** — One ReAct agent per selected source runs concurrently. Each agent searches its own FAISS index, pulls the most relevant chunks, and writes a cited answer.

3. **Synthesis** — A final node merges all the agent answers into one coherent response, attributes facts to their source, and ends with a suggested follow-up.

4. **Web fallback** — If no internal source matched, the user is asked for permission before a DuckDuckGo search is run.

## Data sources

| Source | What's in it |
|---|---|
| **ESG** | GHG Protocol — scope 1/2/3 accounting, project accounting, value chain standards |
| **IFRS** | IFRS 18 — Presentation and Disclosure in Financial Statements |
| **Competitor** | Pfizer Annual Report — financials, strategy, sustainability disclosures |
| **Web** | DuckDuckGo real-time search (only runs with user approval) |

Documents are ingested with full multimodal extraction — text, tables (converted to markdown), and GPT-4 Vision descriptions of charts and figures. So answers can draw on visual and tabular content too, not just prose.

## Stack

- **LLM**: OpenAI `gpt-4.1-mini`
- **Embeddings**: `text-embedding-3-small`
- **Vector store**: FAISS
- **Orchestration**: LangGraph
- **Agents**: LangChain ReAct
- **UI**: Streamlit

## Setup

**Prerequisites**: Python 3.12+ and an OpenAI API key.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Drop your PDFs into the right sub-folders under `documents/`:

```
documents/
├── ESG_documents/
├── IFRS_documents/
└── annual_reports_of_comperitors/
```

Then build the indexes:

```bash
python build_index.py
```

This writes the FAISS indexes to `faiss_index/`. To re-ingest a single source, delete its folder and re-run:

```bash
rm -rf faiss_index/ifrs && python build_index.py
```

## Running

**Streamlit UI** (recommended):
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501).

**CLI**:
```bash
python main.py          # chat loop
python main.py --ingest # ingest first, then chat
```

**Docker**:
```bash
docker compose up --build
```
Needs a `.env` with `OPENAI_API_KEY`. The `faiss_index/` and `documents/` folders are mounted as volumes so you don't have to rebuild the image when you re-ingest.

## What could be better

- Replace FAISS with a managed vector DB (Pinecone, pgvector) for persistence and partial re-indexing
- Add hybrid search (BM25 + dense + re-ranker) for exact-term queries
- Integration tests for graph routing logic using a mocked LLM
