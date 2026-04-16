"""
Agent definitions for the Multi-Agentic RAG system.

Agents:
  - SourceClassificationAgent  – detects which data sources to query
  - ESGAgent / IFRSAgent / CompetitorAgent – ReAct RAG agents per source
  - WebSearchAgent  – DuckDuckGo fallback (no API key required)
"""

from typing import List, Literal, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from settings import BaseLLMSettings

settings = BaseLLMSettings()

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Source Classification Agent
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_DESCRIPTIONS = {
    "esg": (
        "ESG / GHG Protocol documents: greenhouse gas accounting, scope 1/2/3 emissions, "
        "carbon footprint, GHG inventory, climate reporting, stationary combustion, "
        "corporate sustainability, environmental standards."
    ),
    "ifrs": (
        "IFRS (International Financial Reporting Standards): financial statement presentation, "
        "disclosure requirements, accounting standards, lease accounting, revenue recognition, "
        "balance sheet, income statement."
    ),
    "competitor": (
        "Competitor annual reports and public disclosures: market intelligence, competitor "
        "financial performance, business strategy, product pipeline, regulatory filings."
    ),
}


class SourceClassification(BaseModel):
    sources: List[Literal["esg", "ifrs", "competitor"]] = Field(
        description="Relevant data sources. Empty list means no internal source applies."
    )
    is_ambiguous: bool = Field(
        description="True if the query is too vague to answer without clarification."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask the user when the query is ambiguous.",
    )


_CLASSIFIER_PROMPT = """You are a query router for a financial firm's knowledge base.

Available data sources:
- esg: {esg}
- ifrs: {ifrs}
- competitor: {competitor}

Analyse the user query and return:
1. sources – a list of relevant source keys (can be multiple, or empty if none apply).
2. is_ambiguous – True if the query needs clarification before answering.
3. clarification_question – a short question to resolve the ambiguity (or null).

User query: {query}"""


def create_source_classifier():
    """Returns a callable: query (str) → SourceClassification."""
    structured_llm = _llm().with_structured_output(SourceClassification)

    def classify(query: str) -> SourceClassification:
        prompt = _CLASSIFIER_PROMPT.format(
            esg=SOURCE_DESCRIPTIONS["esg"],
            ifrs=SOURCE_DESCRIPTIONS["ifrs"],
            competitor=SOURCE_DESCRIPTIONS["competitor"],
            query=query,
        )
        return structured_llm.invoke(prompt)

    return classify


# ──────────────────────────────────────────────────────────────────────────────
# ReAct RAG Agents (one per source collection)
# ──────────────────────────────────────────────────────────────────────────────

_MULTIMODAL_NOTE = (
    "Documents were ingested with multimodal extraction:\n"
    "  • [TABLE] … [/TABLE]  blocks contain structured table data.\n"
    "  • [Visual Content — p.N] blocks contain GPT-4 Vision descriptions of\n"
    "    charts, diagrams, flowcharts, and figures found on that page.\n"
    "When the retrieved text contains these blocks, incorporate their content\n"
    "into your answer just as you would plain text.\n"
)

_AGENT_SYSTEM_PROMPTS = {
    "esg": (
        "You are an ESG expert specialising in GHG Protocol standards and climate accounting.\n"
        + _MULTIMODAL_NOTE
        + "Use the search_documents tool to retrieve relevant passages from the ESG document library.\n"
        "Cite the document filename and page number for every claim you make.\n"
        "Reproduce tables or visual data when they support your answer.\n"
        "End your answer with one specific follow-up question the user might want to explore next."
    ),
    "ifrs": (
        "You are an IFRS accounting standards expert.\n"
        + _MULTIMODAL_NOTE
        + "Use the search_documents tool to retrieve relevant passages from the IFRS document library.\n"
        "Cite the standard name and paragraph number for every claim you make.\n"
        "Reproduce tables or visual data when they support your answer.\n"
        "End your answer with one specific follow-up question the user might want to explore next."
    ),
    "competitor": (
        "You are a financial analyst specialising in competitive intelligence.\n"
        + _MULTIMODAL_NOTE
        + "Use the search_documents tool to retrieve relevant passages from competitor annual reports.\n"
        "Cite the company name, report year, and section for every claim you make.\n"
        "Reproduce financial tables or visual data when they support your answer.\n"
        "End your answer with one specific follow-up question the user might want to explore next."
    ),
}


def create_rag_agent(source_name: str, persist_dir: str | None = None):
    """Create a ReAct agent with a retriever tool backed by a FAISS index."""
    from pathlib import Path
    persist_dir = persist_dir or settings.FAISS_INDEX_DIR
    index_path = str(Path(persist_dir) / source_name)

    vectorstore = FAISS.load_local(
        index_path,
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVE_DOCS_NUMBER}
    )

    @tool
    def search_documents(query: str) -> str:
        """Retrieve relevant passages from the document collection."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passages found in this document collection."
        parts = []
        for i, doc in enumerate(docs, 1):
            fname = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source {i} — {fname}, p.{page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    return create_react_agent(
        _llm(),
        tools=[search_documents],
        prompt=_AGENT_SYSTEM_PROMPTS[source_name],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Web Search Agent (DuckDuckGo – no API key required)
# ──────────────────────────────────────────────────────────────────────────────

def create_web_agent():
    """Create a ReAct agent that searches the internet via DuckDuckGo."""
    ddg = DuckDuckGoSearchRun()

    @tool
    def web_search(query: str) -> str:
        """Search the internet for up-to-date information."""
        return ddg.run(query)

    return create_react_agent(
        _llm(),
        tools=[web_search],
        prompt=(
            "You are a research assistant with internet access.\n"
            "Use the web_search tool to find accurate, up-to-date information.\n"
            "Always cite the source URLs you used.\n"
            "End your answer with one specific follow-up question."
        ),
    )
