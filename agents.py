"""
Agent definitions for the Multi-Agentic RAG system.

Classes:
  AgentFactory  – creates and caches all agents used in the LangGraph workflow.
    .get_classifier()        → source-routing callable
    .get_rag_agent(source)   → ReAct RAG agent for a document collection
    .get_web_agent()         → DuckDuckGo ReAct agent (internet fallback)
"""

from typing import List, Literal, Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from settings import BaseLLMSettings


# ──────────────────────────────────────────────────────────────────────────────
# Shared output schema (module-level — it's a type, not agent state)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Agent factory
# ──────────────────────────────────────────────────────────────────────────────

class AgentFactory:
    """Creates and caches all agents used in the RAG workflow.

    A single instance is shared across the process so LLM / embedding clients
    and loaded FAISS indexes are not duplicated.
    """

    # ── Source descriptions (used by classifier prompt) ───────────────────────
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

    # ── Classifier prompt ─────────────────────────────────────────────────────
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

    # ── RAG agent prompts ─────────────────────────────────────────────────────
    _MULTIMODAL_NOTE = (
        "Documents were ingested with multimodal extraction:\n"
        "  • [TABLE] … [/TABLE]  blocks contain structured table data.\n"
        "  • [Visual Content] blocks contain GPT-4 Vision descriptions of\n"
        "    charts, diagrams, flowcharts, and figures.\n"
        "Incorporate table and visual content into your answer as you would plain text.\n"
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

    def __init__(self) -> None:
        s = BaseLLMSettings()
        self._settings = s
        self.llm = ChatOpenAI(
            model=s.LLM_MODEL_NAME,
            temperature=s.LLM_TEMPERATURE,
            api_key=s.OPENAI_API_KEY,
        )
        self.embeddings = OpenAIEmbeddings(
            model=s.EMBEDDING_MODEL,
            api_key=s.OPENAI_API_KEY,
        )
        # Lazy-loaded caches
        self._classifier = None
        self._rag_agents: dict = {}
        self._web_agent = None

    # ── Public agent accessors (lazy-init, cached) ────────────────────────────

    def get_classifier(self):
        """Return the source-classification callable (created once)."""
        if self._classifier is None:
            structured_llm = self.llm.with_structured_output(SourceClassification)

            def classify(query: str) -> SourceClassification:
                prompt = self._CLASSIFIER_PROMPT.format(
                    esg=self.SOURCE_DESCRIPTIONS["esg"],
                    ifrs=self.SOURCE_DESCRIPTIONS["ifrs"],
                    competitor=self.SOURCE_DESCRIPTIONS["competitor"],
                    query=query,
                )
                return structured_llm.invoke(prompt)

            self._classifier = classify
        return self._classifier

    def get_rag_agent(self, source_name: str, persist_dir: str | None = None):
        """Return a ReAct RAG agent for *source_name* (created and cached once per source)."""
        if source_name not in self._rag_agents:
            from pathlib import Path
            index_path = str(
                Path(persist_dir or self._settings.FAISS_INDEX_DIR) / source_name
            )
            vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": self._settings.RETRIEVE_DOCS_NUMBER}
            )

            @tool
            def search_documents(query: str) -> str:
                """Retrieve relevant passages from the document collection."""
                docs = retriever.invoke(query)
                if not docs:
                    return "No relevant passages found in this document collection."
                parts = [
                    f"[Source {i} — {doc.metadata.get('filename', 'unknown')}, "
                    f"p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
                    for i, doc in enumerate(docs, 1)
                ]
                return "\n\n---\n\n".join(parts)

            self._rag_agents[source_name] = create_react_agent(
                self.llm,
                tools=[search_documents],
                prompt=self._AGENT_SYSTEM_PROMPTS[source_name],
            )
        return self._rag_agents[source_name]

    def get_web_agent(self):
        """Return the DuckDuckGo web-search agent (created once)."""
        if self._web_agent is None:
            ddg = DuckDuckGoSearchRun()

            @tool
            def web_search(query: str) -> str:
                """Search the internet for up-to-date information."""
                return ddg.run(query)

            self._web_agent = create_react_agent(
                self.llm,
                tools=[web_search],
                prompt=(
                    "You are a research assistant with internet access.\n"
                    "Use the web_search tool to find accurate, up-to-date information.\n"
                    "Always cite the source URLs you used.\n"
                    "End your answer with one specific follow-up question."
                ),
            )
        return self._web_agent
