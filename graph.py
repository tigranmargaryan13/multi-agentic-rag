"""
LangGraph orchestration for the Multi-Agentic RAG system.

Workflow
────────
                ┌─► esg_agent ──────────┐
START → classify ──► ifrs_agent ──────────► synthesize → END
                └─► competitor_agent ───┘

Steps
─────
1. classify   – detects relevant sources; flags ambiguous queries.
2. RAG agents – run in PARALLEL; each skips if its source was not selected.
3. synthesize – if agent_results is not empty  → generate final answer.
              – if agent_results is empty
                  + web_approved=False → set web_permission_required (ask user)
                  + web_approved=True  → run DuckDuckGo then synthesise.
              – if needs_clarification → pass through (caller shows question).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevent FAISS/torch OMP conflict

from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import AgentFactory
from settings import BaseLLMSettings

settings = BaseLLMSettings()
_factory = AgentFactory()   # single shared instance — LLM clients and FAISS indexes cached here

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merge parallel agent results into one dict (no overwrite)."""
    return {**a, **b}


class GraphState(TypedDict):
    query: str
    sources: List[str]
    agent_results: Annotated[Dict[str, str], _merge_dicts]   # merged across parallel branches
    final_answer: str
    needs_clarification: bool
    clarification_question: Optional[str]
    web_permission_required: bool   # set by synthesize when results are empty & not approved
    web_approved: bool              # caller sets True after user grants permission
    chat_history: List[dict]        # accumulated [{role, content}, …] across turns


# ──────────────────────────────────────────────────────────────────────────────
# Agent accessors (thin wrappers — caching lives inside AgentFactory)
# ──────────────────────────────────────────────────────────────────────────────

def _index_exists(source: str) -> bool:
    from pathlib import Path
    return (Path(settings.FAISS_INDEX_DIR) / source / "index.faiss").exists()

def _get_classifier():
    return _factory.get_classifier()

def _get_rag_agent(source: str):
    return _factory.get_rag_agent(source)

def _get_web_agent():
    return _factory.get_web_agent()


# ──────────────────────────────────────────────────────────────────────────────
# Node helpers
# ──────────────────────────────────────────────────────────────────────────────

def _invoke_rag(source: str, query: str) -> str:
    agent = _get_rag_agent(source)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

def _make_rag_node(source: str):
    """RAG agent node. Skips when clarification needed or source not selected."""
    def node(state: GraphState) -> dict:
        if state.get("needs_clarification"):
            return {}
        if source not in state.get("sources", []):
            return {}
        return {"agent_results": {source: _invoke_rag(source, state["query"])}}
    node.__name__ = f"{source}_agent"
    return node


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

def classify_node(state: GraphState) -> dict:
    history = state.get("chat_history", [])

    # Give the classifier context from recent turns so follow-up questions
    # ("tell me more about that", "what about ESG?") resolve correctly.
    if history:
        recent = history[-4:]   # last 2 Q&A pairs
        context = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        )
        query_for_classification = (
            f"[Conversation so far]\n{context}\n\n"
            f"[New question]\n{state['query']}"
        )
    else:
        query_for_classification = state["query"]

    result = _get_classifier()(query_for_classification)
    available = [s for s in result.sources if _index_exists(s)]
    if missing := set(result.sources) - set(available):
        print(f"  [skip {missing} — not ingested yet]")
    return {
        "sources": available,
        "needs_clarification": result.is_ambiguous,
        "clarification_question": result.clarification_question,
        "agent_results": {},
    }


def synthesize_node(state: GraphState) -> dict:
    """
    Combines agent results into a final answer.
    Also handles two special cases:
      • Clarification needed  → pass through (caller shows the question).
      • No RAG results        → trigger web fallback (ask permission or run search).
    """
    from langchain_openai import ChatOpenAI

    # ── 1. Clarification pass-through ─────────────────────────────────────────
    if state.get("needs_clarification"):
        return {}

    results = state.get("agent_results", {})

    # ── 2. No RAG content — web fallback ──────────────────────────────────────
    if not results:
        if not state.get("web_approved"):
            # Signal the caller to ask the user for permission
            return {"web_permission_required": True}
        # Permission already granted — run web search now
        web_result = _get_web_agent().invoke(
            {"messages": [HumanMessage(content=state["query"])]}
        )
        results = {"web": web_result["messages"][-1].content}

    # ── 3. Synthesise ──────────────────────────────────────────────────────────
    labels = {
        "esg": "ESG Documents (GHG Protocol)",
        "ifrs": "IFRS Standards",
        "competitor": "Competitor Annual Reports",
        "web": "Internet Search",
    }
    sections = "\n\n".join(f"### {labels.get(k, k.upper())}\n{v}" for k, v in results.items())

    history = state.get("chat_history", [])
    history_block = ""
    if history:
        recent = history[-6:]   # last 3 Q&A pairs
        history_block = (
            "## Conversation History\n"
            + "\n\n".join(
                f"**{'User' if m['role'] == 'user' else 'Assistant'}**: {m['content'][:400]}"
                for m in recent
            )
            + "\n\n"
        )

    prompt = (
        f"{history_block}"
        "You are a senior financial analyst synthesising answers from multiple specialist agents.\n\n"
        f"User question: {state['query']}\n\n"
        f"Agent answers:\n{sections}\n\n"
        "Write a well-structured final answer that:\n"
        "1. Integrates insights from all sources cohesively.\n"
        "2. Clearly attributes facts to their source.\n"
        "3. Highlights connections or contrasts across sources.\n"
        "4. If the user refers to something from the conversation history, address it directly.\n"
        "5. Ends with one specific follow-up question.\n\n"
        "Final Answer:"
    )
    answer = _factory.llm.invoke(prompt).content

    # Append this turn to chat_history so future turns have context
    updated_history = list(history) + [
        {"role": "user",      "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]
    return {"final_answer": answer, "chat_history": updated_history}


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("classify",         classify_node)
    g.add_node("esg_agent",        _make_rag_node("esg"))
    g.add_node("ifrs_agent",       _make_rag_node("ifrs"))
    g.add_node("competitor_agent", _make_rag_node("competitor"))
    g.add_node("synthesize",       synthesize_node)

    g.add_edge(START, "classify")

    # Fan-out: 3 RAG agents in PARALLEL
    g.add_edge("classify",         "esg_agent")
    g.add_edge("classify",         "ifrs_agent")
    g.add_edge("classify",         "competitor_agent")

    # Fan-in: all three converge at synthesize
    g.add_edge("esg_agent",        "synthesize")
    g.add_edge("ifrs_agent",       "synthesize")
    g.add_edge("competitor_agent", "synthesize")

    g.add_edge("synthesize", END)

    return g.compile()
