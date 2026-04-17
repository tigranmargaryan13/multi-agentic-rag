"""
LangGraph orchestration for the Multi-Agentic RAG system.

Workflow
────────
                        ┌─► esg_agent ──────────┐
START → classify ───────┼─► ifrs_agent ──────────┼─► synthesize → END
                        └─► competitor_agent ────┘

Steps
─────
1. classify   – detects relevant sources; flags ambiguous queries.
2. RAG agents – run in PARALLEL; each skips if its source was not selected.
3. synthesize    – if agent_results is not empty  → generate final answer.
                 – if agent_results is empty
                     + web_approved=False → set web_permission_required (ask user)
                     + web_approved=True  → run DuckDuckGo then synthesise.
                 – if needs_clarification → pass through (caller shows question).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevent FAISS/torch OMP conflict

from pathlib import Path
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import AgentFactory
from prompts import SOURCE_LABELS, build_synthesizer_prompt
from settings import BaseLLMSettings


# ──────────────────────────────────────────────────────────────────────────────
# State  (module-level — it's a type used by both inner and outer graphs)
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


class RAGGraph:
    """
    Encapsulates all node logic, subgraph construction, and outer graph wiring.

    Usage
    ─────
        compiled = RAGGraph().build()
        result   = compiled.invoke({...})
    """

    def __init__(self) -> None:
        self._settings = BaseLLMSettings()
        self._factory  = AgentFactory()

    # ── Agent accessors ───────────────────────────────────────────────────────

    def _index_exists(self, source: str) -> bool:
        return (Path(self._settings.FAISS_INDEX_DIR) / source / "index.faiss").exists()

    def _get_classifier(self):
        return self._factory.get_classifier()

    def _get_rag_agent(self, source: str):
        return self._factory.get_rag_agent(source)

    def _get_web_agent(self):
        return self._factory.get_web_agent()

    # ── Node helpers ──────────────────────────────────────────────────────────

    def _invoke_rag(self, source: str, query: str) -> str:
        result = self._get_rag_agent(source).invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        return result["messages"][-1].content

    def _make_rag_node(self, source: str):
        """Return a node function for *source*. Skips when not selected or clarifying."""
        import threading, time
        def node(state: GraphState) -> dict:
            if state.get("needs_clarification"):
                return {}
            if source not in state.get("sources", []):
                return {}
            t0 = time.time()
            print(f"[{source}] START  t={t0:.3f}  thread={threading.current_thread().name}")
            result = self._invoke_rag(source, state["query"])
            t1 = time.time()
            print(f"[{source}] END    t={t1:.3f}  duration={t1-t0:.2f}s  thread={threading.current_thread().name}")
            return {"agent_results": {source: result}}
        node.__name__ = f"{source}_agent"
        return node

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def classify_node(self, state: GraphState) -> dict:
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

        result    = self._get_classifier()(query_for_classification)
        available = [s for s in result.sources if self._index_exists(s)]
        if missing := set(result.sources) - set(available):
            print(f"  [skip {missing} — not ingested yet]")
        return {
            "sources":                available,
            "needs_clarification":    result.is_ambiguous,
            "clarification_question": result.clarification_question,
            "agent_results":          {},
        }

    def synthesize_node(self, state: GraphState) -> dict:
        """
        Combines agent results into a final answer.
        Special cases:
          • Clarification needed → pass through (caller shows the question).
          • No RAG results       → trigger web fallback (ask permission or run search).
        """
        # ── 1. Clarification pass-through ─────────────────────────────────────
        if state.get("needs_clarification"):
            return {}

        results = state.get("agent_results", {})

        # ── 2. No RAG content — web fallback ──────────────────────────────────
        if not results:
            if not state.get("web_approved"):
                return {"web_permission_required": True}
            web_result = self._get_web_agent().invoke(
                {"messages": [HumanMessage(content=state["query"])]}
            )
            results = {"web": web_result["messages"][-1].content}

        # ── 3. Synthesise ──────────────────────────────────────────────────────
        sections = "\n\n".join(
            f"### {SOURCE_LABELS.get(k, k.upper())}\n{v}" for k, v in results.items()
        )

        history      = state.get("chat_history", [])
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

        prompt = build_synthesizer_prompt(history_block, state["query"], sections)
        answer = self._factory.llm.invoke(prompt).content

        updated_history = list(history) + [
            {"role": "user",      "content": state["query"]},
            {"role": "assistant", "content": answer},
        ]
        return {"final_answer": answer, "chat_history": updated_history}


    def build(self):
        """
        Compile and return the graph.

        Topology
        ────────
                             ┌─► esg_agent ──────────┐
        START → classify ────┼─► ifrs_agent ──────────┼─► synthesize → END
                             └─► competitor_agent ────┘
        """
        g = StateGraph(GraphState)

        g.add_node("classify",         self.classify_node)
        g.add_node("esg_agent",        self._make_rag_node("esg"))
        g.add_node("ifrs_agent",       self._make_rag_node("ifrs"))
        g.add_node("competitor_agent", self._make_rag_node("competitor"))
        g.add_node("synthesize",       self.synthesize_node)

        g.add_edge(START,              "classify")

        # Fan-out: classify → 3 agents in parallel
        g.add_edge("classify",         "esg_agent")
        g.add_edge("classify",         "ifrs_agent")
        g.add_edge("classify",         "competitor_agent")

        # Fan-in: all 3 agents → synthesize
        g.add_edge("esg_agent",        "synthesize")
        g.add_edge("ifrs_agent",       "synthesize")
        g.add_edge("competitor_agent", "synthesize")

        g.add_edge("synthesize",       END)

        return g.compile()


def build_graph():
    return RAGGraph().build()
