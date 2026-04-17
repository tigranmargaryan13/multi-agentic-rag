"""
Streamlit UI for the Multi-Agentic Financial RAG System.
Run with:  streamlit run app.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevent FAISS/torch OMP conflict

import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")   # suppress transformers FutureWarning

import time
import uuid

import streamlit as st
from graph import GraphState

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Financial Knowledge Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Slightly tighter chat bubbles */
.stChatMessage { padding: 0.6rem 0.8rem; }
/* Source badge row */
.badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 999px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: .03em;
}
.badge-esg        { background:#d1fae5; color:#065f46; }
.badge-ifrs       { background:#dbeafe; color:#1e40af; }
.badge-competitor { background:#fef3c7; color:#92400e; }
.badge-web        { background:#ede9fe; color:#5b21b6; }
</style>
""", unsafe_allow_html=True)

SOURCE_ICONS  = {"esg": "📊", "ifrs": "📋", "competitor": "🏢", "web": "🌐"}
SOURCE_LABELS = {"esg": "ESG", "ifrs": "IFRS", "competitor": "Competitor", "web": "Web"}
BADGE_CLASS   = {"esg": "badge-esg", "ifrs": "badge-ifrs",
                 "competitor": "badge-competitor", "web": "badge-web"}


def source_badges(sources: list[str]) -> str:
    parts = [
        f'<span class="badge {BADGE_CLASS.get(s, "")}">'
        f'{SOURCE_ICONS.get(s, "•")} {SOURCE_LABELS.get(s, s.upper())}</span>'
        for s in sources
    ]
    return f'<div class="badge-row">{"".join(parts)}</div>'


# ── Cached resource: load graphs once ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_graph():
    from graph import build_graph
    return build_graph()


main_graph = load_graph()

# ── Session state defaults ────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "thread_id"     not in st.session_state: st.session_state.thread_id     = str(uuid.uuid4())
if "pending_web"   not in st.session_state: st.session_state.pending_web   = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 Financial Knowledge Assistant")
    st.caption("Multi-Agentic · Multimodal · RAG")
    st.divider()

    st.markdown("### 📚 Data Sources")
    st.markdown("""
| | Source | Content |
|---|---|---|
| 📊 | **ESG** | GHG Protocol Standards |
| 📋 | **IFRS** | IFRS — Financial Statements |
| 🏢 | **Competitor** | Pfizer Annual Report |
| 🌐 | **Web** | DuckDuckGo *(on request)* |
""")

    st.divider()

    st.markdown("### 🔀 Agent Architecture")
    try:
        img = main_graph.get_graph().draw_mermaid_png()
        st.image(img, width="stretch")
    except Exception:
        st.code(
            "classify\n"
            "  ├─► ESG Agent ──┐\n"
            "  ├─► IFRS Agent ─┼─► synthesize\n"
            "  └─► Competitor ─┘",
            language=None,
        )

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🔄 New chat", width="stretch"):
        st.session_state.messages     = []
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.pending_web  = None
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Thread `{st.session_state.thread_id[:8]}…`")

    with st.expander("💡 Sample queries"):
        SAMPLE_QUERIES = [
            ("How should a company calculate Scope 3 emissions from purchased goods and services?", ["esg"]),
            ("How does Pfizer report Scope 1 and 2 emissions and does it align with GHG Protocol?",  ["esg", "competitor"]),
            ("How should ESG liabilities be presented under IFRS, and what does Pfizer actually disclose?", ["ifrs", "competitor"]),
            ("Tell me the current price of AAPL stock?", ["web"]),
            ("What are the disclosure requirements for operating segments under IFRS 8 Operating Segments?",           ["ifrs"]),
            ("Tell me about the standards...", ["clarification"])
        ]

        SOURCE_BADGE_MINI = {
            "esg":           "🟢 ESG",
            "ifrs":          "🔵 IFRS",
            "competitor":    "🟡 Competitor",
            "web":           "🟣 Web",
            "clarification": "❓ Clarify",
        }

        for question, sources in SAMPLE_QUERIES:
            badge_str = " · ".join(SOURCE_BADGE_MINI[s] for s in sources)
            if st.button(f"{question}", key=f"sample_{question[:30]}", width="stretch"):
                st.session_state["prefill_query"] = question
                st.rerun()
            st.caption(badge_str)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask your financial question")
st.caption("Queries are routed to specialist agents for ESG, IFRS, and competitor intelligence.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(source_badges(msg["sources"]), unsafe_allow_html=True)

# ── Web permission dialog ─────────────────────────────────────────────────────
if st.session_state.pending_web:
    query = st.session_state.pending_web
    with st.container(border=True):
        st.markdown("### 🌐 No internal sources found")
        st.markdown(
            "I couldn't find an answer in the internal document library for:\n\n"
            f"> *{query}*"
        )
        st.markdown("Would you like me to search the internet?")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("✅ Yes, search web", type="primary", width="stretch"):
            with st.chat_message("assistant"):
                with st.spinner("Searching the internet…"):
                    config  = {"configurable": {"thread_id": st.session_state.thread_id}}
                    initial: GraphState = {
                        "query": query, "sources": [], "agent_results": {},
                        "final_answer": "", "needs_clarification": False,
                        "clarification_question": None, "web_permission_required": False,
                        "web_approved": True,
                        "chat_history": st.session_state.chat_history,
                    }
                    result = main_graph.invoke(initial, config=config)
                answer = result.get("final_answer", "No result found.")
                st.markdown(answer)
                st.markdown(source_badges(["web"]), unsafe_allow_html=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": ["web"]}
            )
            st.session_state.chat_history = result.get("chat_history", st.session_state.chat_history)
            st.session_state.pending_web = None
            st.rerun()

        if c2.button("❌ Skip", width="stretch"):
            reply = ("Understood. Please try rephrasing or ask about "
                     "ESG, IFRS standards, or competitor reports.")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.pending_web = None
            st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
_prefill = st.session_state.pop("prefill_query", None)
if prompt := (st.chat_input("Ask about ESG, IFRS, or competitor annual reports…") or _prefill):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config: dict = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial: GraphState = {
        "query": prompt, "sources": [], "agent_results": {}, "final_answer": "",
        "needs_clarification": False, "clarification_question": None,
        "web_permission_required": False, "web_approved": False,
        "chat_history": st.session_state.chat_history,
    }

    with st.chat_message("assistant"):
        progress_ph = st.empty()   # status / progress widget
        answer_ph   = st.empty()   # streaming answer (rendered below progress)

        sources_found: list[str] = []
        final_answer = ""
        streamed_text = ""
        needs_clarification = False
        clarification_question = None
        web_permission_required = False

        t0 = time.time()

        with progress_ph.status("Processing…", expanded=True) as status:
            for mode, payload in main_graph.stream(
                initial, config=config, stream_mode=["updates", "messages"]
            ):
                # ── Progress updates ──────────────────────────────────────────
                if mode == "updates":
                    node = next(iter(payload))
                    data = payload[node] or {}

                    if node == "classify":
                        sources_found          = data.get("sources", [])
                        needs_clarification    = data.get("needs_clarification", False)
                        clarification_question = data.get("clarification_question")

                        if needs_clarification:
                            st.write("❓ Query is ambiguous — asking for clarification")
                        elif sources_found:
                            labels = " · ".join(
                                f"{SOURCE_ICONS.get(s,'')} **{SOURCE_LABELS.get(s,s.upper())}**"
                                for s in sources_found
                            )
                            st.write(f"🔍 Sources detected: {labels}")
                            st.write("🤖 Agents running in parallel…")
                        else:
                            st.write("⚠️ No matching sources found")

                    elif node in ("esg_agent", "ifrs_agent", "competitor_agent"):
                        src = node.replace("_agent", "")
                        agent_res = data.get("agent_results", {}) if data else {}
                        if agent_res:
                            st.write(
                                f"✅ {SOURCE_ICONS.get(src,'')} "
                                f"**{SOURCE_LABELS.get(src, src.upper())}** agent — complete"
                            )

                    elif node == "synthesize" and data:
                        final_answer            = data.get("final_answer", "")
                        web_permission_required = data.get("web_permission_required", False)

                # ── Token streaming from synthesize ───────────────────────────
                elif mode == "messages":
                    chunk, metadata = payload
                    if (
                        metadata.get("langgraph_node") == "synthesize"
                        and hasattr(chunk, "content")
                        and chunk.content
                    ):
                        streamed_text += chunk.content
                        answer_ph.markdown(streamed_text + "▌")   # ▌ = live cursor

            elapsed = time.time() - t0
            status.update(
                label=f"✓ Done in {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

        # ── Finalise answer placeholder (remove cursor) ───────────────────────
        display = streamed_text or final_answer
        if display:
            answer_ph.markdown(display)

        # ── Render result ─────────────────────────────────────────────────────
        if needs_clarification and clarification_question:
            answer_ph.info(f"**Clarification needed**\n\n{clarification_question}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**Clarification needed**\n\n{clarification_question}",
            })
            st.session_state.chat_history = [
                *st.session_state.chat_history,
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": clarification_question},
            ]

        elif web_permission_required:
            answer_ph.empty()
            st.session_state.pending_web = prompt
            st.rerun()

        elif display:
            if sources_found:
                st.markdown(source_badges(sources_found), unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": display,
                "sources": sources_found,
            })
            st.session_state.chat_history = [
                *st.session_state.chat_history,
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": display},
            ]

        else:
            fallback = "I could not find relevant information for your query."
            answer_ph.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
