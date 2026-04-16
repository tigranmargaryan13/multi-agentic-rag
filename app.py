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
| 📋 | **IFRS** | IFRS 18 — Financial Statements |
| 🏢 | **Competitor** | Pfizer Annual Report |
| 🌐 | **Web** | DuckDuckGo *(on request)* |
""")

    st.divider()

    st.markdown("### 🔀 Agent Architecture")
    try:
        img = main_graph.get_graph().draw_mermaid_png()
        st.image(img, use_container_width=True)
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
    if col1.button("🔄 New chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.pending_web  = None
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Thread `{st.session_state.thread_id[:8]}…`")

    with st.expander("💡 Sample queries"):
        st.markdown("""
- What are the key changes in IFRS 18?
- How do I calculate Scope 2 GHG emissions?
- How does Pfizer report ESG in its annual report?
- What ESG disclosures are required under IFRS?
- Tell me about standards. *(triggers clarification)*
- What's Pfizer's current stock price? *(triggers web fallback)*
""")


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
        if c1.button("✅ Yes, search web", type="primary", use_container_width=True):
            with st.chat_message("assistant"):
                with st.spinner("Searching the internet…"):
                    from graph import GraphState
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

        if c2.button("❌ Skip", use_container_width=True):
            reply = ("Understood. Please try rephrasing or ask about "
                     "ESG, IFRS standards, or competitor reports.")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.pending_web = None
            st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about ESG, IFRS, or competitor annual reports…"):
    from graph import GraphState

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
        # Live progress via LangGraph streaming
        sources_found: list[str] = []
        accumulated_agent_results: dict = {}
        final_answer = ""
        needs_clarification = False
        clarification_question = None
        web_permission_required = False

        t0 = time.time()

        with st.status("Processing…", expanded=True) as status:
            for update in main_graph.stream(initial, config=config, stream_mode="updates"):
                node = next(iter(update))
                data = update[node] or {}

                if node == "classify":
                    sources_found          = data.get("sources", [])
                    needs_clarification    = data.get("needs_clarification", False)
                    clarification_question = data.get("clarification_question")

                    if needs_clarification:
                        st.write("❓ Query is ambiguous — asking for clarification")
                    elif web_permission_required:
                        st.write("🌐 No matching internal sources — web search available")
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
                        accumulated_agent_results.update(agent_res)
                        st.write(
                            f"✅ {SOURCE_ICONS.get(src,'')} "
                            f"**{SOURCE_LABELS.get(src, src.upper())}** agent — complete"
                        )
                    # else: agent skipped (source not selected)

                elif node == "synthesize":
                    if data:
                        final_answer            = data.get("final_answer", "")
                        web_permission_required = data.get("web_permission_required", False)
                        if final_answer:
                            st.write("📝 Synthesising final answer…")

            elapsed = time.time() - t0
            status.update(
                label=f"✓ Done in {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

        # ── Render result ─────────────────────────────────────────────────────
        if needs_clarification and clarification_question:
            st.info(f"**Clarification needed**\n\n{clarification_question}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**Clarification needed**\n\n{clarification_question}",
            })
            # Store the clarification exchange in history so the follow-up
            # answer ("yes, his business activities") has enough context to
            # resolve pronouns and topic references correctly.
            st.session_state.chat_history = [
                *st.session_state.chat_history,
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": clarification_question},
            ]

        elif web_permission_required:
            st.session_state.pending_web = prompt
            st.rerun()

        elif final_answer:
            st.markdown(final_answer)
            if sources_found:
                st.markdown(source_badges(sources_found), unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": sources_found,
            })
            # Persist conversation history for the next turn
            st.session_state.chat_history = [
                *st.session_state.chat_history,
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": final_answer},
            ]

        else:
            fallback = "I could not find relevant information for your query."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
