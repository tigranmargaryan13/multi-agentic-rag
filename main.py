"""
Interactive CLI for the Multi-Agentic RAG System.

Usage:
    python main.py [--ingest]

Flags:
    --ingest   Re-run document ingestion before starting the chat loop.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevent FAISS/torch OMP conflict on macOS

import sys
import uuid

from graph import GraphState, build_graph

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "─" * 72


def print_separator():
    print(f"\n{SEPARATOR}\n")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Goodbye]")
        sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run_chat():
    main_graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "═" * 72)
    print("  Financial Knowledge Assistant  (Multi-Agentic RAG)")
    print("  Sources: ESG (GHG Protocol) | IFRS 18 | Competitor Reports")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 72 + "\n")

    while True:
        query = ask("You: ")
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print_separator()

        # ── Step 1: run the main graph ────────────────────────────────────────
        initial_state: GraphState = {
            "query": query,
            "sources": [],
            "agent_results": {},
            "final_answer": "",
            "needs_clarification": False,
            "clarification_question": None,
            "web_permission_required": False,
            "web_approved": False,
        }

        result = main_graph.invoke(initial_state, config=config)

        # ── Step 2: handle clarification ──────────────────────────────────────
        if result.get("needs_clarification"):
            cq = result.get("clarification_question", "Could you clarify your question?")
            print(f"Assistant: {cq}")
            clarification = ask("\nYou (clarification): ")
            if not clarification:
                continue
            # Re-invoke with the combined query
            query = f"{query} — {clarification}"
            result = main_graph.invoke(
                {**initial_state, "query": query},
                config=config,
            )

        # ── Step 3: handle web fallback ───────────────────────────────────────
        if result.get("web_permission_required"):
            print(
                "Assistant: I couldn't find an answer in the internal documents. "
                "Do you want me to search the internet for this? (yes/no)"
            )
            approval = ask("\nYou: ").lower()
            if approval in {"yes", "y"}:
                result = main_graph.invoke(
                    {**initial_state, "query": query, "web_approved": True},
                    config=config,
                )
            else:
                print("\nAssistant: Understood. Please try rephrasing your question.")
                print_separator()
                continue

        # ── Step 4: present the answer ────────────────────────────────────────
        final = result.get("final_answer", "")
        if final:
            print(f"Assistant:\n\n{final}")
        else:
            print("Assistant: I was unable to generate an answer.")

        # ── Step 5: feedback loop ─────────────────────────────────────────────
        print_separator()
        print(
            "Was this answer helpful? Would you like me to explore other data sources?\n"
            "(Press Enter to continue, or type a follow-up question / source name)"
        )
        feedback = ask("\nYou: ")
        if feedback and feedback.lower() not in {"no", "n", "quit", "exit"}:
            # Treat feedback as a follow-up question
            query = feedback
            follow_up_result = main_graph.invoke(
                {**initial_state, "query": query},
                config=config,
            )
            print_separator()
            follow_answer = follow_up_result.get("final_answer", "")
            if follow_answer:
                print(f"Assistant:\n\n{follow_answer}")

        print_separator()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--ingest" in sys.argv:
        from ingest import ingest
        print("Running document ingestion …\n")
        ingest()
        print()
    run_chat()
