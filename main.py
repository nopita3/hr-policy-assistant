"""
main.py
-------
Command-line runner for the HR support assistant.

Unlike app.py (the Gradio UI), this streams the LangGraph workflow and prints
the result of each node as it executes, so you can see how a query flows through
retrieve -> generate / no_context.

Usage:
    python main.py "What are the standard working hours?"   # single query
    python main.py                                          # interactive prompt
"""

import sys

from services.graph import hr_graph


def run(query: str) -> None:
    """Stream the graph for one query, printing each node's output."""
    print(f"\n=== Query: {query} ===\n")

    for step in hr_graph.stream({"query": query.strip()}):
        for node, update in step.items():
            print(f"--- Node: {node} ---")
            if "has_context" in update:
                print(f"has_context : {update['has_context']}")
            if update.get("documents") is not None:
                sections = [d.metadata.get("section", "General") for d in update["documents"]]
                print(f"retrieved   : {len(update['documents'])} chunk(s) -> {sections}")
            if update.get("context"):
                print(f"context:\n{update['context']}")
            if update.get("answer"):
                print(f"answer:\n{update['answer']}")
            print()


def main() -> None:
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return

    print("HR Support Assistant (CLI). Type a question, or 'quit' to exit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if query:
            run(query)


if __name__ == "__main__":
    main()
