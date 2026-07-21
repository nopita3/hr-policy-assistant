"""
services/graph.py
-----------------
Orchestrates the two agents as a LangGraph workflow.

Flow:
    retrieve  --(relevant context found)-->  generate  --> END
              --(nothing relevant found)-->  no_context --> END

- retrieve   : the Data Retriever agent (RAG). Fetches relevant chunks.
- generate   : the Report Generator agent. Synthesizes a grounded answer.
- no_context : guard node that avoids hallucinating when retrieval comes up empty.

The conditional edge after `retrieve` is what makes this a graph rather than a
plain sequential chain, and it keeps the assistant from inventing policies.
"""

from langgraph.graph import StateGraph, END

from services.schemas import GraphState
from services import tools


# --- Nodes -----------------------------------------------------------------
def retrieve_node(state: GraphState) -> GraphState:
    """Data Retriever agent: pull relevant snippets from the knowledge base."""
    documents, has_context = tools.retrieve(state["query"])
    return {
        "documents": documents,
        "context": tools.format_context(documents),
        "has_context": has_context,
    }


def generate_node(state: GraphState) -> GraphState:
    """Report Generator agent: write a grounded, cited answer from the context."""
    answer = tools.generate_answer(state["query"], state["context"])
    return {"answer": answer}


def no_context_node(state: GraphState) -> GraphState:
    """Fallback when nothing relevant was retrieved - do not hallucinate."""
    return {
        "answer": (
            "I couldn't find anything about that in the HR knowledge base. "
            "Please rephrase your question or contact the HR department directly."
        )
    }


def route_after_retrieve(state: GraphState) -> str:
    """Conditional router: go to the generator only if we found relevant context."""
    return "generate" if state.get("has_context") else "no_context"


# --- Build + compile the graph --------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_context", no_context_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "no_context": "no_context"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph.compile()


# Compiled once at import so the app can reuse it.
hr_graph = build_graph()


def answer_query(query: str) -> str:
    """Convenience entry point used by the Gradio app."""
    if not query or not query.strip():
        return "Please enter a question about the HR policies."
    result = hr_graph.invoke({"query": query.strip()})
    return result["answer"]
