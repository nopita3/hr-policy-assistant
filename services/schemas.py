"""
services/schemas.py
-------------------
Defines the shape of the data that flows through the LangGraph workflow.

`GraphState` is the shared state object. Each node reads some keys and returns
a partial update; LangGraph merges those updates back into the state.
"""

from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    query: str                 # the user's question (input)
    documents: List[Document]  # chunks returned by the Data Retriever
    context: str               # retrieved chunks formatted as text for the LLM
    has_context: bool          # True if retrieval found relevant material
    answer: str                # the final answer from the Report Generator
