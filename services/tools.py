"""
services/tools.py
-----------------
The RAG toolbox used by the two agents.

Data Retriever side:
    - load_and_chunk(): read knowledge_base.txt and split it into
      section-aware chunks (each chunk keeps its section name as metadata).
    - build_vectorstore(): embed the chunks into LangChain's in-memory vector
      store (InMemoryVectorStore) using Gemini embeddings.
    - retrieve(): the actual retrieval "tool" - given a query, return the most
      relevant chunks together with a flag for whether anything relevant was found.

Report Generator side:
    - generate_answer(): inject the retrieved context into a prompt and call the
      Gemini chat model to produce the final, grounded answer.
"""

import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Path to the knowledge base, resolved relative to the project root so it works
# no matter where the app is launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(_PROJECT_ROOT, "documents", "knowledge_base.txt")


# ---------------------------------------------------------------------------
# 1. Load + chunk the knowledge base (section-aware)
# ---------------------------------------------------------------------------
def load_and_chunk(path: str = KNOWLEDGE_BASE_PATH) -> List[Document]:
    """Split knowledge_base.txt on '## ' headers, keeping the section name as
    metadata. Long sections are further split so no chunk is too large."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split into (header, body) blocks on lines that start with "## ".
    blocks = re.split(r"^##\s+", raw, flags=re.MULTILINE)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    documents: List[Document] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        section = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        text = f"{section}\n{body}".strip()

        for piece in splitter.split_text(text):
            documents.append(Document(page_content=piece, metadata={"section": section}))
    return documents


# ---------------------------------------------------------------------------
# 2. Build the in-memory vector store (LangChain InMemoryVectorStore)
# ---------------------------------------------------------------------------
def build_vectorstore(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
) -> InMemoryVectorStore:
    """Embed the chunks into LangChain's in-memory vector store. It lives in
    process memory only (rebuilt on each start). `embeddings` can be injected
    for testing."""
    embeddings = embeddings or config.get_embeddings()
    return InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> InMemoryVectorStore:
    """Build the vector store once and cache it for the process lifetime."""
    return build_vectorstore(load_and_chunk())


# ---------------------------------------------------------------------------
# 3. The retrieval tool (Data Retriever agent)
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = config.TOP_K) -> Tuple[List[Document], bool]:
    """Return the top-k relevant chunks and a flag for whether any cleared the
    similarity threshold. The Data Retriever does NOT answer - it only fetches."""
    vectorstore = get_vectorstore()
    # similarity_search_with_score returns (Document, cosine_similarity);
    # higher = closer, results already ranked best-first.
    scored = vectorstore.similarity_search_with_score(query, k=k)

    if not scored:
        return [], False

    best_similarity = scored[0][1]
    docs = [doc for doc, _ in scored]
    has_context = best_similarity >= config.MIN_SIMILARITY
    return docs, has_context


def format_context(documents: List[Document]) -> str:
    """Format retrieved chunks into a labeled context block for the LLM."""
    parts = []
    for doc in documents:
        section = doc.metadata.get("section", "General")
        parts.append(f"[{section}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 4. The generation tool (Report Generator agent)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an HR support assistant for company employees.
Answer the employee's question using ONLY the HR policy context provided below.

Rules:
- Base every statement strictly on the context. Do not invent policies, numbers, or approvals.
- If the context does not contain the answer, say you don't have that information in the HR knowledge base and suggest contacting HR directly.
- Write a clear, well-structured answer in English. Do not simply repeat the raw snippets.
- Cite the policy section(s) you used in square brackets, e.g. [Sick (Medical) Leave].
- Be concise and non-redundant.

HR policy context:
{context}
"""


def generate_answer(query: str, context: str) -> str:
    """Call Gemini with the retrieved context injected to produce the final answer."""
    llm = config.get_llm()
    messages = [
        ("system", _SYSTEM_PROMPT.format(context=context)),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return _extract_text(response.content)


def _extract_text(content) -> str:
    """Normalize AIMessage.content to plain text.

    Some Gemini models return a list of content blocks (e.g. text blocks plus
    thought-signature metadata) instead of a plain string; join just the text.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()
