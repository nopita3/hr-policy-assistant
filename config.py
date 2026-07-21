"""
config.py
---------
Central place to load and manage the Gemini models used across the app.

- Loads GOOGLE_API_KEY from a local .env file (via python-dotenv) when running
  locally, and falls back to real environment variables (e.g. Hugging Face
  Spaces Secrets) when deployed. No code change is needed between the two.
- Exposes small factory functions so the rest of the code never hard-codes a
  model name or reaches for the API key directly.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load variables from .env if present (no-op on Hugging Face, which injects
# secrets as real environment variables).
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Locally: put it in a .env file as GOOGLE_API_KEY=your_gemini_key. "
        "On Hugging Face Spaces: add it under Settings > Secrets."
    )

# Make sure the underlying google-genai SDK can also see the key.
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Model names -----------------------------------------------------------
# Generation: gemini-2.5-flash is the stable, free-tier Flash model.
# (Swap to "gemini-flash-latest" for the newest Flash if you prefer.)
CHAT_MODEL = "gemini-3.1-flash-lite"

# Embeddings: gemini-embedding-001 is GA, free-tier, and multilingual.
EMBEDDING_MODEL = "gemini-embedding-001"

# Retrieval settings (imported by the tools module).
TOP_K = 4  # how many chunks to pull per query
# InMemoryVectorStore ranks by cosine SIMILARITY (1 = identical, 0 = unrelated).
# If the closest chunk scores below this, the query is treated as "no info found"
# and routed to the fallback node. Kept lenient so real HR questions always pass;
# tune it after seeing real similarities in the console. The LLM prompt is the
# primary anti-hallucination guard - this threshold is a coarse safety net.
MIN_SIMILARITY = 0.65


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return the chat model used by the Report Generator."""
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=temperature)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model used to index and search the knowledge base."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
