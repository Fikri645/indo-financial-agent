"""Central configuration for the Indonesian Financial Research Agent.

All tunables live here so the rest of the codebase stays declarative. Secrets are
read from the environment (.env); see .env.example.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PDF_DIR, VECTORSTORE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# LLM provider (free-tier friendly). Mirrors the Philosopher Chat pattern of
# routing across Groq / Google AI Studio / OpenRouter via a single switch.
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --------------------------------------------------------------------------- #
# Embeddings & retrieval
# --------------------------------------------------------------------------- #
# Multilingual model covers Bahasa Indonesia financial text well. Swap for the
# Netmonk-fine-tuned IBM Granite embedding if you want to showcase that.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 6          # final chunks fed to the LLM
RETRIEVAL_CANDIDATES = 20    # candidate pool before reranking/fusion

# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
DEFAULT_EXCHANGE_SUFFIX = ".JK"   # Yahoo Finance suffix for IDX-listed equities
FINANCIALS_YEARS = 4              # how many annual periods to pull


def normalize_ticker(ticker: str) -> str:
    """Ensure an IDX ticker carries the Yahoo Finance ``.JK`` suffix.

    >>> normalize_ticker("bbri")
    'BBRI.JK'
    >>> normalize_ticker("TLKM.JK")
    'TLKM.JK'
    """
    t = ticker.strip().upper()
    if "." not in t:
        t = f"{t}{DEFAULT_EXCHANGE_SUFFIX}"
    return t
