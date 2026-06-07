"""LangChain tools exposed to the tool-calling worker agents.

These are the *actions* a genuine ReAct agent can choose to invoke. Unlike the
deterministic worker nodes (which call one function each), an agent given these
tools decides which to call, with what arguments, and how many times.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.tools.news import search_news


@tool
def search_financial_news(query: str) -> str:
    """Cari berita keuangan/pasar Indonesia terbaru.

    Gunakan query yang fokus, mis. nama perusahaan atau kode saham BEI plus topik
    ("BBRI dividen", "Telkom laba kuartal"). Kembalikan headline + ringkasan.

    Args:
        query: kata kunci pencarian berita (Bahasa Indonesia lebih baik).
    """
    items = search_news(query, max_results=6)
    if not items:
        return (
            "Tidak ada berita ditemukan untuk query ini. "
            "Coba query lain yang lebih umum (mis. hanya kode saham)."
        )
    lines = [f"- {it.title}: {it.snippet[:200]}" for it in items]
    return "\n".join(lines)
