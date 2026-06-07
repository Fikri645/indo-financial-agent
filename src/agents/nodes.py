"""Agent node implementations for the LangGraph supervisor graph.

Each node is a plain Python function that accepts AgentState and returns a
partial update dict — LangGraph merges updates into the shared state.

Node map
--------
supervisor_node   — deterministic router: decides which worker to invoke next
financial_node    — yfinance fundamentals + rule-based risk scorer
document_node     — Docling PDF parse + hybrid retrieval of analyst sections
news_node         — recent Indonesian financial news (DuckDuckGo / Tavily)
risk_analyst_node — synthesises all evidence → structured RiskReport via LLM

Design notes
------------
- The supervisor is *deterministic*, not LLM-based. The workflow order is fixed
  (financial → news → document → risk_analyst) so an LLM router adds latency
  with no benefit. The LLM budget is reserved for the synthesis step.
- All external calls (yfinance, PDF parser, search) are isolated to worker nodes
  so they can be patched individually in tests.
- Every node returns a safe fallback if its primary operation fails; the graph
  never crashes on a single tool failure.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.risk import evaluate_ratios, overall_risk
from src.schemas import (
    CompanyFinancials,
    FinancialRatios,
    RiskFlag,
    RiskLevel,
    RiskReport,
)

logger = logging.getLogger(__name__)


# ============================================================================ #
# Supervisor (deterministic router)
# ============================================================================ #

def supervisor_node(state: AgentState) -> dict:
    """Route to the next worker based on what evidence has already been collected.

    Execution order:
        financial_agent → news_agent → document_agent → risk_analyst → __end__

    ``document_agent`` is skipped when no ``pdf_path`` is provided.
    """
    has_financials = state.get("financials") is not None
    # None = not yet fetched; [] or [...] = already ran (may have 0 results)
    has_news = state.get("news_headlines") is not None
    # doc is "done" when worker ran (None→[] or None→[...]) OR no PDF to parse
    pdf_path = state.get("pdf_path")
    has_docs = state.get("doc_chunks") is not None or pdf_path is None
    has_report = state.get("risk_report") is not None

    if has_report:
        next_node = "__end__"
    elif not has_financials:
        next_node = "financial_agent"
    elif not has_news:
        next_node = "news_agent"
    elif not has_docs:
        next_node = "document_agent"
    else:
        next_node = "risk_analyst"

    logger.info("Supervisor → %s", next_node)
    return {"next": next_node}


# ============================================================================ #
# Financial data worker
# ============================================================================ #

def financial_node(state: AgentState) -> dict:
    """Fetch yfinance fundamentals and run the deterministic risk scorer."""
    from src.tools.financial_data import fetch_financials

    ticker = state["ticker"]
    logger.info("financial_node: fetching %s", ticker)

    try:
        cf: CompanyFinancials = fetch_financials(ticker)
        data = cf.model_dump()

        # Human-readable summary for the message log
        rev = cf.total_revenue
        cur = cf.currency or ""
        name = cf.company_name or ticker
        if rev is not None:
            summary = (
                f"Data fundamental {name} berhasil diambil.\n"
                f"  Revenue: {rev:,.0f} {cur} | "
                f"Net Income: {cf.net_income:,.0f} {cur}\n"
                f"  Current Ratio: {cf.ratios.current_ratio} | "
                f"DER: {cf.ratios.debt_to_equity} | "
                f"ROE: {cf.ratios.roe}"
            )
        else:
            summary = f"Data fundamental {name} diambil (beberapa field tidak tersedia)."

    except Exception as exc:
        logger.warning("financial_node error: %s", exc)
        data = None
        summary = f"Gagal mengambil data fundamental {ticker}: {exc}"

    return {
        "financials": data,
        "messages": [AIMessage(content=summary, name="financial_agent")],
    }


# ============================================================================ #
# Document (PDF) worker
# ============================================================================ #

def document_node(state: AgentState) -> dict:
    """Parse the annual-report PDF and retrieve analyst-relevant sections."""
    from src.tools.document import DocumentStore, parse_pdf

    pdf_path = state.get("pdf_path")
    ticker = state["ticker"]

    if not pdf_path:
        return {
            "doc_chunks": [],
            "messages": [AIMessage(
                content="Tidak ada PDF laporan keuangan yang diberikan — skip.",
                name="document_agent",
            )],
        }

    logger.info("document_node: parsing %s", pdf_path)
    try:
        markdown = parse_pdf(pdf_path)
        store = DocumentStore()
        store.add_markdown(markdown, source=pdf_path)

        # Retrieve sections most relevant to credit / risk analysis
        queries = [
            "going concern kelangsungan usaha",
            "risiko keuangan utang liabilitas",
            "transaksi pihak berelasi",
            "opini auditor",
        ]
        chunks: list[str] = []
        seen: set[str] = set()
        for q in queries:
            for chunk in store.search(q, top_k=2):
                text = chunk.text[:800]
                if text not in seen:
                    chunks.append(f"[{chunk.section}] {text}")
                    seen.add(text)

        summary = (
            f"PDF berhasil diparsing untuk {ticker}. "
            f"{len(chunks)} chunk relevan diambil."
        )
    except Exception as exc:
        logger.warning("document_node error: %s", exc)
        chunks = []
        summary = f"Gagal parsing PDF {pdf_path}: {exc}"

    return {
        "doc_chunks": chunks,
        "messages": [AIMessage(content=summary, name="document_agent")],
    }


# ============================================================================ #
# News worker
# ============================================================================ #

def news_node(state: AgentState) -> dict:
    """Fetch recent Indonesian financial news for the company."""
    from src.tools.news import fetch_news

    ticker = state["ticker"]
    fin = state.get("financials")
    company = (fin.get("company_name") or ticker) if fin else ticker
    logger.info("news_node: searching news for %s", company)

    try:
        items = fetch_news(company, max_results=6, ticker=ticker)
        headlines = [f"{it.title} — {it.snippet[:200]}" for it in items]
        summary = f"Ditemukan {len(headlines)} berita terbaru untuk {company}."
    except Exception as exc:
        logger.warning("news_node error: %s", exc)
        headlines = []
        summary = f"Gagal mengambil berita untuk {company}: {exc}"

    return {
        "news_headlines": headlines,
        "messages": [AIMessage(content=summary, name="news_agent")],
    }


# ============================================================================ #
# Risk Analyst synthesis (LLM → RiskReport)
# ============================================================================ #

_ANALYST_SYSTEM = """\
Kamu adalah analis risiko keuangan senior. Tulis laporan risiko terstruktur \
untuk {ticker} dalam Bahasa Indonesia berdasarkan data yang telah dikumpulkan.

Panduan:
- summary: narasi eksekutif 3-5 kalimat, bahasa Indonesia, non-teknis.
- overall_risk: harus konsisten dengan flags (banyak flag HIGH/SEVERE → SEVERE/HIGH).
- flags: hanya red/amber flags nyata dari data; jangan mengarang.
- positives: aspek positif yang memitigasi risiko.
- sources: sebutkan semua sumber yang digunakan (yfinance, news_search, \
financial_report_pdf).
- key_ratios: salin dari data fundamental yang tersedia.
"""


def risk_analyst_node(state: AgentState) -> dict:
    """Synthesise all collected evidence into a validated RiskReport via LLM."""
    from src.agents.llm import get_llm

    ticker = state["ticker"]
    fin_data = state.get("financials") or {}
    doc_chunks = state.get("doc_chunks") or []
    news = state.get("news_headlines") or []

    # ---- 1. Deterministic rule-based scorer (quantitative baseline) ----------
    flags: list[RiskFlag] = []
    quant_risk = RiskLevel.MODERATE
    if fin_data.get("ratios"):
        try:
            ratios = FinancialRatios(**fin_data["ratios"])
            flags = evaluate_ratios(ratios)
            quant_risk = overall_risk(flags)
        except Exception as exc:
            logger.warning("Rule-based scoring failed: %s", exc)

    # ---- 2. Build context block for LLM -------------------------------------
    context_parts: list[str] = []

    if fin_data.get("ratios"):
        r = fin_data["ratios"]
        ratios_str = "\n".join([
            f"  Current Ratio      : {r.get('current_ratio')}",
            f"  Quick Ratio        : {r.get('quick_ratio')}",
            f"  Debt-to-Equity     : {r.get('debt_to_equity')}",
            f"  Interest Coverage  : {r.get('interest_coverage')}",
            f"  Net Profit Margin  : {r.get('net_profit_margin')}",
            f"  ROE                : {r.get('roe')}",
            f"  Revenue Growth YoY : {r.get('revenue_growth')}",
            f"  Net Income Growth  : {r.get('net_income_growth')}",
        ])
        rule_flags_json = json.dumps(
            [f.model_dump() for f in flags], ensure_ascii=False, indent=2
        )
        context_parts.append(
            f"## Data Fundamental\n{ratios_str}\n\n"
            f"Quantitative risk (rule-based): **{quant_risk.value.upper()}**\n"
            f"Rule flags:\n```json\n{rule_flags_json}\n```"
        )

    if news:
        context_parts.append(
            "## Berita Terbaru\n"
            + "\n".join(f"- {h}" for h in news[:5])
        )

    if doc_chunks:
        context_parts.append(
            "## Ekstrak Laporan Keuangan PDF\n"
            + "\n\n---\n\n".join(doc_chunks[:4])
        )

    full_context = "\n\n".join(context_parts) or f"Data terbatas untuk {ticker}."

    # ---- 3. LLM → structured RiskReport -------------------------------------
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(RiskReport)

    messages = [
        SystemMessage(content=_ANALYST_SYSTEM.format(ticker=ticker)),
        HumanMessage(content=full_context),
    ]

    try:
        report: RiskReport = structured_llm.invoke(messages)
    except Exception as exc:
        logger.error("LLM structured output failed (%s) — using rule-based fallback", exc)
        report = _fallback_report(ticker, fin_data, flags, quant_risk, news, doc_chunks)

    return {
        "risk_report": report.model_dump(),
        "messages": [AIMessage(
            content=(
                f"Risk report selesai untuk {ticker}. "
                f"Overall risk: **{report.overall_risk.value.upper()}**. "
                f"{len(report.flags)} flag ditemukan."
            ),
            name="risk_analyst",
        )],
    }


# ============================================================================ #
# Fallback: build RiskReport from rule-based results (no LLM)
# ============================================================================ #

def _fallback_report(
    ticker: str,
    fin_data: dict,
    flags: list[RiskFlag],
    risk_level: RiskLevel,
    news: list[str],
    doc_chunks: list[str],
) -> RiskReport:
    """Construct a minimal RiskReport from rule-based output when LLM fails."""
    ratios = (
        FinancialRatios(**fin_data["ratios"])
        if fin_data.get("ratios")
        else FinancialRatios()
    )
    sources = ["yfinance"]
    if news:
        sources.append("news_search")
    if doc_chunks:
        sources.append("financial_report_pdf")

    return RiskReport(
        ticker=ticker,
        company_name=fin_data.get("company_name"),
        overall_risk=risk_level,
        summary=(
            f"Analisis risiko {ticker} berdasarkan data kuantitatif. "
            f"Level risiko keseluruhan: {risk_level.value.upper()}. "
            f"Ditemukan {len(flags)} flag risiko dari data fundamental."
        ),
        key_ratios=ratios,
        flags=flags,
        positives=[],
        sources=sources,
    )
