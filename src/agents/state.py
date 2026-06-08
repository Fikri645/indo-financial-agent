"""Shared state for the LangGraph multi-agent graph.

Every node reads from and writes to AgentState. Using a TypedDict (not a plain
dict) gives static type checking and makes the data contract explicit — each
field has one owner and one consumer.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ---- conversation history (LangGraph-managed; append-only via add_messages) ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- task inputs -------------------------------------------------------- #
    ticker: str          # IDX ticker, e.g. "BBRI" or "BBRI.JK"
    pdf_path: Optional[str]  # optional path to the annual-report PDF
    use_quarterly: bool  # True → use quarterly statements; False → annual (default)

    # ---- evidence collected by worker nodes --------------------------------- #
    # None  = worker has NOT run yet  (supervisor will dispatch it)
    # []    = worker ran but returned no results  (supervisor moves on)
    # [..] = worker ran and returned results
    financials: Optional[dict]           # CompanyFinancials.model_dump() from financial_node
    doc_chunks: Optional[list[str]]      # analyst-relevant PDF excerpts from document_node
    news_headlines: Optional[list[str]]  # recent news snippets from news_node

    # ---- final output ------------------------------------------------------- #
    risk_report: Optional[dict]   # RiskReport.model_dump() from risk_analyst_node

    # ---- routing ------------------------------------------------------------ #
    next: str  # supervisor's routing decision → next node name
