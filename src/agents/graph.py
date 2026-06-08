"""LangGraph StateGraph — Indonesian Financial Research Agent.

The supervisor node deterministically routes through:
  data_gather_agent (financial + news in parallel)
    → document_agent (only when a PDF is provided)
    → risk_analyst
    → END

The only LLM call is in risk_analyst_node, which synthesises all collected
evidence into a validated RiskReport (Pydantic structured output).

Usage
-----
    # As a module (CLI):
    python -m src.agents.graph BBRI [path/to/BBRI_2024.pdf]

    # Programmatic:
    from src.agents.graph import graph
    result = graph.invoke(initial_state)
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    document_node,
    parallel_data_node,
    risk_analyst_node,
    supervisor_node,
)
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Routing edge
# --------------------------------------------------------------------------- #

_RouteTarget = Literal[
    "data_gather_agent", "document_agent", "risk_analyst", "__end__"
]


def _route(state: AgentState) -> _RouteTarget:
    """Read the supervisor's decision from state and forward to the right node."""
    return state.get("next", "__end__")


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_graph() -> StateGraph:
    """Construct and compile the multi-agent supervisor graph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_gather_agent", parallel_data_node)  # financial + news concurrently
    builder.add_node("document_agent", document_node)
    builder.add_node("risk_analyst", risk_analyst_node)

    # Entry point: always hit supervisor first
    builder.add_edge(START, "supervisor")

    # Supervisor dispatches conditionally
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {
            "data_gather_agent": "data_gather_agent",
            "document_agent": "document_agent",
            "risk_analyst": "risk_analyst",
            "__end__": END,
        },
    )

    # All workers return to supervisor after completing their task
    builder.add_edge("data_gather_agent", "supervisor")
    builder.add_edge("document_agent", "supervisor")
    builder.add_edge("risk_analyst", "supervisor")  # supervisor sees report → routes to END

    return builder.compile()


# Compiled graph — import and use directly:
#   from src.agents.graph import graph
graph = build_graph()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _make_initial_state(
    ticker: str,
    pdf_path: str | None = None,
    use_quarterly: bool = False,
) -> AgentState:
    return {
        "messages": [HumanMessage(content=f"Analisis risiko keuangan perusahaan {ticker}")],
        "ticker": ticker,
        "pdf_path": pdf_path,
        "use_quarterly": use_quarterly,
        "financials": None,
        "doc_chunks": None,      # None = not yet fetched (sentinel)
        "news_headlines": None,  # None = not yet fetched (sentinel)
        "risk_report": None,
        "next": "",
    }


def _run_and_print(ticker: str, pdf_path: str | None = None) -> None:
    initial = _make_initial_state(ticker, pdf_path)

    print(f"\n{'='*60}")
    print("  Indonesian Financial Research Agent")
    print(f"  Ticker  : {ticker.upper()}")
    print(f"  PDF     : {pdf_path or '(tidak ada)'}")
    print(f"{'='*60}\n")

    # stream_mode="values" yields the full cumulative state after each step.
    # Track message count to print only newly appended messages each step.
    final_state: AgentState | None = None
    prev_msg_count = 0
    for step in graph.stream(initial, {"recursion_limit": 25}, stream_mode="values"):
        msgs = step.get("messages") or []
        for msg in msgs[prev_msg_count:]:
            name = getattr(msg, "name", None) or "agent"
            print(f"  [{name}] {msg.content}")
        prev_msg_count = len(msgs)
        final_state = step

    # Print final structured report
    report = (final_state or {}).get("risk_report")
    if report:
        print(f"\n{'='*60}")
        print("  RISK REPORT (JSON)")
        print(f"{'='*60}")
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print("\n[!] Risk report tidak berhasil dibuat.")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    _ticker = sys.argv[1] if len(sys.argv) > 1 else "BBRI"
    _pdf = sys.argv[2] if len(sys.argv) > 2 else None
    _run_and_print(_ticker, _pdf)
