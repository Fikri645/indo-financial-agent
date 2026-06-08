"""Unit tests for the LangGraph agent graph.

All external I/O (yfinance, news, PDF parser, LLM) is mocked so tests run
offline, deterministically, and fast. LangGraph itself is required — tests are
skipped gracefully when the package is not installed (e.g. in the lightweight
CI environment).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if LangGraph is not installed (lightweight CI env).
pytest.importorskip("langgraph")

from src.agents.nodes import (  # noqa: E402 — after importorskip guard
    _fallback_report,
    financial_node,
    news_node,
    document_node,
    parallel_data_node,
    supervisor_node,
    risk_analyst_node,
)
from src.schemas import (  # noqa: E402
    CompanyFinancials,
    FinancialRatios,
    RiskLevel,
    RiskReport,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _base_state(**overrides):
    """Return a minimal AgentState-compatible dict."""
    state = {
        "messages": [],
        "ticker": "BBRI",
        "pdf_path": None,
        "use_quarterly": False,
        "financials": None,
        "doc_chunks": None,      # None = not yet fetched
        "news_headlines": None,  # None = not yet fetched
        "risk_report": None,
        "next": "",
    }
    state.update(overrides)
    return state


def _fake_financials() -> CompanyFinancials:
    """Synthetic CompanyFinancials for a healthy company."""
    return CompanyFinancials(
        ticker="BBRI.JK",
        company_name="Bank Rakyat Indonesia",
        currency="IDR",
        period_end="2024-12-31",
        ratios=FinancialRatios(
            current_ratio=1.5,
            quick_ratio=1.2,
            debt_to_equity=0.4,
            interest_coverage=8.0,
            net_profit_margin=0.20,
            roe=0.15,
            revenue_growth=0.10,
            net_income_growth=0.08,
        ),
        total_revenue=186_000_000_000,
        net_income=56_000_000_000,
        total_assets=1_800_000_000_000,
        total_liabilities=720_000_000_000,
        total_equity=1_080_000_000_000,
    )


def _fake_risk_report(ticker="BBRI") -> RiskReport:
    return RiskReport(
        ticker=ticker,
        company_name="Bank Rakyat Indonesia",
        overall_risk=RiskLevel.MODERATE,
        summary="Perusahaan dalam kondisi keuangan moderat.",
        key_ratios=FinancialRatios(roe=0.15, debt_to_equity=0.4),
        flags=[],
        positives=["ROE positif"],
        sources=["yfinance"],
    )


# ============================================================================ #
# Supervisor routing — deterministic, no external deps
# ============================================================================ #

class TestSupervisorNode:
    def test_routes_to_data_gather_when_nothing_gathered(self):
        # Neither financials nor news → data_gather_agent fetches both concurrently
        result = supervisor_node(_base_state())
        assert result["next"] == "data_gather_agent"

    def test_routes_to_data_gather_when_only_news_missing(self):
        # financials done but news still None → still needs data_gather
        state = _base_state(financials={"ticker": "BBRI.JK"}, news_headlines=None)
        result = supervisor_node(state)
        assert result["next"] == "data_gather_agent"

    def test_routes_to_data_gather_when_only_financials_missing(self):
        # news done but financials still None → still needs data_gather
        state = _base_state(financials=None, news_headlines=["headline 1"])
        result = supervisor_node(state)
        assert result["next"] == "data_gather_agent"

    def test_routes_to_risk_analyst_when_both_data_ready_no_pdf(self):
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=["headline 1"],
            pdf_path=None,         # no PDF → doc considered done
        )
        result = supervisor_node(state)
        assert result["next"] == "risk_analyst"

    def test_routes_to_risk_analyst_when_news_empty_no_pdf(self):
        # news_headlines=[] → fetched but 0 results → treated as done
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=[],     # fetched, empty result — still done
            pdf_path=None,
        )
        result = supervisor_node(state)
        assert result["next"] == "risk_analyst"

    def test_routes_to_document_agent_when_pdf_provided_but_not_parsed(self):
        # doc_chunks=None → not fetched yet (pdf_path is set)
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=["headline 1"],
            pdf_path="/data/BBRI.pdf",
            doc_chunks=None,        # None = not yet fetched
        )
        result = supervisor_node(state)
        assert result["next"] == "document_agent"

    def test_routes_to_risk_analyst_after_document_parsed(self):
        # doc_chunks=[] → fetched (even if empty) → done
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=["headline 1"],
            pdf_path="/data/BBRI.pdf",
            doc_chunks=["[risk] Some risk section text."],
        )
        result = supervisor_node(state)
        assert result["next"] == "risk_analyst"

    def test_routes_to_risk_analyst_when_pdf_doc_empty(self):
        # doc_chunks=[] (fetched, empty) should still be considered done
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=["headline 1"],
            pdf_path="/data/BBRI.pdf",
            doc_chunks=[],          # fetched but 0 chunks → still done
        )
        result = supervisor_node(state)
        assert result["next"] == "risk_analyst"

    def test_routes_to_end_when_report_ready(self):
        state = _base_state(
            financials={"ticker": "BBRI.JK"},
            news_headlines=["headline 1"],
            risk_report={"ticker": "BBRI"},
        )
        result = supervisor_node(state)
        assert result["next"] == "__end__"


# ============================================================================ #
# parallel_data_node
# ============================================================================ #

class TestParallelDataNode:
    def test_populates_both_financials_and_news(self):
        fake = _fake_financials()
        from langchain_core.messages import AIMessage as _AI
        fake_news_agent = MagicMock()
        fake_news_agent.invoke.return_value = {"messages": [_AI(content="Sentimen positif.")]}

        with (
            patch("src.tools.financial_data.fetch_financials", return_value=fake),
            patch("langgraph.prebuilt.create_react_agent", return_value=fake_news_agent),
            patch("src.agents.llm.get_llm", return_value=MagicMock()),
        ):
            result = parallel_data_node(_base_state())

        assert result["financials"] is not None
        assert result["news_headlines"] is not None
        assert len(result["messages"]) == 2  # one from each agent

    def test_financial_failure_does_not_block_news(self):
        """Even if yfinance fails, news should still run."""
        from langchain_core.messages import AIMessage as _AI
        fake_news_agent = MagicMock()
        fake_news_agent.invoke.return_value = {"messages": [_AI(content="Berita terkini.")]}

        with (
            patch("src.tools.financial_data.fetch_financials", side_effect=RuntimeError("down")),
            patch("langgraph.prebuilt.create_react_agent", return_value=fake_news_agent),
            patch("src.agents.llm.get_llm", return_value=MagicMock()),
        ):
            result = parallel_data_node(_base_state())

        assert result["financials"] is None          # graceful failure
        assert result["news_headlines"] is not None  # news still ran
        assert len(result["messages"]) == 2

    def test_messages_preserve_agent_names(self):
        """Messages returned by parallel_data_node keep their original agent names."""
        fake = _fake_financials()
        from langchain_core.messages import AIMessage as _AI
        fake_news_agent = MagicMock()
        fake_news_agent.invoke.return_value = {"messages": [_AI(content="News OK.")]}

        with (
            patch("src.tools.financial_data.fetch_financials", return_value=fake),
            patch("langgraph.prebuilt.create_react_agent", return_value=fake_news_agent),
            patch("src.agents.llm.get_llm", return_value=MagicMock()),
        ):
            result = parallel_data_node(_base_state())

        names = {msg.name for msg in result["messages"]}
        assert "financial_agent" in names
        assert "news_agent" in names


# ============================================================================ #
# financial_node
# ============================================================================ #

class TestFinancialNode:
    def test_success_populates_financials(self):
        fake = _fake_financials()
        with patch("src.tools.financial_data.fetch_financials", return_value=fake):
            result = financial_node(_base_state())

        assert result["financials"] is not None
        assert result["financials"]["ticker"] == "BBRI.JK"
        assert result["financials"]["total_revenue"] == 186_000_000_000
        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0].name == "financial_agent"

    def test_failure_returns_none_financials(self):
        with patch(
            "src.tools.financial_data.fetch_financials",
            side_effect=RuntimeError("network down"),
        ):
            result = financial_node(_base_state())

        assert result["financials"] is None
        assert "Gagal" in result["messages"][0].content

    def test_message_contains_key_ratios(self):
        fake = _fake_financials()
        with patch("src.tools.financial_data.fetch_financials", return_value=fake):
            result = financial_node(_base_state())
        content = result["messages"][0].content
        assert "ROE" in content or "Revenue" in content


# ============================================================================ #
# news_node
# ============================================================================ #

class TestNewsNode:
    def _make_news_items(self):
        from src.tools.news import NewsItem
        return [
            NewsItem(title="BBRI Catat Laba Besar", snippet="Laba naik 15%", url="http://a.com"),
            NewsItem(title="Dividen BBRI", snippet="Dividen interim diumumkan", url="http://b.com"),
        ]

    def _fake_agent(self, synthesis: str, n_tool_calls: int = 1):
        """A fake ReAct agent whose invoke() returns a synthesised final message."""
        from langchain_core.messages import AIMessage as _AI, ToolMessage as _Tool
        msgs = [_Tool(content="tool result", tool_call_id="t1") for _ in range(n_tool_calls)]
        msgs.append(_AI(content=synthesis))
        agent = MagicMock()
        agent.invoke.return_value = {"messages": msgs}
        return agent

    def test_agent_path_populates_synthesis(self):
        fake_agent = self._fake_agent("Sentimen positif: BBRI laba naik 15%.", n_tool_calls=2)
        with (
            patch("langgraph.prebuilt.create_react_agent", return_value=fake_agent),
            patch("src.agents.llm.get_llm", return_value=MagicMock()),
        ):
            result = news_node(_base_state())

        assert len(result["news_headlines"]) == 1
        assert "BBRI laba naik" in result["news_headlines"][0]
        assert result["messages"][0].name == "news_agent"

    def test_falls_back_to_deterministic_when_agent_fails(self):
        items = self._make_news_items()
        with (
            patch("langgraph.prebuilt.create_react_agent", side_effect=RuntimeError("no key")),
            patch("src.tools.news.fetch_news", return_value=items) as mock_fetch,
        ):
            result = news_node(_base_state(financials=_fake_financials().model_dump()))

        # Fallback path used fetch_news with the company name
        assert len(result["news_headlines"]) == 2
        assert "BBRI Catat Laba Besar" in result["news_headlines"][0]
        assert "Bank Rakyat Indonesia" in mock_fetch.call_args[0][0]

    def test_total_failure_returns_empty(self):
        with (
            patch("langgraph.prebuilt.create_react_agent", side_effect=RuntimeError("no key")),
            patch("src.tools.news.fetch_news", side_effect=RuntimeError("timeout")),
        ):
            result = news_node(_base_state())

        assert result["news_headlines"] == []
        assert "Gagal" in result["messages"][0].content


# ============================================================================ #
# document_node
# ============================================================================ #

class TestDocumentNode:
    def test_no_pdf_path_returns_empty_chunks(self):
        result = document_node(_base_state(pdf_path=None))
        assert result["doc_chunks"] == []
        assert "skip" in result["messages"][0].content.lower() or \
               "tidak ada" in result["messages"][0].content.lower()

    def test_success_populates_chunks(self):
        fake_markdown = (
            "# Going Concern\nAdanya ketidakpastian material dalam kelangsungan usaha.\n\n"
            "# Risiko Keuangan\nRasio utang meningkat signifikan tahun ini."
        )

        mock_store = MagicMock()
        mock_store.search.return_value = [
            MagicMock(text="Ketidakpastian material.", section="going_concern")
        ]

        # DocumentStore is a local import inside document_node → patch at its source module
        with (
            patch("src.tools.document.parse_pdf", return_value=fake_markdown),
            patch("src.tools.document.DocumentStore", return_value=mock_store),
        ):
            result = document_node(_base_state(pdf_path="/data/BBRI.pdf"))

        assert len(result["doc_chunks"]) >= 1
        assert result["messages"][0].name == "document_agent"

    def test_parse_failure_returns_empty_chunks(self):
        with patch("src.tools.document.parse_pdf", side_effect=FileNotFoundError("not found")):
            result = document_node(_base_state(pdf_path="/data/missing.pdf"))
        assert result["doc_chunks"] == []
        assert "Gagal" in result["messages"][0].content


# ============================================================================ #
# risk_analyst_node — LLM mocked
# ============================================================================ #

class TestRiskAnalystNode:
    def _state_with_all_data(self):
        fin = _fake_financials().model_dump()
        return _base_state(
            financials=fin,
            news_headlines=["BBRI Laba Naik — laba naik 15% YoY"],
            doc_chunks=["[risk] Risiko keuangan moderat."],  # non-None = fetched
        )

    def test_success_produces_risk_report(self):
        fake_report = _fake_risk_report()
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_report

        with patch("src.agents.llm.get_llm", return_value=mock_llm):
            result = risk_analyst_node(self._state_with_all_data())

        assert result["risk_report"] is not None
        assert result["risk_report"]["overall_risk"] == RiskLevel.MODERATE.value
        assert result["messages"][0].name == "risk_analyst"

    def test_llm_failure_uses_fallback(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")

        with patch("src.agents.llm.get_llm", return_value=mock_llm):
            result = risk_analyst_node(self._state_with_all_data())

        # Fallback should still produce a report
        assert result["risk_report"] is not None
        assert "overall_risk" in result["risk_report"]

    def test_fallback_includes_all_sources(self):
        report = _fallback_report(
            ticker="BBRI",
            fin_data=_fake_financials().model_dump(),
            flags=[],
            risk_level=RiskLevel.LOW,
            news=["headline"],
            doc_chunks=["[risk] text"],
        )
        assert "yfinance" in report.sources
        assert "news_search" in report.sources
        assert "financial_report_pdf" in report.sources

    def test_empty_financials_still_produces_report(self):
        """Graph must not crash when yfinance returned None."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("skip")

        with patch("src.agents.llm.get_llm", return_value=mock_llm):
            result = risk_analyst_node(_base_state())  # no financials

        assert result["risk_report"] is not None


# ============================================================================ #
# Full graph smoke test (all I/O mocked)
# ============================================================================ #

class TestFullGraph:
    def test_graph_produces_risk_report(self):
        from src.agents.graph import graph

        fake_fin = _fake_financials()
        fake_report = _fake_risk_report()
        from src.tools.news import NewsItem
        fake_news = [NewsItem(title="Test", snippet="snippet", url="http://x.com")]

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_report

        # Fake news ReAct agent → returns a synthesis message
        from langchain_core.messages import AIMessage as _AI
        fake_news_agent = MagicMock()
        fake_news_agent.invoke.return_value = {
            "messages": [_AI(content="Sentimen netral untuk BBRI.")]
        }

        with (
            patch("src.tools.financial_data.fetch_financials", return_value=fake_fin),
            patch("src.tools.news.fetch_news", return_value=fake_news),
            patch("langgraph.prebuilt.create_react_agent", return_value=fake_news_agent),
            patch("src.agents.llm.get_llm", return_value=mock_llm),
        ):
            initial = {
                "messages": [],
                "ticker": "BBRI",
                "pdf_path": None,
                "financials": None,
                "doc_chunks": None,      # None = not yet fetched
                "news_headlines": None,  # None = not yet fetched
                "risk_report": None,
                "next": "",
            }
            from langchain_core.messages import HumanMessage
            initial["messages"] = [HumanMessage(content="Analisis BBRI")]

            result = graph.invoke(initial, {"recursion_limit": 20})

        assert result["risk_report"] is not None
        assert result["risk_report"]["ticker"] == "BBRI"
        # All workers should have been called
        assert result["financials"] is not None
        assert len(result["news_headlines"]) >= 1
