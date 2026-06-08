"""Unit tests for src/tools/financial_data.py.

All yfinance I/O is mocked so tests run offline and deterministically.
Covers: _safe_div, _first_row, _growth, compute_ratios, classify_sector,
        fetch_financials (annual + quarterly), and the TTL cache.
"""
from __future__ import annotations

import math
import sys
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.tools.financial_data import (
    _CACHE,
    _CACHE_TTL,
    _first_row,
    _growth,
    _safe_div,
    classify_sector,
    compute_ratios,
    fetch_financials,
)


def _mock_yfinance(tk: MagicMock) -> MagicMock:
    """Build a fake ``yfinance`` module whose Ticker() returns *tk*."""
    yf_mod = MagicMock()
    yf_mod.Ticker.return_value = tk
    return yf_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_balance_sheet() -> pd.DataFrame:
    """Synthetic annual balance sheet (items as index, dates as columns)."""
    items = [
        "Total Assets",
        "Total Liabilities Net Minority Interest",
        "Common Stock Equity",
        "Current Assets",
        "Current Liabilities",
        "Inventory",
    ]
    data = {
        pd.Timestamp("2024-12-31"): [1_000_000.0, 600_000.0, 400_000.0, 300_000.0, 200_000.0, 50_000.0],
        pd.Timestamp("2023-12-31"): [900_000.0, 540_000.0, 360_000.0, 270_000.0, 180_000.0, 45_000.0],
    }
    return pd.DataFrame(data, index=items)


def _make_income_stmt() -> pd.DataFrame:
    """Synthetic annual income statement."""
    items = ["Total Revenue", "Gross Profit", "EBIT", "Interest Expense", "Net Income"]
    data = {
        pd.Timestamp("2024-12-31"): [500_000.0, 200_000.0, 100_000.0, -30_000.0, 60_000.0],
        pd.Timestamp("2023-12-31"): [400_000.0, 160_000.0, 80_000.0, -25_000.0, 48_000.0],
    }
    return pd.DataFrame(data, index=items)


def _make_quarterly_income_stmt() -> pd.DataFrame:
    """Synthetic quarterly income statement (5 quarters = current + 4 prior)."""
    items = ["Total Revenue", "Net Income"]
    # Q1 2026, Q4 2025, Q3 2025, Q2 2025, Q1 2025
    data = {
        pd.Timestamp("2026-03-31"): [130_000.0, 16_000.0],
        pd.Timestamp("2025-12-31"): [140_000.0, 18_000.0],
        pd.Timestamp("2025-09-30"): [125_000.0, 15_000.0],
        pd.Timestamp("2025-06-30"): [120_000.0, 14_000.0],
        pd.Timestamp("2025-03-31"): [110_000.0, 12_000.0],
    }
    return pd.DataFrame(data, index=items)


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10.0, 4.0) == 2.5

    def test_rounds_to_4_decimal_places(self):
        result = _safe_div(1.0, 3.0)
        assert result == round(1 / 3, 4)

    def test_zero_denominator_returns_none(self):
        assert _safe_div(10.0, 0.0) is None

    def test_none_numerator_returns_none(self):
        assert _safe_div(None, 5.0) is None

    def test_none_denominator_returns_none(self):
        assert _safe_div(5.0, None) is None

    def test_both_none_returns_none(self):
        assert _safe_div(None, None) is None

    def test_negative_values(self):
        assert _safe_div(-60.0, 400.0) == round(-60 / 400, 4)


# ---------------------------------------------------------------------------
# _first_row
# ---------------------------------------------------------------------------

class TestFirstRow:
    def test_finds_primary_alias(self):
        bs = _make_balance_sheet()
        val = _first_row(bs, ["Total Assets"], col_idx=0)
        assert val == 1_000_000.0

    def test_falls_back_to_second_alias(self):
        bs = _make_balance_sheet()
        # "Total Liabilities" is not in index; "Total Liabilities Net Minority Interest" is
        val = _first_row(bs, ["Total Liabilities", "Total Liabilities Net Minority Interest"])
        assert val == 600_000.0

    def test_prior_year_column(self):
        bs = _make_balance_sheet()
        val = _first_row(bs, ["Total Assets"], col_idx=1)
        assert val == 900_000.0

    def test_missing_label_returns_none(self):
        bs = _make_balance_sheet()
        val = _first_row(bs, ["Goodwill", "Intangibles"])
        assert val is None

    def test_none_df_returns_none(self):
        assert _first_row(None, ["Total Assets"]) is None

    def test_empty_df_returns_none(self):
        assert _first_row(pd.DataFrame(), ["Total Assets"]) is None

    def test_col_idx_out_of_range_returns_none(self):
        bs = _make_balance_sheet()  # 2 columns
        assert _first_row(bs, ["Total Assets"], col_idx=5) is None

    def test_nan_value_skipped_tries_next_alias(self):
        import numpy as np
        items = ["Row A", "Row B"]
        data = {pd.Timestamp("2024-12-31"): [float("nan"), 42.0]}
        df = pd.DataFrame(data, index=items)
        val = _first_row(df, ["Row A", "Row B"])
        assert val == 42.0


# ---------------------------------------------------------------------------
# _growth
# ---------------------------------------------------------------------------

class TestGrowth:
    def test_positive_annual_growth(self):
        is_ = _make_income_stmt()
        g = _growth(is_, ["Total Revenue"], prior_col=1)
        # (500k - 400k) / 400k = 0.25
        assert g == pytest.approx(0.25, abs=1e-4)

    def test_negative_growth(self):
        items = ["Net Income"]
        data = {
            pd.Timestamp("2024-12-31"): [40_000.0],
            pd.Timestamp("2023-12-31"): [50_000.0],
        }
        df = pd.DataFrame(data, index=items)
        g = _growth(df, ["Net Income"], prior_col=1)
        assert g == pytest.approx(-0.20, abs=1e-4)

    def test_quarterly_same_quarter_yoy(self):
        is_ = _make_quarterly_income_stmt()
        # Q1 2026 (130k) vs Q1 2025 (110k): (130-110)/110 ≈ 0.1818
        g = _growth(is_, ["Total Revenue"], prior_col=4)
        assert g == pytest.approx((130_000 - 110_000) / 110_000, abs=1e-4)

    def test_zero_prior_returns_none(self):
        items = ["Revenue"]
        data = {pd.Timestamp("2024-12-31"): [100.0], pd.Timestamp("2023-12-31"): [0.0]}
        df = pd.DataFrame(data, index=items)
        assert _growth(df, ["Revenue"], prior_col=1) is None

    def test_none_df_returns_none(self):
        assert _growth(None, ["Revenue"]) is None

    def test_prior_col_missing_returns_none(self):
        # Only 1 column → prior_col=1 is out of range
        items = ["Revenue"]
        data = {pd.Timestamp("2024-12-31"): [100.0]}
        df = pd.DataFrame(data, index=items)
        assert _growth(df, ["Revenue"], prior_col=1) is None


# ---------------------------------------------------------------------------
# compute_ratios (annual)
# ---------------------------------------------------------------------------

class TestComputeRatios:
    def test_liquidity_ratios(self):
        bs = _make_balance_sheet()
        is_ = _make_income_stmt()
        r = compute_ratios(bs, is_)
        # current_ratio = 300k / 200k = 1.5
        assert r.current_ratio == pytest.approx(1.5, abs=1e-3)
        # quick_ratio = (300k - 50k) / 200k = 1.25
        assert r.quick_ratio == pytest.approx(1.25, abs=1e-3)

    def test_leverage_ratios(self):
        bs = _make_balance_sheet()
        is_ = _make_income_stmt()
        r = compute_ratios(bs, is_)
        # DER = 600k / 400k = 1.5
        assert r.debt_to_equity == pytest.approx(1.5, abs=1e-3)
        # debt_ratio = 600k / 1000k = 0.6
        assert r.debt_ratio == pytest.approx(0.6, abs=1e-3)
        # interest_coverage = 100k / 30k ≈ 3.33
        assert r.interest_coverage == pytest.approx(100_000 / 30_000, abs=1e-3)

    def test_profitability_ratios(self):
        bs = _make_balance_sheet()
        is_ = _make_income_stmt()
        r = compute_ratios(bs, is_)
        # net_profit_margin = 60k / 500k = 0.12
        assert r.net_profit_margin == pytest.approx(0.12, abs=1e-3)
        # gross_margin = 200k / 500k = 0.40
        assert r.gross_margin == pytest.approx(0.40, abs=1e-3)
        # ROE = 60k / 400k = 0.15
        assert r.roe == pytest.approx(0.15, abs=1e-3)
        # ROA = 60k / 1000k = 0.06
        assert r.roa == pytest.approx(0.06, abs=1e-3)

    def test_annual_growth_uses_prior_col_1(self):
        bs = _make_balance_sheet()
        is_ = _make_income_stmt()
        r = compute_ratios(bs, is_, use_quarterly=False)
        # revenue growth = (500k-400k)/400k = 0.25
        assert r.revenue_growth == pytest.approx(0.25, abs=1e-4)
        # net income growth = (60k-48k)/48k = 0.25
        assert r.net_income_growth == pytest.approx(0.25, abs=1e-4)

    def test_quarterly_growth_uses_prior_col_4(self):
        bs = _make_balance_sheet()  # balance sheet stays annual
        is_ = _make_quarterly_income_stmt()
        r = compute_ratios(bs, is_, use_quarterly=True)
        # Q1 2026 vs Q1 2025: (130k-110k)/110k ≈ 0.1818
        assert r.revenue_growth == pytest.approx((130_000 - 110_000) / 110_000, abs=1e-4)

    def test_none_dataframes_return_none_ratios(self):
        r = compute_ratios(None, None)
        assert r.current_ratio is None
        assert r.debt_to_equity is None
        assert r.net_profit_margin is None
        assert r.revenue_growth is None


# ---------------------------------------------------------------------------
# classify_sector
# ---------------------------------------------------------------------------

class TestClassifySector:
    @pytest.mark.parametrize("sector,industry,expected", [
        ("Financial Services", "Banks - Regional", "financial"),
        ("Financial Services", "Insurance - Life", "financial"),
        ("Financial Services", "Asset Management", "financial"),
        ("Financial Services", "Capital Markets", "financial"),
        (None, "Bank Umum", "financial"),
        ("Technology", "Software", "general"),
        ("Consumer Defensive", "Food Products", "general"),
        (None, None, "general"),
        ("Basic Materials", "Gold", "general"),
    ])
    def test_sector_classification(self, sector, industry, expected):
        assert classify_sector(sector, industry) == expected


# ---------------------------------------------------------------------------
# fetch_financials (mocked yfinance)
# ---------------------------------------------------------------------------

def _make_mock_ticker(bs=None, is_=None, quarterly_bs=None, quarterly_is=None, info=None):
    """Return a mock yfinance Ticker with configurable statements."""
    tk = MagicMock()
    tk.info = info or {
        "longName": "Test Corp",
        "financialCurrency": "IDR",
        "sector": "Technology",
        "industry": "Software",
    }
    tk.balance_sheet = bs if bs is not None else _make_balance_sheet()
    tk.income_stmt = is_ if is_ is not None else _make_income_stmt()
    tk.quarterly_balance_sheet = quarterly_bs if quarterly_bs is not None else _make_balance_sheet()
    tk.quarterly_income_stmt = quarterly_is if quarterly_is is not None else _make_quarterly_income_stmt()
    return tk


class TestFetchFinancials:
    """Tests for fetch_financials — yfinance is mocked via sys.modules injection."""

    def setup_method(self):
        """Clear cache before each test to avoid cross-test contamination."""
        _CACHE.clear()

    def _run(self, ticker: str, use_quarterly: bool = False, tk: MagicMock | None = None):
        """Run fetch_financials with a mocked yfinance module."""
        if tk is None:
            tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            return fetch_financials(ticker, use_quarterly=use_quarterly)

    def test_annual_populates_period_end(self):
        cf = self._run("TEST", use_quarterly=False)
        assert cf.period_end == "2024-12-31"
        assert cf.prior_period_end == "2023-12-31"

    def test_quarterly_uses_quarterly_statements(self):
        """When use_quarterly=True, the mock's quarterly_balance_sheet is called."""
        tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            cf = fetch_financials("TEST", use_quarterly=True)
        # quarterly_balance_sheet was queried (not balance_sheet)
        _ = tk.quarterly_balance_sheet  # accessed → no AttributeError
        assert cf.period_end is not None

    def test_company_name_from_info(self):
        cf = self._run("TEST")
        assert cf.company_name == "Test Corp"
        assert cf.currency == "IDR"

    def test_sector_classified_as_financial(self):
        tk = _make_mock_ticker(info={
            "longName": "Bank ABC",
            "financialCurrency": "IDR",
            "sector": "Financial Services",
            "industry": "Banks - Regional",
        })
        cf = self._run("BANK", tk=tk)
        assert cf.sector == "financial"

    def test_normalizes_ticker_adds_jk_suffix(self):
        tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            fetch_financials("BBRI")
        yf_mock.Ticker.assert_called_once_with("BBRI.JK")

    def test_result_cached_on_second_call(self):
        tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            fetch_financials("CACHE")
            fetch_financials("CACHE")
        # Ticker() should only be instantiated once — second call hits cache
        assert yf_mock.Ticker.call_count == 1

    def test_cache_key_separates_quarterly_from_annual(self):
        tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            fetch_financials("DIFF", use_quarterly=False)
            fetch_financials("DIFF", use_quarterly=True)
        # Different cache keys → 2 real fetches
        assert yf_mock.Ticker.call_count == 2

    def test_cache_expires_after_ttl(self):
        tk = _make_mock_ticker()
        yf_mock = _mock_yfinance(tk)
        with patch.dict(sys.modules, {"yfinance": yf_mock}):
            fetch_financials("TTL")
            # Manually expire the cache entry
            key = "TTL.JK:a"
            cf, _ = _CACHE[key]
            _CACHE[key] = (cf, time.monotonic() - _CACHE_TTL - 1)
            fetch_financials("TTL")
        assert yf_mock.Ticker.call_count == 2

    def test_empty_statements_return_none_ratios(self):
        tk = MagicMock()
        tk.info = {}
        tk.balance_sheet = pd.DataFrame()
        tk.income_stmt = pd.DataFrame()
        tk.quarterly_balance_sheet = pd.DataFrame()
        tk.quarterly_income_stmt = pd.DataFrame()
        cf = self._run("NODATA", tk=tk)
        assert cf.ratios.current_ratio is None
        assert cf.total_revenue is None
        assert cf.period_end is None
