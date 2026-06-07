"""Unit tests for ratio computation using synthetic yfinance-shaped DataFrames."""
import pandas as pd
import pytest

from src.tools.financial_data import classify_sector, compute_ratios


@pytest.mark.parametrize(
    "sector,industry,expected",
    [
        ("Financial Services", "Banks - Regional", "financial"),
        (None, "Insurance - Life", "financial"),
        ("Financial Services", "Asset Management", "financial"),
        ("Technology", "Software - Infrastructure", "general"),
        ("Communication Services", "Telecom Services", "general"),
        (None, None, "general"),
    ],
)
def test_classify_sector(sector, industry, expected):
    assert classify_sector(sector, industry) == expected


@pytest.fixture
def balance_sheet():
    # Columns = periods (newest first), index = line items — mirrors yfinance.
    return pd.DataFrame(
        {
            "2025-12-31": [1500.0, 1000.0, 300.0, 5000.0, 2000.0, 3000.0],
            "2024-12-31": [1400.0, 950.0, 280.0, 4800.0, 1900.0, 2900.0],
        },
        index=[
            "Current Assets",
            "Current Liabilities",
            "Inventory",
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Stockholders Equity",
        ],
    )


@pytest.fixture
def income_stmt():
    return pd.DataFrame(
        {
            "2025-12-31": [4000.0, 1600.0, 800.0, 100.0, 500.0],
            "2024-12-31": [3500.0, 1400.0, 700.0, 90.0, 400.0],
        },
        index=["Total Revenue", "Gross Profit", "EBIT", "Interest Expense", "Net Income"],
    )


def test_liquidity_ratios(balance_sheet, income_stmt):
    r = compute_ratios(balance_sheet, income_stmt)
    assert r.current_ratio == pytest.approx(1.5)          # 1500 / 1000
    assert r.quick_ratio == pytest.approx(1.2)            # (1500 - 300) / 1000


def test_leverage_ratios(balance_sheet, income_stmt):
    r = compute_ratios(balance_sheet, income_stmt)
    assert r.debt_to_equity == pytest.approx(0.6667, abs=1e-3)   # 2000 / 3000
    assert r.debt_ratio == pytest.approx(0.4)                    # 2000 / 5000
    assert r.interest_coverage == pytest.approx(8.0)            # 800 / 100


def test_profitability_ratios(balance_sheet, income_stmt):
    r = compute_ratios(balance_sheet, income_stmt)
    assert r.net_profit_margin == pytest.approx(0.125)    # 500 / 4000
    assert r.gross_margin == pytest.approx(0.4)           # 1600 / 4000
    assert r.roe == pytest.approx(0.1667, abs=1e-3)       # 500 / 3000
    assert r.roa == pytest.approx(0.1)                    # 500 / 5000


def test_growth_ratios(balance_sheet, income_stmt):
    r = compute_ratios(balance_sheet, income_stmt)
    assert r.revenue_growth == pytest.approx(0.1429, abs=1e-3)   # (4000-3500)/3500
    assert r.net_income_growth == pytest.approx(0.25)           # (500-400)/400


def test_missing_fields_return_none():
    empty = pd.DataFrame()
    r = compute_ratios(empty, empty)
    assert r.current_ratio is None
    assert r.roe is None


def test_zero_denominator_is_safe():
    bs = pd.DataFrame(
        {"2025": [1000.0, 0.0]},
        index=["Current Assets", "Current Liabilities"],
    )
    r = compute_ratios(bs, pd.DataFrame())
    assert r.current_ratio is None   # division by zero -> None, not a crash
