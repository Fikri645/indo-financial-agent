"""Structured financial-data tool — pulls IDX company fundamentals via yfinance
and derives the ratios a credit/risk analyst actually looks at.

This is the *quantitative* leg of the agent. The PDF tool (document.py) provides
the qualitative leg, and news.py provides market context.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src import config
from src.schemas import CompanyFinancials, FinancialRatios


# yfinance row labels are not perfectly stable; we try a list of aliases per item.
_BS_ALIASES = {
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "total_assets": ["Total Assets"],
    "total_liabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    ],
    "total_equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ],
}

_IS_ALIASES = {
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebit": ["EBIT", "Operating Income"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}


def _first_row(df: Optional[pd.DataFrame], aliases: list[str], col_idx: int = 0) -> Optional[float]:
    """Return the value for the first matching row label at ``col_idx`` (newest period)."""
    if df is None or df.empty or col_idx >= df.shape[1]:
        return None
    for label in aliases:
        if label in df.index:
            val = df.iloc[df.index.get_loc(label), col_idx]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
    return None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 4)


def _growth(df: Optional[pd.DataFrame], aliases: list[str]) -> Optional[float]:
    """YoY growth using the two newest columns (col 0 = latest, col 1 = prior)."""
    latest = _first_row(df, aliases, 0)
    prior = _first_row(df, aliases, 1)
    if latest is None or prior is None or prior == 0:
        return None
    return round((latest - prior) / abs(prior), 4)


def compute_ratios(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame) -> FinancialRatios:
    """Derive the standard liquidity / leverage / profitability / growth ratios."""
    # Balance-sheet items (latest period)
    cur_assets = _first_row(balance_sheet, _BS_ALIASES["current_assets"])
    cur_liab = _first_row(balance_sheet, _BS_ALIASES["current_liabilities"])
    inventory = _first_row(balance_sheet, _BS_ALIASES["inventory"]) or 0.0
    total_assets = _first_row(balance_sheet, _BS_ALIASES["total_assets"])
    total_liab = _first_row(balance_sheet, _BS_ALIASES["total_liabilities"])
    total_equity = _first_row(balance_sheet, _BS_ALIASES["total_equity"])

    # Income-statement items (latest period)
    revenue = _first_row(income_stmt, _IS_ALIASES["total_revenue"])
    gross = _first_row(income_stmt, _IS_ALIASES["gross_profit"])
    ebit = _first_row(income_stmt, _IS_ALIASES["ebit"])
    interest = _first_row(income_stmt, _IS_ALIASES["interest_expense"])
    net_income = _first_row(income_stmt, _IS_ALIASES["net_income"])

    quick_assets = (cur_assets - inventory) if cur_assets is not None else None

    return FinancialRatios(
        current_ratio=_safe_div(cur_assets, cur_liab),
        quick_ratio=_safe_div(quick_assets, cur_liab),
        debt_to_equity=_safe_div(total_liab, total_equity),
        debt_ratio=_safe_div(total_liab, total_assets),
        interest_coverage=_safe_div(ebit, abs(interest) if interest else None),
        net_profit_margin=_safe_div(net_income, revenue),
        gross_margin=_safe_div(gross, revenue),
        roe=_safe_div(net_income, total_equity),
        roa=_safe_div(net_income, total_assets),
        revenue_growth=_growth(income_stmt, _IS_ALIASES["total_revenue"]),
        net_income_growth=_growth(income_stmt, _IS_ALIASES["net_income"]),
    )


def fetch_financials(ticker: str) -> CompanyFinancials:
    """Fetch fundamentals for an IDX-listed company and compute risk ratios.

    Parameters
    ----------
    ticker : str
        IDX ticker, with or without the ``.JK`` suffix (e.g. ``BBRI`` or ``BBRI.JK``).
    """
    import yfinance as yf  # local import: keeps module import cheap/testable

    symbol = config.normalize_ticker(ticker)
    tk = yf.Ticker(symbol)

    notes: list[str] = []
    try:
        info = tk.info or {}
    except Exception:  # pragma: no cover - network dependent
        info = {}
        notes.append("Could not fetch company info from Yahoo Finance.")

    balance_sheet = _safe_statement(tk, "balance_sheet", notes)
    income_stmt = _safe_statement(tk, "income_stmt", notes)

    ratios = compute_ratios(balance_sheet, income_stmt)

    period_end = None
    if balance_sheet is not None and not balance_sheet.empty:
        period_end = str(balance_sheet.columns[0].date()) if hasattr(
            balance_sheet.columns[0], "date"
        ) else str(balance_sheet.columns[0])

    return CompanyFinancials(
        ticker=symbol,
        company_name=info.get("longName") or info.get("shortName"),
        currency=info.get("financialCurrency") or info.get("currency"),
        period_end=period_end,
        ratios=ratios,
        total_revenue=_first_row(income_stmt, _IS_ALIASES["total_revenue"]),
        net_income=_first_row(income_stmt, _IS_ALIASES["net_income"]),
        total_assets=_first_row(balance_sheet, _BS_ALIASES["total_assets"]),
        total_liabilities=_first_row(balance_sheet, _BS_ALIASES["total_liabilities"]),
        total_equity=_first_row(balance_sheet, _BS_ALIASES["total_equity"]),
        notes=notes,
    )


def _safe_statement(tk, attr: str, notes: list[str]) -> Optional[pd.DataFrame]:
    try:
        df = getattr(tk, attr)
        if df is None or df.empty:
            notes.append(f"{attr} unavailable for this ticker.")
            return None
        return df
    except Exception:  # pragma: no cover - network dependent
        notes.append(f"Failed to fetch {attr}.")
        return None


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "BBRI"
    data = fetch_financials(sym)
    print(json.dumps(data.model_dump(), indent=2, ensure_ascii=False))
