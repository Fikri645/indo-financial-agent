"""Structured financial-data tool — pulls IDX company fundamentals via yfinance
and derives the ratios a credit/risk analyst actually looks at.

This is the *quantitative* leg of the agent. The PDF tool (document.py) provides
the qualitative leg, and news.py provides market context.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import pandas as pd

from src import config
from src.schemas import CompanyFinancials, FinancialRatios

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache for fetched financials (avoids re-hitting
# yfinance on repeated analysis of the same ticker within a session).
# ---------------------------------------------------------------------------
_CACHE: dict[str, Tuple[CompanyFinancials, float]] = {}
_CACHE_TTL: float = 600.0  # 10 minutes


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


# yfinance sector/industry labels that mean "balance sheet works differently"
# (banks, insurers, capital markets). For these, liquidity/leverage thresholds
# tuned for industrials produce misleading flags (e.g. a bank's DER is naturally
# high because customer deposits are liabilities).
_FINANCIAL_KEYWORDS = (
    "bank", "insurance", "asuransi", "financial", "capital markets",
    "credit", "mortgage", "asset management", "diversified financ",
)


def classify_sector(sector: Optional[str], industry: Optional[str]) -> str:
    """Collapse yfinance sector/industry into 'financial' or 'general'."""
    blob = f"{sector or ''} {industry or ''}".lower()
    if any(kw in blob for kw in _FINANCIAL_KEYWORDS):
        return "financial"
    return "general"


def _growth(
    df: Optional[pd.DataFrame], aliases: list[str], prior_col: int = 1
) -> Optional[float]:
    """YoY growth between col 0 (latest) and ``prior_col``.

    Annual:    prior_col=1  (prior fiscal year)
    Quarterly: prior_col=4  (same quarter prior year — avoids seasonal noise)
    """
    latest = _first_row(df, aliases, 0)
    prior = _first_row(df, aliases, prior_col)
    if latest is None or prior is None or prior == 0:
        return None
    return round((latest - prior) / abs(prior), 4)


def compute_ratios(
    balance_sheet: pd.DataFrame,
    income_stmt: pd.DataFrame,
    use_quarterly: bool = False,
) -> FinancialRatios:
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
    # Quarterly: compare same quarter prior year (col 4) to avoid seasonal noise.
    prior_col = 4 if use_quarterly else 1

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
        revenue_growth=_growth(income_stmt, _IS_ALIASES["total_revenue"], prior_col),
        net_income_growth=_growth(income_stmt, _IS_ALIASES["net_income"], prior_col),
    )


def fetch_financials(ticker: str, use_quarterly: bool = False) -> CompanyFinancials:
    """Fetch fundamentals for an IDX-listed company and compute risk ratios.

    Results are cached in-process for ``_CACHE_TTL`` seconds to avoid
    redundant yfinance round-trips when the same ticker is analysed repeatedly.

    Parameters
    ----------
    ticker : str
        IDX ticker, with or without the ``.JK`` suffix (e.g. ``BBRI`` or ``BBRI.JK``).
    use_quarterly : bool
        When True, use the most recent quarterly statements instead of annual.
        Growth ratios compare the same quarter in the prior year (YoY, not QoQ).
    """
    import yfinance as yf  # local import: keeps module import cheap/testable

    symbol = config.normalize_ticker(ticker)
    cache_key = f"{symbol}:{'q' if use_quarterly else 'a'}"
    now = time.monotonic()
    if cache_key in _CACHE:
        cached, ts = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return cached

    tk = yf.Ticker(symbol)

    notes: list[str] = []
    try:
        info = tk.info or {}
    except Exception:  # pragma: no cover - network dependent
        info = {}
        notes.append("Could not fetch company info from Yahoo Finance.")

    if use_quarterly:
        balance_sheet = _safe_statement(tk, "quarterly_balance_sheet", notes)
        income_stmt = _safe_statement(tk, "quarterly_income_stmt", notes)
    else:
        balance_sheet = _safe_statement(tk, "balance_sheet", notes)
        income_stmt = _safe_statement(tk, "income_stmt", notes)

    ratios = compute_ratios(balance_sheet, income_stmt, use_quarterly)

    # period_end  = latest period (col 0)
    # prior_period_end = comparison period for growth (col 1 annual, col 4 quarterly)
    period_end = None
    prior_period_end = None
    prior_col = 4 if use_quarterly else 1
    if balance_sheet is not None and not balance_sheet.empty:
        period_end = str(balance_sheet.columns[0].date()) if hasattr(
            balance_sheet.columns[0], "date"
        ) else str(balance_sheet.columns[0])
        if balance_sheet.shape[1] > prior_col:
            prior_period_end = str(balance_sheet.columns[prior_col].date()) if hasattr(
                balance_sheet.columns[prior_col], "date"
            ) else str(balance_sheet.columns[prior_col])

    raw_sector = info.get("sector")
    raw_industry = info.get("industry")

    result = CompanyFinancials(
        ticker=symbol,
        company_name=info.get("longName") or info.get("shortName"),
        currency=info.get("financialCurrency") or info.get("currency"),
        sector=classify_sector(raw_sector, raw_industry),
        industry=raw_industry,
        period_end=period_end,
        prior_period_end=prior_period_end,
        ratios=ratios,
        total_revenue=_first_row(income_stmt, _IS_ALIASES["total_revenue"]),
        net_income=_first_row(income_stmt, _IS_ALIASES["net_income"]),
        total_assets=_first_row(balance_sheet, _BS_ALIASES["total_assets"]),
        total_liabilities=_first_row(balance_sheet, _BS_ALIASES["total_liabilities"]),
        total_equity=_first_row(balance_sheet, _BS_ALIASES["total_equity"]),
        notes=notes,
    )
    _CACHE[cache_key] = (result, time.monotonic())
    return result


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
