"""Pydantic schemas — the contract for every structured object in the pipeline.

Using strict schemas (rather than free-form dicts) is what lets the LLM emit a
risk report we can validate, render, and unit-test deterministically.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

    @classmethod
    def _missing_(cls, value):
        """Case-insensitive coercion — LLMs often emit 'LOW'/'High' etc."""
        if isinstance(value, str):
            v = value.strip().lower()
            for member in cls:
                if member.value == v:
                    return member
        return None


class FinancialRatios(BaseModel):
    """Computed from the income statement + balance sheet (latest period)."""

    # Liquidity
    current_ratio: Optional[float] = Field(None, description="Current Assets / Current Liabilities")
    quick_ratio: Optional[float] = Field(
        None, description="(Current Assets - Inventory) / Current Liabilities"
    )
    # Leverage / solvency
    debt_to_equity: Optional[float] = Field(
        None, description="Total Liabilities / Total Equity (DER)"
    )
    debt_ratio: Optional[float] = Field(None, description="Total Liabilities / Total Assets")
    interest_coverage: Optional[float] = Field(None, description="EBIT / Interest Expense")
    # Profitability
    net_profit_margin: Optional[float] = Field(None, description="Net Income / Revenue")
    gross_margin: Optional[float] = Field(None, description="Gross Profit / Revenue")
    roe: Optional[float] = Field(None, description="Net Income / Total Equity")
    roa: Optional[float] = Field(None, description="Net Income / Total Assets")
    # Growth (YoY)
    revenue_growth: Optional[float] = Field(None, description="Revenue YoY growth")
    net_income_growth: Optional[float] = Field(None, description="Net income YoY growth")


class CompanyFinancials(BaseModel):
    """Structured snapshot pulled from yfinance."""

    ticker: str
    company_name: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = Field(None, description="Coarse sector: financial | general")
    industry: Optional[str] = Field(None, description="Raw yfinance industry label")
    period_end: Optional[str] = Field(None, description="Latest annual period end (ISO date)")
    prior_period_end: Optional[str] = Field(
        None, description="Prior annual period end (ISO date) — used for YoY growth label"
    )
    ratios: FinancialRatios = Field(default_factory=FinancialRatios)
    # Raw headline figures (latest period), in reporting currency
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    notes: list[str] = Field(
        default_factory=list, description="Data-quality notes / missing fields"
    )


class RiskFlag(BaseModel):
    """A single red/amber flag identified by the agent."""

    category: str = Field(..., description="e.g. liquidity, leverage, profitability, governance")
    severity: RiskLevel
    finding: str = Field(..., description="What was observed")
    evidence: str = Field(..., description="The metric / quote that supports it")
    source: str = Field(..., description="yfinance | financial_report_pdf | news")


class RiskReport(BaseModel):
    """The agent's final structured deliverable."""

    ticker: str
    company_name: Optional[str] = None
    overall_risk: RiskLevel
    summary: str = Field(..., description="Executive narrative (Bahasa Indonesia)")
    key_ratios: FinancialRatios = Field(default_factory=FinancialRatios)
    flags: list[RiskFlag] = Field(default_factory=list)
    positives: list[str] = Field(default_factory=list, description="Mitigating / positive signals")
    sources: list[str] = Field(default_factory=list, description="All sources consulted")
