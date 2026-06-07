"""Deterministic, rule-based risk scoring from financial ratios.

This runs *before* the LLM and gives the agent a defensible quantitative baseline.
The LLM (Risk Analyst node) then layers qualitative findings from PDF + news on top.
Keeping a rule-based core means the numeric verdict is reproducible and unit-testable.

Sector awareness
----------------
Liquidity and leverage thresholds tuned for industrials are misleading for
financial firms (banks/insurers): customer deposits make a bank's debt-to-equity
naturally high, and banks do not report a meaningful current ratio. Each rule is
therefore tagged with the sectors it applies to, so a bank is not penalised for a
balance sheet that is normal for its business model.
"""
from __future__ import annotations

from typing import Optional

from src.schemas import FinancialRatios, RiskFlag, RiskLevel

# Sector tags. "general" = non-financial corporates; "financial" = banks/insurers.
_ALL = ("general", "financial")
_GENERAL = ("general",)

# Each rule: (attr, comparison, threshold, category, severity, finding, sectors)
# A rule fires only when the company's sector is in its ``sectors`` tuple.
_RULES = [
    # --- Liquidity (industrials only — banks have no meaningful current ratio) --
    ("current_ratio", "<", 1.0, "liquidity", RiskLevel.HIGH,
     "Current ratio below 1.0 — short-term liabilities exceed current assets", _GENERAL),
    ("current_ratio", "<", 1.5, "liquidity", RiskLevel.MODERATE,
     "Current ratio below 1.5 — thin liquidity cushion", _GENERAL),
    ("quick_ratio", "<", 1.0, "liquidity", RiskLevel.MODERATE,
     "Quick ratio below 1.0 — reliant on inventory to cover current liabilities", _GENERAL),
    # --- Leverage (DER thresholds are industrial-only; banks excluded) ----------
    ("debt_to_equity", ">", 2.0, "leverage", RiskLevel.HIGH,
     "Debt-to-equity above 2.0 — heavily leveraged capital structure", _GENERAL),
    ("debt_to_equity", ">", 1.0, "leverage", RiskLevel.MODERATE,
     "Debt-to-equity above 1.0 — meaningful leverage", _GENERAL),
    ("interest_coverage", "<", 1.5, "leverage", RiskLevel.HIGH,
     "Interest coverage below 1.5x — earnings barely cover interest", _GENERAL),
    ("interest_coverage", "<", 3.0, "leverage", RiskLevel.MODERATE,
     "Interest coverage below 3.0x — limited buffer on debt servicing", _GENERAL),
    # --- Profitability & growth (apply to ALL sectors) -------------------------
    ("net_profit_margin", "<", 0.0, "profitability", RiskLevel.HIGH,
     "Negative net margin — the company is loss-making", _ALL),
    ("roe", "<", 0.0, "profitability", RiskLevel.HIGH,
     "Negative ROE — eroding shareholder equity", _ALL),
    ("revenue_growth", "<", -0.10, "growth", RiskLevel.MODERATE,
     "Revenue contracted more than 10% YoY", _ALL),
    ("net_income_growth", "<", -0.25, "growth", RiskLevel.MODERATE,
     "Net income fell more than 25% YoY", _ALL),
]

_SEVERITY_WEIGHT = {
    RiskLevel.LOW: 0,
    RiskLevel.MODERATE: 1,
    RiskLevel.HIGH: 3,
    RiskLevel.SEVERE: 5,
}


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def evaluate_ratios(
    ratios: FinancialRatios, sector: Optional[str] = "general"
) -> list[RiskFlag]:
    """Apply threshold rules to ratios, returning the triggered flags.

    Only the most severe flag per ratio is kept (avoids double-counting the
    moderate + high rule for the same metric). Rules that do not apply to the
    company's ``sector`` (e.g. DER/liquidity rules for a bank) are skipped.
    """
    sector = sector or "general"
    flags: list[RiskFlag] = []
    fired_attrs: set[str] = set()

    # Sort rules so HIGH fires before MODERATE for the same attribute.
    ordered = sorted(_RULES, key=lambda r: -_SEVERITY_WEIGHT[r[4]])
    for attr, op, threshold, category, severity, finding, sectors in ordered:
        if sector not in sectors:
            continue
        if attr in fired_attrs:
            continue
        value = getattr(ratios, attr, None)
        if value is None:
            continue
        triggered = (value < threshold) if op == "<" else (value > threshold)
        if triggered:
            flags.append(
                RiskFlag(
                    category=category,
                    severity=severity,
                    finding=finding,
                    evidence=f"{attr} = {_fmt(value)}",
                    source="yfinance",
                )
            )
            fired_attrs.add(attr)
    return flags


def overall_risk(flags: list[RiskFlag]) -> RiskLevel:
    """Aggregate individual flags into an overall risk level via weighted score."""
    score = sum(_SEVERITY_WEIGHT[f.severity] for f in flags)
    if score == 0:
        return RiskLevel.LOW
    if score <= 2:
        return RiskLevel.MODERATE
    if score <= 6:
        return RiskLevel.HIGH
    return RiskLevel.SEVERE
