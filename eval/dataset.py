"""Evaluation dataset — IDX tickers with known characteristics.

These are not ground-truth risk labels (which would need analyst consensus);
they encode *checkable expectations* — sector classification and, for banks, the
requirement that a naturally-high DER is not mis-flagged. The evaluators turn
these into scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalCase:
    ticker: str
    name: str
    expected_sector: Optional[str] = None   # "financial" | "general" | None
    note: str = ""


# A small, diverse slice of IDX: two banks (financial), three non-financials.
EVAL_CASES: list[EvalCase] = [
    EvalCase("BBRI", "Bank Rakyat Indonesia", "financial",
             "Large state bank — high DER is normal, must not flag leverage"),
    EvalCase("BBCA", "Bank Central Asia", "financial",
             "Private bank — same sector expectation"),
    EvalCase("TLKM", "Telkom Indonesia", "general",
             "Telco — standard corporate balance sheet"),
    EvalCase("ASII", "Astra International", "general",
             "Conglomerate — standard corporate"),
    EvalCase("UNVR", "Unilever Indonesia", "general",
             "Consumer goods — typically strong margins, can have high DER"),
]
