"""Evaluators for RiskReport quality.

Each evaluator takes a ``RiskReport`` (and optional context) and returns an
``EvalResult`` with a 0.0–1.0 score plus a human-readable reason. They split into:

  * **Deterministic** checks — schema validity, internal consistency, grounding of
    flags, source coverage. Fast, offline, fully unit-testable.
  * **LLM-as-judge** — groundedness of the narrative summary vs the underlying
    data. The standard technique for evaluating generated text at scale.

Keeping evaluators as pure functions means the eval harness is itself testable
(see tests/test_eval.py) — you can trust the scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.risk import overall_risk
from src.schemas import FinancialRatios, RiskLevel, RiskReport

# Ordering of risk levels for "distance" comparisons.
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MODERATE: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.SEVERE: 3,
}

_RATIO_NAMES = set(FinancialRatios.model_fields.keys())


@dataclass
class EvalResult:
    key: str
    score: float          # 0.0–1.0
    comment: str

    @property
    def passed(self) -> bool:
        return self.score >= 0.5


# --------------------------------------------------------------------------- #
# Deterministic evaluators
# --------------------------------------------------------------------------- #

def eval_schema_valid(report: RiskReport, **_) -> EvalResult:
    """The report must be a well-formed RiskReport with required fields populated."""
    try:
        RiskReport.model_validate(report.model_dump())
        ok = bool(report.ticker) and bool(report.summary) and report.overall_risk
        return EvalResult(
            "schema_valid",
            1.0 if ok else 0.0,
            "Valid RiskReport with ticker/summary/overall_risk"
            if ok else "Missing required fields",
        )
    except Exception as exc:
        return EvalResult("schema_valid", 0.0, f"Validation error: {exc}")


def eval_risk_consistency(report: RiskReport, **_) -> EvalResult:
    """overall_risk should be within one level of the flag-implied risk.

    Recomputes the rule-based aggregate from the report's own flags and compares.
    A 2+ level mismatch (e.g. SEVERE flags but LOW verdict) scores 0.
    """
    implied = overall_risk(report.flags)
    distance = abs(_RISK_ORDER[report.overall_risk] - _RISK_ORDER[implied])
    score = {0: 1.0, 1: 0.6}.get(distance, 0.0)
    return EvalResult(
        "risk_consistency",
        score,
        f"verdict={report.overall_risk.value}, flag-implied={implied.value}, "
        f"distance={distance}",
    )


def eval_flags_grounded(report: RiskReport, **_) -> EvalResult:
    """Every flag's evidence should cite a real ratio name or a number."""
    if not report.flags:
        return EvalResult("flags_grounded", 1.0, "No flags to ground (vacuously true)")
    grounded = 0
    for f in report.flags:
        ev = (f.evidence or "").lower()
        has_number = bool(re.search(r"\d", ev))
        has_ratio = any(name in ev for name in _RATIO_NAMES)
        if has_number or has_ratio:
            grounded += 1
    score = grounded / len(report.flags)
    return EvalResult(
        "flags_grounded",
        score,
        f"{grounded}/{len(report.flags)} flags cite a number or ratio name",
    )


def eval_sources_present(report: RiskReport, **_) -> EvalResult:
    """At least the quantitative source (yfinance) should be cited."""
    srcs = {s.lower() for s in report.sources}
    ok = any("yfinance" in s or "financial" in s for s in srcs)
    return EvalResult(
        "sources_present",
        1.0 if ok else 0.0,
        f"sources={sorted(srcs)}",
    )


def eval_summary_quality(report: RiskReport, **_) -> EvalResult:
    """Summary should be substantive and in Bahasa Indonesia (heuristic)."""
    s = (report.summary or "").strip()
    if len(s) < 50:
        return EvalResult("summary_quality", 0.0, f"Too short ({len(s)} chars)")
    indo_markers = ("risiko", "perusahaan", "keuangan", "dan", "yang", "memiliki")
    hits = sum(1 for w in indo_markers if w in s.lower())
    score = min(1.0, hits / 3)
    return EvalResult(
        "summary_quality",
        score,
        f"{len(s)} chars, {hits} Indonesian markers",
    )


def eval_sector_correct(
    report: RiskReport, financials: Optional[dict] = None,
    expected_sector: Optional[str] = None, **_
) -> EvalResult:
    """If we expect a sector (e.g. bank=financial), the classifier should match,
    and a financial firm must NOT carry a leverage flag from DER."""
    if not expected_sector or not financials:
        return EvalResult("sector_correct", 1.0, "No sector expectation set")
    actual = financials.get("sector")
    if actual != expected_sector:
        return EvalResult(
            "sector_correct", 0.0,
            f"expected sector={expected_sector}, got={actual}",
        )
    if expected_sector == "financial":
        has_lev = any(f.category == "leverage" for f in report.flags)
        return EvalResult(
            "sector_correct",
            0.0 if has_lev else 1.0,
            "Bank wrongly flagged for leverage" if has_lev
            else "Bank correctly not penalised for leverage",
        )
    return EvalResult("sector_correct", 1.0, f"sector={actual} as expected")


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #

_JUDGE_PROMPT = """\
Kamu adalah evaluator. Nilai apakah RINGKASAN risiko di bawah ini DIDUKUNG oleh
DATA yang diberikan (rasio + flags). Jangan menilai gaya bahasa — hanya apakah
klaim dalam ringkasan konsisten dengan data dan tidak mengarang angka.

Beri skor 1 (didukung penuh), 0.5 (sebagian), atau 0 (mengarang/kontradiktif).
Jawab HANYA dalam format: SKOR|alasan singkat
"""


def eval_groundedness_llm(
    report: RiskReport, llm=None, **_
) -> EvalResult:
    """LLM-as-judge: is the narrative summary grounded in the ratios + flags?"""
    if llm is None:
        return EvalResult("groundedness_llm", -1.0, "Skipped (no LLM provided)")
    from langchain_core.messages import HumanMessage, SystemMessage

    data_block = (
        f"RASIO: {report.key_ratios.model_dump()}\n"
        f"FLAGS: {[f.model_dump() for f in report.flags]}\n"
        f"RINGKASAN: {report.summary}"
    )
    try:
        resp = llm.invoke([
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=data_block),
        ])
        text = (resp.content or "").strip()
        m = re.match(r"\s*(1(?:\.0)?|0(?:\.5)?|0(?:\.0)?)\s*\|?\s*(.*)", text)
        if not m:
            return EvalResult("groundedness_llm", 0.5, f"Unparseable judge output: {text[:80]}")
        score = float(m.group(1))
        return EvalResult("groundedness_llm", score, m.group(2)[:120] or "judge")
    except Exception as exc:
        return EvalResult("groundedness_llm", -1.0, f"Judge error: {exc}")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DETERMINISTIC_EVALUATORS: list[Callable[..., EvalResult]] = [
    eval_schema_valid,
    eval_risk_consistency,
    eval_flags_grounded,
    eval_sources_present,
    eval_summary_quality,
    eval_sector_correct,
]

LLM_EVALUATORS: list[Callable[..., EvalResult]] = [
    eval_groundedness_llm,
]


def run_all(
    report: RiskReport,
    financials: Optional[dict] = None,
    expected_sector: Optional[str] = None,
    llm=None,
) -> list[EvalResult]:
    """Run every evaluator and return the results (LLM judge included if llm given)."""
    ctx = {"financials": financials, "expected_sector": expected_sector, "llm": llm}
    results = [ev(report, **ctx) for ev in DETERMINISTIC_EVALUATORS]
    if llm is not None:
        results += [ev(report, **ctx) for ev in LLM_EVALUATORS]
    return results
