"""Tests for the evaluation harness itself.

An eval harness is only trustworthy if its scorers are correct — so we unit-test
the evaluators with crafted RiskReports whose expected scores are obvious.
"""
from unittest.mock import MagicMock

from eval.evaluators import (
    eval_flags_grounded,
    eval_groundedness_llm,
    eval_risk_consistency,
    eval_schema_valid,
    eval_sector_correct,
    eval_sources_present,
    eval_summary_quality,
    run_all,
)
from src.schemas import FinancialRatios, RiskFlag, RiskLevel, RiskReport


def _report(**overrides) -> RiskReport:
    base = dict(
        ticker="BBRI",
        company_name="Bank Rakyat Indonesia",
        overall_risk=RiskLevel.MODERATE,
        summary=(
            "Perusahaan memiliki risiko keuangan moderat dengan leverage yang "
            "terkendali dan pertumbuhan pendapatan yang positif."
        ),
        key_ratios=FinancialRatios(debt_to_equity=0.5, roe=0.15),
        flags=[],
        positives=["ROE sehat"],
        sources=["yfinance"],
    )
    base.update(overrides)
    return RiskReport(**base)


# --- schema_valid ---------------------------------------------------------- #

def test_schema_valid_passes_for_good_report():
    assert eval_schema_valid(_report()).score == 1.0


def test_schema_valid_fails_for_empty_summary():
    assert eval_schema_valid(_report(summary="")).score == 0.0


# --- risk_consistency ------------------------------------------------------ #

def test_risk_consistency_perfect_when_aligned():
    # No flags → implied LOW; verdict LOW → distance 0 → score 1.0
    r = _report(overall_risk=RiskLevel.LOW, flags=[])
    assert eval_risk_consistency(r).score == 1.0


def test_risk_consistency_penalises_large_mismatch():
    # SEVERE flags but verdict LOW → distance 3 → score 0
    severe_flags = [
        RiskFlag(category="leverage", severity=RiskLevel.SEVERE,
                 finding="x", evidence="debt_to_equity = 9", source="yfinance"),
        RiskFlag(category="liquidity", severity=RiskLevel.SEVERE,
                 finding="y", evidence="current_ratio = 0.2", source="yfinance"),
    ]
    r = _report(overall_risk=RiskLevel.LOW, flags=severe_flags)
    assert eval_risk_consistency(r).score == 0.0


# --- flags_grounded -------------------------------------------------------- #

def test_flags_grounded_all_cite_numbers():
    flags = [
        RiskFlag(category="leverage", severity=RiskLevel.HIGH,
                 finding="DER tinggi", evidence="debt_to_equity = 5.6", source="yfinance"),
    ]
    assert eval_flags_grounded(_report(flags=flags)).score == 1.0


def test_flags_grounded_penalises_ungrounded():
    flags = [
        RiskFlag(category="governance", severity=RiskLevel.HIGH,
                 finding="buruk", evidence="kata-kata tanpa angka", source="news"),
    ]
    assert eval_flags_grounded(_report(flags=flags)).score == 0.0


def test_flags_grounded_vacuously_true_when_no_flags():
    assert eval_flags_grounded(_report(flags=[])).score == 1.0


# --- sources_present ------------------------------------------------------- #

def test_sources_present_requires_yfinance():
    assert eval_sources_present(_report(sources=["yfinance", "news"])).score == 1.0
    assert eval_sources_present(_report(sources=["news"])).score == 0.0


# --- summary_quality ------------------------------------------------------- #

def test_summary_quality_rewards_indonesian_substance():
    assert eval_summary_quality(_report()).score >= 0.5


def test_summary_quality_penalises_too_short():
    assert eval_summary_quality(_report(summary="Singkat.")).score == 0.0


# --- sector_correct -------------------------------------------------------- #

def test_sector_correct_bank_without_leverage_flag():
    fin = {"sector": "financial"}
    r = _report(flags=[])
    res = eval_sector_correct(r, financials=fin, expected_sector="financial")
    assert res.score == 1.0


def test_sector_correct_bank_with_leverage_flag_fails():
    fin = {"sector": "financial"}
    flags = [RiskFlag(category="leverage", severity=RiskLevel.HIGH,
                      finding="DER", evidence="debt_to_equity = 6", source="yfinance")]
    res = eval_sector_correct(_report(flags=flags), financials=fin,
                              expected_sector="financial")
    assert res.score == 0.0


def test_sector_correct_misclassification_fails():
    fin = {"sector": "general"}
    res = eval_sector_correct(_report(), financials=fin, expected_sector="financial")
    assert res.score == 0.0


def test_sector_correct_skipped_without_expectation():
    assert eval_sector_correct(_report()).score == 1.0


# --- groundedness_llm (mocked judge) --------------------------------------- #

def test_groundedness_llm_parses_judge_score():
    judge = MagicMock()
    judge.invoke.return_value = MagicMock(content="1|ringkasan didukung penuh oleh data")
    res = eval_groundedness_llm(_report(), llm=judge)
    assert res.score == 1.0


def test_groundedness_llm_handles_partial_score():
    judge = MagicMock()
    judge.invoke.return_value = MagicMock(content="0.5|sebagian klaim tidak didukung")
    res = eval_groundedness_llm(_report(), llm=judge)
    assert res.score == 0.5


def test_groundedness_llm_skipped_without_llm():
    res = eval_groundedness_llm(_report(), llm=None)
    assert res.score == -1.0  # sentinel = skipped


# --- run_all integration --------------------------------------------------- #

def test_run_all_deterministic_only():
    results = run_all(_report(), financials={"sector": "financial"},
                      expected_sector="financial", llm=None)
    keys = {r.key for r in results}
    assert "schema_valid" in keys
    assert "groundedness_llm" not in keys  # judge skipped when llm=None
    assert all(r.score >= 0.5 for r in results)


def test_run_all_includes_judge_when_llm_given():
    judge = MagicMock()
    judge.invoke.return_value = MagicMock(content="1|ok")
    results = run_all(_report(), llm=judge)
    assert any(r.key == "groundedness_llm" for r in results)
