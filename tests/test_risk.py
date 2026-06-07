"""Unit tests for the deterministic rule-based risk scorer."""
from src.risk import evaluate_ratios, overall_risk
from src.schemas import FinancialRatios, RiskLevel


def test_healthy_company_has_no_flags():
    ratios = FinancialRatios(
        current_ratio=2.5,
        quick_ratio=1.8,
        debt_to_equity=0.4,
        interest_coverage=10.0,
        net_profit_margin=0.15,
        roe=0.18,
        revenue_growth=0.12,
        net_income_growth=0.10,
    )
    flags = evaluate_ratios(ratios)
    assert flags == []
    assert overall_risk(flags) == RiskLevel.LOW


def test_insolvent_company_flags_high():
    ratios = FinancialRatios(
        current_ratio=0.7,
        debt_to_equity=3.5,
        interest_coverage=1.0,
        net_profit_margin=-0.05,
        roe=-0.08,
    )
    flags = evaluate_ratios(ratios)
    categories = {f.category for f in flags}
    assert "liquidity" in categories
    assert "leverage" in categories
    assert "profitability" in categories
    assert overall_risk(flags) == RiskLevel.SEVERE


def test_only_most_severe_flag_per_metric():
    # current_ratio = 0.7 trips both the <1.0 (HIGH) and <1.5 (MODERATE) rules;
    # only the HIGH one should be kept.
    ratios = FinancialRatios(current_ratio=0.7)
    flags = [f for f in evaluate_ratios(ratios) if f.category == "liquidity"]
    assert len(flags) == 1
    assert flags[0].severity == RiskLevel.HIGH


def test_moderate_leverage_only():
    ratios = FinancialRatios(debt_to_equity=1.5, interest_coverage=4.0)
    flags = evaluate_ratios(ratios)
    assert overall_risk(flags) == RiskLevel.MODERATE


def test_none_ratios_produce_no_flags():
    assert evaluate_ratios(FinancialRatios()) == []


def test_bank_high_der_not_flagged_as_leverage():
    """A bank with DER 5.6 (normal for banks) must NOT be flagged for leverage."""
    ratios = FinancialRatios(
        debt_to_equity=5.6,   # would be HIGH for a general corporate
        net_profit_margin=0.30,
        roe=0.17,
        revenue_growth=0.08,
    )
    general_flags = evaluate_ratios(ratios, sector="general")
    financial_flags = evaluate_ratios(ratios, sector="financial")

    # General corporate: DER 5.6 trips the HIGH leverage rule.
    assert any(f.category == "leverage" for f in general_flags)
    # Bank: leverage rule is suppressed → no leverage flag.
    assert not any(f.category == "leverage" for f in financial_flags)
    assert overall_risk(financial_flags) == RiskLevel.LOW


def test_bank_still_flagged_for_losses():
    """Profitability rules still apply to banks (sector-agnostic)."""
    ratios = FinancialRatios(debt_to_equity=6.0, net_profit_margin=-0.05, roe=-0.03)
    flags = evaluate_ratios(ratios, sector="financial")
    categories = {f.category for f in flags}
    assert "profitability" in categories
    assert "leverage" not in categories  # bank leverage still suppressed


def test_general_sector_is_default():
    """Calling without a sector keeps the original (general) behaviour."""
    ratios = FinancialRatios(debt_to_equity=3.0)
    assert evaluate_ratios(ratios) == evaluate_ratios(ratios, sector="general")
