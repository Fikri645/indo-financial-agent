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
