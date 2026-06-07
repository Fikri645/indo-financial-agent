"""Smoke-test the data layer end-to-end: fetch fundamentals + rule-based risk.

Usage:
    python scripts/fetch_company.py BBRI
    python scripts/fetch_company.py TLKM.JK
"""
import json
import sys

from src.risk import evaluate_ratios, overall_risk
from src.tools.financial_data import fetch_financials


def main(ticker: str) -> None:
    print(f"Fetching fundamentals for {ticker} ...")
    fin = fetch_financials(ticker)

    flags = evaluate_ratios(fin.ratios)
    level = overall_risk(flags)

    print("\n=== COMPANY ===")
    print(json.dumps(fin.model_dump(), indent=2, ensure_ascii=False))

    print("\n=== RULE-BASED RISK ===")
    print(f"Overall (quantitative only): {level.value.upper()}")
    for f in flags:
        print(f"  [{f.severity.value}] {f.category}: {f.finding}  ({f.evidence})")
    if not flags:
        print("  No quantitative red flags from ratio thresholds.")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BBRI"
    main(sym)
