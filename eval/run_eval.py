"""Run the evaluation harness over the dataset and print a scorecard.

Usage
-----
    # Full eval (runs the live graph on each ticker, applies all evaluators):
    python -m eval.run_eval

    # Quick: skip the LLM-as-judge evaluator (deterministic checks only):
    python -m eval.run_eval --no-judge

    # Limit cases (faster smoke):
    python -m eval.run_eval --limit 2

    # Also log the run to LangSmith (requires LANGCHAIN_API_KEY):
    python -m eval.run_eval --langsmith

Results are written to ``reports/eval_results.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from eval.dataset import EVAL_CASES, EvalCase
from eval.evaluators import EvalResult, run_all
from src import config
from src.schemas import RiskReport

logger = logging.getLogger(__name__)


def _run_graph(case: EvalCase) -> tuple[RiskReport | None, dict | None]:
    """Run the full agent graph for one ticker; return (report, financials)."""
    from langchain_core.messages import HumanMessage

    from src.agents.graph import graph

    initial = {
        "messages": [HumanMessage(content=f"Analisis risiko {case.ticker}")],
        "ticker": case.ticker,
        "pdf_path": None,
        "financials": None,
        "doc_chunks": None,
        "news_headlines": None,
        "risk_report": None,
        "next": "",
    }
    final = graph.invoke(initial, {"recursion_limit": 25})
    report_dict = final.get("risk_report")
    report = RiskReport.model_validate(report_dict) if report_dict else None
    return report, final.get("financials")


def _scorecard(rows: list[dict]) -> str:
    """Render an aggregate scorecard across all cases and evaluators."""
    if not rows:
        return "No results."
    # Collect evaluator keys
    keys: list[str] = []
    for r in rows:
        for er in r["results"]:
            if er["key"] not in keys:
                keys.append(er["key"])

    lines = ["", "=" * 72, "  EVALUATION SCORECARD", "=" * 72]
    header = f"  {'ticker':<8}" + "".join(f"{k[:14]:>16}" for k in keys)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    per_key_totals = {k: [] for k in keys}
    for r in rows:
        rmap = {er["key"]: er["score"] for er in r["results"]}
        cells = ""
        for k in keys:
            v = rmap.get(k)
            if v is None or v < 0:
                cells += f"{'—':>16}"
            else:
                cells += f"{v:>16.2f}"
                per_key_totals[k].append(v)
        lines.append(f"  {r['ticker']:<8}{cells}")

    lines.append("  " + "-" * (len(header) - 2))
    avg_cells = ""
    for k in keys:
        vals = per_key_totals[k]
        avg_cells += f"{(sum(vals)/len(vals)):>16.2f}" if vals else f"{'—':>16}"
    lines.append(f"  {'AVG':<8}{avg_cells}")

    # Overall mean of all scored cells
    all_scores = [v for vals in per_key_totals.values() for v in vals]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    lines.append("=" * 72)
    lines.append(f"  OVERALL MEAN SCORE: {overall:.3f}  ({len(rows)} cases)")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval harness.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge groundedness evaluator.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate the first N cases.")
    parser.add_argument("--langsmith", action="store_true",
                        help="Also log this eval run to LangSmith.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    cases = EVAL_CASES[: args.limit] if args.limit else EVAL_CASES

    # Optional LLM judge
    judge_llm = None
    if not args.no_judge:
        try:
            from src.agents.llm import get_llm
            judge_llm = get_llm(temperature=0.0)
        except Exception as exc:
            logger.warning("Could not init judge LLM (%s) — running deterministic only", exc)

    if args.langsmith:
        config_msg = "ON" if config.GROQ_API_KEY else "no LLM key!"
        print(f"[LangSmith logging requested — project={config.LLM_PROVIDER}, LLM={config_msg}]")

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Evaluating {case.ticker} ({case.name})...")
        try:
            report, financials = _run_graph(case)
        except Exception as exc:
            print(f"   ! graph failed: {exc}")
            rows.append({
                "ticker": case.ticker,
                "results": [EvalResult("graph_run", 0.0, str(exc)).__dict__],
            })
            continue

        if report is None:
            print("   ! no report produced")
            rows.append({
                "ticker": case.ticker,
                "results": [EvalResult("graph_run", 0.0, "No report").__dict__],
            })
            continue

        results = run_all(
            report,
            financials=financials,
            expected_sector=case.expected_sector,
            llm=judge_llm,
        )
        for er in results:
            mark = "✓" if er.passed else ("·" if er.score < 0 else "✗")
            print(f"   {mark} {er.key:<18} {er.score:>5.2f}  {er.comment}")
        rows.append({"ticker": case.ticker, "results": [er.__dict__ for er in results]})

    print(_scorecard(rows))

    # Persist
    out = config.REPORTS_DIR / "eval_results.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(rows),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out}")

    if args.langsmith:
        _log_to_langsmith(rows)


def _log_to_langsmith(rows: list[dict]) -> None:
    """Best-effort: record aggregate scores to LangSmith as a dataset run."""
    try:
        from langsmith import Client
        client = Client()
        ds_name = "indo-financial-agent-eval"
        # Create dataset if missing
        try:
            ds = client.create_dataset(ds_name, description="Agent eval cases")
        except Exception:
            ds = client.read_dataset(dataset_name=ds_name)
        for r in rows:
            scores = {er["key"]: er["score"] for er in r["results"]}
            client.create_example(
                inputs={"ticker": r["ticker"]},
                outputs={"scores": scores},
                dataset_id=ds.id,
            )
        print(f"[LangSmith] logged {len(rows)} examples to dataset '{ds_name}'")
    except Exception as exc:
        print(f"[LangSmith] logging failed: {exc}")


if __name__ == "__main__":
    main()
