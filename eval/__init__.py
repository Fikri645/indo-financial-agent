"""Evaluation harness for the Indonesian Financial Research Agent.

Two layers:
  - evaluators.py : deterministic + LLM-as-judge scorers (pure, unit-tested)
  - dataset.py    : the evaluation cases (tickers + expectations)
  - run_eval.py   : runs the graph over the dataset, prints a scorecard, and can
                    optionally push results to LangSmith.
"""
