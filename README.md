# 🇮🇩 Indonesian Financial Research Agent

[![CI](https://github.com/Fikri645/indo-financial-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Fikri645/indo-financial-agent/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/fikri0o0/indo-financial-agent)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

> A **LangGraph multi-agent system** that researches an IDX-listed company end-to-end —
> pulling structured fundamentals, parsing its financial-report PDF, scanning recent
> news — and produces a **structured risk assessment** in Bahasa Indonesia.

This is an *agentic* system, not a chatbot: a supervisor delegates to specialist
worker agents, each of which autonomously selects and calls tools, and a final
analyst node synthesizes everything into a validated `RiskReport`.

---

## 🎯 What it does

Give it a ticker (e.g. `BBRI`) and it will:

1. **Fetch fundamentals** — income statement & balance sheet via `yfinance` (`.JK`),
   then compute liquidity / leverage / profitability / growth ratios.
2. **Read the report** — parse the annual-report / financial-statement PDF with
   **Docling** (table-aware) and retrieve analyst-relevant sections (going concern,
   related-party transactions, risk factors) via hybrid search.
3. **Scan the news** — recent Indonesian market coverage (free DuckDuckGo, or Tavily).
4. **Assess risk** — a deterministic rule-based scorer establishes the quantitative
   baseline, then the LLM layers qualitative findings into a structured, sourced report.

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
   "Analisis    ──► │   SUPERVISOR    │  routing + final synthesis
    risiko BBRI"    └────────┬────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
    │ FINANCIAL     │ │ DOCUMENT    │ │ NEWS         │
    │ DATA AGENT    │ │ AGENT       │ │ AGENT        │
    │ yfinance +    │ │ Docling +   │ │ web search   │
    │ ratio engine  │ │ hybrid RAG  │ │ (sentiment)  │
    └───────────────┘ └─────────────┘ └──────────────┘
            └────────────────┼────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  RISK ANALYST   │  rules + LLM → RiskReport (Pydantic)
                    └─────────────────┘
```

---

## 🧱 Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | **LangGraph** (supervisor multi-agent) |
| LLM | Free-tier: **Groq** (Llama 3.3 70B) / **Gemini 2.0 Flash** / **OpenRouter** |
| Structured data | **yfinance** (IDX `.JK`) |
| PDF parsing | **Docling** (TableFormer, 97.9% table accuracy) |
| Retrieval | **ChromaDB** + multilingual-e5 + BM25 hybrid (RRF) → **BGE cross-encoder rerank** |
| Output contract | **Pydantic** schemas |
| UI | **Gradio** on Hugging Face Spaces |

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add at least one free LLM key (Groq recommended)

# Data layer (works today):
make fetch TICKER=BBRI        # fundamentals + rule-based risk
make ingest PDF=data/pdfs/BBRI_2024.pdf   # parse a report PDF

# Full agent (in progress):
make agent TICKER=BBRI
make app                      # Gradio UI
```

Get free PDFs from IDX:
[idx.co.id › financial statements & annual report](https://www.idx.co.id/en/listed-companies/financial-statements-and-annual-report).

---

## 📊 Project status

- [x] **Data layer** — yfinance fundamentals + ratio engine + Docling PDF parsing +
      hybrid retrieval + rule-based risk scorer
- [x] **Agent nodes** — financial / document / news worker agents, deterministic supervisor;
      parallel financial + news fetch via `ThreadPoolExecutor`
- [x] **Risk Analyst synthesis node** — rule-based scorer + LLM structured output → `RiskReport`
      with Pydantic validation and graceful LLM-failure fallback
- [x] **Gradio UI** — streaming progress, 4-tab report layout (summary / ratios / flags / sources),
      quarterly toggle, Markdown export, PDF upload, Enter-to-submit
- [x] **Tool-calling News Agent** — genuine ReAct agent (`create_react_agent`): the LLM
      composes its own queries, calls the search tool repeatedly, refines on empty results,
      then synthesises headlines + sentiment (deterministic fallback if the agent path fails)
- [x] **LangSmith tracing** — automatic via `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`
      (LangGraph traces every run; no code changes required)
- [x] **Eval harness** — deterministic scorers (schema, risk-consistency, flag-grounding,
      sector-correctness) + **LLM-as-judge** groundedness, run over a labelled IDX dataset
      with an aggregate scorecard (`python -m eval.run_eval`)
- [x] **Hugging Face Spaces** — live at [fikri0o0/indo-financial-agent](https://huggingface.co/spaces/fikri0o0/indo-financial-agent)

---

## ⚠️ Limitations

- Ratio thresholds are tuned for **non-financial corporates**; banks & insurers
  (different balance-sheet structure) need sector-specific rules.
- `yfinance` data depth varies by ticker; missing fields degrade gracefully to `None`.
- **Not investment advice** — a portfolio/engineering demonstration only.

---

## 📏 Evaluation

The agent is graded, not just demoed. `eval/` runs the full graph over a labelled
slice of IDX tickers and scores each report:

| Evaluator | Checks |
|---|---|
| `schema_valid` | output is a well-formed `RiskReport` |
| `risk_consistency` | `overall_risk` matches the flag-implied severity |
| `flags_grounded` | every flag cites a real ratio / number |
| `sources_present` | the quantitative source is credited |
| `summary_quality` | substantive narrative in Bahasa Indonesia |
| `sector_correct` | banks classified `financial`, not mis-flagged for DER |
| `groundedness_llm` | **LLM-as-judge**: is the summary supported by the data? |

```bash
python -m eval.run_eval              # full scorecard (live graph + LLM judge)
python -m eval.run_eval --no-judge   # deterministic checks only
python -m eval.run_eval --limit 2    # quick smoke
```

> The harness paid for itself immediately: it caught a Groq enum-case bug that
> silently dropped bank reports to the rule-based fallback (`summary_quality`
> 0.33 → 1.00 after the fix).

## 🧪 Development

```bash
make test     # pytest (offline; no network/model required) — 124 tests
make lint     # flake8
```

## License

MIT © 2026 Muhammad Fikri Wahidin
