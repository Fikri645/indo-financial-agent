# 🇮🇩 Indonesian Financial Research Agent

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
| Retrieval | **ChromaDB** + multilingual-e5 embeddings + BM25 hybrid (RRF) |
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
      hybrid retrieval + rule-based risk scorer (**20 unit tests green**)
- [ ] Agent nodes (financial / document / news worker agents)
- [ ] Supervisor graph + Risk Analyst synthesis node
- [ ] LangSmith tracing + agent evaluation harness
- [ ] Gradio UI + Hugging Face Spaces deployment

---

## ⚠️ Limitations

- Ratio thresholds are tuned for **non-financial corporates**; banks & insurers
  (different balance-sheet structure) need sector-specific rules.
- `yfinance` data depth varies by ticker; missing fields degrade gracefully to `None`.
- **Not investment advice** — a portfolio/engineering demonstration only.

---

## 🧪 Development

```bash
make test     # pytest (offline; no network/model required)
make lint     # flake8
```

## License

MIT © 2026 Muhammad Fikri Wahidin
