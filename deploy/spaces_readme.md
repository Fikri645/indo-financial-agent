---
title: Indonesian Financial Research Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.15.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🇮🇩 Indonesian Financial Research Agent

A **LangGraph multi-agent system** that researches IDX-listed companies end-to-end and produces a structured risk assessment in Bahasa Indonesia.

## What it does

Give it a ticker (e.g. `BBRI`) and it will:

1. **📊 Financial Agent** — fetch fundamentals via yfinance (.JK) + compute 10 financial ratios
2. **📰 News Agent** — ReAct tool-calling agent: compose queries autonomously, scan recent news, synthesise sentiment
3. **📄 Document Agent** — parse annual-report PDF with Docling (table-aware) + hybrid RAG retrieval
4. **🧠 Risk Analyst** — synthesise all evidence → structured `RiskReport` (Pydantic-validated, Bahasa Indonesia)

## Architecture

```
START → SUPERVISOR → financial_agent → news_agent → document_agent → risk_analyst → END
```

The supervisor is deterministic (if/else routing, not LLM-based) — LLM budget is reserved for the synthesis step only.

## Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (supervisor multi-agent) |
| LLM | Groq Llama 3.3 70B (free tier) / Gemini 2.0 Flash / OpenRouter |
| Structured data | yfinance (IDX .JK) |
| PDF parsing | Docling (TableFormer, table-aware) |
| Retrieval | ChromaDB + multilingual-e5 + BM25 hybrid (RRF) → BGE cross-encoder rerank |
| Output contract | Pydantic schemas |
| Eval | Deterministic scorers + LLM-as-judge (mean score 0.93) |

> ⚠️ Not investment advice — portfolio/engineering demonstration only.
