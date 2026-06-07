"""Gradio UI for the Indonesian Financial Research Agent.

Streams real-time progress while the LangGraph graph runs, then renders the
final RiskReport as a structured multi-tab layout.

Run locally:
    python app/gradio_app.py          # or: make app
Deploy:
    Push to Hugging Face Spaces (see app.py at project root).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

# Allow running as a script directly from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr  # noqa: E402

# --------------------------------------------------------------------------- #
# Risk-level display helpers
# --------------------------------------------------------------------------- #

_RISK_EMOJI = {
    "low": "🟢",
    "moderate": "🟡",
    "high": "🟠",
    "severe": "🔴",
}

_RISK_LABEL = {
    "low": "RENDAH",
    "moderate": "MODERAT",
    "high": "TINGGI",
    "severe": "KRITIS",
}

_SEVERITY_EMOJI = {
    "low": "🟢",
    "moderate": "🟡",
    "high": "🟠",
    "severe": "🔴",
}

_RATIO_LABELS = {
    "current_ratio": ("Current Ratio", None),
    "quick_ratio": ("Quick Ratio", None),
    "debt_to_equity": ("Debt-to-Equity (DER)", None),
    "debt_ratio": ("Debt Ratio", None),
    "interest_coverage": ("Interest Coverage", None),
    "net_profit_margin": ("Net Profit Margin", "pct"),
    "gross_margin": ("Gross Margin", "pct"),
    "roe": ("ROE", "pct"),
    "roa": ("ROA", "pct"),
    "revenue_growth": ("Revenue Growth YoY", "pct"),
    "net_income_growth": ("Net Income Growth YoY", "pct"),
}


# --------------------------------------------------------------------------- #
# Report formatting helpers
# --------------------------------------------------------------------------- #

def _fmt_val(val: float | None, fmt: str | None) -> str:
    if val is None:
        return "*N/A*"
    if fmt == "pct":
        return f"{val:.1%}"
    return f"{val:.2f}"


def _format_summary(report: dict, ticker: str) -> str:
    risk_val = report.get("overall_risk", "moderate")
    emoji = _RISK_EMOJI.get(risk_val, "⚪")
    label = _RISK_LABEL.get(risk_val, risk_val.upper())
    company = report.get("company_name") or ticker

    return (
        f"## {emoji} {company} ({ticker.upper()})\n\n"
        f"**Tingkat Risiko Keseluruhan: {label}**\n\n"
        f"---\n\n"
        f"{report.get('summary', '*Ringkasan tidak tersedia.*')}"
    )


def _format_ratios(report: dict) -> str:
    ratios = report.get("key_ratios") or {}
    rows: list[str] = []
    for key, (label, fmt) in _RATIO_LABELS.items():
        val = ratios.get(key)
        if val is not None:
            rows.append(f"| {label} | {_fmt_val(val, fmt)} |")
    if not rows:
        return "*Data rasio keuangan tidak tersedia.*"
    return "| Metrik | Nilai |\n|---|---|\n" + "\n".join(rows)


def _format_flags(report: dict) -> str:
    flags = report.get("flags") or []
    if not flags:
        return "✅ **Tidak ada flag risiko signifikan yang ditemukan.**"
    parts: list[str] = []
    for f in flags:
        sev = f.get("severity", "moderate")
        sem = _SEVERITY_EMOJI.get(sev, "⚪")
        cat = f.get("category", "").replace("_", " ").title()
        parts.append(
            f"### {sem} [{sev.upper()}] {cat}\n"
            f"{f.get('finding', '')}\n\n"
            f"> **Evidence:** {f.get('evidence', '')}  \n"
            f"> **Source:** `{f.get('source', '')}`"
        )
    return "\n\n---\n\n".join(parts)


def _format_positives(report: dict) -> str:
    positives = report.get("positives") or []
    sources = report.get("sources") or []
    md = ""
    if positives:
        md += "## ✅ Faktor Positif / Mitigasi Risiko\n\n"
        md += "\n".join(f"- {p}" for p in positives)
        md += "\n\n"
    else:
        md += "## ✅ Faktor Positif\n\n*Tidak ada catatan positif eksplisit.*\n\n"
    if sources:
        md += "## 📚 Sumber Data\n\n"
        md += "\n".join(f"- `{s}`" for s in sources)
    return md


# --------------------------------------------------------------------------- #
# Core analysis function (streaming generator)
# --------------------------------------------------------------------------- #

_LOADING = "⏳"
_EMPTY = ""


def _pdf_path(pdf_file) -> str | None:
    """Extract file path from Gradio's file component (handles Gradio 5 types)."""
    if pdf_file is None:
        return None
    if isinstance(pdf_file, str):
        return pdf_file
    if hasattr(pdf_file, "name"):       # Gradio ≤4 NamedTemporaryFile
        return pdf_file.name
    if isinstance(pdf_file, dict):      # some Gradio 5 modes
        return pdf_file.get("path") or pdf_file.get("name")
    return str(pdf_file)


def analyze(
    ticker: str,
    pdf_file,
) -> Generator[tuple[str, str, str, str], None, None]:
    """Stream agent progress updates, then yield the final formatted report."""
    from langchain_core.messages import HumanMessage

    # ---- input validation ---------------------------------------------------
    ticker = (ticker or "").strip().upper()
    if not ticker:
        yield "❌ Masukkan kode saham BEI (contoh: BBRI).", _EMPTY, _EMPTY, _EMPTY
        return

    pdf_path = _pdf_path(pdf_file)

    # ---- build initial graph state ------------------------------------------
    initial_state = {
        "messages": [HumanMessage(content=f"Analisis risiko keuangan perusahaan {ticker}")],
        "ticker": ticker,
        "pdf_path": pdf_path,
        "financials": None,
        "doc_chunks": [],
        "news_headlines": [],
        "risk_report": None,
        "next": "",
    }

    yield (
        f"{_LOADING} Memulai analisis **{ticker}**...",
        _EMPTY, _EMPTY, _EMPTY,
    )

    # ---- import graph (lazy — keeps startup fast) ---------------------------
    try:
        from src.agents.graph import graph
    except Exception as exc:
        yield (
            f"❌ Gagal memuat agent graph: {exc}",
            _EMPTY, _EMPTY, _EMPTY,
        )
        return

    # ---- stream graph execution ---------------------------------------------
    # stream_mode="values" yields the full cumulative state after each step.
    # Track message count to yield only newly appended messages.
    final_state: dict | None = None
    prev_msg_count = 0
    try:
        for step in graph.stream(
            initial_state, {"recursion_limit": 25}, stream_mode="values"
        ):
            msgs = step.get("messages") or []
            for msg in msgs[prev_msg_count:]:
                name = getattr(msg, "name", None) or "supervisor"
                content = (msg.content or "")[:300]
                yield (f"{_LOADING} `[{name}]` {content}", _EMPTY, _EMPTY, _EMPTY)
            prev_msg_count = len(msgs)
            final_state = step
    except Exception as exc:
        yield (
            f"❌ Error saat menjalankan agent: {exc}",
            _EMPTY, _EMPTY, _EMPTY,
        )
        return

    # ---- render final report ------------------------------------------------
    report = (final_state or {}).get("risk_report")
    if not report:
        yield (
            "❌ Laporan risiko tidak berhasil dibuat. "
            "Pastikan API key LLM sudah dikonfigurasi di `.env`.",
            _EMPTY, _EMPTY, _EMPTY,
        )
        return

    yield (
        _format_summary(report, ticker),
        _format_ratios(report),
        _format_flags(report),
        _format_positives(report),
    )


# --------------------------------------------------------------------------- #
# Gradio Blocks layout
# --------------------------------------------------------------------------- #

_DESCRIPTION = """
**LangGraph multi-agent system** yang menganalisis risiko keuangan perusahaan
IDX secara otomatis:

1. **Financial Agent** — fundamental yfinance (.JK) + rasio keuangan
2. **News Agent** — berita terbaru (DuckDuckGo)
3. **Document Agent** — parsing laporan keuangan PDF (Docling, table-aware)
4. **Risk Analyst** — sintesis semua data → laporan risiko terstruktur (Bahasa Indonesia)

> ⚠️ *Bukan saran investasi — demonstrasi teknikal portofolio.*
"""

_EXAMPLES = [
    ["BBRI", None],
    ["TLKM", None],
    ["ASII", None],
    ["GOTO", None],
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="🇮🇩 Indonesian Financial Research Agent",
        theme=gr.themes.Soft(),
        css=".output-markdown { font-size: 0.95rem; }",
    ) as demo:

        gr.Markdown("# 🇮🇩 Indonesian Financial Research Agent")
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            # ---- left panel: inputs ----------------------------------------
            with gr.Column(scale=1, min_width=280):
                ticker_input = gr.Textbox(
                    label="Kode Saham BEI",
                    placeholder="Contoh: BBRI",
                    max_lines=1,
                    autofocus=True,
                )
                pdf_input = gr.File(
                    label="📄 Laporan Keuangan PDF (Opsional)",
                    file_types=[".pdf"],
                    file_count="single",
                )
                analyze_btn = gr.Button(
                    "🔍 Analisis Risiko", variant="primary", size="lg"
                )

                gr.Markdown("""
**Cara penggunaan:**
1. Ketik kode saham BEI (tanpa `.JK`)
2. *Opsional:* upload PDF laporan keuangan tahunan dari
   [IDX](https://www.idx.co.id/en/listed-companies/financial-statements-and-annual-report)
3. Klik **Analisis Risiko**

**Diperlukan:** minimal satu API key di `.env`:
```
GROQ_API_KEY=...     # gratis, direkomendasikan
GOOGLE_API_KEY=...   # Gemini 2.0 Flash
```
""")

                gr.Examples(
                    examples=_EXAMPLES,
                    inputs=[ticker_input, pdf_input],
                    label="Contoh ticker",
                )

            # ---- right panel: outputs --------------------------------------
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Ringkasan Risiko"):
                        summary_out = gr.Markdown(
                            label="Ringkasan",
                            value="*Hasil analisis akan muncul di sini...*",
                            elem_classes=["output-markdown"],
                        )
                    with gr.Tab("📈 Rasio Keuangan"):
                        ratios_out = gr.Markdown(
                            value="",
                            elem_classes=["output-markdown"],
                        )
                    with gr.Tab("⚠️ Risk Flags"):
                        flags_out = gr.Markdown(
                            value="",
                            elem_classes=["output-markdown"],
                        )
                    with gr.Tab("✅ Positif & Sumber"):
                        positives_out = gr.Markdown(
                            value="",
                            elem_classes=["output-markdown"],
                        )

        analyze_btn.click(
            fn=analyze,
            inputs=[ticker_input, pdf_input],
            outputs=[summary_out, ratios_out, flags_out, positives_out],
            show_progress="full",
        )

        ticker_input.submit(
            fn=analyze,
            inputs=[ticker_input, pdf_input],
            outputs=[summary_out, ratios_out, flags_out, positives_out],
            show_progress="full",
        )

        gr.Markdown("""
---
Built with LangGraph · yfinance · Docling · Groq
[GitHub](https://github.com/Fikri645/indo-financial-agent) ·
[Hugging Face](https://huggingface.co/fikri0o0)
""")

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
