"""Gradio UI — Indonesian Financial Research Agent (Enhanced).

Streams real-time agent progress in a dedicated log panel, then renders the
final RiskReport as a rich, sector-aware multi-tab layout.

Run locally:
    python app/gradio_app.py      # or: make app
Deploy:
    Push to Hugging Face Spaces (see app.py at project root).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Generator

# Allow running as a script directly from any working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────

# (badge_color, background_color, label)
_RISK_STYLE: dict[str, tuple[str, str, str]] = {
    "low":      ("#16a34a", "#f0fdf4", "✅ RISIKO RENDAH"),
    "moderate": ("#ca8a04", "#fefce8", "⚠️ RISIKO MODERAT"),
    "high":     ("#ea580c", "#fff7ed", "🟠 RISIKO TINGGI"),
    "severe":   ("#dc2626", "#fef2f2", "🔴 RISIKO KRITIS"),
}

_SEV_STYLE: dict[str, tuple[str, str, str]] = {
    "low":      ("#16a34a", "#f0fdf4", "RENDAH"),
    "moderate": ("#ca8a04", "#fefce8", "MODERAT"),
    "high":     ("#ea580c", "#fff7ed", "TINGGI"),
    "severe":   ("#dc2626", "#fef2f2", "KRITIS"),
}

_AGENT_ICONS: dict[str, str] = {
    "financial_agent": "📊",
    "news_agent":      "📰",
    "document_agent":  "📄",
    "risk_analyst":    "🧠",
    "supervisor":      "🎯",
}

# ─── Ratio groups (key, display-label, format, sector-scope) ─────────────────
_RATIO_GROUPS: list[tuple[str, list[tuple[str, str, str | None, str]]]] = [
    ("💧 Likuiditas", [
        ("current_ratio",    "Current Ratio",       None,  "general"),
        ("quick_ratio",      "Quick Ratio",         None,  "general"),
    ]),
    ("⚖️ Leverage / Solvabilitas", [
        ("debt_to_equity", "Debt-to-Equity (DER)", None, "general"),
        ("debt_ratio", "Debt Ratio", None, "all"),
        ("interest_coverage", "Interest Coverage", None, "all"),
    ]),
    ("💰 Profitabilitas", [
        ("net_profit_margin", "Net Profit Margin", "pct", "all"),
        ("gross_margin", "Gross Margin", "pct", "all"),
        ("roe", "Return on Equity (ROE)", "pct", "all"),
        ("roa", "Return on Assets (ROA)", "pct", "all"),
    ]),
    ("📈 Pertumbuhan YoY", [
        ("revenue_growth", "Revenue Growth", "pct", "all"),
        ("net_income_growth", "Net Income Growth", "pct", "all"),
    ]),
]

# (operator, threshold, benchmark_label)
_BENCHMARKS: dict[str, tuple[str, float, str]] = {
    "current_ratio":     (">", 1.5,  "≥ 1.5 sehat"),
    "quick_ratio":       (">", 1.0,  "≥ 1.0 sehat"),
    "debt_to_equity":    ("<", 2.0,  "< 2.0 aman"),
    "interest_coverage": (">", 2.0,  "≥ 2.0× aman"),
    "net_profit_margin": (">", 0.05, "> 5%"),
    "gross_margin":      (">", 0.20, "> 20%"),
    "roe":               (">", 0.10, "> 10%"),
    "roa":               (">", 0.03, "> 3%"),
    "revenue_growth":    (">", 0.0,  "positif"),
    "net_income_growth": (">", 0.0,  "positif"),
}

_CSS = """
/* Progress log — dark terminal feel */
.progress-log { font-family: 'Courier New', monospace; font-size: 0.83rem;
                line-height: 1.6; }

/* Tabs */
.tab-nav button { font-weight: 600 !important; }

/* General output markdown */
.output-md  { font-size: 0.93rem; line-height: 1.75; }

/* Thinner divider */
hr { margin: 12px 0 !important; opacity: 0.3; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_val(val: float | None, fmt: str | None) -> str:
    if val is None:
        return "*N/A*"
    return f"{val:.1%}" if fmt == "pct" else f"{val:.2f}"


def _status_icon(key: str, val: float | None, sector: str) -> str:
    """Return ✅/❌/— based on benchmark. Bank skips liquidity & leverage."""
    if val is None:
        return "—"
    # For banks, liquidity/leverage benchmarks are not applicable
    if sector == "financial" and key in ("current_ratio", "quick_ratio", "debt_to_equity"):
        return "*N/A*"
    bench = _BENCHMARKS.get(key)
    if not bench:
        return "—"
    op, threshold, _ = bench
    passed = (val > threshold) if op == ">" else (val < threshold)
    return "✅" if passed else "❌"


def _key_metrics_html(ratios: dict, sector: str, accent: str) -> str:
    """3–4 prominent metric boxes shown inside the summary card."""
    if sector == "financial":
        keys = [
            ("roe", "ROE", "pct"),
            ("roa", "ROA", "pct"),
            ("net_profit_margin", "Net Margin", "pct"),
            ("net_income_growth", "Income Growth", "pct"),
        ]
    else:
        keys = [
            ("current_ratio", "Current Ratio", None),
            ("debt_to_equity", "DER", None),
            ("net_profit_margin", "Net Margin", "pct"),
            ("roe", "ROE", "pct"),
        ]

    boxes = ""
    for k, lbl, fmt in keys:
        val = ratios.get(k)
        if val is None:
            continue
        boxes += (
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;'
            f'padding:10px 14px;flex:1;min-width:100px;text-align:center;">'
            f'<div style="font-size:1.35em;font-weight:bold;color:{accent};">'
            f'{_fmt_val(val, fmt).replace("*", "")}</div>'
            f'<div style="font-size:0.72em;color:#64748b;margin-top:2px;">{lbl}</div>'
            f'</div>'
        )
    if not boxes:
        return ""
    return (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 4px;">'
        f'{boxes}</div>'
    )


def _format_summary(
    report: dict, ticker: str, financials: dict | None = None
) -> str:
    risk_val = (report.get("overall_risk") or "moderate").lower()
    accent, bg, badge_label = _RISK_STYLE.get(
        risk_val, ("#6b7280", "#f9fafb", "❓ TIDAK DIKETAHUI")
    )
    company = report.get("company_name") or ticker
    sector = (financials or {}).get("sector", "general")
    industry = (financials or {}).get("industry") or ""
    period = (financials or {}).get("period_end") or ""

    sector_label = "🏦 Perbankan / Keuangan" if sector == "financial" else "🏭 Korporat Umum"
    sector_colors = (
        ("#1e40af", "#dbeafe") if sector == "financial" else ("#065f46", "#d1fae5")
    )
    period_str = f" · {period}" if period else ""
    industry_str = (
        f"<p style='color:#64748b;font-size:0.84em;margin:2px 0 0;'>{industry}</p>"
        if industry else ""
    )

    metrics_html = _key_metrics_html(report.get("key_ratios") or {}, sector, accent)

    num_flags = len(report.get("flags") or [])
    flags_summary = (
        f"&nbsp;·&nbsp; <strong>{num_flags} flag risiko</strong>"
        if num_flags else "&nbsp;·&nbsp; ✅ tidak ada flag risiko"
    )

    summary_text = report.get("summary") or "*Ringkasan tidak tersedia.*"

    card = (
        f'<div style="background:{bg};border-left:6px solid {accent};'
        f'border-radius:10px;padding:20px 22px;margin-bottom:16px;">'
        # badges row
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">'
        f'<span style="background:{accent};color:white;padding:5px 16px;border-radius:20px;'
        f'font-weight:bold;font-size:0.95em;">{badge_label}</span>'
        f'<span style="background:{sector_colors[1]};color:{sector_colors[0]};padding:3px 10px;'
        f'border-radius:12px;font-size:0.80em;font-weight:600;">{sector_label}</span>'
        f'</div>'
        # company heading
        f'<h2 style="margin:0 0 2px;color:#1e293b;font-size:1.3em;">'
        f'{company} <span style="color:#94a3b8;font-weight:400;font-size:0.75em;">'
        f'({ticker.upper()}{period_str})</span>'
        f'</h2>'
        f'{industry_str}'
        # metrics
        f'{metrics_html}'
        # footer meta
        f'<div style="font-size:0.78em;color:#94a3b8;margin-top:8px;">'
        f'Overall risk: <strong>{risk_val.upper()}</strong>{flags_summary}'
        f'</div>'
        f'</div>'
    )

    return card + "\n\n" + summary_text


def _format_ratios(report: dict, financials: dict | None = None) -> str:
    ratios = report.get("key_ratios") or {}
    sector = (financials or {}).get("sector", "general")
    parts: list[str] = []
    has_any = False

    for group_title, fields in _RATIO_GROUPS:
        rows: list[str] = []
        for key, label, fmt, _ in fields:
            val = ratios.get(key)
            if val is None:
                continue
            has_any = True
            status = _status_icon(key, val, sector)
            if sector == "financial" and key in ("current_ratio", "quick_ratio", "debt_to_equity"):
                bench_str = "*tidak relevan (bank)*"
            elif key in _BENCHMARKS:
                bench_str = _BENCHMARKS[key][2]
            else:
                bench_str = "—"
            rows.append(
                f"| {label} | **{_fmt_val(val, fmt).replace('*', '')}** "
                f"| {bench_str} | {status} |"
            )
        if rows:
            header = "| Metrik | Nilai | Benchmark | Status |\n|---|---:|---|:---:|"
            parts.append(f"### {group_title}\n\n{header}\n" + "\n".join(rows))

    if not has_any:
        return "*Data rasio keuangan tidak tersedia.*"

    if sector == "financial":
        parts.append(
            "\n> 💡 **Catatan Bank:** Untuk lembaga keuangan, Debt-to-Equity (DER) yang "
            "tinggi adalah **normal** — simpanan nasabah dicatat sebagai liabilitas. "
            "Current Ratio dan DER tidak menjadi patokan risiko untuk sektor ini."
        )

    return "\n\n".join(parts)


def _format_flags(report: dict) -> str:
    flags = report.get("flags") or []
    if not flags:
        return (
            '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;'
            'padding:20px;text-align:center;">'
            '<div style="font-size:2em;">✅</div>'
            '<strong style="color:#16a34a;">Tidak Ada Flag Risiko Signifikan</strong>'
            '<p style="color:#4b5563;margin:8px 0 0;">Perusahaan ini tidak menunjukkan '
            'flag risiko merah/kuning dari data yang tersedia.</p>'
            '</div>'
        )

    parts = [f"## ⚠️ {len(flags)} Flag Risiko Ditemukan\n"]
    for f in flags:
        sev = (f.get("severity") or "moderate").lower()
        col, bg, sev_label = _SEV_STYLE.get(sev, ("#6b7280", "#f9fafb", "—"))
        cat = f.get("category", "").replace("_", " ").title()
        finding = f.get("finding", "")
        evidence = f.get("evidence", "")
        source = f.get("source", "")

        parts.append(
            f'<div style="background:{bg};border-left:5px solid {col};'
            f'border-radius:0 10px 10px 0;padding:14px 18px;margin:14px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="background:{col};color:white;padding:2px 11px;'
            f'border-radius:10px;font-size:0.78em;font-weight:bold;">{sev_label}</span>'
            f'<strong style="color:#1e293b;font-size:0.97em;">{cat}</strong>'
            f'</div>'
            f'<p style="margin:0 0 8px;color:#374151;line-height:1.6;">{finding}</p>'
            f'<div style="font-size:0.82em;color:#6b7280;">'
            f'📊 <em>{evidence}</em>'
            f'{f"&nbsp;·&nbsp;🔗 <code>{source}</code>" if source else ""}'
            f'</div>'
            f'</div>'
        )

    return "\n".join(parts)


def _format_positives(report: dict) -> str:
    positives = report.get("positives") or []
    sources = report.get("sources") or []
    parts: list[str] = []

    if positives:
        items = "\n".join(f"- ✅ {p}" for p in positives)
        parts.append(f"## 💚 Faktor Positif / Mitigasi Risiko\n\n{items}")
    else:
        parts.append(
            "## 💚 Faktor Positif\n\n"
            "*Tidak ada catatan positif eksplisit dari analisis ini.*"
        )

    if sources:
        _src_desc = {
            "yfinance":             "data fundamental & rasio keuangan",
            "news_search":          "pencarian berita terkini",
            "financial_report_pdf": "laporan keuangan / annual report PDF",
        }
        src_lines = []
        for s in sources:
            desc = next((v for k, v in _src_desc.items() if k in s.lower()), "")
            src_lines.append(f"- `{s}`" + (f" — {desc}" if desc else ""))
        parts.append("## 📚 Sumber Data\n\n" + "\n".join(src_lines))

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis function (streaming generator)
# ─────────────────────────────────────────────────────────────────────────────

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
) -> Generator[tuple[str, str, str, str, str], None, None]:
    """Stream agent progress to a dedicated log panel, then yield the full report.

    Yields 5-tuples: (progress_log, summary, ratios, flags, positives).
    During streaming only progress_log is updated; the 4 report tabs fill at end.
    """
    from langchain_core.messages import HumanMessage

    # ── input validation ──────────────────────────────────────────────────────
    ticker = (ticker or "").strip().upper()
    if not ticker:
        yield (
            "❌ Masukkan kode saham BEI (contoh: BBRI).",
            _EMPTY, _EMPTY, _EMPTY, _EMPTY,
        )
        return

    pdf_path = _pdf_path(pdf_file)

    # ── initial graph state ───────────────────────────────────────────────────
    # IMPORTANT: use None as sentinel ("not yet fetched"), not []
    # The supervisor checks `is not None` to decide whether each leg has run.
    initial_state = {
        "messages": [HumanMessage(content=f"Analisis risiko keuangan perusahaan {ticker}")],
        "ticker":        ticker,
        "pdf_path":      pdf_path,
        "financials":    None,
        "doc_chunks":    None,   # None = not yet fetched (sentinel)
        "news_headlines": None,  # None = not yet fetched (sentinel)
        "risk_report":   None,
        "next":          "",
    }

    # ── live log accumulator ──────────────────────────────────────────────────
    log_lines: list[str] = []

    def _log(line: str) -> None:
        log_lines.append(line)

    def _get_log() -> str:
        return "\n\n".join(log_lines)

    def _yield_progress() -> tuple[str, str, str, str, str]:
        return (_get_log(), _EMPTY, _EMPTY, _EMPTY, _EMPTY)

    ts = datetime.now().strftime("%H:%M:%S")
    _log(f"**{ts}** — Memulai analisis **{ticker}**"
         + (f" dengan PDF `{Path(pdf_path).name}`" if pdf_path else "") + " ...")
    yield _yield_progress()

    # ── lazy graph import (keeps startup fast) ────────────────────────────────
    try:
        from src.agents.graph import graph
    except Exception as exc:
        _log(f"❌ Gagal memuat agent graph: `{exc}`")
        yield _yield_progress()
        return

    # ── stream graph execution ────────────────────────────────────────────────
    # stream_mode="values" yields the full cumulative state after each step.
    # We track prev_msg_count to log only newly appended messages per step.
    final_state: dict | None = None
    prev_msg_count = 0
    step_count = 0

    try:
        for step in graph.stream(
            initial_state, {"recursion_limit": 25}, stream_mode="values"
        ):
            msgs = step.get("messages") or []
            for msg in msgs[prev_msg_count:]:
                name = getattr(msg, "name", None) or "supervisor"
                icon = _AGENT_ICONS.get(name, "🔄")
                content = (msg.content or "")[:200].replace("\n", " ")
                _log(f"{icon} **[{name}]** {content}")
                step_count += 1
            prev_msg_count = len(msgs)
            final_state = step
            yield _yield_progress()

    except Exception as exc:
        _log(f"❌ Error saat menjalankan agent: `{exc}`")
        yield _yield_progress()
        return

    # ── render final report ───────────────────────────────────────────────────
    report = (final_state or {}).get("risk_report")
    financials = (final_state or {}).get("financials")

    if not report:
        _log(
            "❌ Laporan risiko tidak berhasil dibuat. "
            "Pastikan API key LLM sudah dikonfigurasi di `.env`."
        )
        yield _yield_progress()
        return

    ts2 = datetime.now().strftime("%H:%M:%S")
    _log(
        f"✅ **Analisis selesai** · {step_count} langkah · {ts2} "
        f"· overall risk: **{(report.get('overall_risk') or '?').upper()}**"
    )

    yield (
        _get_log(),
        _format_summary(report, ticker, financials),
        _format_ratios(report, financials),
        _format_flags(report),
        _format_positives(report),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Blocks layout
# ─────────────────────────────────────────────────────────────────────────────

_DESCRIPTION = """\
**LangGraph multi-agent system** yang menganalisis risiko keuangan perusahaan
IDX listed secara otomatis — gratis, open-source, Bahasa Indonesia.

| Agent | Tugas |
|---|---|
| 📊 **Financial Agent** | Fundamental yfinance `.JK` + 10 rasio keuangan |
| 📰 **News Agent** | ReAct tool-calling: cari & sintesis berita terbaru |
| 📄 **Document Agent** | Parse laporan keuangan PDF (Docling, table-aware) |
| 🧠 **Risk Analyst** | Sintesis semua data → laporan risiko terstruktur |

> ⚠️ *Bukan saran investasi — demonstrasi portofolio teknikal.*
"""

_EXAMPLES = [
    ["BBRI", None],
    ["BBCA", None],
    ["TLKM", None],
    ["ASII", None],
    ["GOTO", None],
    ["UNVR", None],
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="🇮🇩 Indonesian Financial Research Agent",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=_CSS,
    ) as demo:

        gr.Markdown("# 🇮🇩 Indonesian Financial Research Agent")
        gr.Markdown(_DESCRIPTION)

        with gr.Row(equal_height=False):

            # ── Left panel: inputs ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                ticker_input = gr.Textbox(
                    label="Kode Saham BEI",
                    placeholder="Contoh: BBRI, TLKM, ASII",
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
---
**Cara penggunaan:**
1. Ketik kode saham BEI (tanpa `.JK`)
2. *Opsional:* upload PDF laporan keuangan dari
   [IDX](https://www.idx.co.id/en/listed-companies/financial-statements-and-annual-report)
3. Klik **Analisis Risiko** atau tekan **Enter**

**Minimal 1 API key di `.env`:**
```
GROQ_API_KEY=...     # gratis, direkomendasikan
GOOGLE_API_KEY=...   # Gemini 2.0 Flash
```
""")

                gr.Examples(
                    examples=_EXAMPLES,
                    inputs=[ticker_input, pdf_input],
                    label="Contoh ticker IDX",
                )

            # ── Right panel: progress log + result tabs ───────────────────────
            with gr.Column(scale=2):

                # Progress log — updates live during streaming
                gr.Markdown("### 📋 Log Analisis")
                progress_md = gr.Markdown(
                    value=(
                        "*Siap menganalisis. Masukkan kode saham dan klik "
                        "**Analisis Risiko**.*"
                    ),
                    elem_classes=["progress-log"],
                )

                gr.Markdown("---")

                # Report tabs — fill in when analysis completes
                with gr.Tabs():
                    with gr.Tab("📊 Ringkasan"):
                        summary_out = gr.Markdown(
                            value=(
                                "*Hasil analisis akan muncul di sini setelah "
                                "analisis selesai.*"
                            ),
                            elem_classes=["output-md"],
                        )
                    with gr.Tab("📈 Rasio Keuangan"):
                        ratios_out = gr.Markdown(
                            value="",
                            elem_classes=["output-md"],
                        )
                    with gr.Tab("⚠️ Risk Flags"):
                        flags_out = gr.Markdown(
                            value="",
                            elem_classes=["output-md"],
                        )
                    with gr.Tab("✅ Positif & Sumber"):
                        positives_out = gr.Markdown(
                            value="",
                            elem_classes=["output-md"],
                        )

        _outputs = [progress_md, summary_out, ratios_out, flags_out, positives_out]

        analyze_btn.click(
            fn=analyze,
            inputs=[ticker_input, pdf_input],
            outputs=_outputs,
            show_progress="hidden",   # we have our own log
        )
        ticker_input.submit(
            fn=analyze,
            inputs=[ticker_input, pdf_input],
            outputs=_outputs,
            show_progress="hidden",
        )

        gr.Markdown("""
---
Built with **LangGraph** · **yfinance** · **Docling** · **Groq** · **Gradio** &nbsp;|&nbsp;
[GitHub](https://github.com/Fikri645/indo-financial-agent) ·
[Hugging Face](https://huggingface.co/fikri0o0) &nbsp;|&nbsp;
MIT © 2026 Muhammad Fikri Wahidin
""")

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
