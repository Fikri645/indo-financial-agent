"""Parse a financial-report PDF with Docling and preview the chunks.

Usage:
    python scripts/ingest_pdf.py data/pdfs/BBRI_2024_annual_report.pdf
"""
import sys

from src.tools.document import DocumentStore


def main(pdf_path: str) -> None:
    store = DocumentStore()
    print(f"Parsing {pdf_path} with Docling (first run downloads layout models) ...")
    n = store.add_pdf(pdf_path)
    print(f"Indexed {n} chunks.")

    for query in ["going concern kelangsungan usaha", "pihak berelasi", "faktor risiko"]:
        print(f"\n--- top hits for: {query!r} ---")
        for c in store.search(query, top_k=2):
            preview = c.text[:200].replace("\n", " ")
            print(f"[{c.section}] {preview}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide a PDF path. See scripts/ingest_pdf.py docstring.")
        sys.exit(1)
    main(sys.argv[1])
