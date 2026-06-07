"""Document tool — parses financial-statement / annual-report PDFs with Docling,
chunks them section-aware, and exposes hybrid retrieval (dense + BM25).

This is the *qualitative* leg of the agent: going-concern notes, related-party
transactions, management discussion — the signals that never show up in an API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src import config


@dataclass
class DocChunk:
    """A retrievable slice of a parsed document."""

    text: str
    section: str = "body"
    page: Optional[int] = None
    source: str = ""
    metadata: dict = field(default_factory=dict)


# Headings that signal an analyst-relevant section in an IDX annual report.
_SECTION_PATTERNS = {
    "going_concern": r"going concern|kelangsungan usaha",
    "related_party": r"related part|pihak berelasi|pihak yang berelasi",
    "debt": r"\bdebt\b|liabilit|utang|pinjaman|obligasi",
    "auditor": r"opini audit|independent auditor|auditor independen",
    "md&a": r"management discussion|analisis.*manajemen|laporan direksi",
    "risk": r"risk factor|faktor risiko|manajemen risiko",
}


def detect_section(text: str) -> str:
    """Tag a chunk with the most relevant analyst section, else 'body'."""
    low = text.lower()
    for name, pat in _SECTION_PATTERNS.items():
        if re.search(pat, low):
            return name
    return "body"


def parse_pdf(pdf_path: str | Path) -> str:
    """Convert a PDF to Markdown using Docling (layout + table aware).

    On HF Spaces (detected via ``SPACE_ID`` env var) a lightweight pipeline
    is used: OCR and TableFormer are disabled.  IDX annual reports are digital
    PDFs so text is already embedded — the fast path is sufficient and avoids
    downloading the 1-2 GB ML model files on first use.

    On local deployments the full pipeline runs (TableFormer recovers the
    merged-cell financial tables that pdfplumber mangles).
    """
    import os

    from docling.document_converter import DocumentConverter  # local import

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if os.getenv("SPACE_ID"):
        # HF Spaces free tier: use fast, model-free pipeline.
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,           # text is embedded in digital PDFs
                do_table_structure=False,  # skip TableFormer ML model
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        except Exception:
            # API may differ across docling versions — fall back to default.
            converter = DocumentConverter()
    else:
        # Local: full pipeline with TableFormer for accurate table extraction.
        converter = DocumentConverter()

    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def chunk_markdown(
    markdown: str,
    source: str = "",
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[DocChunk]:
    """Split Markdown into overlapping, section-tagged chunks.

    We split on Markdown headings first (keeps tables/sections intact) then pack
    paragraphs up to ``chunk_size`` characters with ``overlap`` carry-over.
    """
    # Split on headings while keeping the heading with its body.
    blocks = re.split(r"\n(?=#{1,6}\s)", markdown)
    chunks: list[DocChunk] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        section = detect_section(block)
        if len(block) <= chunk_size:
            chunks.append(DocChunk(text=block, section=section, source=source))
            continue
        # Pack long blocks with overlap.
        start = 0
        while start < len(block):
            piece = block[start : start + chunk_size]
            chunks.append(DocChunk(text=piece, section=section, source=source))
            start += chunk_size - overlap
    return chunks


class DocumentStore:
    """Hybrid retriever over parsed PDF chunks (dense embeddings + BM25 fusion).

    Mirrors the Philosopher Chat retrieval stack: cheap recall first, then fuse.
    Embeddings are lazy-loaded so the module imports without a GPU/model present.
    """

    def __init__(self, collection_name: str = "financial_docs"):
        self.collection_name = collection_name
        self._chunks: list[DocChunk] = []
        self._vectorstore = None
        self._bm25 = None
        self._tokenized: list[list[str]] = []
        self._reranker = None  # lazy-loaded cross-encoder

    # -- ingest -------------------------------------------------------------- #
    def add_pdf(self, pdf_path: str | Path) -> int:
        """Parse a PDF and index its chunks. Returns the number of chunks added."""
        md = parse_pdf(pdf_path)
        new_chunks = chunk_markdown(md, source=str(pdf_path))
        self._chunks.extend(new_chunks)
        self._build_indexes()
        return len(new_chunks)

    def add_markdown(self, markdown: str, source: str = "inline") -> int:
        """Index already-parsed Markdown (useful for tests / cached parses)."""
        new_chunks = chunk_markdown(markdown, source=source)
        self._chunks.extend(new_chunks)
        self._build_indexes()
        return len(new_chunks)

    def _build_indexes(self) -> None:
        from rank_bm25 import BM25Okapi

        self._tokenized = [c.text.lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        # Dense index is built lazily on first query to avoid loading the model
        # during ingest-only flows.
        self._vectorstore = None

    def _ensure_vectorstore(self):
        if self._vectorstore is not None or not self._chunks:
            return
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        texts = [c.text for c in self._chunks]
        metadatas = [{"section": c.section, "source": c.source} for c in self._chunks]
        self._vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
        )

    # -- retrieve ------------------------------------------------------------ #
    def search(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[DocChunk]:
        """Hybrid retrieve → fuse (RRF) → cross-encoder rerank → top_k.

        1. Recall a wide candidate pool with BM25 + dense embeddings.
        2. Fuse the two rankings with Reciprocal Rank Fusion (cheap, robust).
        3. Rerank the fused candidates with a multilingual cross-encoder, which
           scores true query-document relevance far better than fusion alone.
        """
        if not self._chunks:
            return []

        # Sparse (BM25) ranking
        bm25_order = self._bm25_rank(query)
        # Dense ranking (best-effort; falls back to BM25 if model unavailable)
        dense_order = self._dense_rank(query)

        fused = _reciprocal_rank_fusion([bm25_order, dense_order])
        # Take a wider candidate pool, then let the reranker pick the final top_k.
        candidate_idx = [i for i, _ in fused[: config.RETRIEVAL_CANDIDATES]]

        reranked = self._rerank(query, candidate_idx, top_k)
        if reranked is not None:
            return reranked
        # Reranker unavailable → fall back to fusion order.
        return [self._chunks[i] for i in candidate_idx[:top_k]]

    def _rerank(
        self, query: str, candidate_idx: list[int], top_k: int
    ) -> Optional[list[DocChunk]]:
        """Cross-encoder rerank. Returns None if the reranker can't be loaded."""
        if not config.RERANK_ENABLED or not candidate_idx:
            return None
        try:
            self._ensure_reranker()
            if self._reranker is None:
                return None
            pairs = [[query, self._chunks[i].text] for i in candidate_idx]
            scores = self._reranker.predict(pairs)
            ranked = sorted(
                zip(candidate_idx, scores), key=lambda x: x[1], reverse=True
            )
            return [self._chunks[i] for i, _ in ranked[:top_k]]
        except Exception:  # pragma: no cover - model/network dependent
            return None

    def _ensure_reranker(self) -> None:
        if self._reranker is not None:
            return
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(config.RERANKER_MODEL)

    def _bm25_rank(self, query: str) -> list[int]:
        if self._bm25 is None:
            return list(range(len(self._chunks)))
        scores = self._bm25.get_scores(query.lower().split())
        return [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    def _dense_rank(self, query: str) -> list[int]:
        try:
            self._ensure_vectorstore()
            if self._vectorstore is None:
                return []
            results = self._vectorstore.similarity_search(
                query, k=min(config.RETRIEVAL_CANDIDATES, len(self._chunks))
            )
            texts = [c.text for c in self._chunks]
            order = []
            for doc in results:
                if doc.page_content in texts:
                    order.append(texts.index(doc.page_content))
            return order
        except Exception:  # pragma: no cover - model/network dependent
            return []


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """RRF: score = sum(1 / (k + rank)) across rankings. Higher is better."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
