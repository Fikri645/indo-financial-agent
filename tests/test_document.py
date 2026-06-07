"""Unit tests for document chunking + hybrid retrieval (offline, no model load)."""
from src.tools.document import (
    DocumentStore,
    chunk_markdown,
    detect_section,
    _reciprocal_rank_fusion,
)

SAMPLE_MD = """# Laporan Posisi Keuangan
Total aset perusahaan tumbuh menjadi Rp 5 triliun.

# Catatan Kelangsungan Usaha
Manajemen menilai terdapat keraguan atas going concern entitas.

# Pihak Berelasi
Transaksi dengan pihak berelasi mencakup pinjaman kepada entitas induk.

# Faktor Risiko
Risiko nilai tukar dan risiko likuiditas menjadi perhatian utama.
"""


def test_detect_section_recognizes_going_concern():
    assert detect_section("ada keraguan atas kelangsungan usaha") == "going_concern"
    assert detect_section("transaksi pihak berelasi") == "related_party"
    assert detect_section("opini audit wajar tanpa pengecualian") == "auditor"
    assert detect_section("biasa saja teks netral") == "body"


def test_chunk_markdown_splits_on_headings():
    chunks = chunk_markdown(SAMPLE_MD, source="test.pdf")
    assert len(chunks) == 4
    sections = {c.section for c in chunks}
    assert "going_concern" in sections
    assert "related_party" in sections
    assert "risk" in sections
    assert all(c.source == "test.pdf" for c in chunks)


def test_chunk_long_block_uses_overlap():
    long_text = "# Heading\n" + ("kata " * 1000)
    chunks = chunk_markdown(long_text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_rrf_fuses_rankings():
    # Item 2 ranks top in both lists -> should win the fusion.
    fused = _reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]])
    assert fused[0][0] == 2


def test_documentstore_bm25_retrieval():
    store = DocumentStore()
    store.add_markdown(SAMPLE_MD, source="test.pdf")
    results = store.search("kelangsungan usaha going concern", top_k=2)
    assert len(results) >= 1
    # The going-concern chunk should surface for that query.
    assert any(c.section == "going_concern" for c in results)


def test_empty_store_returns_empty():
    assert DocumentStore().search("apa saja") == []
