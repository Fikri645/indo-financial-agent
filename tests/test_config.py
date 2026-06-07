from src.config import normalize_ticker


def test_normalize_adds_jk_suffix():
    assert normalize_ticker("bbri") == "BBRI.JK"
    assert normalize_ticker("TLKM") == "TLKM.JK"


def test_normalize_keeps_existing_suffix():
    assert normalize_ticker("BBRI.JK") == "BBRI.JK"
    assert normalize_ticker("aapl.us") == "AAPL.US"


def test_normalize_strips_whitespace():
    assert normalize_ticker("  asii  ") == "ASII.JK"
