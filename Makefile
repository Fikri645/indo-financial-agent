.PHONY: help install fetch ingest agent app test lint clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  install   pip install -r requirements.txt"
	@echo "  fetch     fetch fundamentals + rule-based risk (TICKER=BBRI)"
	@echo "  ingest    parse a financial-report PDF (PDF=path/to.pdf)"
	@echo "  agent     run the full LangGraph agent on a ticker (TICKER=BBRI)"
	@echo "  app       launch the Gradio UI"
	@echo "  test      run unit tests"
	@echo "  lint      flake8"
	@echo "  clean     remove caches"

install:
	$(PY) -m pip install -r requirements.txt

TICKER ?= BBRI
fetch:
	$(PY) scripts/fetch_company.py $(TICKER)

PDF ?=
ingest:
	$(PY) scripts/ingest_pdf.py $(PDF)

agent:
	$(PY) -m src.agents.graph $(TICKER)

app:
	$(PY) app/gradio_app.py

test:
	$(PY) -m pytest tests/ -v

lint:
	flake8 src/ tests/ scripts/

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache data/vectorstore/*
