VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup init sources ingest analyze backtest serve test clean

help:
	@echo "make setup     - create the virtualenv and install dependencies"
	@echo "make init      - create the database schema"
	@echo "make sources   - report which data sources are reachable"
	@echo "make ingest    - download history for the default competitions"
	@echo "make backtest  - evaluate the models on historical matches"
	@echo "make serve     - run the web application on http://127.0.0.1:8000"
	@echo "make test      - run the test suite"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

init:
	$(VENV)/bin/mebet init

sources:
	$(VENV)/bin/mebet sources

ingest:
	$(VENV)/bin/mebet ingest ENG.1 ESP.1 ITA.1 GER.1 FRA.1 --seasons 10 --players

backtest:
	$(VENV)/bin/mebet backtest --competition ENG.1 --from 2024-08-01 --to 2026-05-31

serve:
	$(VENV)/bin/mebet serve

test:
	$(VENV)/bin/pytest

clean:
	rm -rf data/http_cache logs/*.log .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
