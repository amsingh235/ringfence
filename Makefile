# Ringfence — Abuse-Ring Sentinel
#
#   make setup   install dependencies into .venv
#   make all     run the whole pipeline: data -> graph -> candidates -> features -> model
#   make test    run the test suite
#   make run     start the FastAPI service on :8000
#   make demo    start the Streamlit dashboard on :8501
#
# Every stage is also runnable on its own; they are ordered by dependency and
# each writes its artifacts to data/ before the next reads them.

PY      := python
VENV    := .venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN     := $(VENV)/Scripts
endif
PYTHON  := $(BIN)/python
export PYTHONPATH := $(CURDIR)

.PHONY: help setup data graph candidates features train all test lint run demo clean clean-memory docker

help:
	@echo "Ringfence — Abuse-Ring Sentinel"
	@echo ""
	@echo "  make setup       create .venv and install requirements"
	@echo "  make all         data -> graph -> candidates -> features -> model"
	@echo "  make test        run pytest (test-scale universe, isolated temp dir)"
	@echo "  make run         FastAPI on http://localhost:8000"
	@echo "  make demo        Streamlit on http://localhost:8501"
	@echo ""
	@echo "  make data        generate the synthetic universe"
	@echo "  make graph       build the collision-capped identity graph"
	@echo "  make candidates  generate candidates and measure ring recall"
	@echo "  make features    compute 31 features and profile latency"
	@echo "  make train       cross-world CV and cost-optimal threshold"
	@echo "  make small       run the whole pipeline at test scale (fast)"
	@echo ""
	@echo "  make clean         remove generated artifacts"
	@echo "  make clean-memory  reset case memory (re-arms the failure demo)"

setup:
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Setup complete. Copy .env.example to .env if you want the LLM summary layer."

# ── pipeline, in dependency order ─────────────────────────────────────────

data:
	$(PYTHON) -m src.data_generator

graph:
	$(PYTHON) -m src.graph_builder

candidates:
	$(PYTHON) -m src.candidate_generator

features:
	$(PYTHON) -m src.feature_engine

train:
	$(PYTHON) -m src.scorer

all: data graph candidates features train
	@echo ""
	@echo "Pipeline complete. Artifacts in data/processed/. Now: make run && make demo"

small:
	$(PYTHON) -m src.data_generator --small
	$(MAKE) graph candidates features train

# ── quality ───────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q src frontend tests

# ── serving ───────────────────────────────────────────────────────────────

run:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

demo:
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

docker:
	docker compose up --build

# ── housekeeping ──────────────────────────────────────────────────────────

clean:
	rm -rf data/raw/*.csv data/processed/* data/ground_truth/*.csv
	@echo "Artifacts removed. Run 'make all' to rebuild."

clean-memory:
	rm -f data/processed/case_memory.sqlite data/processed/alerts.sqlite
	@echo "Case memory cleared — the failure-recovery demo is re-armed."
