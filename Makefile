.PHONY: install test eval eval-fail serve

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) -m llm_eval run --dataset datasets/rag_faq.jsonl --out reports/

eval-fail:
	$(PYTHON) -m llm_eval run --dataset datasets/golden_fail.jsonl --out reports/golden-fail/

serve:
	$(PYTHON) -m llm_eval serve --host 127.0.0.1 --port 8000
