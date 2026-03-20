.PHONY: test lint demo

PYTHON ?= $(shell command -v python3.11 || command -v python3.10 || command -v python3.9 || command -v python3)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests examples

demo:
	PYTHONPATH=.:src $(PYTHON) examples/quickstart.py
