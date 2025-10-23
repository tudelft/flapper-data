PYTHON := .venv/bin/python

FLAG ?= hover1

process:
	FLIGHT=$(FLAG) $(PYTHON) process_data.py

rerun:
	FLIGHT=$(FLAG) $(PYTHON) rerun_visuals.py
