PYTHON := .venv/bin/python3.11

process-%:
	FLIGHT=$* $(PYTHON) process_data.py

rerun-%:
	FLIGHT=$* $(PYTHON) rerun_visuals.py

# Default targets
process: FLAG = hover1
process:
	FLIGHT=$(FLAG) $(PYTHON) process_data.py

rerun: FLAG = hover1
rerun:
	FLIGHT=$(FLAG) $(PYTHON) rerun_visuals.py