PYTHON := .venv/bin/python3.11

# Define your list of flights
FLIGHTS := hover1 hover2 climb1 climb2 lateral1 lateral2 longitudinal1 longitudinal2 yaw1 yaw2

# Process a specific flight
process-%:
	FLIGHT=$* $(PYTHON) process_data.py

# Process all flights
process-all:
	@for flight in $(FLIGHTS); do \
		echo "Processing $$flight..."; \
		FLIGHT=$$flight $(PYTHON) process_data.py; \
	done

# Rerun visuals for a specific flight
rerun-%:
	FLIGHT=$* $(PYTHON) rerun_visuals.py

# Rerun visuals for all flights
rerun-all:
	@for flight in $(FLIGHTS); do \
		echo "Rerunning visuals for $$flight..."; \
		FLIGHT=$$flight $(PYTHON) rerun_visuals.py; \
	done

# Default targets (single flight)
process: FLAG = hover1
process:
	FLIGHT=$(FLAG) $(PYTHON) process_data.py

rerun: FLAG = hover1
rerun:
	FLIGHT=$(FLAG) $(PYTHON) rerun_visuals.py