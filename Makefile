# AutoSteer Automation

.PHONY: setup run clean test

# Detect shell for uv activation
SHELL := /bin/bash

# Setup the environment using uv
setup:
	@echo "--> [Setup] Creating virtual environment with uv..."
	uv venv
	@echo "--> [Setup] Installing dependencies..."
	uv pip install -e .

# Run the experiment
run:
	@echo "--> [Experiment] Launching AutoSteer..."
	@# Ensure we are using the venv python
	uv run python src/main.py

# Clean artifacts
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run Marimo in edit mode
marimo-edit:
	uv run marimo edit notebook/deep_dive.py

# Run Marimo in app mode
marimo-run:
	uv run marimo run notebook/deep_dive.py
