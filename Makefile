# Variables
PYTHON = python3
PIP = pip
# Path to entry point assuming it is moved into src/frontend
APP_PATH = src/frontend/App.py

.PHONY: version

version:
	@echo "Current Project Version:"
	@python3 -c "from src.version import __version__; print(__version__)"

# Helper to quickly bump version (Example: make bump V=1.0.1)
bump:
	echo "__version__ = \"$(V)\"" > src/version/version.py
	echo "__build_date__ = \"$$(date +%Y-%m-%d)\"" >> src/version/version.py

.PHONY: help install run test clean lint

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies and project"
	@echo "  make run        Launch the Chainlit application"
	@echo "  make test       Run unit tests in the /tests directory"
	@echo "  make clean      Remove caches, logs, and local index data"
	@echo "  make lint       Run code formatting checks"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

run:
	chainlit run $(APP_PATH) -w

test:
	pytest tests/

clean:
	rm -rf ./logs/*.log
	rm -rf ./data/index_data/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	black src/ tests/
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
