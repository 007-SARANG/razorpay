# Trikon -- development tasks.
#
# PYTEST_DISABLE_PLUGIN_AUTOLOAD is set because some system Python installs (e.g. a ROS
# distribution on PATH) register global pytest plugins that fail to import. Trikon needs
# no third-party plugins, so disabling autoload keeps tests reproducible anywhere.

VENV := .venv
PY   := $(VENV)/bin/python
PORT ?= 8000

.PHONY: help install test lint run stress sweep report dashboard doctor clean docker

help:
	@echo "make install    create venv and install (editable, with dev extras)"
	@echo "make test       run the test suite"
	@echo "make run        reconcile 1000 orders and print the report"
	@echo "make stress     accuracy + throughput across five batch sizes"
	@echo "make sweep      auto-accept threshold tradeoff table"
	@echo "make report     write data/reports/report.json"
	@echo "make dashboard  serve the dashboard on PORT=$(PORT)"
	@echo "make doctor     show config and probe the LLM provider"

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest tests/ -q

lint:
	$(PY) -m compileall -q trikon api && echo "compile OK"

run:
	$(PY) -m trikon.cli run --orders 1000 --exceptions 5 --cash

stress:
	$(PY) -m trikon.cli stress --sizes 120 500 1000 5000 10000

sweep:
	$(PY) -m trikon.cli sweep --orders 1000

report:
	$(PY) -m trikon.cli run --orders 1000 --json-out data/reports/report.json

dashboard:
	@echo "Dashboard: http://localhost:$(PORT)"
	$(PY) -m uvicorn api.main:app --port $(PORT)

doctor:
	$(PY) -m trikon.cli doctor

docker:
	docker build -t trikon:latest .
	@echo "run: docker run --rm -p 8000:8000 trikon:latest"

clean:
	rm -rf data/batches data/reports .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
