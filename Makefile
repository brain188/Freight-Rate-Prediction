# Freight Rate Prediction Challenge
#
# `make all` takes a fresh clone through install, training, prediction, and the
# assessment's own scorer. Run `make help` to see every target.

PYTHON ?= python
CONFIG ?= config/config.yaml

SUBMISSION := validation_predictions.csv
DECEMBER   := data/december_chart_inputs.csv
CHART      := scorer_results/candidate_december.png

.DEFAULT_GOAL := help
.PHONY: help setup train predict score test lint all clean clean-outputs

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:  ## Train the model and save it to models/
	$(PYTHON) entrypoint/train.py --config $(CONFIG)

predict:  ## Write both output files from the saved model
	$(PYTHON) entrypoint/predict.py --config $(CONFIG)

score:  ## Validate both files and draw the December chart
	$(PYTHON) score.py --predictions $(SUBMISSION) --december-predictions $(DECEMBER)

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Check formatting and style
	$(PYTHON) -m ruff check src tests entrypoint
	$(PYTHON) -m ruff format --check src tests entrypoint

all: setup train predict score  ## Everything, from install to the finished chart
	@echo ""
	@echo "Done. Submission: $(SUBMISSION)"
	@echo "      Chart:      $(CHART)"

clean-outputs:  ## Remove generated predictions, model, and charts
	rm -rf models scorer_results logs
	rm -f $(SUBMISSION)

clean: clean-outputs  ## Remove outputs and Python caches
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

# --- Flows and CI ----------------------------------------------------------

monitor:  ## Check drift and live accuracy once
	$(PYTHON) flows/monitor.py --days 7

retrain:  ## Train a candidate and promote it only if it beats production
	$(PYTHON) flows/retrain.py

retrain-dry:  ## Train and compare without promoting
	$(PYTHON) flows/retrain.py --dry-run

schedule:  ## Register both flows on their schedules with Prefect
	$(PYTHON) flows/deployments.py

ci:  ## Everything the pipeline runs, locally
	$(PYTHON) -m ruff check src tests serving monitoring flows
	$(PYTHON) -m pytest -q
	$(PYTHON) flows/retrain.py --dry-run

# --- Docker ----------------------------------------------------------------

up:  ## Build and start the whole stack
	docker compose up --build -d
	@echo ""
	@echo "  API        http://localhost:8000/docs"
	@echo "  Dashboard  http://localhost:8050"
	@echo "  MLflow     http://localhost:5000"

down:  ## Stop the stack, keeping the data
	docker compose down

nuke:  ## Stop the stack and wipe the databases
	docker compose down -v

ps:  ## Show what is running
	docker compose ps

tail:  ## Follow the logs
	docker compose logs -f