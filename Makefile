# Freight Rate Prediction
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