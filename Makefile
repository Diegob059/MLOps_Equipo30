#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo30
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies and local package in editable mode
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files, caches and build artifacts
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache build dist *.egg-info

## Lint source code using ruff / flake8
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff check src/ tests/
	$(PYTHON_INTERPRETER) -m ruff format --check src/ tests/

## Format source code using ruff / black
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix src/ tests/
	$(PYTHON_INTERPRETER) -m ruff format src/ tests/

## Run unit tests with pytest
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

#################################################################################
# PROJECT PIPELINES (CCDS v1)                                                   #
#################################################################################

## Download raw data from MiBici portal
.PHONY: download_data
download_data:
	$(PYTHON_INTERPRETER) -m src.data.p2_download_raw_data

## Build interim dataset and engineer features
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m src.data.p2_dataset
	$(PYTHON_INTERPRETER) -m src.features.p2_features

## Train all models and log experiments to MLflow (P1, P2 and P3)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m src.models.run_all_experiments

## Train specifically Question 2 risk classifier
.PHONY: train_p2
train_p2:
	$(PYTHON_INTERPRETER) -m src.models.p2_train

## Launch local MLflow Tracking server on port 5005
.PHONY: mlflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5005

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

## Prepare Question 1 hourly flows and features
.PHONY: data_p1
data_p1:
	$(PYTHON_INTERPRETER) -m src.data.p1_dataset
	$(PYTHON_INTERPRETER) -m src.features.p1_features

## Train Question 1 and log the four comparable MLflow runs
.PHONY: train_p1
train_p1:
	$(PYTHON_INTERPRETER) -m src.models.p1_train
