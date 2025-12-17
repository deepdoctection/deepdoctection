.PHONY: clean clean-test clean-pyc clean-build develop help venv start pytest test install-dd

UNAME_S := $(shell uname -s)

PYTHON=python3

# optional: R=1 enables --recreate (-r) for tox
ifdef R
TOX_R = -r
endif

help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*##' Makefile | sed 's/:.*##/: /' | sort

analyze: ## Run type checks for all packages. Set R=1 to recreate tox envs
	tox -c packages/tox.ini -e py310-type-all $(TOX_R)

check-lint-and-format: ## Run lint and format checks for all packages. Set R=1 to recreate tox envs
	tox -c packages/tox.ini -e py310-check-format-and-lint-all $(TOX_R)

clean: clean-build clean-pyc clean-test ## Clean build, Python, and test artifacts

clean-build: ## Remove build artifacts
	rm -fr build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc: ## Remove Python bytecode and cache files
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts (top-level and packages)
	# top-level
	rm -fr .coverage coverage htmlcov .pytest_cache
	rm -fr .hypothesis
	# everything under packages (dd_core, dd_datasets, deepdoctection, etc.)
	find packages -maxdepth 3 -type d \( \
		-name '.tox' -o \
		-name '.hypothesis' -o \
		-name '.pytest_cache' -o \
		-name 'coverage' -o \
		-name 'htmlcov' -o \
		-name 'dist' -o \
		-name 'build' \
	\) -exec rm -rf {} +
	find packages -maxdepth 4 -type f \( \
		-name '.coverage' -o \
		-name 'coverage.xml' \
	\) -exec rm -f {} +

qa: analyze check-lint-and-format ## Run full QA suite (types, lint, format)

format: ## Auto-format all Python code. Set R=1 to recreate tox envs
	tox -c packages/tox.ini -e py310-format-all $(TOX_R)

install-dd-dev: check-venv ## Install dev editable packages with extras
	pip install --no-build-isolation "detectron2 @ git+https://github.com/deepdoctection/detectron2.git"
	pip install -e ./packages/dd_core[full,dev,test]
	pip install -e ./packages/dd_datasets[full,dev,test]
	pip install -e ./packages/deepdoctection[full,dev,test]

package_actions: check-venv ## Build packages for dd_core, dd_datasets, deepdoctection.
	@echo "--> Generating packages for dd_core, dd_datasets, deepdoctection"
	pip install --upgrade build
	cd packages/dd_core && $(PYTHON) -m build
	cd packages/dd_datasets && $(PYTHON) -m build
	cd packages/deepdoctection && $(PYTHON) -m build

# default env if ENV is not provided
TOX_ENV ?= py310-core-test

test: ## Run tests via tox. TOX_ENV selectable and R=1 to recreate tox envs
	tox -c packages/tox.ini -e $(TOX_ENV) $(TOX_R)


up-req: check-venv ## Update requirements.txt for all packages from pyproject.toml
	@echo "--> Updating Python requirements from pyproject.toml"
	pip-compile \
		--output-file packages/dd_core/requirements.txt \
		--resolver=backtracking \
		--extra full \
		--extra dev \
		--extra test \
		packages/dd_core/pyproject.toml -c /dev/null
	# dd_datasets: full+dev+test
	pip-compile \
		--output-file packages/dd_datasets/requirements.txt \
		--resolver=backtracking \
		--extra full \
		--extra dev \
		--extra test \
		packages/dd_datasets/pyproject.toml -c /dev/null
	# deepdoctection: full+dev+test
	pip-compile \
		--output-file packages/deepdoctection/requirements.txt \
		--resolver=backtracking \
		--extra full \
		--extra dev \
		--extra test \
		packages/deepdoctection/pyproject.toml -c /dev/null
	@echo "--> Done updating Python requirements"

up-req-docs: check-venv ## Update docs requirements from deepdoctection extras
	@echo "--> Updating docs requirements from pyproject.toml"
	pip-compile \
		--output-file docs/requirements.txt \
		--resolver=backtracking \
		--extra docs \
		packages/deepdoctection/pyproject.toml -c /dev/null
	@echo "--> Done updating docs requirements"


venv: ## Create a virtual environment in ./venv
	$(PYTHON) -m venv venv --system-site-packages

check-venv:
ifndef VIRTUAL_ENV
	$(error Please activate virtualenv first)
endif