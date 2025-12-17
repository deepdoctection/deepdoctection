.PHONY: clean clean-test clean-pyc clean-build develop help venv start pytest test install-dd

UNAME_S := $(shell uname -s)

PYTHON=python3

# optional: R=1 enables --recreate (-r) for tox
ifdef R
TOX_R = -r
endif

analyze:
	tox -c packages/tox.ini -e py310-type-all $(TOX_R)

check-lint-and-format:
	tox -c packages/tox.ini -e py310-check-format-and-lint-all $(TOX_R)

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr packages/.tox/
	rm -fr packages/.hypothesis/
	rm -fr packages/.coverage
	rm -fr packages/coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

qa: analyze check-lint-and-format

format:
	tox -e py310-format-all -r

install-dd-dev-pt: check-venv
	@echo "--> Installing pt"
	pip install -e ".[pt]"
	@echo "--> Installing dev, test dependencies"
	pip install -e ".[dev, test]"
	@echo "--> Done installing dev, test dependencies"
	pip install -e ".[docs]"
	@echo "--> Done installing docs dependencies"
	@echo ""

isort:
	isort  deepdoctection tests setup.py

lint:
	pylint deepdoctection tests tests_d2

package:
	@echo "--> Generating package"
	pip install --upgrade build
	$(PYTHON) -m build

package_actions: check-venv
	@echo "--> Generating package"
	pip install --upgrade build
	$(PYTHON) -m build

# default env if ENV is not provided
TOX_ENV ?= py310-core-test

test:
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2


up-req: check-venv
	@echo "--> Updating Python requirements"
	pip-compile  --output-file requirements.txt  setup.py
	@echo "--> Done updating Python requirements"

up-req-docs: check-venv
	@echo "--> Updating Python requirements"
	pip-compile  --output-file docs/requirements.txt  --extra docs  setup.py
	@echo "--> Done updating Python requirements"

install-dd: check-venv
	@echo "--> Installing detectron2 without build isolation"
	pip install --no-build-isolation 'detectron2 @ git+https://github.com/deepdoctection/detectron2.git'
	@echo "--> Installing local packages"
	pip install -e  ./packages/shared_test_utils
	pip install -e ./packages/dd_core[full]
	pip install -e ./packages/dd_datasets[full]
	@echo "--> Installing deepdoctection (full) into active venv"
	pip install -e ./packages/deepdoctection[full]
	@echo "--> Done"

venv:
	$(PYTHON) -m venv venv --system-site-packages

check-venv:
ifndef VIRTUAL_ENV
	$(error Please activate virtualenv first)
endif