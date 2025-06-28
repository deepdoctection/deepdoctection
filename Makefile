.PHONY: clean clean-test clean-pyc clean-build develop help venv start pytest test

UNAME_S := $(shell uname -s)


PYTHON=python3

black:
	black --line-length 120 deepdoctection tests setup.py

analyze:
	mypy -p deepdoctection -p tests -p tests_d2

check-format:
	black --line-length 120 --check deepdoctection tests setup.py
	isort --check tests setup.py

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

format-and-qa: format lint analyze

format: black isort

install-dd-dev-pt: check-venv
	@echo "--> Installing pt"
	pip install -e ".[pt]"
	@echo "--> Installing dev, test dependencies"
	pip install -e ".[dev, test]"
	@echo "--> Done installing dev, test dependencies"
	pip install -e ".[docs]"
	@echo "--> Done installing docs dependencies"
	@echo ""

install-dd-dev-tf: check-venv
	@echo "--> Installing source-all-tf"
	pip install -e ".[tf]"
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

# all tests - this will never succeed in full due to dependency conflicts
test:
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2

# tests that does only require the basic detup
test-basic:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m basic tests

# tests that require additional dependencies not based on DL libraries
test-additional: test-basic
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m additional tests

# analyzer with legacy configurations
test-pt-legacy:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "pt_legacy" tests

# tests with full TF setup
test-tf: test-additional
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "tf_deps" tests

# tests with full PT setup
test-pt: test-additional
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "pt_deps" tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2

test-gpu:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "requires_gpu" tests

up-req: check-venv
	@echo "--> Updating Python requirements"
	pip-compile  --output-file requirements.txt  setup.py
	@echo "--> Done updating Python requirements"

up-req-docs: check-venv
	@echo "--> Updating Python requirements"
	pip-compile  --output-file docs/requirements.txt  --extra docs  setup.py
	@echo "--> Done updating Python requirements"



venv:
	$(PYTHON) -m venv venv --system-site-packages

check-venv:
ifndef VIRTUAL_ENV
	$(error Please activate virtualenv first)
endif