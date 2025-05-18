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

format-and-qa: format qa

format: black isort

install-dd-dev-pt: check-venv
	@echo "--> Installing source-all-pt"
	pip install -e ".[source-pt]"
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

install-dd-test: check-venv
	@echo "--> Installing test dependencies"
	pip install -e ".[test]"
	@echo "--> Done installing test dependencies"
	@echo ""

isort:
	isort  deepdoctection tests setup.py

lint:
	pylint deepdoctection tests tests_d2

package: check-venv
	@echo "--> Generating package"
	pip install --upgrade build
	$(PYTHON) -m build

package_actions: check-venv
	@echo "--> Generating package"
	pip install --upgrade build
	$(PYTHON) -m build

qa: lint analyze test-basic

# all tests - this will never succeed in full due to dependency conflicts
test:
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2

test-basic:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m basic tests

test-additional: test-basic
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m additional tests

test-pt-legacy:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "pt_legacy" tests

test-tf-legacy:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "tf_legacy" tests

test-tf: test-additional
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "tf_deps" tests

test-pt: test-additional
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "pt_deps" tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2

test-gpu:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "requires_gpu" tests

up-pip: check-venv
	@echo "--> Updating pip"
	pip install pip
	pip install --upgrade pip pip-tools
	pip install wheel
	@echo "--> Done updating pip"

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