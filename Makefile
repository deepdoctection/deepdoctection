.PHONY: clean clean-test clean-pyc clean-build develop help venv start pytest test

UNAME_S := $(shell uname -s)
PRODIGY_PATH=


PYTHON=python3.8


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

venv:
	$(PYTHON) -m venv venv --system-site-packages

format-and-qa: format qa

format:
	black --line-length 120 deepdoctection tests setup.py
	isort --skip deep_doctection tests setup.py

check-format:
	black --line-length 120 --check deepdoctection tests setup.py
	isort --check tests setup.py

lint:
	pylint deepdoctection tests tests_d2

analyze:
	mypy -p deepdoctection -p tests -p tests_d2

test:
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests
	pytest --cov=deepdoctection --cov-branch --cov-report=html tests_d2

test-des-tf:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "not requires_tf" tests

test-des-pt:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "not requires_pt" tests

test-integration:
	pytest --cov=deepdoctection --cov-branch --cov-report=html -m "integration" tests

qa: check-format lint analyze test

up-reqs: up-pip up-req-files install-dd

up-reqs-dev: up-reqs install-dd-all-dev

up-pip: check-venv
	@echo "--> Updating pip"
	pip install pip
	pip install --upgrade pip pip-tools
	pip install wheel
	@echo "--> Done updating pip"

up-pipx: check-venv
	@echo "--> Installing pipx"
	pip install pipx
	@echo "--> Done installing pipx"

up-req-files: check-venv
	@echo "--> Updating Python requirements"
	pip-compile --output-file requirements.txt setup.py
	@echo "--> Done updating Python requirements"

install-dd: check-venv
	@echo "--> Installing dependencies"
	pip install -r requirements.txt .
	@echo "--> Done installing dependencies"
	@echo ""

install-dd-tf: install-dd
	@echo "--> Installing tensorflow dependencies"
	pip install ".[tf]"
	@echo "--> Done installing tensorflow dependencies"
	@echo ""

install-dd-pt: install-dd
	@echo "--> Installing PT dependencies"
	pip install ".[pt]"
	@echo "--> Done installing PT dependencies"
	@echo ""

install-dd-all: check-venv install-dd-tf install-dd-pt

install-dd-all-dev: check-venv install-dd-dev install-dd-test

install-dd-dev: check-venv
	@echo "--> Installing requirements dependencies"
	pip install -r requirements.txt -e .
	@echo "--> Installing tensorflow dependencies"
	pip install -e ".[tf]"
	@echo "--> Installing pytorch dependencies"
	pip install -e ".[pt]"
	@echo "--> Installing aws dependencies"
	pip install -e ".[aws]"
	@echo "--> Installing dev dependencies"
	pip install -e ".[dev]"
	@echo "--> Done installing dev dependencies"
	@echo ""

install-dd-test: check-venv
	@echo "--> Installing test dependencies"
	pip install -e ".[test]"
	pip install -U pytest
	@echo "--> Done installing test dependencies"
	@echo ""

install-jupyterlab-setup: check-venv
	@echo "--> Installing Jupyter Lab"
	pip install jupyterlab>=3.0.0
	@echo "--> Done installing Jupyter Lab"

install-prodigy-setup: check-venv install-jupyterlab-setup
	@echo "--> Installing Prodigy"
	pip install $(PRODIGY_PATH)
	@echo "--> Done installing Prodigy"
	@echo "--> Installing Jupyter Lab Prodigy plugin"
	pip install jupyterlab-prodigy
	jupyter labextension list
	@echo "--> Done installing Jupyter Lab Prodigy plugin"
	@echo ""

install-kernel-dd: check-venv
	@echo "--> Installing IPkernel setup and setup kernel deep-doctection"
	pip install --user ipykernel
	$(PYTHON) -m ipykernel install --user --name=deep-doc
	@echo "--> Done installing kernel deep-doctection"

install-docker-env:  check-venv up-reqs-dev install-kernel-dd


check-venv:
ifndef VIRTUAL_ENV
	$(error Please activate virtualenv first)
endif