.PHONY: clean clean-test clean-pyc clean-build develop help venv start pytest test

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
PLATFORM=linux
PRODIGY_PATH=
endif


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
	black --line-length 120 deep_doctection tests setup.py
	isort --skip deep_doctection tests setup.py

check-format:
	black --line-length 120 --check deep_doctection tests setup.py
	isort --check tests setup.py

lint:
	pylint deep_doctection tests

analyze:
	mypy -p deep_doctection -p tests

test:
	pytest --cov=deep_doctection --cov-branch --cov-report=html tests

test-des-tf:
	pytest --cov=deep_doctection --cov-branch --cov-report=html -m "not requires_tf" tests

qa: check-format lint analyze test

up-reqs: up-pip up-req-files install-dependencies

up-reqs-dev: up-reqs install-dev-dependencies

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

install-dependencies: check-venv
	@echo "--> Installing dependencies"
	pip install -r requirements.txt -e .
	@echo "--> Done installing dependencies"
	@echo ""

install-tf-dependencies: install-dependencies
	@echo "--> Installing tensorflow dependencies"
	pip install -e ".[tf]"
	@echo "--> Done installing tensorflow dependencies"
	@echo ""

install-transformers-dependencies: install-dependencies
	@echo "--> Installing HF transformers dependencies"
	pip install -e ".[hf]"
	@echo "--> Done installing HF transformers dependencies"
	@echo ""

install-dev-dependencies: install-tf-dependencies install-transformers-dependencies
	@echo "--> Installing dev dependencies"
	pip install -e ".[dev]"
	pip install -e ".[test]"
	pip install -U pytest
	@echo "--> Done installing dependencies"
	@echo ""

install-jupyterlab-setup: check-venv
	@echo "--> Installing Jupyter Lab"
	pip install "jupyterlab>=2.0.0,<3.0.0"
	@echo "--> Done installing Jupyter Lab"

install-prodigy-setup: check-venv install-jupyterlab-setup
	@echo "--> Installing Prodigy"
	pip install $(PRODIGY_PATH)
	@echo "--> Done installing Prodigy"
	@echo "--> Installing Jupyter Lab Prodigy plugin"
	jupyter labextension install jupyterlab-prodigy
	@echo "--> Done installing Jupyter Lab Prodigy plugin"
	@echo ""

install-kernel-deepdoc: check-venv up-reqs-dev
	@echo "--> Installing IPkernel setup and setup kernel deepdoctection"
	pip install --user ipykernel
	$(PYTHON) -m ipykernel install --user --name=deep-doc
	@echo "--> Done installing kernel deepdoctection"

install-docker-env:  check-venv up-reqs-dev install-kernel-deepdoc

install-aws-dependencies: check-venv
	@echo "--> Installing aws dependencies"
	pip install -e ".[aws]"
	@echo "--> Done installing aws dependencies"

check-venv:
ifndef VIRTUAL_ENV
	$(error Please activate virtualenv first)
endif