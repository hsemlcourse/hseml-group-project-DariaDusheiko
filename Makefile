.PHONY: lint lint-ruff lint-flake8 help

help:
	@echo "Targets:"
	@echo "  make lint        - run ruff and flake8"
	@echo "  make lint-ruff   - ruff check ."
	@echo "  make lint-flake8 - flake8 ."

lint: lint-ruff lint-flake8

lint-ruff:
	ruff check . --line-length 120

lint-flake8:
	flake8 . --max-line-length=120 --extend-exclude=.venv,notebooks,data
