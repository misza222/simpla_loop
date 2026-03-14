.PHONY: help install test test-fast lint format typecheck check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	uv pip install -e ".[dev]"

test: ## Run all tests with coverage
	uv run pytest

test-fast: ## Run tests without coverage (faster)
	uv run pytest --no-cov -q

lint: ## Run ruff linter and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and auto-fix lint issues
	uv run ruff check . --fix
	uv run ruff format .

typecheck: ## Run mypy type checking
	uv run mypy src/simpla_loop

check: lint typecheck test ## Run all checks (lint + typecheck + test)

pre-commit: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ .eggs/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
