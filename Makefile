SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE_NAME ?= frankenst-ai
CONTAINER_NAME ?= frankenst-ai-mcp
MCP_PORT ?= 8000
FUNCTION_APP_IMAGE_NAME ?= frankenst-ai-functions
FUNCTION_APP_CONTAINER_NAME ?= frankenst-ai-functions-app
FUNCTION_APP_PORT ?= 8080

LINT_PATHS := src tests .github/scripts main.py
FRANKSTATE_TESTS := tests/unit_test/frankstate
COMMENT_RATIO_MAX ?= 0.15
AUDIT_REQUIREMENTS := .audit-requirements.txt
AUDIT_IGNORE :=

.PHONY: help install install-dev lock format format-check lint type test test-frankstate cov cov-frankstate \
	config comment-ratio yaml-check audit hooks pre-commit build release-prepare ci clean mcp-server \
	function-app-build function-app-run function-app-stop function-app-logs \
	docker-build docker-run docker-stop docker-prune docker-rebuild

help: ## Show every target
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Sync the venv with the locked runtime dependencies
	uv sync --frozen

install-dev: ## Sync the venv with the examples extra and the dev group
	uv sync --frozen --extra examples --group dev

lock: ## Refresh uv.lock from pyproject.toml
	uv lock

format: ## Apply ruff format and safe fixes over the whole tree
	uv run ruff format $(LINT_PATHS)
	uv run ruff check --fix $(LINT_PATHS)

format-check: ## Fail if ruff format would change anything
	uv run ruff format --check $(LINT_PATHS)

lint: ## ruff check over the whole tree
	uv run ruff check $(LINT_PATHS)

type: ## Strict mypy over [tool.mypy].files
	uv run mypy

test: ## Run every test
	uv run pytest -q

test-frankstate: ## Run only the published slice's tests
	uv run pytest -q $(FRANKSTATE_TESTS)

cov: ## Whole-tree coverage against [tool.coverage.report].fail_under
	uv run pytest -q --cov --cov-report=term-missing

cov-frankstate: ## Published slice coverage, the 90% release gate
	uv run pytest -q $(FRANKSTATE_TESTS) --cov=src/frankstate --cov-report=term-missing --cov-fail-under=90

config: ## Construct settings once, so a broken env fails here and not in a layout
	PYTHONPATH=src uv run python -c "from config.settings import get_settings; get_settings()"

comment-ratio: ## Fail when '#' comment lines exceed COMMENT_RATIO_MAX of a file (frankstate reported only)
	uv run python .github/scripts/comment_ratio.py --max $(COMMENT_RATIO_MAX) --report-only src/frankstate src main.py .github/scripts

yaml-check: ## Fail on any .yml file (house rule: .yaml)
	@BAD=$$(find . -name '*.yml' -not -path './.venv/*' -not -path './node_modules/*' -not -path './.git/*'); \
	test -z "$$BAD" || { echo "Use .yaml, not .yml:"; echo "$$BAD"; exit 1; }

audit: ## pip-audit over the locked runtime dependencies (AUDIT_IGNORE lists accepted advisories)
	uv export --quiet --frozen --no-dev --no-emit-project --output-file $(AUDIT_REQUIREMENTS)
	uvx pip-audit --requirement $(AUDIT_REQUIREMENTS) $(AUDIT_IGNORE)

hooks: ## Install the pre-commit hooks
	uv run pre-commit install

pre-commit: ## Run every pre-commit hook over the whole tree
	uv run pre-commit run --all-files

build: clean ## Build wheel + sdist and validate their metadata
	uv build
	uv run twine check dist/*

release-prepare: ## Set VERSION in pyproject + uv.lock and build dist (called by semantic-release)
	@test -n "$(VERSION)" || { echo "VERSION=X.Y.Z is required"; exit 1; }
	uv version $(VERSION) --no-sync
	$(MAKE) build

ci: yaml-check lint format-check type comment-ratio cov cov-frankstate audit build pre-commit ## Everything CI runs, in CI order

clean: ## Remove build, dist, cache and audit artifacts
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache .coverage src/frankstate.egg-info $(AUDIT_REQUIREMENTS)

mcp-server: ## Run the FastMCP HTTP server locally
	PYTHONPATH=src uv run python src/services/mcp/server_oaklang_agent.py

function-app-build: ## Build the Azure Functions image
	docker build -f src/services/functions/Dockerfile -t $(FUNCTION_APP_IMAGE_NAME) .

function-app-run: function-app-build ## Run the Azure Functions container
	docker run --rm -it -p $(FUNCTION_APP_PORT):80 --name $(FUNCTION_APP_CONTAINER_NAME) $(FUNCTION_APP_IMAGE_NAME)

function-app-stop: ## Stop the Azure Functions container
	@docker stop $(FUNCTION_APP_CONTAINER_NAME) || echo "Container $(FUNCTION_APP_CONTAINER_NAME) is not running."

function-app-logs: ## Tail the Azure Functions container logs
	docker logs $(FUNCTION_APP_CONTAINER_NAME)

docker-build: ## Build the root image (MCP server)
	docker build -t $(IMAGE_NAME) .

docker-run: docker-build ## Run the root image
	docker run --rm -it -p $(MCP_PORT):$(MCP_PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

docker-stop: ## Stop the root container
	@docker stop $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) is not running."

docker-prune: ## Remove unused Docker objects
	docker system prune -f

docker-rebuild: docker-prune docker-build ## Prune and rebuild the root image
