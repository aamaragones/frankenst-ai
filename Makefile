SHELL := /bin/bash

PYTHON ?= python
IMAGE_NAME ?= frankenst-ai
CONTAINER_NAME ?= frankenst-ai-app
STREAMLIT_PORT ?= 8501
FUNCTION_APP_IMAGE_NAME ?= frankenst-ai-functions
FUNCTION_APP_CONTAINER_NAME ?= frankenst-ai-functions-app
FUNCTION_APP_PORT ?= 8080

FRANKSTATE_TESTS := tests/unit_test/frankstate
PACKAGE_PATHS := src/frankstate tests/unit_test/frankstate

.PHONY: help install install-dev lock format lint type test test-frankstate build clean streamlit mcp-server function-app-build function-app-run function-app-stop function-app-logs docker-build docker-run docker-stop docker-prune docker-rebuild

help:
	@echo "Available targets:"
	@echo "  quick-start    - uv venv .venv && source .venv/bin/activate"
	@echo "  install        - Sync the active virtual environment with uv"
	@echo "  install-dev    - Sync the active virtual environment with examples and dev extras"
	@echo "  lock           - Refresh uv.lock from pyproject.toml"
	@echo "  format         - Format the published package slice with ruff"
	@echo "  lint           - Run ruff checks on the published package slice"
	@echo "  type           - Run mypy on the published package slice"
	@echo "  test           - Run the full repository test suite"
	@echo "  test-frankstate - Run only the installable frankstate package tests"
	@echo "  build          - Build wheel/sdist and validate dist metadata"
	@echo "  clean          - Remove build, dist, cache and egg-info artifacts"
	@echo "  streamlit      - Run the local Streamlit app"
	@echo "  mcp-server     - Run the local FastMCP HTTP server"
	@echo "  function-app-build - Build the Azure Functions container image"
	@echo "  function-app-run   - Run the Azure Functions container locally"
	@echo "  function-app-stop  - Stop the Azure Functions container"
	@echo "  function-app-logs  - Tail logs from the Azure Functions container"
	@echo "  docker-build   - Build the root Docker image"
	@echo "  docker-run     - Build and run the root Docker image"
	@echo "  docker-stop    - Stop the running root Docker container"
	@echo "  docker-prune   - Remove unused Docker objects"
	@echo "  docker-rebuild - Prune Docker caches and rebuild the image"

install:
	uv sync --frozen

install-dev:
	uv sync --frozen --extra examples --group dev

lock:
	uv lock

format:
	uv run ruff format $(PACKAGE_PATHS)
	uv run ruff check --fix $(PACKAGE_PATHS)

lint:
	uv run ruff check $(PACKAGE_PATHS)

type:
	uv run mypy src/frankstate

test:
	uv run pytest -q

test-frankstate:
	uv run pytest -q $(FRANKSTATE_TESTS)

build: clean
	uv build
	uv run twine check dist/*

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache src/frankstate.egg-info

streamlit:
	uv run streamlit run app.py

mcp-server:
	PYTHONPATH=src uv run python src/services/mcp/server_oaklang_agent.py

function-app-build:
	docker build -f src/services/functions/Dockerfile -t $(FUNCTION_APP_IMAGE_NAME) .

function-app-run: function-app-build
	docker run --rm -it -p $(FUNCTION_APP_PORT):80 --name $(FUNCTION_APP_CONTAINER_NAME) $(FUNCTION_APP_IMAGE_NAME)

function-app-stop:
	@docker stop $(FUNCTION_APP_CONTAINER_NAME) || echo "Container $(FUNCTION_APP_CONTAINER_NAME) is not running."

function-app-logs:
	docker logs $(FUNCTION_APP_CONTAINER_NAME)

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run: docker-build
	docker run --rm -it -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

docker-stop:
	@docker stop $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) is not running."

docker-prune:
	docker system prune -f

docker-rebuild: docker-prune docker-build
