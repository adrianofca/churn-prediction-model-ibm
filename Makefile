.PHONY: install lint format test run

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

run:
	uv run uvicorn api.main:app --reload
