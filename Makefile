.PHONY: install seed seed-pg run mcp lint typecheck test test-pg check eval docker-up

PG_URL ?= postgresql://localhost/speech_demo
PG_TEST_URL ?= postgresql://localhost/s2s_test

install:            ## install runtime + dev dependencies
	uv sync

seed:               ## create the SQLite demo database at ./data/demo.db
	uv run python -m speech_to_sql.demo.seed --url sqlite:///./data/demo.db --if-missing

seed-pg:            ## load the demo dataset into $(PG_URL) (drops existing demo tables)
	uv run python -m speech_to_sql.demo.seed --url $(PG_URL) --reset

run:                ## run the API + web UI + MCP endpoint on :8000 (reads .env)
	uv run speech-to-sql

mcp:                ## run as an MCP server over stdio (for local agents)
	uv run speech-to-sql mcp

lint:
	uv run ruff check src tests evals
	uv run ruff format --check src tests evals

typecheck:
	uv run mypy src

test:               ## unit + API tests (no network, no API keys needed)
	uv run pytest -q

test-pg:            ## also run Postgres integration tests against a disposable DB
	S2S_TEST_PG_DSN=$(PG_TEST_URL) uv run pytest -q

check: lint typecheck test

eval:               ## execution-accuracy evals (calls the Claude API; costs money)
	uv run python evals/run_evals.py

docker-up:          ## Postgres + demo data + read-only role + service
	docker compose up --build
