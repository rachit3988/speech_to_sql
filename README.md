# Speech-to-SQL Agent

Ask a database questions out loud and get back an answer, the SQL behind it, and the rows.

```
🎤 "who were our top five customers by revenue last year"
 → Deepgram Nova-3 transcript
 → Claude Opus 5.5 agent: looks up stored values → writes SQL → runs it → checks the result
 → a short answer you can read aloud, plus the SQL and the result rows
```

- **Agentic, not one-shot.** Claude explores the schema, resolves misheard names against the data
  ("globe x" → `Globex Inc`), runs its query, reads the result, and fixes its own mistakes before answering.
- **Read-only by construction.** Three independent safety layers between the model and your data (below).
- **Production plumbing.** Async FastAPI, streaming progress over SSE, auth, rate limiting, structured logs,
  Prometheus metrics, health/readiness probes, bounded retries and timeouts everywhere, Docker, CI.
- **Usable by other agents.** An MCP server (`/mcp` over HTTP, or stdio) lets Claude Code, Claude
  Desktop, the Claude API's MCP connector, or any MCP client hand data questions to this agent.
- **Measured.** An execution-accuracy eval suite with spoken-style questions, plus 100+ offline tests.

## Architecture

```mermaid
flowchart LR
    UI[Web UI / API client] -->|audio| STT[Deepgram Nova-3<br/>keyterm prompting]
    STT -->|transcript, editable| UI
    UI -->|question + history| API[FastAPI<br/>auth · rate limit · SSE]
    Agents[Other agents<br/>MCP clients] -->|"/mcp (HTTP) or stdio"| MCP[MCP server<br/>ask_database · run_sql …]
    MCP --> Agent
    API --> Agent[SQLAgent<br/>Claude Opus 5.5 tool runner]
    Agent -->|describe_tables| Catalog[(Schema catalog<br/>TTL cache)]
    Agent -->|search_column_values| DB
    Agent -->|run_sql| Guard[SQL guard<br/>sqlglot AST checks] --> DB[(Postgres<br/>read-only role + txn)]
    Agent -->|FinalAnswer JSON| API
```

| Module | Responsibility |
| --- | --- |
| `stt/` | `Transcriber` protocol; Deepgram REST adapter with retries/backoff and domain keyterms |
| `agent/` | Prompt construction, the three tools, and the bounded agent loop |
| `sql/guard.py` | Parses every statement and rejects anything that isn't a single side-effect-free SELECT over allowed tables/columns; enforces a row cap |
| `db/` | `Database` protocol; Postgres (asyncpg) and SQLite adapters; schema introspection and catalog cache |
| `api/` | App factory, routes, SSE streaming, request-ID middleware |
| `mcp_server.py` | MCP tools for other agents; Streamable HTTP endpoint and stdio entry point |
| `runtime.py`, `security.py` | Service wiring shared by REST and MCP; API-key auth and rate limiting |
| `web/` | Dependency-free browser UI (mic recording, live progress, result tables, optional read-aloud) |
| `evals/` | Execution-accuracy evals against the demo database |

### How the agent works

The agent runs on the Anthropic SDK's tool runner with Claude Opus 5.5:

- **Tools:** `describe_tables` (columns, keys, comments), `search_column_values` (find how a spoken value is
  stored, with trigram fuzzy matching on Postgres), `run_sql` (guarded execution returning a result preview).
  All tools use `strict: true` schemas; tool errors go back to the model so it can correct itself.
- **Structured final answer:** the last turn is a `FinalAnswer` JSON object (`answered` /
  `needs_clarification` / `cannot_answer`, a speakable summary, the SQL, and assumptions). The final SQL is
  re-validated and its result served from the per-run cache, so the rows you get are the rows the model saw.
- **Adaptive thinking + effort:** effort defaults to `medium` and is configurable (`low` cuts latency for voice).
- **Prompt caching:** the system prompt (instructions + schema) is byte-stable and cached; per-request data
  (date, history, question) lives in the user turn. Within a run, the growing conversation is cached too.
- **Refusal fallback:** server-side `fallbacks: "default"` reroutes the rare classifier refusal to an
  appropriate model instead of failing the request (disable with `S2S_LLM_REFUSAL_FALLBACK=false`).
- **Bounds:** step budget (`S2S_AGENT_MAX_ITERATIONS`), wall-clock deadline (`S2S_AGENT_TIMEOUT_S`),
  truncated-turn detection, and immediate abort if the database becomes unavailable.
- **Follow-ups without server state:** clients send the previous turns (question, SQL, answer); the server is
  stateless and scales horizontally.

## Safety model

Model-written SQL never runs with more than read access. Each layer is independent:

1. **Database role** — connect as a least-privilege role (`deploy/postgres/readonly_role.sql`): SELECT-only
   grants, `default_transaction_read_only`, role-level `statement_timeout`, optional column-level grants.
   The schema catalog only includes tables/columns the role can actually SELECT, so revoked columns
   disappear from the agent's view automatically.
2. **Transaction** — every query runs in an explicit `READ ONLY` transaction with `SET LOCAL statement_timeout`
   and a client-side deadline; results are fetched through a cursor capped at `max_result_rows + 1`.
3. **SQL guard** — sqlglot AST validation: exactly one statement; root must be a query; no DML/DDL anywhere
   (including data-modifying CTEs), no `SELECT INTO`, no row locks; denylisted functions (`pg_*`, `dblink*`,
   `query_to_xml`, `set_config`, `nextval`, …) and schema-qualified function calls rejected; every table must be
   in the catalog (no `pg_catalog` / `information_schema`); hidden columns blocked including `*` and whole-row
   references; `LIMIT` injected or clamped. The *re-generated* SQL is what executes.

Also: tool results are framed as data, not instructions; hidden tables/columns
(`S2S_HIDDEN_TABLES`, `S2S_HIDDEN_COLUMNS`) never reach the prompt; transcripts are not logged unless
`S2S_LOG_PAYLOADS=true`.

> Treat the guard as defense in depth, not the primary control. Always connect with the read-only role and
> use database grants (or a reporting replica / views) for anything that must stay private.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
cp .env.example .env   # then set ANTHROPIC_API_KEY (and DEEPGRAM_API_KEY for voice)
```

**Fastest path (SQLite demo data):**

```bash
make seed
```

```bash
make run
```

Open http://localhost:8000. Without `DEEPGRAM_API_KEY` the UI runs text-only in dev.

**Postgres (recommended, matches production):**

```bash
createdb speech_demo && make seed-pg PG_URL=postgresql://localhost/speech_demo
```

```bash
psql -d speech_demo -v app_password="'change-me'" -f deploy/postgres/readonly_role.sql
```

Then set `S2S_DATABASE_URL=postgresql://s2s_readonly:change-me@localhost/speech_demo` in `.env` and `make run`.

**Docker (Postgres + demo data + read-only role + service):**

```bash
ANTHROPIC_API_KEY=sk-ant-... DEEPGRAM_API_KEY=... docker compose up --build
```

## Use from other agents (MCP)

The service is also an [MCP](https://modelcontextprotocol.io) server, so other agents can call it as a
tool when they need data. It exposes four read-only tools, each annotated `readOnlyHint`:

| Tool | Use |
| --- | --- |
| `ask_database(question, context?, max_rows?)` | The full agent: plans, writes, runs and checks SQL; returns `status`, `answer`, `rows`, `columns`, `sql`, `assumptions` as structured output, with progress notifications while it works |
| `describe_database(tables?)` | Schema (tables, columns, keys, comments) so a caller can check the data exists |
| `find_values(table, column, search_term)` | How a value is stored (`"acme"` → `Acme Corporation`) |
| `run_sql(sql, max_rows?)` | One guarded read-only SELECT, for callers that write their own SQL (no LLM cost) |

All four go through the same SQL guard, read-only transaction and database role as the REST API.
Set `S2S_MCP_DATA_DESCRIPTION` to one line describing the data: it goes into the server instructions and
tool descriptions, which is what tells a calling agent *when* to use this server. `run_sql` and
`find_values` can be turned off with `S2S_MCP_ALLOW_RAW_SQL=false`.

**Remote (HTTP).** `speech-to-sql` serves MCP at `/mcp` (Streamable HTTP, stateless, so any replica can
serve any request). Clients authenticate with `Authorization: Bearer <key>` or `X-API-Key` using a key
from `S2S_API_KEYS`. To add it to Claude Code:

```bash
claude mcp add --transport http sql-agent https://sql-agent.example.com/mcp --header "Authorization: Bearer $S2S_AGENT_KEY"
```

From the Claude API, Anthropic connects to the server for you (it must be reachable from the internet):

```python
client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{"type": "url", "url": "https://sql-agent.example.com/mcp",
                  "name": "sql-agent", "authorization_token": os.environ["S2S_AGENT_KEY"]}],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "sql-agent"}],
    messages=[{"role": "user", "content": "Which region grew fastest last quarter?"}],
)
```

Any MCP client works the same way, e.g. with the MCP Python SDK:

```python
async with Client("http://localhost:8000/mcp") as sql_agent:
    result = await sql_agent.call_tool("ask_database", {"question": "top 5 products by revenue in 2025"})
    print(result.structured_content["answer"])
```

**Local (stdio).** `speech-to-sql mcp` runs the same tools over stdin/stdout for an agent that launches
the process itself (logs go to stderr; voice is not loaded). It reads `.env` from its working directory,
so `uv --directory` points it at this project's configuration. For Claude Code:

```bash
claude mcp add --transport stdio sql-agent -- uv --directory /path/to/speech_to_sql run speech-to-sql mcp
```

Or commit a `.mcp.json` to a project so every teammate's Claude Code gets it:

```json
{
  "mcpServers": {
    "sql-agent": {
      "type": "http",
      "url": "https://sql-agent.example.com/mcp",
      "headers": {"Authorization": "Bearer ${S2S_AGENT_KEY}"}
    }
  }
}
```

Notes: `ask_database` usually takes 5–60 s (up to `S2S_AGENT_TIMEOUT_S`), so give calling clients a read
timeout above that. Without API keys, `/mcp` only accepts requests addressed to localhost (DNS-rebinding
protection); add public host names with `S2S_MCP_ALLOWED_HOSTS` or, better, enable `S2S_API_KEYS`. Each
`ask_database` / `run_sql` call counts against the caller's rate limit.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/transcribe` | multipart `audio` → `{text, confidence, duration_s}` |
| `POST` | `/v1/query` | `{question, history?, from_speech?}` → `QueryResponse` |
| `POST` | `/v1/query/stream` | same body → SSE: `thinking`, `tool_call`, `tool_result`, `fallback`, then `result` or `error` |
| `POST` | `/v1/voice-query` | multipart `audio` (+ `history` JSON) → `QueryResponse` with `transcript` |
| `GET` | `/v1/meta` | non-secret client config (auth required? voice enabled?) |
| `POST` | `/mcp` | MCP endpoint for other agents (see above) |
| `GET` | `/healthz`, `/readyz`, `/metrics` | liveness, readiness (DB + catalog), Prometheus |

`QueryResponse` carries `status`, `answer`, `clarification_question`, `assumptions`, `sql`, `columns`,
`rows`, `row_count`, `truncated`, `model`, `tool_calls`, token `usage`, and per-stage `timings_ms`. Errors
are `{"error": {"code", "message", "request_id"}}`. Interactive docs at `/docs` when `S2S_ENVIRONMENT=dev`.

```bash
curl -s localhost:8000/v1/query -H 'Content-Type: application/json' -d '{"question": "orders by status"}'
```

```bash
curl -s localhost:8000/v1/voice-query -F audio=@question.webm
```

## Configuration

All settings are environment variables (or `.env`), documented in
[`config.py`](src/speech_to_sql/config.py). The ones you will most likely touch:

| Variable | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | – | read by the Anthropic SDK |
| `DEEPGRAM_API_KEY` | – | required for voice in prod |
| `S2S_DATABASE_URL` | `sqlite:///./data/demo.db` | `postgresql://…` or `sqlite:///…` |
| `S2S_DB_SCHEMAS` | `public` | Postgres schemas the agent may see |
| `S2S_HIDDEN_TABLES` / `S2S_HIDDEN_COLUMNS` | – | e.g. `customers.email,users.ssn` |
| `S2S_LLM_MODEL` / `S2S_LLM_EFFORT` | `claude-opus-5-5` / `medium` | effort: `low` … `max` |
| `S2S_STT_KEYTERMS` | – | product names, jargon to boost in transcription |
| `S2S_API_KEYS` | – | comma-separated; empty disables auth (dev only) |
| `S2S_RATE_LIMIT_PER_MINUTE` | `30` | per API key (or client IP), per replica |
| `S2S_MAX_RESULT_ROWS` / `S2S_STATEMENT_TIMEOUT_S` | `500` / `10` | result cap / per-query timeout |
| `S2S_TIMEZONE` | `UTC` | for "today" in prompts and the Postgres session time zone |
| `S2S_MCP_DATA_DESCRIPTION` | – | one line telling calling agents what the data is |
| `S2S_MCP_MAX_ROWS` / `S2S_MCP_ALLOW_RAW_SQL` | `100` / `true` | row cap for MCP callers / expose `run_sql` + `find_values` |
| `S2S_MCP_ALLOWED_HOSTS` | – | extra Host names for `/mcp` (needed without API keys) |

## Testing and evals

```bash
make check     # ruff + mypy --strict + pytest (no network or API keys needed)
```

```bash
make test-pg   # adds Postgres integration tests (uses a disposable DB: createdb s2s_test)
```

The unit and API tests drive the real agent loop against a scripted Claude API (mock transport), so tool
dispatch, retries, step budgets, refusals, SSE streaming and request shapes are all covered offline.

`make eval` runs `evals/cases.jsonl` — 20 spoken-style questions (spelled-out numbers, misheard names,
relative dates, a privacy probe, a write attempt) — against the Postgres demo DB and reports execution
accuracy, p50/p95 latency, tokens and estimated cost. It calls the Claude API (roughly 2–5 calls per case).
Use it to compare effort levels or prompt changes before shipping them:

```bash
S2S_DATABASE_URL=postgresql://localhost/speech_demo uv run python evals/run_evals.py --effort low
```

## Running in production

- Run one Uvicorn worker per container and scale containers horizontally; the service is stateless. (With
  multiple workers per container, Prometheus needs multiprocess mode.)
- Put it behind a TLS-terminating proxy; set `FORWARDED_ALLOW_IPS` so client IPs are correct. SSE needs
  proxy buffering off (the app sends `X-Accel-Buffering: no` and heartbeats every 15 s).
- Set `S2S_ENVIRONMENT=prod` (JSON logs, no `/docs`), `S2S_API_KEYS`, and point `S2S_DATABASE_URL` at the
  read-only role — ideally on a read replica.
- The rate limiter is per replica; enforce a global limit at the gateway as well.
- Watch `s2s_agent_outcomes_total`, `s2s_sql_rejected_total`, `s2s_stage_seconds`, and
  `s2s_llm_tokens_total{kind="cache_read"}` (a drop means something is invalidating the prompt cache).
- Schema changes are picked up after `S2S_CATALOG_TTL_S` (default 10 min).

## Extending

- **Another database:** implement the `Database` protocol in `db/base.py` (`connect`, `introspect`,
  `execute`, `search_values`) with read-only enforcement, set its sqlglot `dialect`, and register it in
  `db/__init__.py`.
- **Another speech provider:** implement `Transcriber` in `stt/base.py` and add it to `create_transcriber`.
- **Another tool for calling agents:** add it in `mcp_server.py` with `@server.tool(...)`; reuse
  `Services` (database, guard, catalog, agent) from `runtime.py` so it inherits the same safety layers.
- **Large schemas:** above `S2S_MAX_SCHEMA_PROMPT_CHARS` the prompt lists table names only and the agent
  pulls columns on demand with `describe_tables`.
