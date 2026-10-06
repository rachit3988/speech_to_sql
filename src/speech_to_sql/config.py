"""Application settings, loaded from environment variables (prefix ``S2S_``) and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="S2S_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_by_name=True,
    )

    # --- Service ---------------------------------------------------------------
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    # Log user questions / transcripts verbatim. Off by default: they can contain PII.
    log_payloads: bool = False
    # API keys accepted in the X-API-Key header. Empty = auth disabled (dev only).
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    rate_limit_per_minute: int = 30
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    serve_web_ui: bool = True
    # Used to tell the agent today's date (for "last month", "this year", ...).
    timezone: str = "UTC"

    # --- Database --------------------------------------------------------------
    # postgresql://user:pass@host:5432/db  or  sqlite:///path/to/file.db
    database_url: SecretStr = SecretStr("sqlite:///./data/demo.db")
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    # Schemas the agent may see and query (Postgres). Ignored for SQLite.
    db_schemas: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["public"])
    # Tables / columns hidden from the agent and rejected by the SQL guard.
    # Format: "table" or "schema.table"; columns as "table.column" or "schema.table.column".
    hidden_tables: Annotated[list[str], NoDecode] = Field(default_factory=list)
    hidden_columns: Annotated[list[str], NoDecode] = Field(default_factory=list)
    statement_timeout_s: float = 10.0
    max_result_rows: int = 500
    catalog_ttl_s: float = 600.0
    # Above this size the system prompt lists tables only; the agent pulls columns via a tool.
    max_schema_prompt_chars: int = 60_000

    # --- LLM (Claude) ------------------------------------------------------------
    # Read from the environment or .env. If unset, the SDK falls back to its own resolution
    # (ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile, workload identity federation).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "medium"
    llm_max_tokens: int = 16_000
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3
    # Server-side refusal fallback ("default" routes by refusal category).
    llm_refusal_fallback: bool = True
    # "summarized" returns readable reasoning summaries the UI can show as progress.
    llm_thinking_display: Literal["summarized", "omitted"] = "summarized"
    agent_max_iterations: int = 12
    agent_timeout_s: float = 180.0
    # Rows of each exploratory query shown to the model (the client gets max_result_rows).
    agent_preview_rows: int = 25
    max_history_turns: int = 5

    # --- MCP (lets other agents use this one as a tool) ------------------------------
    mcp_enabled: bool = True
    # One line telling calling agents what this data is, so they know when to use it,
    # e.g. "Acme's e-commerce data: customers, orders, products, revenue (2024-2026)".
    mcp_data_description: str | None = None
    # Cap on rows returned to calling agents (keeps their context small).
    mcp_max_rows: int = 100
    # Also expose run_sql / find_values for agents that prefer to write SQL themselves.
    mcp_allow_raw_sql: bool = True
    # Extra Host header values accepted on /mcp, e.g. "sql-agent.example.com". Localhost is always
    # accepted. "*" disables the check (DNS-rebinding protection) entirely.
    mcp_allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Speech-to-text ------------------------------------------------------------
    stt_provider: Literal["deepgram", "none"] = "deepgram"
    deepgram_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPGRAM_API_KEY")
    deepgram_model: str = "nova-3"
    deepgram_language: str = "en"
    deepgram_base_url: str = "https://api.deepgram.com"
    # Domain vocabulary boosted during transcription (product names, metric names...).
    stt_keyterms: Annotated[list[str], NoDecode] = Field(default_factory=list)
    stt_timeout_s: float = 30.0
    stt_max_retries: int = 2
    max_audio_bytes: int = 10 * 1024 * 1024

    @field_validator(
        "api_keys",
        "cors_origins",
        "db_schemas",
        "hidden_tables",
        "hidden_columns",
        "stt_keyterms",
        "mcp_allowed_hosts",
        mode="before",
    )
    @classmethod
    def _csv(cls, value: object) -> object:
        return _split_csv(value)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
