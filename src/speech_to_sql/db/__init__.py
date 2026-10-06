from speech_to_sql.config import Settings
from speech_to_sql.db.base import Database, QueryResult
from speech_to_sql.db.catalog import CatalogProvider, SchemaCatalog, TableInfo

__all__ = [
    "CatalogProvider",
    "Database",
    "QueryResult",
    "SchemaCatalog",
    "TableInfo",
    "create_database",
]


def create_database(settings: Settings) -> Database:
    """Build the adapter for ``settings.database_url`` (not yet connected)."""
    url = settings.database_url.get_secret_value()
    if url.startswith(("postgresql://", "postgres://", "postgresql+asyncpg://")):
        from speech_to_sql.db.postgres import PostgresDatabase

        return PostgresDatabase(
            url,
            schemas=settings.db_schemas,
            hidden_tables=settings.hidden_tables,
            hidden_columns=settings.hidden_columns,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            default_timeout_s=settings.statement_timeout_s,
            timezone=settings.timezone,
        )
    if url.startswith("sqlite:///"):
        from speech_to_sql.db.sqlite import SQLiteDatabase

        return SQLiteDatabase(
            url.removeprefix("sqlite:///"),
            hidden_tables=settings.hidden_tables,
            hidden_columns=settings.hidden_columns,
            pool_size=settings.db_pool_max_size,
        )
    raise ValueError("S2S_DATABASE_URL must start with postgresql:// or sqlite:///")
