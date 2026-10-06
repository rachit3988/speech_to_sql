-- Dedicated least-privilege role for the speech-to-SQL service.
-- Run as a superuser / database owner:
--   psql -d <database> -v app_password="'change-me'" -f deploy/postgres/readonly_role.sql
--
-- Layers of protection this role provides, independent of the application's SQL guard:
--   * SELECT-only privileges, no ownership, no CREATE on any schema;
--   * every session defaults to read-only transactions;
--   * a hard per-statement timeout and an idle-in-transaction timeout;
--   * optional column-level grants to keep sensitive columns out of reach.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 's2s_readonly') THEN
    CREATE ROLE s2s_readonly LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  END IF;
END
$$;

ALTER ROLE s2s_readonly PASSWORD :app_password;
ALTER ROLE s2s_readonly CONNECTION LIMIT 50;
ALTER ROLE s2s_readonly SET default_transaction_read_only = on;
ALTER ROLE s2s_readonly SET statement_timeout = '15s';
ALTER ROLE s2s_readonly SET idle_in_transaction_session_timeout = '30s';
ALTER ROLE s2s_readonly SET search_path = public;

-- On PostgreSQL < 15 every role can CREATE in schema public by default. Uncomment if that
-- applies to you and nothing else depends on it:
-- REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE :"DBNAME" TO s2s_readonly;
GRANT USAGE ON SCHEMA public TO s2s_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO s2s_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO s2s_readonly;

-- Column-level restriction example: hide customers.email from the agent.
-- The service only exposes columns this role can SELECT, so revoked columns
-- disappear from the agent's schema automatically.
REVOKE SELECT ON public.customers FROM s2s_readonly;
GRANT SELECT (id, full_name, company, city, country, segment, signup_date)
  ON public.customers TO s2s_readonly;
