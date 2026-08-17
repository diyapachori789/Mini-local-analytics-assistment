"""Central configuration for the Mini Local Analytics Assistant.

This module must remain importable without any environment configuration so the
rest of the codebase can be imported and unit-tested offline. Required secrets
are therefore validated on demand via :func:`require_groq_api_key`, not at import
time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# Loaded before any os.getenv call below, otherwise settings placed in .env are
# read before they exist and silently fall back to their defaults.
load_dotenv(PROJECT_ROOT / ".env")

# Where the DuckDB files live. Defaults to the project root, so a local run is
# unchanged. Docker sets DATA_HOME to a mounted directory: DuckDB writes a
# write-ahead log beside its database, and bind-mounting individual files would
# leave that .wal inside the container and lose it on recreation.
DATA_HOME = Path(os.getenv("DATA_HOME") or PROJECT_ROOT)

DATABASE_NAME = DATA_HOME / "analytics.duckdb"
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "sample_opportunities.csv"

# --- Database --------------------------------------------------------------

TABLE_NAME = "opportunities"
TABLE_SCHEMA = "main"

# Block DuckDB from touching the filesystem once the table is loaded. This is a
# one-way latch: DuckDB refuses to re-enable it while the database is running,
# so generated SQL cannot turn it back on.
RESTRICT_FILE_ACCESS = True

# --- LLM -------------------------------------------------------------------

# The single source of truth for which model every stage uses: routing, SQL
# generation, conversation, and grounded answers all read MODEL_NAME.
#
# Override with GROQ_MODEL in .env or the environment to switch models without
# touching code, for example when a model's daily token quota is exhausted.
#
# 2026-08-17: moved to openai/gpt-oss-120b. Groq retired both llama-3.1-8b-instant
# (what this defaulted to) and llama-3.3-70b-versatile, so every request failed
# with "model not found". gpt-oss-120b was chosen over the other suggested
# replacement, qwen/qwen3.6-27b, because Qwen writes its chain of thought into
# message.content: the routing JSON arrives wrapped in <think> blocks and the
# reply is truncated at the token ceiling before the answer begins. Reading it
# would mean loosening the plan parser, which is the one place that must stay
# strict. gpt-oss returns reasoning in a separate field, leaving content clean.
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0
# Without an explicit timeout a hung request blocks the CLI indefinitely.
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2

# --- Result handling -------------------------------------------------------

# Hard cap on the number of rows pulled out of DuckDB into Python.
#
# The cap is applied by stopping the fetch, never by rewriting the query. The
# statement DuckDB executes is byte-identical to the one that passed validation,
# so the user's own LIMIT and ORDER BY keep their exact meaning. A query that
# asks for 3 rows still returns 3; a query that would return a million returns
# the first MAX_RESULT_ROWS with `truncated` set.
#
# Phase 5 will send results to the model, where row count drives token cost.
# That step should pass its own smaller max_rows rather than change this value.
MAX_RESULT_ROWS = 1000

# Rows the CLI prints. Kept separate from MAX_RESULT_ROWS so the display stays
# readable while the full capped result remains available to the caller.
DISPLAY_ROWS = 20

# --- Answer generation -----------------------------------------------------

# Rows included in the answer prompt. Bounded separately from MAX_RESULT_ROWS
# because rows are tokens here: answer quality does not improve with hundreds of
# rows, and the prompt must stay small. When fewer rows are sent than the query
# returned, the prompt says so explicitly so the model cannot imply the sample
# is the whole result.
ANSWER_MAX_ROWS = 50

# Upper bound on the length of the generated answer.
ANSWER_MAX_TOKENS = 500

# Returned verbatim when a query matches no rows. Answered without an API call:
# there is nothing to summarise, so a request would only add cost and give the
# model room to invent data.
NO_DATA_ANSWER = "No matching records were found."

# --- Query history ---------------------------------------------------------

# History lives in its own DuckDB file, deliberately not in analytics.duckdb.
#
# initialize_database() rebuilds the opportunities table from CSV on every run,
# and analytics.duckdb is disposable: deleting it is a supported way to reset the
# dataset. History stored there would be lost with it. A separate file also keeps
# its single-writer lock independent, and analytics connections latch
# enable_external_access off, which blocks ATTACH entirely.
HISTORY_DATABASE = DATA_HOME / "history.duckdb"
HISTORY_TABLE = "query_history"

# Rows returned by GET /api/history when the caller does not ask for a size.
HISTORY_DEFAULT_LIMIT = 50

# Server-side ceiling. A browser cannot request an unbounded history page.
HISTORY_MAX_LIMIT = 100

# --- Persistent conversations ---------------------------------------------

# Conversations are intentionally stored beside the existing safe history
# records, in ``history.duckdb``.  They provide semantic orientation for a
# follow-up, not a second data source: the planner still has to retrieve every
# current figure from DuckDB.
#
# Keep this window deliberately small.  It is applied before the first model
# call and excludes result rows, SQL, schema text, paths, logs, and provider
# payloads.  The limits are duplicated defensively in the repository so a
# malformed caller cannot turn history into an unbounded prompt.
CONVERSATION_CONTEXT_MAX_MESSAGES = 6
CONVERSATION_CONTEXT_MAX_CHARS = 2_400
CONVERSATION_CONTEXT_MAX_MESSAGE_CHARS = 600
CONVERSATION_TITLE_MAX_CHARS = 64
CONVERSATION_DEFAULT_LIMIT = 100
CONVERSATION_MESSAGE_LIMIT = 200

# --- Web server ------------------------------------------------------------

# Loopback by default, so running web_app.py directly is unchanged and stays
# unreachable from the network. A container must bind 0.0.0.0 to receive its
# published port, so Docker sets WEB_HOST=0.0.0.0; Compose still publishes the
# port only to 127.0.0.1 on the host, so it is not exposed publicly.
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"

# Addresses a server binds but a browser cannot open.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})


def _resolve_port(raw: str) -> int:
    """Return a valid TCP port, falling back to 8000 on anything unusable."""
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return 8000
    return port if 1 <= port <= 65535 else 8000


WEB_PORT = _resolve_port(os.getenv("WEB_PORT", "8000"))

# What to print at startup: a wildcard bind is not a browsable address.
BROWSABLE_HOST = "127.0.0.1" if WEB_HOST in _WILDCARD_HOSTS else WEB_HOST
WEB_URL = f"http://{BROWSABLE_HOST}:{WEB_PORT}"

# --- Charts ----------------------------------------------------------------

CHARTS_DIR = PROJECT_ROOT / "charts"

# Above this many categories a vertical bar chart becomes unreadable, so the
# bars are drawn horizontally instead.
CHART_HORIZONTAL_THRESHOLD = 6

# A pie chart stops communicating anything once the slices get thin. Beyond this
# many categories, a bar chart is used instead.
CHART_MAX_PIE_SLICES = 8

CHART_FIGURE_SIZE = (10, 6)
CHART_DPI = 120

# --- Logging ---------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

# Web-layer events are additionally written here so a browser session can be
# reviewed without reading through CLI and analytics entries. app.log remains
# the complete record; this is a filtered view, not a replacement.
WEB_LOG_FILE = LOG_DIR / "web_app.log"
# File captures detail; the console stays quiet so CLI output is readable.
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "WARNING").upper()
LOG_MAX_BYTES = 1_048_576
LOG_BACKUP_COUNT = 3

# --- Secrets ---------------------------------------------------------------

# Read the Groq API key from the environment. Never log or print this value.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


def require_groq_api_key() -> str:
    """Return the Groq API key, raising a clear error when it is not configured.

    Call this at application startup to fail fast, and from any code path that is
    about to contact the Groq API.
    """
    if not GROQ_API_KEY:
        raise ConfigurationError(
            "Missing GROQ_API_KEY. Create a .env file in the project root "
            "containing 'GROQ_API_KEY=<your key>' before running the assistant."
        )
    return GROQ_API_KEY
