"""SQL cleaning and safety validation.

This module is deliberately free of any dependency on the database, the LLM
client and the configuration. That matters for three reasons:

* ``llm.py`` validates model output and ``database.py`` guards execution. If the
  rules lived in either module the other would need a circular import.
* The rules can be unit-tested with no API key and no database.
* There is exactly one definition of "safe SQL" in the project.

The validator inspects SQL *structure*. String literals, quoted identifiers and
comments are masked out first so that a legitimate text search such as
``WHERE notes ILIKE '%delete%'`` is not mistaken for a DELETE statement.
"""

from __future__ import annotations

import re

# Sentinel the model returns when a question cannot be answered from the schema.
INVALID_QUESTION = "INVALID_QUESTION"

# Single source of truth for statement types that must never execute. The
# matching regex is derived from this tuple so the two can never drift apart.
FORBIDDEN_KEYWORDS = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "MERGE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "GRANT",
    "REVOKE",
    "COPY",
    # DuckDB-specific statements that could reach the filesystem or extensions.
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "EXPORT",
    "IMPORT",
)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)

# System catalogs and metadata functions. These are readable with a plain
# SELECT, so the keyword ban above does not stop them: "SELECT * FROM
# duckdb_tables()" returns the full CREATE TABLE statement, and
# information_schema.columns returns every column and type. The assistant
# answers business questions, so the shape of the database is not its to hand
# out. Internal code that legitimately needs the schema (database.get_schema)
# queries the connection directly and never passes through this validator.
BLOCKED_SOURCES = (
    "information_schema",
    "pg_catalog",
    "sqlite_master",
    "sqlite_temp_master",
    "pg_tables",
    "pg_class",
)

_BLOCKED_SOURCE_PATTERN = re.compile(
    r"\b(?:" + "|".join(BLOCKED_SOURCES) + r")\b"
    # DuckDB exposes its catalog through table functions: duckdb_tables(),
    # duckdb_columns(), pragma_table_info() and friends.
    r"|\b(?:duckdb_\w+|pragma_\w+)\s*\(",
    re.IGNORECASE,
)

# Statement prefixes that are allowed. Anything else is rejected outright.
ALLOWED_PREFIXES = ("SELECT", "WITH")

# A complete fenced code block with any language tag (```sql, ```SQL, ```duckdb
# or a bare ```), plus tolerant patterns for an unbalanced fence.
_CODE_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+-]*\s*(?P<body>.*?)\s*```$", re.DOTALL)
_LEADING_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?")
_TRAILING_FENCE_RE = re.compile(r"\r?\n?```$")

# Punctuation and quoting a model may append to the refusal sentinel. Stripping
# these stops a plain refusal from being reported as a SQL syntax problem.
_REFUSAL_TRIM = " \t\r\n;.!?,:\"'`*"


def is_refusal(text: str) -> bool:
    """Return True when model output is the INVALID_QUESTION sentinel.

    Tolerates surrounding whitespace, quoting and trailing punctuation, so
    ``INVALID_QUESTION;`` and ``"INVALID_QUESTION."`` are still recognised as a
    refusal rather than falling through to SQL validation and producing a
    misleading "must start with SELECT or WITH" error.
    """
    if not isinstance(text, str):
        return False
    return text.strip().strip(_REFUSAL_TRIM).upper() == INVALID_QUESTION


def mask_literals_and_comments(sql: str) -> str:
    """Blank out string literals, quoted identifiers and comments.

    Safety checks must inspect SQL structure, never the text a user is searching
    for. Scanning the raw statement rejects legitimate queries such as
    ``WHERE notes ILIKE '%delete%'`` and ``WHERE notes LIKE '%;%'`` because the
    literal contains a banned word or a semicolon.

    Masking is deliberately conservative: an unterminated quote consumes the rest
    of the statement, which then fails the terminator check and is rejected.
    """
    masked: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]

        if char in ("'", '"'):
            quote = char
            masked.append(quote * 2)
            index += 1
            while index < length:
                if sql[index] == quote:
                    # A doubled quote is an escaped quote, not a terminator.
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
        elif char == "-" and sql.startswith("--", index):
            while index < length and sql[index] != "\n":
                index += 1
            masked.append(" ")
        elif char == "/" and sql.startswith("/*", index):
            index += 2
            while index < length and not sql.startswith("*/", index):
                index += 1
            index += 2
            masked.append(" ")
        else:
            masked.append(char)
            index += 1

    return "".join(masked)


def clean_sql(sql: str) -> str:
    """Clean model output by removing code fences and extra whitespace."""
    if not isinstance(sql, str):
        raise ValueError("SQL output must be a string.")

    cleaned = sql.strip()

    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group("body")
    else:
        # Tolerate an unbalanced fence.
        cleaned = _LEADING_FENCE_RE.sub("", cleaned)
        cleaned = _TRAILING_FENCE_RE.sub("", cleaned)

    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip()


def validate_cleaned_sql(cleaned: str) -> str:
    """Validate SQL that has already passed through :func:`clean_sql`.

    Raises ``ValueError`` with a specific message for each rule that fails.
    """
    if not cleaned:
        raise ValueError("Generated SQL is empty.")

    # Inspect structure only: literals, identifiers and comments are blanked out.
    inspectable = mask_literals_and_comments(cleaned)

    if _FORBIDDEN_PATTERN.search(inspectable):
        raise ValueError("Generated SQL contains a disallowed statement.")

    if _BLOCKED_SOURCE_PATTERN.search(inspectable):
        raise ValueError("Generated SQL reads database metadata, which is not allowed.")

    if not inspectable.upper().startswith(ALLOWED_PREFIXES):
        raise ValueError("Generated SQL must start with SELECT or WITH.")

    if inspectable.count(";") != 1:
        raise ValueError("Generated SQL must contain exactly one statement ending with a semicolon.")

    if not inspectable.rstrip().endswith(";"):
        raise ValueError("Generated SQL must end with a semicolon.")

    return cleaned


def validate_sql(sql: str) -> str:
    """Clean and validate SQL, returning the safe statement.

    Raises ``ValueError`` if the statement is not a single read-only
    SELECT/WITH query.
    """
    return validate_cleaned_sql(clean_sql(sql))
