"""What kind of request is this, and what data would help answer it?

The first model call used to return either SQL or the bare sentinel
``INVALID_QUESTION``. That made one model responsible for three different
decisions at once - understand the question, judge whether it is legitimate, and
translate it - with only two possible outputs. Anything that was not a direct
lookup fell into the sentinel, so ``How can I change the win rate?`` was refused
while ``can i change the win rate?`` happened to survive. The difference was
phrasing, not meaning.

This module separates the decision from the translation. The first call now
returns a small structured object: an *intent* and, when data would help, the
SQL that retrieves it. Routing is therefore semantic - it comes from the model
reading the question against the live schema - and no metric, column or phrase
is named anywhere in this file.

Parsing is deliberately forgiving. A model that replies with bare SQL, or with
the old sentinel, is still understood. A formatting slip must never turn an
answerable question into a refusal.

Nothing here decides that SQL is safe. A plan is a proposal; ``sql_guard`` and
the database layer remain the authorities on what may execute.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sql_guard import ALLOWED_PREFIXES, clean_sql, is_refusal


class Intent(str, Enum):
    """Why the assistant is being addressed.

    These values are internal routing states. They are never shown to a user;
    the browser sees an answer or a friendly refusal, never a label.
    """

    #: A figure, list, ranking, breakdown or comparison. Needs SQL.
    DATA_QUERY = "DATA_QUERY"
    #: A question *about* a measure - what it means, what moves it, why it looks
    #: the way it does. Usually still needs SQL, because current figures are the
    #: evidence the explanation is built on.
    DATA_EXPLANATION = "DATA_EXPLANATION"
    #: Depends on an earlier turn that this assistant cannot see.
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    #: Not about this dataset at all.
    UNSUPPORTED = "UNSUPPORTED"
    #: Asks to modify or administer the database, or to reveal its structure.
    UNSAFE = "UNSAFE"


#: Intents the assistant will actually answer. Everything else is refused.
ANSWERABLE: tuple[Intent, ...] = (Intent.DATA_QUERY, Intent.DATA_EXPLANATION)


# Wording drifts between model versions and between prompts. Accepting the
# obvious synonyms costs nothing and stops a rephrased label from being read as
# an unknown intent.
_INTENT_ALIASES: dict[str, Intent] = {
    "data_query": Intent.DATA_QUERY,
    "analytics_query": Intent.DATA_QUERY,
    "analytics": Intent.DATA_QUERY,
    "query": Intent.DATA_QUERY,
    "data": Intent.DATA_QUERY,
    "data_explanation": Intent.DATA_EXPLANATION,
    "analytics_explanation": Intent.DATA_EXPLANATION,
    "conceptual_analytics": Intent.DATA_EXPLANATION,
    "conceptual": Intent.DATA_EXPLANATION,
    "explanation": Intent.DATA_EXPLANATION,
    "explain": Intent.DATA_EXPLANATION,
    "insufficient_context": Intent.INSUFFICIENT_CONTEXT,
    "needs_context": Intent.INSUFFICIENT_CONTEXT,
    "missing_context": Intent.INSUFFICIENT_CONTEXT,
    "follow_up": Intent.INSUFFICIENT_CONTEXT,
    "unsupported": Intent.UNSUPPORTED,
    "out_of_scope": Intent.UNSUPPORTED,
    "off_topic": Intent.UNSUPPORTED,
    "invalid": Intent.UNSUPPORTED,
    "invalid_question": Intent.UNSUPPORTED,
    "unsafe": Intent.UNSAFE,
    "unsafe_sql": Intent.UNSAFE,
    "admin": Intent.UNSAFE,
    "metadata": Intent.UNSAFE,
    "schema": Intent.UNSAFE,
}

# The outermost brace pair. Greedy on purpose: a model that adds a stray
# sentence around the object still yields the whole object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# Asking for the shape of the database rather than the business data in it.
#
# The router handles this correctly most of the time, but "most of the time" is
# not a safety property. Asked to "show the database schema" a model can produce
# a perfectly legal SELECT * that the guard has no reason to reject, and the
# answer stage then reads the column names out loud. That is a leak assembled
# entirely from safe parts, so it has to be stopped before any of them run.
#
# Deliberately narrow. Every term here describes database structure and has no
# ordinary business reading, so "describe the pipeline" and "show every field
# for OPP-1003" are untouched. Anything phrased more obliquely still has to get
# past the router, which classifies these as UNSAFE.
_STRUCTURE_REQUEST_RE = re.compile(
    r"\b(?:"
    r"schemas?|ddl|create\s+table|information_schema|sqlite_master|pg_catalog"
    r"|duckdb_\w+|table\s+structure|column\s+names?|field\s+names?"
    r"|data\s*types?|table\s+list|list\s+of\s+tables?"
    r"|database\s+(?:structure|metadata|design|layout)"
    r")\b"
    r"|\b(?:show|list|display|print|describe|get)\s+(?:me\s+|all\s+|the\s+|every\s+)*"
    r"tables?\b",
    re.IGNORECASE,
)


def is_structure_request(question: str) -> bool:
    """True when a question asks about the database's shape rather than its data.

    Checked before the model is called, so such a question costs nothing and
    cannot be argued into an answer.
    """
    if not isinstance(question, str):
        return False
    return bool(_STRUCTURE_REQUEST_RE.search(question))


@dataclass(frozen=True)
class QueryPlan:
    """One routing decision: what this is, and the SQL that supports it."""

    intent: Intent
    sql: Optional[str] = None

    @property
    def needs_data(self) -> bool:
        """True when answering this plan requires a database execution."""
        return self.sql is not None

    @property
    def is_answerable(self) -> bool:
        """True when the assistant will answer rather than refuse."""
        return self.intent in ANSWERABLE


def _coerce_intent(value: object) -> Optional[Intent]:
    """Map a model-supplied label onto an Intent, or None if unrecognised."""
    if isinstance(value, Intent):
        return value
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    return _INTENT_ALIASES.get(key)


def _coerce_sql(value: object) -> Optional[str]:
    """Normalise a model-supplied SQL field to a statement or None.

    JSON null, an empty string, the legacy refusal sentinel and placeholder
    words all mean the same thing here: there is no query to run.
    """
    if not isinstance(value, str):
        return None
    cleaned = clean_sql(value)
    if not cleaned or is_refusal(cleaned):
        return None
    if cleaned.strip().lower() in ("null", "none", "n/a"):
        return None
    return cleaned


def _reconcile(intent: Intent, sql: Optional[str]) -> QueryPlan:
    """Apply the invariants that make a plan safe to act on.

    A refusing intent never carries SQL, so a mislabelled-but-populated reply
    cannot smuggle a statement past the router. A DATA_QUERY without SQL cannot
    be executed, so it becomes a refusal rather than an empty success.
    """
    if intent not in ANSWERABLE:
        return QueryPlan(intent=intent, sql=None)
    if intent is Intent.DATA_QUERY and sql is None:
        return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)
    return QueryPlan(intent=intent, sql=sql)


def _parse_bare_text(text: str) -> QueryPlan:
    """Understand a reply that is not JSON.

    Covers the legacy contract - raw SQL, or the ``INVALID_QUESTION`` sentinel -
    and any future call that returns a statement without the wrapper.
    """
    if is_refusal(text):
        return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)
    if text.upper().startswith(ALLOWED_PREFIXES):
        return QueryPlan(intent=Intent.DATA_QUERY, sql=text)
    return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)


def parse_plan(text: str) -> QueryPlan:
    """Read one first-stage reply into a :class:`QueryPlan`.

    Never raises. Anything unintelligible becomes ``UNSUPPORTED``, which the
    caller turns into a friendly refusal rather than an error.
    """
    if not isinstance(text, str):
        return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)

    cleaned = clean_sql(text)
    if not cleaned:
        return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)

    match = _JSON_OBJECT_RE.search(cleaned)
    if match is None:
        return _parse_bare_text(cleaned)

    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return _parse_bare_text(cleaned)

    if not isinstance(payload, dict):
        return _parse_bare_text(cleaned)

    sql = _coerce_sql(payload.get("sql"))
    intent = _coerce_intent(payload.get("intent"))

    if intent is None:
        # An unfamiliar label is not a reason to discard usable SQL: the guard
        # still has the final say on whether it runs.
        intent = Intent.DATA_QUERY if sql is not None else Intent.UNSUPPORTED

    return _reconcile(intent, sql)
