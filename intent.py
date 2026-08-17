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
    #: A legitimate social or capability interaction that needs no analytics.
    #: A schema-free answer stage supplies the brief assistant reply within the
    #: existing two-call budget.
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    #: An ordinary question that is not about this dataset - general knowledge,
    #: a joke, a definition from the wider world. Answered briefly and without
    #: pretending to have data, because refusing every one of these made an
    #: assistant that could not hold a conversation.
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    #: An analytics request whose grouping or measure is genuinely ambiguous.
    #: Asking one short question back is more useful than guessing, and costs
    #: no query.
    CLARIFICATION = "CLARIFICATION"
    #: Depends on an earlier turn that this assistant cannot see.
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    #: Not about this dataset at all.
    UNSUPPORTED = "UNSUPPORTED"
    #: Asks to modify or administer the database, or to reveal its structure.
    UNSAFE = "UNSAFE"


#: Intents the assistant will actually answer. Everything else is refused.
ANSWERABLE: tuple[Intent, ...] = (
    Intent.DATA_QUERY,
    Intent.DATA_EXPLANATION,
    Intent.GENERAL_CONVERSATION,
    Intent.OUT_OF_DOMAIN,
    Intent.CLARIFICATION,
)

#: Intents answered by the schema-free conversational stage rather than by data.
#: They share one call slot, so adding them costs no extra request.
CONVERSATIONAL: tuple[Intent, ...] = (
    Intent.GENERAL_CONVERSATION,
    Intent.OUT_OF_DOMAIN,
    Intent.CLARIFICATION,
)


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
    "general_conversation": Intent.GENERAL_CONVERSATION,
    "meta_conversation": Intent.GENERAL_CONVERSATION,
    "follow_up_conversation": Intent.GENERAL_CONVERSATION,
    "out_of_domain": Intent.OUT_OF_DOMAIN,
    "out_of_domain_conversation": Intent.OUT_OF_DOMAIN,
    "general_knowledge": Intent.OUT_OF_DOMAIN,
    "clarification": Intent.CLARIFICATION,
    "clarify": Intent.CLARIFICATION,
    "ambiguous": Intent.CLARIFICATION,
    "conversation": Intent.GENERAL_CONVERSATION,
    "conversational": Intent.GENERAL_CONVERSATION,
    "small_talk": Intent.GENERAL_CONVERSATION,
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

_CONVERSATION_RESPONSE_MAX_CHARS = 1200
_CONVERSATION_RESPONSE_NUMBER_RE = re.compile(r"\d")

# A number as prose writes it.
_CONVERSATION_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_-])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_])")

# Vocabulary that makes a number a claim about *this business*. A conversational
# reply has no query result behind it, so "your win rate is 45%" would be an
# invented fact; "there are 3 broad types of machine learning" is not about the
# data at all. Blanket-rejecting every digit was the simpler rule and it made
# ordinary questions unanswerable, so the distinction is drawn here instead.
_BUSINESS_CLAIM_RE = re.compile(
    r"\b(?:opportunit\w*|deal|deals|pipeline|revenue|amount|amounts|win\s*rate|"
    r"won|lost|closed|stage|stages|owner|owners|account|accounts|region|regions|"
    r"industry|industries|quarter|forecast|sales|customer|customers|"
    r"conversion|average|total|totals|percent|percentage|%)\b",
    re.IGNORECASE,
)
_CONVERSATION_RESPONSE_UNSAFE_RE = re.compile(
    r"```|"
    r"(?:^|\n)\s*(?:select|with|insert|update|delete|drop|alter|create|attach|"
    r"detach|pragma)\b|"
    r"\b(?:information_schema|duckdb_tables|sqlite_master|pg_catalog)\b|"
    r"\b(?:database\s+schema|table\s+structure|column\s+names?|field\s+names?|"
    r"data\s+types?|list\s+of\s+tables?)\b|"
    r"\b(?:data_query|data_explanation|general_conversation|out_of_domain|"
    r"clarification|insufficient_context|unsupported|unsafe)\b|"
    r"\b(?:(?:[A-Za-z0-9]+_)?api[_ -]?key|access[_ -]?token|password|secret)\b|"
    r"\bgsk_[A-Za-z0-9_-]{8,}\b|\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"\bbearer\s+[A-Za-z0-9._~+/=-]{6,}\b|"
    # An address in a reply is either an identifier picked up from the
    # surroundings or a name inferred from one. A social reply has no reason
    # to contain one, and "greet them by the name in their email" is exactly
    # the shortcut this assistant must not take.
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"(?:[A-Za-z]:[\\/]|/(?:home|tmp|var|app|users|root|opt|usr|etc|mnt|srv|"
    r"bin|lib|private)(?:/|\b))",
    re.IGNORECASE,
)


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


def business_claim_numbers(text: object) -> list[str]:
    """Numbers in prose that are being asserted as facts about this business.

    A number counts when dataset vocabulary sits close to it. That keeps a
    general-knowledge answer usable while still catching a figure invented
    about the user's own pipeline, which is the case that actually matters:
    the conversational stage has no query result behind it, so any such number
    came from the model rather than from DuckDB.
    """
    if not isinstance(text, str):
        return []
    claims: list[str] = []
    for match in _CONVERSATION_NUMBER_RE.finditer(text):
        window = text[max(0, match.start() - 60) : match.end() + 60]
        if _BUSINESS_CLAIM_RE.search(window):
            claims.append(match.group(0))
    return claims


def safe_conversation_response(
    value: object, *, grounded_numbers: "frozenset[str] | set[str] | tuple[str, ...]" = ()
) -> Optional[str]:
    """Return bounded plain prose that is safe to expose as a social reply.

    A conversational reply crosses an explicit output boundary before it can
    reach an adapter: SQL, internal route labels, credentials, and local paths
    are rejected rather than displayed.

    Figures about the business are rejected too, unless they already appear in
    ``grounded_numbers`` - the numbers persisted earlier in this conversation,
    which reached it from DuckDB. That is what lets the assistant recap "the
    win rate we saw was 23.67%" without letting it invent one.
    """
    if not isinstance(value, str):
        return None
    response = " ".join(value.split()).strip()
    if not response or _CONVERSATION_RESPONSE_UNSAFE_RE.search(response):
        return None

    # The same schema check the data-backed answers use. This stage is never
    # given the schema, so a leak here could only come from the transcript, but
    # applying one filter at every output boundary is what stops a gap opening
    # between them - the pattern rules alone missed "the opportunity_id column".
    if contains_schema_disclosure(response):
        return None

    allowed = {str(n).replace(",", "") for n in grounded_numbers}
    for claim in business_claim_numbers(response):
        if claim.replace(",", "") not in allowed:
            return None
    return response[:_CONVERSATION_RESPONSE_MAX_CHARS]


def numbers_in_context(text: object) -> frozenset[str]:
    """Figures already stated in this conversation, so a recap may repeat them.

    Everything here was written by a previous turn that was itself grounded, so
    quoting it back is reporting, not inventing.
    """
    if not isinstance(text, str):
        return frozenset()
    return frozenset(m.group(0).replace(",", "") for m in _CONVERSATION_NUMBER_RE.finditer(text))


# Words that turn a business noun into an implementation detail. "amount" is
# what a deal is worth; "the amount column" is how the table is built.
_SCHEMA_VOCABULARY = (
    r"column|field|attribute|property|table|schema|database|dataset\s+field"
    r"|data\s*type|datatype|varchar|bigint|boolean|integer|numeric|timestamp"
)

# "the opportunity_id column", "the is_won field", "a boolean field called x".
_IDENTIFIER_WITH_VOCABULARY = re.compile(
    r"(?:\b(?:" + _SCHEMA_VOCABULARY + r")\b[^.]{0,40}?\b{ident}\b)"
    r"|(?:\b{ident}\b[^.]{{0,40}}?\b(?:" + _SCHEMA_VOCABULARY + r")\b)",
    re.IGNORECASE,
)

# A lower_snake_case word. Column names and query aliases look like this;
# ordinary prose and this dataset's values never do.
_SNAKE_CASE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# "the columns are a, b", "fields include x and y" - an enumeration is a
# disclosure even when the names that follow are ordinary words.
_SCHEMA_ENUMERATION_RE = re.compile(
    r"\b(?:column|field|attribute|property)s?\b\s*"
    r"(?:\w+\s+){0,3}?(?:are|is|include[sd]?|named|called|consist|comprise)\b",
    re.IGNORECASE,
)


def _identifier_pattern(identifier: str) -> re.Pattern[str]:
    """Build the 'identifier near schema vocabulary' matcher for one column."""
    quoted = re.escape(identifier)
    return re.compile(
        _IDENTIFIER_WITH_VOCABULARY.pattern.replace("{ident}", quoted).replace(
            "{{0,40}}", "{0,40}"
        ),
        re.IGNORECASE,
    )


def contains_schema_disclosure(
    text: object, identifiers: "frozenset[str] | set[str] | tuple[str, ...]" = ()
) -> bool:
    """True when prose exposes the shape of the table rather than its data.

    Three signals, chosen so that business language survives and
    implementation language does not:

    * A snake_case name that is a real column. Prose says "the account name";
      only a description of the table says ``account_name``. That single
      underscore is what separates the two, which is why this needs no list of
      phrasings to keep up to date.
    * A real column name sitting next to schema vocabulary - "the amount column",
      "the is_won field". The bare word "amount" is business language and is
      left alone; the same word introduced as a column is not.
    * An enumeration such as "the columns are ..." or "fields include ...",
      which discloses the shape whatever names follow.

    ``identifiers`` comes from the live schema, so a column added later is
    protected without touching this function.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    if _SCHEMA_ENUMERATION_RE.search(text):
        return True

    # Any snake_case token, whether or not it is a table column. Result aliases
    # are invented per query - win_rate_pct, total_amount - so a list of real
    # columns cannot cover them, and it was exactly such an alias that reached a
    # user. No value in this dataset contains an underscore, and business prose
    # does not write one, so the shape alone is the signal.
    if _SNAKE_CASE_TOKEN_RE.search(text):
        return True

    lowered = text.lower()
    for identifier in identifiers:
        name = str(identifier).strip().lower()
        if not name:
            continue
        # A compound identifier is never ordinary prose.
        if "_" in name and re.search(rf"\b{re.escape(name)}\b", lowered):
            return True
        if _identifier_pattern(name).search(text):
            return True
    return False


def _reconcile(intent: Intent, sql: Optional[str]) -> QueryPlan:
    """Apply the invariants that make a plan safe to act on.

    A refusing intent never carries SQL, so a mislabelled-but-populated reply
    cannot smuggle a statement past the router. A DATA_QUERY without SQL cannot
    be executed, so it becomes a refusal rather than an empty success.
    """
    if intent in CONVERSATIONAL:
        return QueryPlan(intent=intent, sql=None)
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
