"""Friendly replies for questions the assistant cannot answer.

A refusal is a normal outcome, not a failure: the request succeeded, there is
simply nothing to compute. These replies are written locally from templates, so
no third model call is made and no quota is spent to say "I can't do that".

Selection is fully deterministic. The category comes from the wording of the
question, and the template within a category comes from a stable hash of the
normalized question, so the same question always produces the same reply and
tests never flake.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum

from intent import Intent


class RefusalCategory(str, Enum):
    """Why a question could not be answered."""

    METADATA = "metadata"
    UNSAFE_SQL = "unsafe_sql"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CONTEXT = "needs_context"
    UNSUPPORTED = "unsupported"


# Asking about the shape of the database rather than the business data in it.
_METADATA_RE = re.compile(
    r"\b(show\s+(all\s+)?tables?|list\s+(all\s+)?tables?|table\s+list"
    r"|describe\s+\w+|describe\s+the\s+table|schema|schemas"
    r"|what\s+(columns?|fields?|tables?)\b|which\s+(columns?|fields?|tables?)\b"
    r"|column\s+names?|field\s+names?|data\s+types?|table\s+structure"
    r"|database\s+(structure|metadata|design|layout)"
    r"|information_schema|duckdb_tables|sqlite_master|pg_catalog)\b",
    re.IGNORECASE,
)

# Statements that change or administer the database rather than read from it.
_UNSAFE_SQL_RE = re.compile(
    r"\b(drop|delete|truncate|insert|update|alter|create|replace"
    r"|attach|detach|copy|grant|revoke|pragma|vacuum|install|load"
    r"|exec|execute|merge|import|export)\b",
    re.IGNORECASE,
)

# Vocabulary that belongs to this dataset. Used to tell an out-of-scope question
# apart from a badly phrased business one.
_DOMAIN_RE = re.compile(
    r"\b(opportunit\w*|deal|deals|pipeline|revenue|amount|amounts|deal\s+value"
    r"|win\s*rate|won|lost|close[ds]?|closing|stage|stages|owner|owners|rep|reps"
    r"|account|accounts|region|regions|industry|industries|lead\s*source"
    r"|quarter|month|forecast|sales|customer|customers|opp-\d+)\b",
    re.IGNORECASE,
)

# Subjects that clearly have nothing to do with the dataset.
_OFF_TOPIC_RE = re.compile(
    r"\b(weather|joke|jokes|president|prime\s+minister|news|sport|sports|football"
    r"|movie|movies|song|songs|music|recipe|cook|poem|story|capital\s+of"
    r"|who\s+are\s+you|your\s+name|how\s+are\s+you|time\s+is\s+it|date\s+today"
    r"|translate|meaning\s+of\s+life|stock\s+price|bitcoin|horoscope)\b",
    re.IGNORECASE,
)

# Grouped so that a category always yields a reply about that category. Wording
# is intentionally suggestive: a refusal should teach what to ask next.
_TEMPLATES: dict[RefusalCategory, tuple[str, ...]] = {
    RefusalCategory.METADATA: (
        "I'm designed to answer business questions about the opportunities dataset "
        "rather than expose database structure directly. Try asking about pipeline, "
        "win rate, stages, owners, accounts, regions, or trends.",
        "I can help analyze your opportunity data, but I don't provide database "
        "metadata or table listings. Try asking something like “show the first 5 "
        "opportunities” or “win rate by region.”",
        "I'm focused on business analytics rather than database administration. Ask "
        "me about opportunity counts, stages, owners, accounts, amounts, win rates, "
        "or trends.",
    ),
    RefusalCategory.UNSAFE_SQL: (
        "I can only run safe read-only analytics questions. Try asking something "
        "like “show open pipeline by stage” or “win rate by region.”",
        "That request isn't supported as a direct database operation. I can safely "
        "analyze the opportunity data using business questions in plain English.",
        "I can safely analyze the opportunities dataset, but I won't run "
        "administrative or data-changing database commands. Try asking about counts, "
        "pipeline, stages, owners, regions, or account performance.",
    ),
    RefusalCategory.OUT_OF_SCOPE: (
        "I'm focused on analytics for the opportunities dataset. Try asking a "
        "business question such as “What is the win rate by region?”",
        "That request is outside the analytics scope of this assistant. I can help "
        "with sales questions such as pipeline by stage, closed-won deals, account "
        "performance, or regional trends.",
        "I'm here to help you understand your sales pipeline. Ask about revenue, "
        "opportunity counts, owners, regions, stages, or performance.",
    ),
    RefusalCategory.NEEDS_CONTEXT: (
        "I'd need a little more detail to answer that safely. Try asking with the "
        "region, owner, account, stage, or measure you have in mind.",
        "I need a clearer business question for that request. For example, ask "
        "“win rate for EMEA” rather than only “what about EMEA?”.",
    ),
    RefusalCategory.UNSUPPORTED: (
        "I couldn't treat that as a supported analytics question. Try something like "
        "“which region has the highest win rate?” or “show open pipeline by owner.”",
    ),
}


def _normalize(question: str) -> str:
    """Collapse whitespace and case so equivalent phrasings hash identically."""
    if not isinstance(question, str):
        return ""
    return re.sub(r"\s+", " ", question).strip().lower()


def classify_refusal(question: str) -> RefusalCategory:
    """Decide why a question is being refused, from its wording alone.

    Order matters. Metadata is checked first because "describe the opportunities
    table" also contains a domain word, and asking for structure is the more
    specific intent.
    """
    normalized = _normalize(question)
    if not normalized:
        return RefusalCategory.UNSUPPORTED

    if _METADATA_RE.search(normalized):
        return RefusalCategory.METADATA

    if _UNSAFE_SQL_RE.search(normalized):
        return RefusalCategory.UNSAFE_SQL

    if _OFF_TOPIC_RE.search(normalized):
        return RefusalCategory.OUT_OF_SCOPE

    # No dataset vocabulary at all: the question is about something else.
    if not _DOMAIN_RE.search(normalized):
        return RefusalCategory.OUT_OF_SCOPE

    # Recognisably about this data, just not answerable as asked.
    return RefusalCategory.UNSUPPORTED


def category_for_intent(intent: Intent, question: str) -> RefusalCategory:
    """Choose the reply category for a routed question.

    The router says *why* a request is being turned down; the wording of the
    question picks the most specific reply within that reason. An unsafe request
    that also reads as a metadata request gets the metadata reply, because that
    is the more useful thing to say back.
    """
    if intent is Intent.UNSAFE:
        if _METADATA_RE.search(_normalize(question)):
            return RefusalCategory.METADATA
        return RefusalCategory.UNSAFE_SQL

    if intent is Intent.INSUFFICIENT_CONTEXT:
        return RefusalCategory.NEEDS_CONTEXT

    # UNSUPPORTED, or any answerable intent that failed later for another
    # reason: the wording is all there is to go on.
    return classify_refusal(question)


def refusal_message(question: str, category: RefusalCategory | None = None) -> str:
    """Return the friendly reply for a question that cannot be answered.

    Deterministic: the same question always yields the same reply. Variation
    across different questions comes from a stable digest rather than `random`,
    which keeps tests reproducible.

    ``category`` lets a caller that already knows why it is refusing say so.
    Without it the category is inferred from the wording alone.
    """
    if category is None:
        category = classify_refusal(question)
    templates = _TEMPLATES[category]
    if len(templates) == 1:
        return templates[0]

    digest = hashlib.sha256(_normalize(question).encode("utf-8")).digest()
    return templates[digest[0] % len(templates)]
