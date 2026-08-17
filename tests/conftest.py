"""Shared pytest fixtures.

The offline tests must run with no API key and no network access. Only the tests
marked ``live`` contact Groq, and they are deselected by default (see
``pyproject.toml``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project modules importable without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import database  # noqa: E402  (import after sys.path setup)
import chart  # noqa: E402
import history_repository  # noqa: E402
import llm  # noqa: E402


@pytest.fixture(autouse=True)
def no_live_api_calls(request, monkeypatch):
    """Make an unintended API call impossible rather than merely unlikely.

    An offline test that stubs the wrong seam used to fall through to the real
    provider and quietly pass while spending quota. Blocking the client itself
    turns that mistake into an immediate, obvious failure that names the fix.
    Tests marked ``live`` opt out, which is the whole point of the marker.
    """
    if "live" in request.keywords:
        return

    def refuse_to_connect():
        raise AssertionError(
            "An offline test tried to reach the language model. Stub the "
            "generator the code under test actually calls, or mark the test "
            "with @pytest.mark.live."
        )

    monkeypatch.setattr(llm, "_get_client", refuse_to_connect)


@pytest.fixture(autouse=True)
def isolated_history_database(tmp_path, monkeypatch):
    """Point every test at a throwaway history database.

    Applied automatically so that no test - including CLI and web tests that
    persist history as a side effect - can ever write to the production
    ``history.duckdb`` in the project root.
    """
    monkeypatch.setattr(history_repository, "HISTORY_DATABASE", tmp_path / "history.duckdb")
    return tmp_path / "history.duckdb"


@pytest.fixture(autouse=True)
def isolated_chart_directory(tmp_path, monkeypatch):
    """Keep automatic chart rendering out of the real project chart gallery."""
    target = tmp_path / "charts"
    monkeypatch.setattr(chart, "CHARTS_DIR", target)
    return target


@pytest.fixture(scope="session")
def _database_session():
    """Open the DuckDB connection once for the whole session.

    DuckDB permits a single writer, so the database is opened once rather than
    per test, and closed at the end.
    """
    database.initialize_database()
    yield database
    database.close_connection()


@pytest.fixture
def initialized_database(_database_session):
    """Guarantee this test starts with a live database connection.

    Some tests deliberately close the connection to exercise the uninitialized
    guards (see :func:`clean_database_state`). Because the session fixture is
    created only once, it cannot repair that itself, so the connection is
    restored here when a previous test has closed it. Without this, test
    ordering silently decides whether later tests pass.
    """
    if database._connection is None:
        database.initialize_database()
    return database


@pytest.fixture
def clean_database_state():
    """Run a test with no active connection.

    The connection is closed before and after; ``initialized_database`` restores
    it for any later test that needs one.
    """
    database.close_connection()
    yield
    database.close_connection()
