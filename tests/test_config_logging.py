"""Configuration loading and the logging strategy."""

from __future__ import annotations

import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

import config
import logging_config
from config import ConfigurationError
from logging_config import NOISY_LOGGERS, SecretRedactingFilter, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestConfiguration:
    def test_modules_import_without_an_api_key(self):
        """config must never raise at import: offline tests depend on it."""
        script = (
            "import sys; sys.path.insert(0, r'%s');\n"
            "import config, database, llm, sql_guard;\n"
            "print('IMPORT_OK')\n" % PROJECT_ROOT
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PATH": "", "SYSTEMROOT": "C:\\Windows", "GROQ_API_KEY": ""},
        )
        assert "IMPORT_OK" in result.stdout, result.stderr

    def test_require_api_key_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        with pytest.raises(ConfigurationError, match="Missing GROQ_API_KEY"):
            config.require_groq_api_key()

    def test_require_api_key_returns_the_key(self, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_example")
        assert config.require_groq_api_key() == "gsk_example"

    def test_expected_paths_and_constants(self):
        assert config.CSV_PATH.exists(), "sample dataset is missing"
        assert config.TABLE_NAME == "opportunities"
        assert config.TABLE_SCHEMA == "main"
        assert config.LLM_TEMPERATURE == 0, "generation must stay deterministic"
        assert config.LLM_TIMEOUT_SECONDS > 0, "a request must not hang forever"


class TestSecretRedaction:
    """The API key must never reach a log handler."""

    SECRET = "gsk_supersecret_test_value"

    @pytest.fixture
    def redactor(self, monkeypatch):
        monkeypatch.setattr(logging_config, "GROQ_API_KEY", self.SECRET)
        return SecretRedactingFilter()

    def _record(self, msg, args=None):
        return logging.LogRecord("t", logging.ERROR, "p", 1, msg, args, None)

    def test_scrubs_key_from_message(self, redactor):
        record = self._record(f"auth failed for {self.SECRET}")
        redactor.filter(record)
        assert self.SECRET not in record.getMessage()
        assert "***REDACTED***" in record.getMessage()

    def test_scrubs_key_from_tuple_args(self, redactor):
        record = self._record("header: %s", (f"Bearer {self.SECRET}",))
        redactor.filter(record)
        assert self.SECRET not in record.getMessage()

    def test_scrubs_key_from_dict_args(self, redactor):
        # logging expects mapping-style args wrapped in a one-tuple; LogRecord
        # unwraps it and stores the mapping on record.args.
        record = self._record("header: %(auth)s", ({"auth": self.SECRET},))
        assert isinstance(record.args, dict)
        redactor.filter(record)
        assert self.SECRET not in record.getMessage()

    def test_leaves_unrelated_messages_alone(self, redactor):
        record = self._record("nothing sensitive here")
        redactor.filter(record)
        assert record.getMessage() == "nothing sensitive here"

    def test_filter_always_allows_the_record_through(self, redactor):
        assert redactor.filter(self._record("anything")) is True


class TestLoggingSetup:
    def test_setup_is_idempotent(self):
        setup_logging()
        first = len(logging.getLogger().handlers)
        setup_logging()
        assert len(logging.getLogger().handlers) == first

    def test_third_party_loggers_are_quietened(self):
        setup_logging()
        for name in NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING

    def test_handlers_carry_the_redacting_filter(self):
        """Both handlers we install must redact.

        Only the handlers created by setup_logging are checked: the test runner
        attaches its own capture handler to the root logger, which is not ours to
        configure.
        """
        setup_logging()
        redacting = [
            handler
            for handler in logging.getLogger().handlers
            if any(isinstance(f, SecretRedactingFilter) for f in handler.filters)
        ]
        assert any(
            isinstance(handler, RotatingFileHandler) for handler in redacting
        ), "file handler is missing the redaction filter"
        assert len(redacting) >= 2, "expected console and file handlers to redact"

    def test_log_directory_is_created(self):
        setup_logging()
        assert config.LOG_DIR.exists()


class TestWebLogging:
    """The dedicated web log is additive: app.log keeps the complete record."""

    @pytest.fixture
    def web_log(self, tmp_path, monkeypatch):
        """Redirect the web log to a temporary file and reset the guard."""
        target = tmp_path / "web_app.log"
        monkeypatch.setattr(logging_config, "WEB_LOG_FILE", target)
        monkeypatch.setattr(logging_config, "_web_configured", False)
        yield target
        # Detach the handler so later tests do not write to a deleted path.
        for name in logging_config.WEB_LOGGER_NAMES:
            target_logger = logging.getLogger(name)
            for handler in list(target_logger.handlers):
                target_logger.removeHandler(handler)
                handler.close()

    def test_creates_the_web_log_file(self, web_log):
        assert logging_config.setup_web_logging(force=True) is True
        logging.getLogger("web_app").info("web event")
        assert web_log.exists()
        assert "web event" in web_log.read_text(encoding="utf-8")

    def test_uses_a_rotating_handler_with_bounds(self, web_log):
        logging_config.setup_web_logging(force=True)
        handlers = [
            handler
            for handler in logging.getLogger("web_app").handlers
            if isinstance(handler, RotatingFileHandler)
        ]
        assert handlers, "the web log must rotate"
        assert handlers[0].maxBytes == config.LOG_MAX_BYTES
        assert handlers[0].backupCount == config.LOG_BACKUP_COUNT

    def test_werkzeug_records_are_captured(self, web_log):
        logging_config.setup_web_logging(force=True)
        logging.getLogger("werkzeug").info("127.0.0.1 - - GET / HTTP/1.1 200")
        assert "GET /" in web_log.read_text(encoding="utf-8")

    def test_the_api_key_is_redacted(self, web_log, monkeypatch):
        secret = "gsk_web_log_secret_value"
        monkeypatch.setattr(logging_config, "GROQ_API_KEY", secret)
        logging_config.setup_web_logging(force=True)
        logging.getLogger("web_app").error("auth failed for %s", secret)

        contents = web_log.read_text(encoding="utf-8")
        assert secret not in contents
        assert "***REDACTED***" in contents

    def test_handlers_are_not_stacked_on_repeat_calls(self, web_log):
        logging_config.setup_web_logging(force=True)
        first = len(logging.getLogger("web_app").handlers)
        logging_config.setup_web_logging(force=True)
        assert len(logging.getLogger("web_app").handlers) == first

    def test_records_still_reach_the_main_log(self, web_log):
        """app.log stays complete; the web log is a filtered view, not a move."""
        logging_config.setup_web_logging(force=True)
        assert logging.getLogger("web_app").propagate is True

    def test_missing_directory_degrades_without_raising(self, tmp_path, monkeypatch):
        # A path whose parent is a file cannot be opened.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(logging_config, "LOG_DIR", blocker)
        monkeypatch.setattr(logging_config, "WEB_LOG_FILE", blocker / "web_app.log")
        monkeypatch.setattr(logging_config, "_web_configured", False)
        assert logging_config.setup_web_logging(force=True) is False

    def test_web_app_uses_a_fixed_logger_name(self):
        """__name__ becomes __main__ when run directly, which would miss the handler."""
        import web_app

        assert web_app.logger.name == "web_app"
