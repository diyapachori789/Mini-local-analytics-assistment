"""Project-wide logging configuration.

Call :func:`setup_logging` once, at application entry point. Every other module
obtains its own logger with ``logging.getLogger(__name__)`` and never configures
handlers itself.

Two handlers are installed:

* a rotating file handler writing full detail to ``logs/app.log``
* a console handler kept at WARNING by default so normal CLI output stays clean

Both levels are overridable via the ``LOG_FILE_LEVEL`` and ``LOG_CONSOLE_LEVEL``
environment variables.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import (
    GROQ_API_KEY,
    LOG_BACKUP_COUNT,
    LOG_CONSOLE_LEVEL,
    LOG_DIR,
    LOG_FILE,
    LOG_FILE_LEVEL,
    LOG_MAX_BYTES,
    WEB_LOG_FILE,
)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that log request detail at INFO. Left at their defaults
# they bury the application's own events and put HTTP metadata in the log file.
NOISY_LOGGERS = ("httpx", "httpcore", "groq", "urllib3")

# Loggers whose records also belong in the dedicated web log. "web_app" is the
# adapter itself; "werkzeug" carries the request lines and the bound address.
WEB_LOGGER_NAMES = ("web_app", "werkzeug")

# Guard against duplicate handlers if setup_logging is called more than once.
_configured = False
_web_configured = False


class SecretRedactingFilter(logging.Filter):
    """Scrub the Groq API key from any log record before it is emitted.

    Defence in depth: no module is supposed to log the key, but an exception
    string from an HTTP client can embed request details. This guarantees the
    secret never reaches a handler even if that happens.
    """

    _REPLACEMENT = "***REDACTED***"

    def filter(self, record: logging.LogRecord) -> bool:
        if not GROQ_API_KEY:
            return True

        if isinstance(record.msg, str) and GROQ_API_KEY in record.msg:
            record.msg = record.msg.replace(GROQ_API_KEY, self._REPLACEMENT)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: self._redact(value) for key, value in record.args.items()
                }
            else:
                record.args = tuple(self._redact(value) for value in record.args)

        return True

    def _redact(self, value: object) -> object:
        """Replace the API key inside a single log argument."""
        if isinstance(value, str) and GROQ_API_KEY in value:
            return value.replace(GROQ_API_KEY, self._REPLACEMENT)
        return value


def _resolve_level(name: str, fallback: int) -> int:
    """Translate a level name into a logging constant, falling back if invalid."""
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else fallback


def setup_logging(*, force: bool = False) -> logging.Logger:
    """Configure root logging for the application and return the root logger."""
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_level = _resolve_level(LOG_FILE_LEVEL, logging.INFO)
    console_level = _resolve_level(LOG_CONSOLE_LEVEL, logging.WARNING)

    # The root logger must pass through the more permissive of the two levels.
    root.setLevel(min(file_level, console_level))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    redactor = SecretRedactingFilter()

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redactor)
    root.addHandler(console_handler)

    # A failure to open the log file must never stop the application; degrade to
    # console-only logging and report it once.
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("File logging disabled, could not open %s: %s", LOG_FILE, exc)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return root


def setup_web_logging(*, force: bool = False) -> bool:
    """Add a rotating web-layer log alongside the main application log.

    Records from the web adapter and Werkzeug are written to ``logs/web_app.log``
    in addition to propagating to the root handlers, so ``logs/app.log`` stays the
    complete record and this file is a focused view of one browser session.

    The same redaction filter is applied, so the API key can never reach this
    file either. Returns whether the handler is installed; a failure to open the
    file degrades to the existing logging rather than stopping the server.
    """
    global _web_configured

    if _web_configured and not force:
        return True

    redactor = SecretRedactingFilter()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            WEB_LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Web logging disabled, could not open %s: %s", WEB_LOG_FILE, exc
        )
        return False

    handler.setLevel(_resolve_level(LOG_FILE_LEVEL, logging.INFO))
    handler.setFormatter(formatter)
    handler.addFilter(redactor)

    for name in WEB_LOGGER_NAMES:
        target = logging.getLogger(name)
        # Never stack a second handler onto the same logger across calls.
        already = any(
            isinstance(existing, RotatingFileHandler)
            and getattr(existing, "baseFilename", None) == handler.baseFilename
            for existing in target.handlers
        )
        if not already:
            target.addHandler(handler)
        if target.level == logging.NOTSET:
            target.setLevel(logging.INFO)

    _web_configured = True
    return True
