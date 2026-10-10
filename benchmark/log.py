"""Progress log for framework runs.

Historical experiment modules print their own lines. This logger is only
the framework runner. A line names a stage, such as a trial or a model
training call. It does not name an individual circuit evaluation.

``configure_logging`` attaches one handler to the ``benchmark`` logger.
It does not call ``logging.basicConfig`` and it does not replace handlers
already attached to the root logger.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "benchmark"
_HANDLER_MARK = "_benchmark_handler"


def get_logger() -> logging.Logger:
    """Return the framework logger. Does not attach a handler."""
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach one stderr handler when this process has not attached one yet.

    A second call keeps the existing handler and returns the same logger.
    """
    logger = get_logger()
    logger.setLevel(level)
    if not any(getattr(handler, _HANDLER_MARK, False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_MARK, True)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def log_event(event: str, **fields) -> None:
    """Record one stage at INFO. ``event`` is the first field on the line."""
    get_logger().info(_format(event, fields))


def log_failure(event: str, error: BaseException, **fields) -> None:
    """Record a failure and its traceback. Does not catch the exception."""
    fields.setdefault("error_type", type(error).__name__)
    get_logger().error(_format(event, fields), exc_info=error)


def _format(event: str, fields: dict) -> str:
    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={_render(value)}")
    return " ".join(parts)


def _render(value) -> str:
    text = str(value).replace("\n", "\\n")
    if text == "" or any(character.isspace() or character in '="\'' for character in text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text
