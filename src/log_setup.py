"""
Shared logging configuration.

Call ``configure_logging(log_file)`` once at module startup in each entry-point.
Subsequent calls for the *same* logger name are no-ops because the handlers are
already attached.

Environment variables
---------------------
LOG_LEVEL   Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO.
LOG_DIR     Directory for the rotating file handler.
            Defaults to /app/logs.
"""

import logging
import os


def configure_logging(log_file: str, log_name: str | None = None) -> logging.Logger:
    """Configure and return a named logger that writes to *log_file*.

    Parameters
    ----------
    log_file:
        Filename (not a full path) of the log file to create inside ``LOG_DIR``.
        Example: ``"scraper.log"``  →  written to ``<LOG_DIR>/scraper.log``.
    log_name:
        Logger name passed to ``logging.getLogger``.  When *None*, the stem of
        *log_file* is used (e.g. ``"scraper"`` for ``"scraper.log"``).

    Returns
    -------
    logging.Logger
        A configured logger ready to use.
    """
    level = getattr(
        logging,
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )

    if log_name is None:
        log_name = os.path.splitext(log_file)[0]

    logger = logging.getLogger(log_name)

    # No-op if this logger already has handlers attached
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
