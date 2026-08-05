"""Logging setup for the pipeline.

Call setup_logging() once at the start of an entrypoint, then use get_logger()
everywhere else. Console output stays readable; the log file keeps the detail.
"""

from __future__ import annotations

import logging
import logging.config
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.config import DEFAULT_LOGGING_PATH, PROJECT_ROOT

if TYPE_CHECKING:
    import pandas as pd

# Fallback format, used only if config/logging.yaml cannot be read.
_FALLBACK_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_FALLBACK_DATEFMT = "%H:%M:%S"

_is_configured = False


def setup_logging(
    path: str | Path | None = None,
    *,
    level: str | None = None,
    force: bool = False,
) -> None:
    """Configure logging from config/logging.yaml.

    Safe to call more than once — later calls do nothing unless force is set.

    Args:
        path: Logging config file. Defaults to config/logging.yaml.
        level: Overrides the console level, e.g. "DEBUG" when debugging a run.
        force: Reapply the configuration even if it was already set up.
    """
    global _is_configured

    if _is_configured and not force:
        return

    config_path = Path(path).resolve() if path else DEFAULT_LOGGING_PATH

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        _apply_fallback(level)
        logging.getLogger(__name__).warning(
            "Could not load %s (%s). Using basic console logging.", config_path, exc
        )
        _is_configured = True
        return

    # Handlers write to relative paths, so anchor them to the project root and
    # make sure the directories exist before dictConfig opens the files.
    for handler in config.get("handlers", {}).values():
        if filename := handler.get("filename"):
            resolved = (PROJECT_ROOT / filename).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(resolved)

    if level:
        config["handlers"]["console"]["level"] = level.upper()

    logging.config.dictConfig(config)
    _is_configured = True


def _apply_fallback(level: str | None) -> None:
    """Set up plain console logging when the YAML config is unavailable.

    Args:
        level: Log level name, defaulting to INFO.
    """
    logging.basicConfig(
        level=getattr(logging, (level or "INFO").upper(), logging.INFO),
        format=_FALLBACK_FORMAT,
        datefmt=_FALLBACK_DATEFMT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging first if nobody has yet.

    Args:
        name: Logger name. Pass __name__ from the calling module.

    Returns:
        A logger ready to use.
    """
    if not _is_configured:
        setup_logging()
    return logging.getLogger(name)


@contextmanager
def log_step(logger: logging.Logger, description: str) -> Generator[None, None, None]:
    """Log the start and end of a pipeline step, with how long it took.

    Failures are logged with a traceback before the exception is re-raised, so
    a crashed run leaves a usable record behind.

    Args:
        logger: Logger to write to.
        description: What the step does, e.g. "Cleaning training data".

    Yields:
        None. Run the step inside the with block.
    """
    logger.info("START  %s", description)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("FAILED %s after %.2fs", description, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("DONE   %s (%.2fs)", description, elapsed)


def log_dataframe(logger: logging.Logger, df: pd.DataFrame, label: str) -> None:
    """Log the shape and missing-value count of a dataframe.

    Useful between pipeline stages to catch rows silently disappearing.

    Args:
        logger: Logger to write to.
        df: Dataframe to describe.
        label: Name for the dataframe in the log line.
    """
    missing = int(df.isna().sum().sum())
    logger.info(
        "%s: %s rows x %s columns, %s missing values",
        label,
        f"{len(df):,}",
        df.shape[1],
        f"{missing:,}",
    )


def log_config(logger: logging.Logger, config: Any) -> None:
    """Log the settings that decide how a run behaves.

    Recorded at the start of every run so results can be traced back to the
    exact configuration that produced them.

    Args:
        logger: Logger to write to.
        config: A loaded Config object.
    """
    logger.info("Config loaded from %s", config.source_file)
    logger.info("Random seed        : %s", config.project.random_seed)
    logger.info(
        "Training window    : %s to %s",
        config.split.train_start,
        config.split.train_end,
    )
    logger.info(
        "Holdout window     : %s to %s",
        config.split.holdout_start,
        config.split.holdout_end,
    )
    logger.info(
        "Estimator          : %s (target transform: %s)",
        config.model.estimator,
        config.model.target_transform,
    )
    logger.info(
        "Rate-per-mile keep : %.2f to %.2f $/mile",
        config.cleaning.rpm_lower,
        config.cleaning.rpm_upper,
    )