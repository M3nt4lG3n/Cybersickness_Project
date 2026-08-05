"""Common/utils.py - small shared helpers used across the pipeline."""

import logging
import sys
from pathlib import Path

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path, **kwargs) -> pd.DataFrame:
    """Read a CSV defensively - returns an empty DataFrame (with a logged
    warning) instead of raising, since raw capture folders occasionally
    contain truncated, empty, or missing exports.

    low_memory=False by default: wide combined/labscribe CSVs (dozens of
    columns) get chunk-read by pandas otherwise, which can infer a
    different dtype per chunk for a column and trigger a DtypeWarning even
    though the data itself is fine. Reading in one pass avoids that.
    """
    logger = get_logger("utils")
    kwargs.setdefault("low_memory", False)
    try:
        df = pd.read_csv(path, **kwargs)
        if df.empty:
            logger.warning(f"CSV loaded but empty: {path}")
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError) as exc:
        logger.warning(f"Could not read CSV {path}: {exc}")
        return pd.DataFrame()
    except Exception as exc:  # malformed CSV, encoding issues, etc.
        logger.warning(f"Unexpected error reading {path}: {exc}")
        return pd.DataFrame()


def match_any_keyword(column_name: str, keywords) -> bool:
    col = str(column_name).lower()
    return any(kw.lower() in col for kw in keywords)


def tag_feature_group(column_name: str, feature_group_keywords: dict) -> str:
    """Return the name of the first feature group whose keywords match
    column_name, or 'unknown' if none match."""
    for group, keywords in feature_group_keywords.items():
        if match_any_keyword(column_name, keywords):
            return group
    return "unknown"
