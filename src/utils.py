"""Shared helpers: project paths, safe CSV loading, validation and reporting.

Nothing in this file knows about text cleaning or machine learning. It only
deals with files, columns and basic dataset checks, so it can be reused by
training, prediction and the Streamlit app.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Resolve paths from this file's location, so the project works no matter
# which folder you start Python or Jupyter from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_csv(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV file and fail with a helpful message if it is missing.

    Args:
        path: Location of the CSV file.
        encoding: Text encoding. The UCI SMS file needs "latin-1".

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download it and place it in "
            "data/raw/. See data/DATASET_NOTES.md for the exact file names."
        )
    return pd.read_csv(path, encoding=encoding)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to CSV, creating parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def validate_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )


def dataset_overview(df: pd.DataFrame, label_col: str) -> dict:
    """Return a small summary used for quick dataset inspection."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_counts": df[label_col].value_counts(dropna=False).to_dict(),
    }
