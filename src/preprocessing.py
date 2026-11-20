"""Text cleaning and dataset preparation for both ML pipelines.

Design rules used in this file:

* Pure functions. Inputs are never modified in place; new objects are returned.
* One cleaning function (clean_text) is shared by training and, later, by the
  prediction code. Using the same function in both places prevents
  training/prediction mismatch.
* No stopword removal and no stemming here. Those are experiments for Phase 7,
  because they do not always help and can remove useful signal.
"""

from __future__ import annotations

import html
import logging
import math
import re

import pandas as pd

from src.utils import validate_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regular expressions (compiled once, reused for speed)
# ---------------------------------------------------------------------------
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
APOSTROPHE_PATTERN = re.compile(r"['’]")  # straight and curly apostrophe
NON_WORD_PATTERN = re.compile(r"[^\w\s]|_")  # punctuation and underscores
WHITESPACE_PATTERN = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Expected columns and labels
# ---------------------------------------------------------------------------
NEWS_REQUIRED_COLUMNS = ("title", "text")
VALID_NEWS_LABELS = {"fake", "real"}
VALID_SMS_LABELS = {"ham", "spam"}


def clean_text(
    text: str | None,
    *,
    lowercase: bool = True,
    strip_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Return a cleaned copy of a single piece of text.

    Steps run in this order:
        1. lowercase
        2. remove HTML tags and decode entities such as &amp;
        3. remove URLs
        4. remove email addresses
        5. remove apostrophes (so "don't" becomes "dont") and other punctuation
        6. collapse repeated whitespace and strip the ends

    Missing values (None, NaN, pd.NA) become an empty string. Other non-string
    values raise TypeError, so bad input is caught early instead of being
    silently converted into something meaningless.

    Args:
        text: The raw text.
        lowercase: Convert to lowercase.
        strip_html: Remove HTML tags and decode HTML entities.
        remove_urls: Remove http(s):// and www. links.
        remove_emails: Remove email addresses.
        remove_punctuation: Remove punctuation. Note that this also removes
            symbols such as "$" and "£", which can be useful spam signals.

    Returns:
        The cleaned string (possibly empty).

    Raises:
        TypeError: If text is not a string or a missing value.
    """
    if text is None or text is pd.NA or (isinstance(text, float) and math.isnan(text)):
        return ""
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects a string, got {type(text).__name__}")

    result = text
    if lowercase:
        result = result.lower()
    if strip_html:
        result = HTML_TAG_PATTERN.sub(" ", result)
        result = html.unescape(result)
    if remove_urls:
        result = URL_PATTERN.sub(" ", result)
    if remove_emails:
        result = EMAIL_PATTERN.sub(" ", result)
    if remove_punctuation:
        result = APOSTROPHE_PATTERN.sub("", result)
        result = NON_WORD_PATTERN.sub(" ", result)

    return WHITESPACE_PATTERN.sub(" ", result).strip()


def remove_duplicates(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Remove duplicate texts and texts that carry conflicting labels.

    Two rules are applied:
        * Exact duplicates (same text and same label) are kept once.
        * A text that appears with more than one label is removed completely,
          because the dataset disagrees with itself about it.

    Duplicates matter because the same text in both the training and test
    sets makes the model look better than it really is. This is called
    duplicate leakage.

    Returns:
        A new DataFrame with a fresh index.
    """
    deduplicated = df.drop_duplicates(subset=[text_col, label_col], keep="first")
    label_counts = deduplicated.groupby(text_col)[label_col].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    cleaned = deduplicated[~deduplicated[text_col].isin(conflicting_texts)]
    return cleaned.reset_index(drop=True)


def _clean_and_deduplicate(
    df: pd.DataFrame,
    raw_col: str,
    valid_labels: set[str],
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Shared steps for both datasets: validate labels, clean, drop empties, dedupe.

    Returns:
        A tuple of (cleaned DataFrame with columns text and label, stats dict).
    """
    unknown_labels = sorted(set(df["label"]) - valid_labels)
    if unknown_labels:
        raise ValueError(
            f"{dataset_name}: unexpected labels {unknown_labels}. "
            f"Expected only {sorted(valid_labels)}."
        )

    input_rows = len(df)
    texts = df[raw_col].fillna("").astype(str).map(clean_text)
    working = pd.DataFrame({"text": texts.to_numpy(), "label": df["label"].to_numpy()})

    non_empty = working[working["text"].str.len() > 0]
    dropped_empty = len(working) - len(non_empty)

    final = remove_duplicates(non_empty, text_col="text", label_col="label")
    dropped_duplicates = len(non_empty) - len(final)

    stats = {
        "input_rows": input_rows,
        "dropped_empty": dropped_empty,
        "dropped_duplicates": dropped_duplicates,
        "output_rows": len(final),
    }
    logger.info("%s preparation: %s", dataset_name, stats)
    return final, stats


def prepare_news_dataset(
    fake_df: pd.DataFrame, real_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Combine the fake and real news tables into one cleaned dataset.

    The title and body are joined into one text column before cleaning,
    because the title often carries strong signal (for example, all-caps
    headlines).

    Args:
        fake_df: Table of fake articles. Needs "title" and "text" columns.
        real_df: Table of real articles. Needs "title" and "text" columns.

    Returns:
        (DataFrame with columns text and label where label is "fake" or "real",
         stats dict).
    """
    validate_columns(fake_df, NEWS_REQUIRED_COLUMNS)
    validate_columns(real_df, NEWS_REQUIRED_COLUMNS)

    combined = pd.concat(
        [fake_df.assign(label="fake"), real_df.assign(label="real")],
        ignore_index=True,
    )
    combined["raw_text"] = (
        combined["title"].fillna("").astype(str)
        + " "
        + combined["text"].fillna("").astype(str)
    )
    return _clean_and_deduplicate(combined, "raw_text", VALID_NEWS_LABELS, "news")


def prepare_sms_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the UCI SMS Spam Collection table.

    The UCI file names its columns v1 (label) and v2 (message) and includes
    extra empty columns. Those are renamed or dropped here.

    Args:
        raw_df: Table loaded from spam.csv. Needs label/message columns, or the
            original v1/v2 names.

    Returns:
        (DataFrame with columns text and label where label is "ham" or "spam",
         stats dict).
    """
    df = raw_df.rename(columns={"v1": "label", "v2": "message"})
    validate_columns(df, ("label", "message"))

    df = df[["label", "message"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.rename(columns={"message": "raw_text"})
    return _clean_and_deduplicate(df, "raw_text", VALID_SMS_LABELS, "sms")
