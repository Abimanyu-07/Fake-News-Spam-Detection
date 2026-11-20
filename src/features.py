"""Feature engineering: convert cleaned text into numeric matrices.

Machine learning models cannot read words. TF-IDF turns each document into a
row of numbers, with one column per vocabulary term. A term's weight is high
when it is frequent in this document and rare across all documents.

Leakage rule: the vectorizer must learn its vocabulary and weights from the
training split only. fit_transform_split() enforces this order: split first,
then fit on training text, then transform the test text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.utils import validate_columns

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class TfidfConfig:
    """Settings for the TF-IDF vectorizer.

    Attributes:
        max_features: Keep only the most frequent N terms. None keeps all.
        ngram_range: (1, 1) = single words; (1, 2) = single words and pairs.
        min_df: Ignore terms in fewer documents than this. An int is a count,
            a float is a fraction of documents. Removes typos and rare noise.
        max_df: Ignore terms in more than this fraction of documents. Removes
            words so common they carry no information.
        sublinear_tf: Use 1 + log(count) instead of the raw count, so a word
            repeated 50 times does not outweigh a word repeated 5 times 10x.
    """

    max_features: int | None = 20000
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 2
    max_df: float = 0.95
    sublinear_tf: bool = True

    def __post_init__(self) -> None:
        low, high = self.ngram_range
        if low < 1 or high < low:
            raise ValueError(f"Invalid ngram_range {self.ngram_range}: need 1 <= low <= high")
        if not 0 < self.max_df <= 1.0:
            raise ValueError(f"max_df must be in (0, 1], got {self.max_df}")
        if isinstance(self.min_df, float) and not 0 < self.min_df < 1.0:
            raise ValueError(f"float min_df must be in (0, 1), got {self.min_df}")
        if isinstance(self.min_df, int) and self.min_df < 1:
            raise ValueError(f"int min_df must be >= 1, got {self.min_df}")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError(f"max_features must be >= 1 or None, got {self.max_features}")


# Starting points only. Phase 7 tunes these with measured experiments.
NEWS_CONFIG = TfidfConfig(max_features=50000, ngram_range=(1, 2), min_df=3, max_df=0.9)
SMS_CONFIG = TfidfConfig(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.9)


@dataclass
class FeatureSplit:
    """Output of fit_transform_split: matrices plus the fitted vectorizer."""

    X_train: csr_matrix
    X_test: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    vectorizer: TfidfVectorizer


def build_vectorizer(config: TfidfConfig) -> TfidfVectorizer:
    """Create an unfitted TF-IDF vectorizer from a config."""
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
    )


def encode_binary_labels(
    labels: pd.Series, positive_label: str, negative_label: str
) -> np.ndarray:
    """Convert text labels to 1 (positive) and 0 (negative).

    Positive means the class we are detecting: "fake" for news, "spam" for SMS.

    Raises:
        ValueError: If any label is neither the positive nor the negative label.
    """
    allowed = {positive_label, negative_label}
    unknown = sorted(set(labels.unique()) - allowed)
    if unknown:
        raise ValueError(f"Unexpected labels {unknown}; expected only {sorted(allowed)}")
    return (labels == positive_label).astype(int).to_numpy()


def fit_transform_split(
    df: pd.DataFrame,
    config: TfidfConfig,
    positive_label: str,
    negative_label: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> FeatureSplit:
    """Split the data, then build TF-IDF features without leakage.

    Steps:
        1. Stratified split, so both sets keep the same class ratio.
        2. Fit the vectorizer on training text only.
        3. Transform test text using the training vocabulary.

    Args:
        df: Cleaned dataset with columns text and label.
        config: Vectorizer settings.
        positive_label: Label encoded as 1.
        negative_label: Label encoded as 0.
        test_size: Fraction of rows kept for testing.
        random_state: Seed, so the split is reproducible.
    """
    validate_columns(df, ("text", "label"))
    y = encode_binary_labels(df["label"], positive_label, negative_label)

    texts_train, texts_test, y_train, y_test = train_test_split(
        df["text"],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = build_vectorizer(config)
    X_train = vectorizer.fit_transform(texts_train)  # learn vocabulary from train only
    X_test = vectorizer.transform(texts_test)

    return FeatureSplit(X_train, X_test, y_train, y_test, vectorizer)


def top_terms(vectorizer: TfidfVectorizer, X: csr_matrix, n: int = 20) -> pd.DataFrame:
    """Return the n terms with the largest total TF-IDF weight across documents.

    Note: total weight favours terms that appear often. It is a descriptive
    view of the data, not a measure of how well a term predicts a label.
    """
    terms = vectorizer.get_feature_names_out()
    weights = np.asarray(X.sum(axis=0)).ravel()
    order = np.argsort(weights)[::-1][:n]
    return pd.DataFrame({"term": terms[order], "total_tfidf": weights[order].round(3)})


def matrix_stats(X: csr_matrix) -> dict:
    """Describe a sparse feature matrix.

    Most entries are zero, because each document uses only a small part of the
    vocabulary. 'density' is the fraction of non-zero entries.
    """
    rows, cols = X.shape
    nonzero = X.nnz
    return {
        "documents": rows,
        "features": cols,
        "non_zero_entries": nonzero,
        "density": nonzero / (rows * cols) if rows and cols else 0.0,
        "avg_terms_per_document": nonzero / rows if rows else 0.0,
    }
