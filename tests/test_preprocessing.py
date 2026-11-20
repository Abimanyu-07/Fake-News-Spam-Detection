"""Tests for src/preprocessing.py. Run with: python -m pytest -v"""

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    prepare_news_dataset,
    prepare_sms_dataset,
    remove_duplicates,
)


class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("HeLLo World") == "hello world"

    def test_removes_html_tags(self):
        assert clean_text("<p>Breaking <b>news</b></p>") == "breaking news"

    def test_decodes_html_entities(self):
        assert clean_text("Tom &amp; Jerry") == "tom jerry"

    def test_removes_http_urls(self):
        assert clean_text("Read more at https://example.com/a?b=1 now") == "read more at now"

    def test_removes_www_urls(self):
        assert clean_text("see www.site.org today") == "see today"

    def test_removes_email_addresses(self):
        assert clean_text("write to john.doe@mail.co.in please") == "write to please"

    def test_removes_punctuation_and_joins_contractions(self):
        assert clean_text("Don't stop!!! Really...") == "dont stop really"

    def test_keeps_digits(self):
        assert clean_text("Call 1234 now") == "call 1234 now"

    def test_removes_currency_symbols_known_tradeoff(self):
        # Documented trade-off: "£" is removed along with other punctuation.
        assert clean_text("Win £1000") == "win 1000"

    def test_normalizes_whitespace(self):
        assert clean_text("  a \n\t b   c  ") == "a b c"

    def test_empty_string_returns_empty_string(self):
        assert clean_text("") == ""

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
    def test_missing_values_become_empty_string(self, missing):
        assert clean_text(missing) == ""

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            clean_text(123)

    def test_steps_can_be_switched_off(self):
        assert clean_text("Hi!", lowercase=False, remove_punctuation=False) == "Hi!"


class TestRemoveDuplicates:
    def test_exact_duplicates_are_kept_once(self):
        df = pd.DataFrame({"text": ["a", "a", "b"], "label": ["fake", "fake", "real"]})
        result = remove_duplicates(df, "text", "label")
        assert list(result["text"]) == ["a", "b"]

    def test_texts_with_conflicting_labels_are_removed_completely(self):
        df = pd.DataFrame({"text": ["x", "x", "y"], "label": ["fake", "real", "fake"]})
        result = remove_duplicates(df, "text", "label")
        assert list(result["text"]) == ["y"]


class TestPrepareNewsDataset:
    def test_combines_title_and_text_and_labels_rows(self):
        fake = pd.DataFrame({"title": ["Shock!"], "text": ["Aliens <b>landed</b>"]})
        real = pd.DataFrame(
            {"title": ["Budget passed"], "text": ["Parliament approved https://x.in/y the budget."]}
        )
        df, stats = prepare_news_dataset(fake, real)

        assert list(df.columns) == ["text", "label"]
        assert set(df["label"]) == {"fake", "real"}
        assert "shock aliens landed" in df["text"].tolist()
        assert "budget passed parliament approved the budget" in df["text"].tolist()
        assert stats["output_rows"] == 2

    def test_handles_missing_title_or_text(self):
        fake = pd.DataFrame({"title": [None, "Only title"], "text": ["Body only", None]})
        real = pd.DataFrame({"title": [None], "text": [None]})  # completely empty row
        df, stats = prepare_news_dataset(fake, real)

        assert sorted(df["text"]) == ["body only", "only title"]
        assert stats["dropped_empty"] == 1

    def test_raises_when_required_column_is_missing(self):
        wrong = pd.DataFrame({"headline": ["x"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_news_dataset(wrong, wrong)


class TestPrepareSmsDataset:
    def test_renames_uci_columns_and_drops_extra_columns(self):
        raw = pd.DataFrame(
            {
                "v1": ["ham", "spam"],
                "v2": ["Ok lar... Call me", "WINNER!! Claim your prize http://x.co"],
                "Unnamed: 2": [None, None],
            }
        )
        df, _ = prepare_sms_dataset(raw)

        assert list(df.columns) == ["text", "label"]
        spam_text = df.loc[df["label"] == "spam", "text"].iloc[0]
        assert spam_text == "winner claim your prize"

    def test_duplicate_messages_are_removed(self):
        raw = pd.DataFrame({"v1": ["ham", "ham"], "v2": ["Hello there", "Hello there"]})
        df, stats = prepare_sms_dataset(raw)
        assert len(df) == 1
        assert stats["dropped_duplicates"] == 1

    def test_rejects_unknown_labels(self):
        raw = pd.DataFrame({"label": ["maybe"], "message": ["hi"]})
        with pytest.raises(ValueError, match="unexpected labels"):
            prepare_sms_dataset(raw)
