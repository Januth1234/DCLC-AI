"""Tests for Sinhala grapheme utilities."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizers.grapheme_utils import (
    split_into_graphemes,
    split_into_words,
    validate_sinhala_range,
    is_sinhala,
    SINHALA_START,
    SINHALA_END,
)


def test_split_into_graphemes_empty():
    assert split_into_graphemes("") == []


def test_split_into_graphemes_sinhala():
    # Sample Sinhala: "අ" (U+0D85)
    text = "\u0d85\u0db1\u0dd2\u0db8\u0dad"
    clusters = split_into_graphemes(text)
    assert len(clusters) >= 1
    assert "".join(clusters) == text


def test_split_into_graphemes_ascii():
    assert split_into_graphemes("hello") == ["h", "e", "l", "l", "o"]


def test_split_into_words_empty():
    assert split_into_words("") == []


def test_split_into_words_spaces():
    assert split_into_words("a b c") == ["a", "b", "c"]


def test_validate_sinhala_range():
    assert validate_sinhala_range("") is True
    assert validate_sinhala_range("\u0d85\u0db1\u0dd2") is True


def test_is_sinhala():
    assert is_sinhala(0x0D85) is True
    assert is_sinhala(0x0D80) is True
    assert is_sinhala(0x0DFF) is True
    assert is_sinhala(0x0041) is False
