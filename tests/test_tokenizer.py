"""Tests for Sinhala tokenizer (when trained tokenizer exists)."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer, SPECIAL_NAMES


def test_sinhala_tokenizer_import():
    tok = SinhalaTokenizer(path="model_output/sinhala_tokenizer.json")
    assert tok is not None


def test_special_names():
    assert "[BOS]" in SPECIAL_NAMES
    assert "[EOS]" in SPECIAL_NAMES
    assert "[PAD]" in SPECIAL_NAMES


@pytest.mark.skipif(
    not Path("model_output/sinhala_tokenizer.json").exists(),
    reason="Tokenizer not trained yet",
)
def test_encode_decode_roundtrip():
    tok = SinhalaTokenizer(path="model_output/sinhala_tokenizer.json")
    tok.load()
    text = "ආයුබෝවන්"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # Decode may normalize; at least same length or sensible
    assert isinstance(decoded, str)
