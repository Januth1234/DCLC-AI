"""Tests for DCLC transformer forward pass."""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.dclc_transformer import DCLCTransformer


@pytest.fixture
def small_model():
    return DCLCTransformer(
        vocab_size=1000,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=256,
        max_seq_len=128,
    )


def test_forward_shape(small_model):
    batch, seq = 2, 16
    input_ids = torch.randint(0, 1000, (batch, seq))
    out = small_model(input_ids)
    assert out.shape == (batch, seq, small_model.hidden_dim)


def test_get_logits_shape(small_model):
    batch, seq = 2, 16
    input_ids = torch.randint(0, 1000, (batch, seq))
    hidden = small_model(input_ids)
    logits = small_model.get_logits(hidden)
    assert logits.shape == (batch, seq, small_model.vocab_size)
