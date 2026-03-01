"""Unified embedding for text and visual tokens."""
import math
import torch
import torch.nn as nn


class UnifiedEmbedding(nn.Module):
    """One embedding table for all token types."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        b, s = input_ids.shape
        x = self.embed(input_ids)
        x = x + self.pos[:, :s]
        return x
