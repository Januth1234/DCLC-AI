"""Transformer decoder block."""
import torch
import torch.nn as nn
from src.models.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """Decoder block: attention + MLP with residuals."""

    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
