"""Causal self-attention."""
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.proj(out)
