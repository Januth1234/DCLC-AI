"""Lightweight graph attention for scene conditioning (optional)."""
import torch
import torch.nn as nn


class GraphAttention(nn.Module):
    """2-layer graph attention over node embeddings."""

    def __init__(self, node_dim: int, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden)

    def forward(self, node_emb, edge_index=None):
        x = self.proj(node_emb)
        for attn in self.layers:
            res, _ = attn(x, x, x)
            x = self.ln(x + res)
        return x.mean(dim=1)
