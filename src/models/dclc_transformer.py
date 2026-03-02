"""DCLC 500M transformer (24 layers, 768 dim)."""
import torch
import torch.nn as nn
from src.models.embedding import UnifiedEmbedding
from src.models.transformer_block import TransformerBlock


class DCLCTransformer(nn.Module):
    """Decoder-only transformer for mixed text/image sequences."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        n_layers: int = 24,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = UnifiedEmbedding(vocab_size, hidden_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _causal_mask(self, seq_len: int, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        if mask is None:
            mask = self._causal_mask(input_ids.size(1), input_ids.device)
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        x = self.ln_f(x)
        return x

    def get_logits(self, hidden_states):
        return self.lm_head(hidden_states)
