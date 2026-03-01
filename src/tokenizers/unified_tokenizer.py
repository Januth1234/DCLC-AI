"""Unified tokenizer: text + visual in one embedding space."""
from pathlib import Path

from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer
from src.tokenizers.visual_vq import VisualVQ
import torch


class UnifiedTokenizer:
    """Text tokens 0..N-1, visual tokens N..N+codebook-1."""

    def __init__(self, text_tokenizer_path="model_output/sinhala_tokenizer.json", codebook_size=8192):
        self.text_tok = SinhalaTokenizer(text_tokenizer_path)
        if Path(text_tokenizer_path).exists():
            self.text_tok.load()
        self.codebook_size = codebook_size
        self.text_vocab_size = self.text_tok.get_vocab_size() or 50000
        self.visual_start_id = self.text_vocab_size
        self.total_vocab = self.text_vocab_size + codebook_size
        self.vq = None

    def set_vq(self, vq: VisualVQ):
        """Set VQ model for image encode/decode."""
        self.vq = vq

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token ids."""
        return self.text_tok.encode(text)

    def decode_text(self, ids: list[int]) -> str:
        """Decode text token ids."""
        return self.text_tok.decode(ids)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent token ids (visual range)."""
        if self.vq is None:
            raise RuntimeError("VQ model not set")
        idx = self.vq.encode_to_ids(image)
        return idx + self.visual_start_id

    def decode_image(self, ids: torch.Tensor) -> torch.Tensor:
        """Decode visual token ids to image."""
        if self.vq is None:
            raise RuntimeError("VQ model not set")
        idx = ids - self.visual_start_id
        return self.vq.decode_from_ids(idx)

    def get_vocab_size(self) -> int:
        return self.total_vocab

    def get_special_token_ids(self) -> dict:
        return self.text_tok.get_special_token_ids()
