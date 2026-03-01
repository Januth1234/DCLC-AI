"""Load and use trained Sinhala tokenizer."""
import json
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

SPECIAL_NAMES = ["[BOS]", "[EOS]", "[PAD]", "[IMG_START]", "[IMG_END]", "[EDIT_START]"]


class SinhalaTokenizer:
    """Sinhala tokenizer wrapper."""

    def __init__(self, path: str = "model_output/sinhala_tokenizer.json"):
        if Tokenizer is None:
            raise ImportError("pip install tokenizers")
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path)) if self.path.exists() else None
        self._special_ids = {}

    def load(self) -> bool:
        """Load tokenizer from file."""
        if self.path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.path))
            for name in SPECIAL_NAMES:
                id_ = self.tokenizer.token_to_id(name)
                if id_ is not None:
                    self._special_ids[name] = id_
            return True
        return False

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token ids."""
        if not self.tokenizer:
            self.load()
        if not self.tokenizer:
            return []
        enc = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        if not self.tokenizer:
            self.load()
        if not self.tokenizer:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        if not self.tokenizer:
            self.load()
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    def get_special_token_ids(self) -> dict[str, int]:
        """Return special token name -> id mapping."""
        if not self._special_ids and self.tokenizer:
            for name in SPECIAL_NAMES:
                id_ = self.tokenizer.token_to_id(name)
                if id_ is not None:
                    self._special_ids[name] = id_
        return dict(self._special_ids)
