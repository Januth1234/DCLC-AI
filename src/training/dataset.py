"""Datasets for DCLC training."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Sinhala text corpus."""

    def __init__(self, path: str, max_len: int = 1024):
        self.path = Path(path)
        self.lines = []
        if self.path.exists():
            self.lines = [l.strip() for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return {"text": self.lines[i][:self.max_len * 4]}


class ImageCaptionDataset(Dataset):
    """Image + Sinhala caption pairs."""

    def __init__(self, json_path: str, image_root: str, max_len: int = 512):
        self.image_root = Path(image_root)
        with open(json_path) as f:
            self.samples = json.load(f) if isinstance(json.load(f), list) else list(json.load(f).values())
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {"image_path": str(self.image_root / s.get("image_path", s.get("path", ""))), "caption": s.get("caption", s.get("text", ""))[:self.max_len * 4]}


class ImageEditDataset(Dataset):
    """Image + edit instruction -> edited image."""

    def __init__(self, json_path: str, image_root: str, max_len: int = 256):
        self.image_root = Path(image_root)
        with open(json_path) as f:
            self.samples = json.load(f)
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "image_path": str(self.image_root / s.get("image_path", s.get("path", ""))),
            "edit_instruction": s.get("edit_instruction", s.get("instruction", ""))[:self.max_len * 4],
            "edited_image_path": str(self.image_root / s.get("edited_image_path", s.get("target", ""))),
        }
