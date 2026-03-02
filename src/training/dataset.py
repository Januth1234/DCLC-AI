"""Datasets for DCLC training."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

try:
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize
except ImportError:
    read_image = None
    resize = None


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
            raw = json.load(f)
        self.samples = raw if isinstance(raw, list) else list(raw.values())
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


class MixedModalDataset(Dataset):
    """Text-only or image+caption for multimodal LM. Each item is a token id sequence (text + optional visual)."""

    def __init__(
        self,
        corpus_path: str,
        captions_json: str,
        image_root: str,
        text_tokenizer,
        vq_model: torch.nn.Module,
        visual_start_id: int,
        max_seq_len: int = 512,
        max_caption_len: int = 200,
        image_size: int = 256,
        text_only_prob: float = 0.4,
    ):
        self.image_root = Path(image_root)
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.image_size = image_size
        self.text_only_prob = text_only_prob
        self.text_tok = text_tokenizer
        self.vq = vq_model
        self.visual_start_id = visual_start_id
        self.special = text_tokenizer.get_special_token_ids()
        self.bos = self.special.get("[BOS]", 1)
        self.eos = self.special.get("[EOS]", 2)
        self.img_start = self.special.get("[IMG_START]", 0)
        self.img_end = self.special.get("[IMG_END]", 0)

        self.text_lines = []
        if Path(corpus_path).exists():
            self.text_lines = [
                l.strip() for l in Path(corpus_path).read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
        self.image_samples = []
        if Path(captions_json).exists():
            with open(captions_json) as f:
                raw = json.load(f)
            self.image_samples = raw if isinstance(raw, list) else list(raw.values())
        self.n_text = len(self.text_lines)
        self.n_image = len(self.image_samples)
        self.length = max(1, self.n_text + self.n_image)

    def __len__(self):
        return self.length

    def _load_image(self, path: str) -> torch.Tensor | None:
        if read_image is None or resize is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            img = read_image(str(p)).float() / 255.0
            if img.dim() == 2:
                img = img.unsqueeze(0).expand(3, -1, -1)
            elif img.size(0) == 1:
                img = img.expand(3, -1, -1)
            img = img.unsqueeze(0)
            img = resize(img, [self.image_size, self.image_size])
            return img
        except Exception:
            return None

    def __getitem__(self, i):
        import random
        use_text = (not self.text_lines) or (self.n_image == 0) or (random.random() < self.text_only_prob)
        if use_text and self.text_lines:
            idx = (i % self.n_text) if self.n_text else 0
            line = self.text_lines[idx][: self.max_seq_len * 4]
            ids = [self.bos] + self.text_tok.encode(line, add_special_tokens=False) + [self.eos]
        else:
            if self.n_image == 0:
                ids = [self.bos, self.eos]
            else:
                idx = (i % self.n_image) if self.n_image else 0
                s = self.image_samples[idx]
                img_path = str(self.image_root / s.get("image_path", s.get("path", "")))
                caption = (s.get("caption", s.get("text", "")) or "")[: self.max_caption_len * 4]
                cap_ids = self.text_tok.encode(caption, add_special_tokens=False)[: self.max_caption_len]
                img = self._load_image(img_path)
                if img is not None and self.vq is not None:
                    with torch.no_grad():
                        vq_ids = self.vq.encode_to_ids(img)
                    vq_flat = (vq_ids.flatten() + self.visual_start_id).tolist()
                    ids = [self.bos] + cap_ids + [self.img_start] + vq_flat + [self.img_end] + [self.eos]
                else:
                    ids = [self.bos] + cap_ids + [self.eos]
        ids = ids[: self.max_seq_len]
        return {"input_ids": ids, "labels": ids}
