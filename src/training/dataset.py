"""Datasets for DCLC training. Includes LAION webdataset support (optional)."""
import io
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize
except ImportError:
    read_image = None
    resize = None

try:
    import webdataset as wds
except ImportError:
    wds = None


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


def _parse_captions_json(raw):
    """Normalize captions JSON to list of samples (handles {"samples": [...]})."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("samples", raw.get("images", raw.get("captions", list(raw.values()))))
    return []


class ImageCaptionDataset(Dataset):
    """Image + Sinhala caption pairs."""

    def __init__(self, json_path: str, image_root: str, max_len: int = 512):
        self.image_root = Path(image_root)
        with open(json_path) as f:
            raw = json.load(f)
        self.samples = _parse_captions_json(raw)
        if not isinstance(self.samples, list):
            self.samples = [self.samples] if self.samples else []
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
            self.image_samples = _parse_captions_json(raw)
            if not isinstance(self.image_samples, list):
                self.image_samples = list(self.image_samples) if self.image_samples else []
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


# --- LAION webdataset support (optional dependency: webdataset) ---


def _laion_tar_paths(folder: str | Path):
    """List .tar paths under folder (non-recursive)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.tar"))


class LAIONWebDataset(IterableDataset):
    """Stream image+caption from webdataset tars (img2dataset output). Yields {"image": tensor, "caption": str}."""

    def __init__(self, tar_dir: str | Path, image_size: int = 256, image_key: str = "jpg", caption_key: str = "txt"):
        self.tar_dir = Path(tar_dir)
        self.image_size = image_size
        self.image_key = image_key
        self.caption_key = caption_key
        self.urls = [str(p) for p in _laion_tar_paths(tar_dir)]

    def __iter__(self):
        if not self.urls or wds is None:
            return
        from PIL import Image
        dataset = wds.WebDataset(
            self.urls,
            handler=wds.handlers.warn_and_continue,
        )
        for sample in dataset:
            if self.image_key not in sample or self.caption_key not in sample:
                continue
            try:
                img_data = sample[self.image_key]
                if isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                else:
                    img = img_data.convert("RGB") if hasattr(img_data, "convert") else Image.open(io.BytesIO(img_data)).convert("RGB")
                import torch
                from torchvision.transforms.functional import to_tensor, resize
                img_t = to_tensor(img).unsqueeze(0)  # (1,3,H,W)
                img_t = resize(img_t, [self.image_size, self.image_size]).squeeze(0)  # (3,256,256)
                cap = sample[self.caption_key]
                if isinstance(cap, bytes):
                    cap = cap.decode("utf-8", errors="replace")
                yield {"image": img_t, "caption": (cap or "").strip()}
            except Exception:
                continue


class LAIONImageDataset(IterableDataset):
    """Stream images only from LAION webdataset tars (for VQ training). Yields tensor (3, H, W)."""

    def __init__(self, tar_dir: str | Path, image_size: int = 256, image_key: str = "jpg", rank: int = 0, world_size: int = 1):
        self.laion = LAIONWebDataset(tar_dir, image_size=image_size, image_key=image_key, caption_key="txt")
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for idx, item in enumerate(self.laion):
            if idx % self.world_size == self.rank:
                yield item["image"]


class CombinedImageIterableDataset(IterableDataset):
    """Mixes map-style image dataset (explicit) with LAION image stream. For VQ training. Yields image tensor."""

    def __init__(
        self,
        explicit_dataset: Dataset,
        laion_tar_dir: str | Path,
        laion_prob: float = 0.5,
        image_size: int = 256,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.explicit_ds = explicit_dataset
        self.laion_tar_dir = Path(laion_tar_dir)
        self.laion_prob = laion_prob
        self.rank = rank
        self.world_size = world_size
        self.laion_images = LAIONImageDataset(laion_tar_dir, image_size=image_size) if _laion_tar_paths(laion_tar_dir) else None

    def __iter__(self):
        import itertools
        n = len(self.explicit_ds)
        it_explicit = itertools.cycle(range(n)) if n else iter(())
        it_laion = iter(self.laion_images) if self.laion_images else None
        idx = 0
        while True:
            if it_laion is not None and random.random() < self.laion_prob:
                try:
                    img = next(it_laion)
                except StopIteration:
                    it_laion = iter(self.laion_images)
                    img = next(it_laion)
                if idx % self.world_size == self.rank:
                    yield img
                idx += 1
            else:
                if n == 0:
                    if it_laion is None:
                        break
                    try:
                        img = next(it_laion)
                    except StopIteration:
                        it_laion = iter(self.laion_images)
                        img = next(it_laion)
                    if idx % self.world_size == self.rank:
                        yield img
                    idx += 1
                    continue
                i = next(it_explicit)
                img = self.explicit_ds[i]
                if img is not None:
                    if idx % self.world_size == self.rank:
                        yield img
                    idx += 1


class LAIONMixedModalDataset(IterableDataset):
    """Wraps LAIONWebDataset + tokenizer + VQ; yields same format as MixedModalDataset (input_ids, labels)."""

    def __init__(
        self,
        tar_dir: str | Path,
        text_tokenizer,
        vq_model: torch.nn.Module,
        visual_start_id: int,
        max_seq_len: int = 512,
        max_caption_len: int = 200,
        image_size: int = 256,
    ):
        self.laion = LAIONWebDataset(tar_dir, image_size=image_size)
        self.text_tok = text_tokenizer
        self.vq = vq_model
        self.visual_start_id = visual_start_id
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.special = text_tokenizer.get_special_token_ids()
        self.bos = self.special.get("[BOS]", 1)
        self.eos = self.special.get("[EOS]", 2)
        self.img_start = self.special.get("[IMG_START]", 0)
        self.img_end = self.special.get("[IMG_END]", 0)

    def __iter__(self):
        for item in self.laion:
            img = item["image"]  # (3, 256, 256)
            caption = item["caption"][: self.max_caption_len * 4]
            cap_ids = self.text_tok.encode(caption, add_special_tokens=False)[: self.max_caption_len]
            img_batch = img.unsqueeze(0)
            with torch.no_grad():
                vq_ids = self.vq.encode_to_ids(img_batch)
            vq_flat = (vq_ids.flatten() + self.visual_start_id).tolist()
            ids = [self.bos] + cap_ids + [self.img_start] + vq_flat + [self.img_end] + [self.eos]
            ids = ids[: self.max_seq_len]
            yield {"input_ids": ids, "labels": ids}


class UnifiedMultimodalDataset(IterableDataset):
    """Mixes MixedModalDataset (explicit + text) with optional LAION stream. laion_prob = fraction of steps from LAION."""

    def __init__(
        self,
        mixed_modal_ds: MixedModalDataset,
        laion_tar_dir: str | Path | None,
        text_tokenizer,
        vq_model: torch.nn.Module,
        visual_start_id: int,
        laion_prob: float = 0.0,
        max_seq_len: int = 512,
        max_caption_len: int = 200,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.mixed = mixed_modal_ds
        self.laion_tar_dir = Path(laion_tar_dir) if laion_tar_dir else None
        self.laion_prob = laion_prob
        self.rank = rank
        self.world_size = world_size
        self._laion_ds = None
        if self.laion_tar_dir and self.laion_tar_dir.exists() and _laion_tar_paths(self.laion_tar_dir):
            self._laion_ds = LAIONMixedModalDataset(
                self.laion_tar_dir, text_tokenizer, vq_model, visual_start_id,
                max_seq_len=max_seq_len, max_caption_len=max_caption_len,
            )

    def __iter__(self):
        import itertools
        it_mixed = itertools.cycle(range(len(self.mixed)))
        it_laion = iter(self._laion_ds) if self._laion_ds else None
        idx = 0
        while True:
            if it_laion is not None and random.random() < self.laion_prob:
                try:
                    sample = next(it_laion)
                except StopIteration:
                    it_laion = iter(self._laion_ds)
                    sample = next(it_laion)
            else:
                i = next(it_mixed)
                sample = self.mixed[i]
            if idx % self.world_size == self.rank:
                yield sample
            idx += 1
