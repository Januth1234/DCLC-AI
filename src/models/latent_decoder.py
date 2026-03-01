"""Decode latent grid to image. Loads pretrained VQ decoder."""
import torch
import torch.nn as nn
from pathlib import Path
from src.tokenizers.visual_vq import VQDecoder, VisualVQ


def load_pretrained_vq(ckpt_path: str = None, codebook_size: int = 8192):
    """Load VQ model from checkpoint or return new."""
    model = VisualVQ(codebook_size=codebook_size, latent_size=16)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    return model


def tokens_to_image(
    token_ids: torch.Tensor,
    decoder: nn.Module,
    size: int = 256,
) -> torch.Tensor:
    """Decode latent token grid to image. size 128 or 256."""
    if decoder is None:
        raise RuntimeError("Decoder not set")
    if hasattr(decoder, "decode_from_ids"):
        img = decoder.decode_from_ids(token_ids)
    else:
        img = decoder(token_ids)
    if size == 128 and img.shape[-1] == 256:
        img = torch.nn.functional.interpolate(img, size=(128, 128), mode="bilinear")
    return img
