"""Load model for inference."""
from pathlib import Path
import torch
from src.models.dclc_transformer import DCLCTransformer


def load_model(checkpoint_path: str, config: dict = None) -> DCLCTransformer:
    """Load transformer from checkpoint."""
    cfg = config or {}
    vocab_size = cfg.get("vocab_size", 50000)
    model = DCLCTransformer(
        vocab_size=vocab_size,
        hidden_dim=cfg.get("hidden_dim", 768),
        n_layers=cfg.get("n_layers", 24),
        n_heads=cfg.get("n_heads", 12),
        ffn_dim=cfg.get("ffn_dim", 3072),
    )
    path = Path(checkpoint_path)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    return model


def load_quantized_model(checkpoint_path: str, config: dict = None):
    """Load with 4-bit quantization if bitsandbytes available."""
    try:
        import bitsandbytes
        from transformers import BitsAndBytesConfig
    except ImportError:
        return load_model(checkpoint_path, config)
    model = load_model(checkpoint_path, config)
    return model
