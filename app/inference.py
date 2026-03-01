"""App inference wrappers. Load model/tokenizer; fallback when missing."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_model = None
_tokenizer = None
_vq_enc = None
_vq_dec = None


def _load():
    global _model, _tokenizer, _vq_enc, _vq_dec
    if _model is not None:
        return True
    try:
        from app.config import load_config
        cfg = load_config()
        model_path = cfg.get("model_path", "output")
        ckpt = next(Path(model_path).glob("checkpoint_*.pt"), None) or Path(model_path) / "checkpoint_1500.pt"
        if not ckpt.exists():
            ckpt = next(Path("output").glob("checkpoint_*.pt"), None)
        if ckpt and ckpt.exists():
            from src.inference.loader import load_model
            _model = load_model(str(ckpt), {"vocab_size": 50000})
        from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer
        tok = SinhalaTokenizer()
        tok.load()
        _tokenizer = tok
    except Exception as e:
        return False
    return _model is not None and _tokenizer is not None


def generate_text(prompt: str) -> str:
    from app.filter import filter_output, content_filter_check
    from app.config import load_config
    cfg = load_config()
    allow_raw = cfg.get("allow_unfiltered", True)
    if not _load():
        return "Model not loaded. Train first or set model_path."
    from src.inference.generator import generate_text as _gen
    out = _gen(_model, _tokenizer, prompt or "සිංහල")
    return filter_output(allow_raw, out)


def generate_image(prompt: str):
    if not _load():
        return None
    from src.inference.generator import generate_image as _gen
    return _gen(_model, _tokenizer, None, prompt or "රූපය")  # vq_dec=None → placeholder


def generate_edit(image, instruction: str):
    if not _load() or image is None:
        return None
    import torch
    from PIL import Image
    import numpy as np
    if hasattr(image, "save"):
        img = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        return None
    from src.inference.generator import generate_edit as _gen
    return _gen(_model, _tokenizer, _vq_enc, _vq_dec, img, instruction or "")


def generate_caption(image):
    """Upload image → Sinhala caption/annotation."""
    if not _load():
        return "Model not loaded. Train with image-caption data first."
    from src.inference.generator import generate_caption as _gen
    import torch
    from PIL import Image
    import numpy as np
    if image is None:
        return ""
    if hasattr(image, "size"):
        arr = np.array(image)
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        return ""
    return _gen(_model, _tokenizer, _vq_enc, img)
