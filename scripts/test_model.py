"""Quick test: load latest checkpoint and generate text from a few Sinhala prompts."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import yaml
import torch
from src.inference.loader import load_model
from src.inference.generator import generate_text
from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer


def resolve_checkpoint(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return str(p)
    checkpoints = list(p.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    def step(f):
        try:
            return int(f.stem.replace("checkpoint_", ""))
        except ValueError:
            return 0
    return str(max(checkpoints, key=step))


def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    checkpoint_path = resolve_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        print(f"No checkpoint found in {checkpoint_dir}. Train first or pass path to a .pt file.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model_cfg = {}
    config_file = root / "configs" / "train_500m_colab.yaml"
    if config_file.exists():
        try:
            with open(config_file) as f:
                model_cfg = (yaml.safe_load(f) or {}).get("model", {})
        except Exception:
            pass

    model = load_model(checkpoint_path, model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tok = SinhalaTokenizer()
    if not tok.load():
        print("Tokenizer not found at model_output/sinhala_tokenizer.json")
        sys.exit(1)

    prompts = [
        "සිංහල",
        "මම ",
        "අද ",
        "කතාවක් ",
    ]
    max_length = 80
    print(f"\nGenerating up to {max_length} tokens per prompt (device: {device}).\n")
    for prompt in prompts:
        out = generate_text(model, tok, prompt, max_length=max_length, temperature=0.85)
        print(f"Prompt: {prompt!r}")
        print(f"Output: {out}\n")


if __name__ == "__main__":
    main()
