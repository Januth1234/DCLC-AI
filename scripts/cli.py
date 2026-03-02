"""CLI entry point for DCLC (generate-text, generate-image, generate-edit, train, app)."""
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def cmd_app(args):
    from app.main import main
    sys.argv = ["app/main.py", "--port", str(getattr(args, "port", 7860))]
    if getattr(args, "unhinged", False):
        sys.argv.append("--unhinged")
    if getattr(args, "offline", False):
        sys.argv.append("--offline")
    main()


def cmd_train(args):
    from scripts.train import main as train_main
    sys.argv = ["train.py", "--config", args.config, "--data-dir", args.data_dir or "data"]
    if args.resume:
        sys.argv += ["--resume", args.resume]
    train_main()


def _resolve_checkpoint(path: str) -> str | None:
    """Return path to a .pt file: if path is a dir, pick latest checkpoint_*.pt."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return str(p)
    # directory: find latest checkpoint_*.pt
    checkpoints = list(p.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    def step(f):
        try:
            return int(f.stem.replace("checkpoint_", ""))
        except ValueError:
            return 0
    latest = max(checkpoints, key=step)
    return str(latest)


def cmd_generate_text(args):
    from src.inference.loader import load_model
    from src.inference.generator import generate_text
    from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer
    from app.config import load_config
    import yaml
    load_config()
    cfg = load_config()
    raw_path = (args.model_path or cfg.get("model_path") or "output")
    checkpoint_path = _resolve_checkpoint(raw_path)
    tok = SinhalaTokenizer()
    tok.load()
    model_cfg = {}
    config_file = root / "configs" / "train_500m_colab.yaml"
    if config_file.exists():
        try:
            with open(config_file) as f:
                model_cfg = (yaml.safe_load(f) or {}).get("model", {})
        except Exception:
            pass
    model = load_model(checkpoint_path, model_cfg) if checkpoint_path else None
    if model is None:
        print("No checkpoint found. Train first or set model_path.", file=sys.stderr)
        return 1
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    out = generate_text(model, tok, args.prompt, max_length=args.max_length or 256)
    print(out)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    return 0


def main():
    p = argparse.ArgumentParser(prog="dclc", description="DCLC CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    # app
    app_p = sub.add_parser("app", help="Launch Gradio app")
    app_p.add_argument("--port", type=int, default=7860)
    app_p.add_argument("--unhinged", action="store_true")
    app_p.add_argument("--offline", action="store_true")
    app_p.set_defaults(func=cmd_app)
    # train
    train_p = sub.add_parser("train", help="Train model")
    train_p.add_argument("--config", default="configs/train_500m_colab.yaml")
    train_p.add_argument("--data-dir", default="data")
    train_p.add_argument("--resume", default=None)
    train_p.set_defaults(func=cmd_train)
    # generate-text
    gt_p = sub.add_parser("generate-text", help="Generate text")
    gt_p.add_argument("prompt", nargs="?", default="")
    gt_p.add_argument("--model-path", default=None)
    gt_p.add_argument("--max-length", type=int, default=256)
    gt_p.add_argument("--output", "-o", default=None)
    gt_p.set_defaults(func=cmd_generate_text)
    # generate-image / generate-edit: stubs
    gi_p = sub.add_parser("generate-image", help="Generate image (requires model + VQ)")
    gi_p.add_argument("prompt", nargs="?", default="")
    gi_p.add_argument("--output", "-o", default=None)
    gi_p.set_defaults(func=lambda a: print("Use Gradio app for image generation.") or 0)
    ge_p = sub.add_parser("generate-edit", help="Edit image (requires model + VQ)")
    ge_p.add_argument("--image", required=False)
    ge_p.add_argument("--instruction", default="")
    ge_p.add_argument("--output", "-o", default=None)
    ge_p.set_defaults(func=lambda a: print("Use Gradio app for image edit.") or 0)

    args = p.parse_args()
    f = getattr(args, "func", None)
    if f is None:
        p.print_help()
        return 0
    return f(args) or 0


if __name__ == "__main__":
    sys.exit(main())
