"""Entry point for DCLC app."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load_config, save_config
from app.ui import build_ui


def main():
    p = argparse.ArgumentParser(description="DCLC Gradio app")
    p.add_argument("--unhinged", action="store_true", help="Force unfiltered mode (default)")
    p.add_argument("--port", type=int, default=7860, help="Gradio server port")
    p.add_argument("--offline", action="store_true", help="Run Gradio in offline mode")
    args = p.parse_args()
    cfg = load_config()
    if args.unhinged:
        cfg["content_filter"] = cfg.get("content_filter") or {}
        cfg["content_filter"]["enabled"] = False
        cfg["allow_unfiltered"] = True
        save_config(cfg)
    demo = build_ui()
    kwargs = {"share": False, "server_port": args.port}
    if args.offline:
        kwargs["server_name"] = "127.0.0.1"
    demo.launch(**kwargs)


if __name__ == "__main__":
    main()
