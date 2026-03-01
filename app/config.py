"""App config: load/save, content filter toggle."""
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "content_filter": {"enabled": False},
    "resolution": 256,
    "model_path": "output",
    "allow_unfiltered": True,
    "seed": None,
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "output" / "app_config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
                return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_content_filter_enabled() -> bool:
    return not load_config().get("allow_unfiltered", True)
