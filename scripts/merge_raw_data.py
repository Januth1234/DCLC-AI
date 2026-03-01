"""
Merge raw/uncensored data into training corpus.
Point config to your data (e.g. Kaggle Dataset at /kaggle/input/mydata/).
No built-in downloads - you provide paths. config.local.yaml is gitignored.
"""
import json
import logging
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.example.yaml"):
    """Load config, prefer config.local.yaml."""
    local = root / "config.local.yaml"
    if local.exists():
        path = str(local)
    else:
        path = str(root / path)
    if Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def merge_into_corpus(corpus_path: str, config: dict) -> None:
    """Append text from explicit_path and raw_image_captions into corpus."""
    data = config.get("data", {})
    explicit_path = data.get("explicit_path", "")
    raw_captions_path = data.get("raw_image_captions_path", "")
    out = Path(corpus_path)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
    existing = out.read_text(encoding="utf-8").splitlines()
    seen = set(existing)
    added = 0

    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            if p.is_file():
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and line not in seen:
                        seen.add(line)
                        existing.append(line)
                        added += 1
            else:
                for f in p.rglob("*.txt"):
                    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if line and line not in seen:
                            seen.add(line)
                            existing.append(line)
                            added += 1
            logger.info("Added %d lines from explicit_path", added)

    if raw_captions_path:
        p = Path(raw_captions_path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("samples", raw.get("images", raw.get("captions", list(raw.values()))))
            if not isinstance(raw, list):
                raw = [raw] if raw else []
            raw_added = 0
            for item in raw:
                if isinstance(item, str):
                    cap = item
                else:
                    cap = item.get("caption", item.get("text", item.get("label", "")))
                if isinstance(cap, str) and cap.strip() and cap.strip() not in seen:
                    seen.add(cap.strip())
                    existing.append(cap.strip())
                    added += 1
                    raw_added += 1
            if raw_added:
                logger.info("Added %d captions from raw_image_captions_path", raw_added)

    if added > 0:
        out.write_text("\n".join(existing), encoding="utf-8")
        logger.info("Corpus total: %d lines", len(existing))


def main():
    cfg = load_config()
    corpus_path = cfg.get("data", {}).get("sinhala_path", "data/sinhala")
    if isinstance(corpus_path, dict):
        corpus_path = corpus_path.get("corpus", "data/sinhala/corpus.txt")
    else:
        corpus_path = str(Path(corpus_path) / "corpus.txt") if not corpus_path.endswith(".txt") else corpus_path
    merge_into_corpus(corpus_path, cfg)
    print(f"Merged. Corpus: {corpus_path}")


if __name__ == "__main__":
    main()
