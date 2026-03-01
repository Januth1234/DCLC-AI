"""Download Sinhala corpus from configured sources."""
import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/sinhala_corpus_sources.yaml") -> dict:
    """Load corpus sources config."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found, using defaults")
        return {"sources": []}
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_huggingface(source: dict) -> list[str]:
    """Fetch from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed")
        return []
    try:
        name = source.get("dataset")
        config = source.get("config")
        split = source.get("split", "train")
        col = source.get("text_column", "text")
        ds = load_dataset(name, config or None, split=split, trust_remote_code=True)
        texts = []
        for row in ds:
            t = row.get(col)
            if t and isinstance(t, str):
                texts.append(t.strip())
        return texts
    except Exception as e:
        logger.warning("HF source %s failed: %s", source.get("name"), e)
        return []


def save_to_text_file(lines: list[str], out_path: str) -> None:
    """Save corpus lines to file. Deduplicates and cleans."""
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        cleaned.append(line)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))
    logger.info("Saved %d lines to %s", len(cleaned), out_path)


def main(data_dir: str = "data/sinhala", config_path: str = "configs/sinhala_corpus_sources.yaml"):
    """Download and aggregate corpus."""
    config = load_config(config_path)
    all_lines = []
    for src in config.get("sources", []):
        if src.get("type") == "huggingface":
            lines = fetch_huggingface(src)
            all_lines.extend(lines)
            logger.info("Fetched %d from %s", len(lines), src.get("name"))
        elif src.get("type") == "http":
            logger.info("Skipping HTTP source %s (manual download)", src.get("name"))
    out = Path(data_dir) / "corpus.txt"
    save_to_text_file(all_lines, str(out))
    return str(out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sinhala")
    p.add_argument("--config", default="configs/sinhala_corpus_sources.yaml")
    args = p.parse_args()
    main(args.data_dir, args.config)
