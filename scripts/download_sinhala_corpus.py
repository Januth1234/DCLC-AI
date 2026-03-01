"""Download Sinhala corpus from configured sources. Intended for Kaggle only."""
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)


def _allow_local():
    """Run on local device only if explicitly allowed."""
    return os.environ.get("ALLOW_LOCAL_CORPUS", "").lower() in ("1", "true", "yes")
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
    """Fetch from HuggingFace. Supports streaming, max_rows, OSCAR language, auth."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed")
        return []
    try:
        name = source.get("dataset")
        config = source.get("config")
        language = source.get("language")
        split = source.get("split", "train")
        col = source.get("text_column", "text")
        streaming = source.get("streaming", False)
        max_rows = source.get("max_rows")
        use_auth = source.get("use_auth_token", False)
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) if use_auth else None
        if use_auth and not token:
            logger.warning("HF source %s needs HF_TOKEN (or HUGGING_FACE_HUB_TOKEN); skipping", source.get("name"))
            return []

        kwargs = {"split": split, "trust_remote_code": True}
        if streaming:
            kwargs["streaming"] = True
        if token:
            kwargs["token"] = token
        if language:
            kwargs["language"] = language  # OSCAR

        if language:
            ds = load_dataset(name, **kwargs)
        else:
            ds = load_dataset(name, config or None, **kwargs)

        texts = []
        count = 0
        for row in ds:
            if max_rows and count >= max_rows:
                break
            t = row.get(col)
            if t and isinstance(t, str) and (s := t.strip()):
                texts.append(s)
                count += 1
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
    is_kaggle = os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    if not is_kaggle and not _allow_local():
        print("ERROR: Corpus download is for Kaggle only. Will not run on local device.")
        print("To override: set ALLOW_LOCAL_CORPUS=1 (or use Kaggle notebook)")
        sys.exit(1)
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
