"""Aggregate Sinhala corpus: run download, merge, dedupe, clean. Intended for Kaggle only."""
import logging
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def _allow_local():
    return os.environ.get("ALLOW_LOCAL_CORPUS", "").lower() in ("1", "true", "yes")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download", root / "scripts" / "download_sinhala_corpus.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
download_main = mod.main

logging.basicConfig(level=logging.INFO)


def main(data_dir: str = "data/sinhala", config_path: str = "configs/sinhala_corpus_sources.yaml"):
    """Run download and save aggregated corpus."""
    is_kaggle = os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    is_colab = os.path.exists("/content") or os.environ.get("COLAB_GPU") is not None
    if not is_kaggle and not is_colab and not _allow_local():
        print("ERROR: Corpus aggregation is for Kaggle only. Will not run on local device.")
        print("To override: set ALLOW_LOCAL_CORPUS=1")
        sys.exit(1)
    out = download_main(data_dir=data_dir, config_path=config_path)
    path = Path(out)
    if path.exists():
        # Stream normalize (strip, drop empty) without loading entire file into RAM (avoids OOM on Colab)
        tmp = path.with_suffix(path.suffix + ".norm.tmp")
        n = 0
        with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                s = line.strip()
                if s:
                    fout.write(s + "\n")
                    n += 1
        tmp.replace(path)
        if n == 0:
            logging.warning("Corpus is empty after download; writing placeholder line.")
            path.write_text("සිංහල පාඨය", encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sinhala")
    p.add_argument("--config", default="configs/sinhala_corpus_sources.yaml")
    args = p.parse_args()
    main(args.data_dir, args.config)
