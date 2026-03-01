"""Aggregate Sinhala corpus: run download, merge, dedupe, clean."""
import logging
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download", root / "scripts" / "download_sinhala_corpus.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
download_main = mod.main

logging.basicConfig(level=logging.INFO)


def main(data_dir: str = "data/sinhala"):
    """Run download and save aggregated corpus."""
    out = download_main(data_dir=data_dir)
    path = Path(out)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        if not lines:
            logging.warning("Corpus is empty after download; writing placeholder line.")
            lines = ["සිංහල පාඨය"]
        path.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sinhala")
    args = p.parse_args()
    main(args.data_dir)
