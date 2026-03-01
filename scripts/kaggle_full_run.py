"""
Kaggle-ready: run full DCLC pipeline (corpus + tokenizer + train).
Enable GPU (P100) in Notebook Settings first.

Run after: !git clone ... && %cd DCLC-AI

Training runs ONLY on Kaggle (not locally). Run this script in a Kaggle notebook.
"""
import os
import subprocess
import sys
from pathlib import Path

def is_kaggle():
    return os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run(cmd: str):
    print(f"\n>>> {cmd}\n")
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    if not is_kaggle():
        print("ERROR: Training only runs on Kaggle. This script must run in a Kaggle notebook with GPU.")
        print("Stopping — no local training.")
        sys.exit(1)
    run("pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate requests beautifulsoup4")
    run("python scripts/fetch_explicit_sources.py")  # required: Literotica, AO3, ASSTR, StoriesOnline
    run("python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala")
    run("python scripts/merge_raw_data.py")  # compulsory: preloaded raw + scraped + optional config.local.yaml
    run("python scripts/train_sinhala_tokenizer.py")
    run("python scripts/train.py --config configs/train_500m_colab.yaml --data-dir data")
    print("\n=== Done. Checkpoints in output/ ===")

if __name__ == "__main__":
    main()
