"""
Kaggle: full pipeline for 2B multimodal (text + image in/out).
  1. Data + tokenizer (same as 500M)
  2. Train VQ on images (data/explicit_media)
  3. Train 2B multimodal LM

Enable GPU. Output: output_2b_multimodal/
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
        print("ERROR: Run on Kaggle with GPU.")
        sys.exit(1)
    run("pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate requests beautifulsoup4")
    run("python scripts/fetch_explicit_sources.py")
    run("python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala")
    run("python scripts/merge_raw_data.py")
    run("python scripts/train_sinhala_tokenizer.py")
    run("python scripts/train_visual_vq.py --epochs 30 --batch-size 8")
    run("python scripts/train_multimodal.py --config configs/train_2b_multimodal.yaml --data-dir data --output-dir output_2b_multimodal")
    print("\n=== Done. Checkpoints in output_2b_multimodal/ ===")

if __name__ == "__main__":
    main()
