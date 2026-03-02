"""
Kaggle: full pipeline then train 2B model.
Same as kaggle_full_run.py but uses configs/train_2b_colab.yaml and output_2b/.

Enable GPU (P100/T4) in Notebook Settings. 16 GB may OOM; if so, in the yaml
set max_seq_len: 256 or use a 30 GB GPU.

Usage in Kaggle notebook (after clone + cd DCLC-AI):
  !python scripts/kaggle_train_2b.py
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
        print("ERROR: 2B training only runs on Kaggle. Use a Kaggle notebook with GPU.")
        sys.exit(1)
    run("pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate requests beautifulsoup4")
    run("python scripts/fetch_explicit_sources.py")
    run("python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala")
    run("python scripts/merge_raw_data.py")
    run("python scripts/train_sinhala_tokenizer.py")
    run("python scripts/train.py --config configs/train_2b_colab.yaml --data-dir data --output-dir output_2b")
    print("\n=== Done. Checkpoints in output_2b/ ===")

if __name__ == "__main__":
    main()
