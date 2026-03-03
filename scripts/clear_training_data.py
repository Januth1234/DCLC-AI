"""
Clear training data, model_output artifacts, and checkpoint dir so the pipeline starts fresh.
Use at notebook start:  python scripts/clear_training_data.py
Optional:  --clear-cache  to also remove HuggingFace datasets cache (~/.cache/huggingface/datasets).
"""
import argparse
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def _rmtree(p: Path) -> None:
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        print(f"Removed dir: {p}")


def _unlink(p: Path) -> None:
    if p.exists():
        p.unlink()
        print(f"Removed file: {p}")


def main(clear_cache: bool = False) -> None:
    # data/
    _unlink(root / "data" / "explicit_scraped.txt")
    _rmtree(root / "data" / "explicit_media")
    _unlink(root / "data" / "sinhala" / "corpus.txt")
    _unlink(root / "data" / "sinhala" / "corpus.txt.tmp")
    _rmtree(root / "data" / "laion")

    # model_output/
    _unlink(root / "model_output" / "sinhala_tokenizer.json")
    _unlink(root / "model_output" / "tokenizer_samples.txt")
    _unlink(root / "model_output" / "visual_vq.pt")

    # output
    _rmtree(root / "output_2b_multimodal_hard")

    if clear_cache:
        hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            print(f"Removed HF datasets cache: {hf_cache}")
        else:
            print("HF datasets cache not found (nothing to clear)")

    print("Clear done. Pipeline can run from scratch.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clear training data and outputs for a fresh run")
    p.add_argument("--clear-cache", action="store_true", help="Also remove ~/.cache/huggingface/datasets")
    args = p.parse_args()
    main(clear_cache=args.clear_cache)
