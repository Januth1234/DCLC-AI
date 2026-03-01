"""Full pipeline: download -> tokenizer -> train. No prompts."""
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--skip-image-bootstrap", action="store_true")
    args = p.parse_args()
    data_dir = args.data_dir
    print("Step 1: aggregate_sinhala_corpus")
    from scripts.aggregate_sinhala_corpus import main as agg
    agg(data_dir=data_dir + "/sinhala" if not data_dir.endswith("sinhala") else data_dir)
    print("Step 2: train_sinhala_tokenizer")
    from scripts.train_sinhala_tokenizer import main as train_tok
    train_tok(corpus_path=data_dir + "/sinhala/corpus.txt" if not data_dir.endswith("sinhala") else data_dir + "/corpus.txt")
    if not args.skip_image_bootstrap:
        print("Step 3: train_visual_tokenizer (optional)")
    print("Step 4: train.py")
    from scripts.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
