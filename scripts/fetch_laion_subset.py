"""
Fetch LAION metadata (parquet) and filter to a subset. Output: data/laion/filtered.parquet.
Use for dev (small) or full 2B (all 128 parts). Prior data sources unchanged; LAION is additive.

Usage:
  Dev (Kaggle):  python scripts/fetch_laion_subset.py --num-parts 3 --max-samples 50000
  Full 2B:       python scripts/fetch_laion_subset.py --subset laion2B-en --num-parts 128
"""
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# LAION2B-en parquet part filenames (hash from HF repo)
LAION2B_EN_PATTERN = "part-{i:05d}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet"
LAION2B_MULTI_PATTERN = "part-{i:05d}-fc82da14-99c9-4ff6-ab6a-ac853ac82819-c000.snappy.parquet"
SUBSETS = {
    "laion2B-en": ("laion/laion2B-en", LAION2B_EN_PATTERN),
    "laion2B-multi": ("laion/laion2B-multi", LAION2B_MULTI_PATTERN),
}


def main():
    p = argparse.ArgumentParser(description="Fetch and filter LAION metadata")
    p.add_argument("--subset", default="laion2B-en", choices=list(SUBSETS.keys()))
    p.add_argument("--num-parts", type=int, default=3, help="Number of parquet parts (1-128). Use 128 for full 2B.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples (dev mode). Omit for full.")
    p.add_argument("--min-size", type=int, default=256, help="Min width/height if columns exist")
    p.add_argument("--out-dir", default="data/laion")
    p.add_argument("--metadata-dir", default=None, help="Subdir for parquet; default out-dir/metadata")
    args = p.parse_args()

    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError:
        print("Install: pip install pandas pyarrow")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface-hub")
        sys.exit(1)

    repo_id, pattern = SUBSETS[args.subset]
    out_dir = Path(args.out_dir)
    meta_dir = Path(args.metadata_dir or str(out_dir / "metadata"))
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for i in range(args.num_parts):
        filename = pattern.format(i=i)
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=meta_dir)
        except Exception as e:
            print(f"Warning: could not download {filename}: {e}")
            continue
        try:
            df = pd.read_parquet(path)
            if "URL" not in df.columns and "url" in df.columns:
                df = df.rename(columns={"url": "URL"})
            if "TEXT" not in df.columns and "caption" in df.columns:
                df = df.rename(columns={"caption": "TEXT"})
            if "TEXT" not in df.columns and "text" in df.columns:
                df = df.rename(columns={"text": "TEXT"})
            dfs.append(df)
        except Exception as e:
            print(f"Warning: read {path}: {e}")
    if not dfs:
        print("No parquet data loaded.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    if "width" in combined.columns and "height" in combined.columns and args.min_size > 0:
        combined = combined[
            (combined["width"].astype(int) >= args.min_size) & (combined["height"].astype(int) >= args.min_size)
        ]
    combined = combined.dropna(subset=["URL", "TEXT"])
    combined = combined[["URL", "TEXT"]].copy()
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.max_samples is not None and len(combined) > args.max_samples:
        combined = combined.head(args.max_samples)
    out_path = out_dir / "filtered.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}")


if __name__ == "__main__":
    main()
