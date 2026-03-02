"""
Fetch LAION metadata and filter to a subset. Output: data/laion/filtered.parquet.
Use for dev (small) or full 2B. Prior data sources unchanged; LAION is additive.

Usage:
  HuggingFace dataset (recommended):  python scripts/fetch_laion_subset.py --hf-dataset laion/relaion2B-en-research-safe --max-samples 50000
  Parquet download:                   python scripts/fetch_laion_subset.py --num-parts 3 --max-samples 50000
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


def fetch_via_hf_dataset(hf_name: str, max_samples: int | None, min_size: int, out_path: Path) -> None:
    """Load via datasets.load_dataset and write filtered.parquet."""
    from datasets import load_dataset
    import pandas as pd

    if max_samples is None:
        max_samples = 50000  # safe default for HF to avoid OOM
    print(f"Loading {hf_name} via HuggingFace datasets (max {max_samples})...")
    ds = load_dataset(hf_name, split="train")
    cols = ds.column_names
    url_col = "URL" if "URL" in cols else ("url" if "url" in cols else None)
    text_col = "TEXT" if "TEXT" in cols else ("caption" if "caption" in cols else ("text" if "text" in cols else None))
    if not url_col or not text_col:
        raise ValueError(f"Need URL and TEXT columns. Found: {cols}")

    rows = []
    for i in range(len(ds)):
        if max_samples and len(rows) >= max_samples:
            break
        row = ds[i]
        url = row.get(url_col) or row.get("URL") or row.get("url") or ""
        text = row.get(text_col) or row.get("TEXT") or row.get("caption") or row.get("text") or ""
        if not url or not isinstance(text, str):
            continue
        if min_size > 0 and "width" in row and "height" in row:
            try:
                w, h = int(row.get("width", 0)), int(row.get("height", 0))
                if w < min_size or h < min_size:
                    continue
            except (TypeError, ValueError):
                pass
        rows.append({"URL": str(url), "TEXT": str(text)[:5000]})
        if (i + 1) % 10000 == 0:
            print(f"  ... {len(rows)} samples so far")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Fetch and filter LAION metadata")
    p.add_argument("--subset", default=None, choices=list(SUBSETS.keys()), help="Use parquet download (ignore if --hf-dataset set)")
    p.add_argument("--hf-dataset", default=None, help="HuggingFace dataset, e.g. laion/relaion2B-en-research-safe")
    p.add_argument("--num-parts", type=int, default=3, help="Number of parquet parts (1-128). Use 128 for full 2B.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples (dev). Use 50000 for HF dataset to avoid OOM.")
    p.add_argument("--min-size", type=int, default=256, help="Min width/height if columns exist")
    p.add_argument("--out-dir", default="data/laion")
    p.add_argument("--metadata-dir", default=None, help="Subdir for parquet; default out-dir/metadata")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "filtered.parquet"

    if args.hf_dataset:
        try:
            import pandas as pd
        except ImportError:
            print("Install: pip install pandas pyarrow")
            sys.exit(1)
        fetch_via_hf_dataset(args.hf_dataset, args.max_samples, args.min_size, out_path)
        return

    try:
        import pandas as pd
    except ImportError:
        print("Install: pip install pandas pyarrow")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface-hub")
        sys.exit(1)

    repo_id, pattern = SUBSETS[args.subset or "laion2B-en"]
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
