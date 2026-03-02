"""
Download LAION images from filtered.parquet using img2dataset. Output: webdataset tars (jpg + txt).
Use data/laion/filtered.parquet from fetch_laion_subset.py. For 2B scale use img2dataset with Spark elsewhere.

Usage:
  python scripts/download_laion_images.py --input data/laion/filtered.parquet --output data/laion/webdataset
"""
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def main():
    p = argparse.ArgumentParser(description="Download LAION images to webdataset tars")
    p.add_argument("--input", default="data/laion/filtered.parquet")
    p.add_argument("--output", default="data/laion/webdataset")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--thread-count", type=int, default=64)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}. Run fetch_laion_subset.py first.")
        sys.exit(1)

    try:
        from img2dataset import download
    except ImportError:
        print("Install: pip install img2dataset")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    download(
        processes_count=args.processes,
        thread_count=args.thread_count,
        url_list=str(in_path),
        image_size=args.image_size,
        output_folder=args.output,
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        number_sample_per_shard=args.shard_size,
        resize_mode="keep_ratio",
        resize_only_if_bigger=True,
    )
    print(f"Done. Webdataset tars in {args.output}")


if __name__ == "__main__":
    main()
