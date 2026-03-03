"""
Kaggle T4 x2 (or 1 GPU): full pipeline for HARD 2B multimodal (ultra-realistic images, incl. explicit).
  - VQ: 16384 codebook, 120 epochs (high-fidelity tokenizer)
  - LM: 100k steps, 80% image-caption mix, AMP + gradient checkpointing
  - Optional LAION: set USE_LAION=1 or pass --laion to add LAION as extra image+text source (prior sources unchanged).
  - Optional --clear / CLEAR_DATA=1: clear data, model_output, and checkpoints before run (fresh start).
  - Optional --high-memory / HIGH_MEMORY=1: use 112GB Sinhala corpus config (more mC4 + Wikipedia).

One command:  python scripts/kaggle_train_2b_multimodal_hard.py
With LAION:   python scripts/kaggle_train_2b_multimodal_hard.py --laion
Fresh + LAION:  python scripts/kaggle_train_2b_multimodal_hard.py --clear --laion
"""
import os
import subprocess
import sys
from pathlib import Path

def is_kaggle():
    return os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

def num_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run(cmd: str):
    print(f"\n>>> {cmd}\n")
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    # Parse flags early for clear and high-memory
    do_clear = "--clear" in sys.argv or os.environ.get("CLEAR_DATA", "").lower() in ("1", "true", "yes")
    clear_cache = "--clear-cache" in sys.argv
    high_memory = "--high-memory" in sys.argv or os.environ.get("HIGH_MEMORY", "").lower() in ("1", "true", "yes")
    multi_gpu = "--multi-gpu" in sys.argv or os.environ.get("MULTI_GPU", "").lower() in ("1", "true", "yes")
    use_laion = "--laion" in sys.argv or os.environ.get("USE_LAION", "").lower() in ("1", "true", "yes")

    if do_clear:
        clear_cmd = "python scripts/clear_training_data.py"
        if clear_cache:
            clear_cmd += " --clear-cache"
        run(clear_cmd)

    run("pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate requests beautifulsoup4")
    if use_laion:
        run("pip install -q webdataset img2dataset pyarrow pandas")
    ngpu = num_gpus()
    if ngpu >= 2 and not multi_gpu:
        multi_gpu = True
        print(f"Detected {ngpu} GPUs — using DDP (torchrun --nproc_per_node={ngpu})")
    nproc = os.environ.get("WORLD_SIZE", str(ngpu)) if multi_gpu else "1"

    run("python scripts/fetch_explicit_sources.py")

    sinhala_config = "configs/sinhala_corpus_sources_112gb.yaml" if high_memory else "configs/sinhala_corpus_sources.yaml"
    if not (ROOT / sinhala_config).exists():
        sinhala_config = "configs/sinhala_corpus_sources.yaml"
    run(f"python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala --config {sinhala_config}")
    run("python scripts/merge_raw_data.py")
    run("python scripts/train_sinhala_tokenizer.py")

    laion_tars_arg = ""
    laion_path_arg = ""
    if use_laion:
        laion_max = int(os.environ.get("LAION_MAX_SAMPLES", "50000"))
        run(f"python scripts/fetch_laion_subset.py --hf-dataset laion/relaion2B-en-research-safe --max-samples {laion_max} --out-dir data/laion")
        run("python scripts/download_laion_images.py --input data/laion/filtered.parquet --output data/laion/webdataset --shard-size 1000")
        laion_tars_arg = " --laion-tars data/laion/webdataset --laion-prob 0.5 --max-steps 5000"
        laion_path_arg = " --laion-path data/laion/webdataset --laion-prob 0.5"

    # VQ: train hard — 16384 codebook, 120 epochs (optionally with LAION)
    vq_cmd = (
        "python scripts/train_visual_vq.py"
        " --epochs 120 --codebook-size 16384 --batch-size 8"
        " --lr 1e-4 --commitment 0.25"
        " --captions data/explicit_media/captions.json"
        " --image-root data/explicit_media --out model_output/visual_vq.pt"
        + laion_tars_arg
    )
    if multi_gpu and int(nproc) >= 2:
        vq_cmd = (
            f"torchrun --nproc_per_node={nproc} scripts/train_visual_vq.py"
            " --epochs 120 --codebook-size 16384 --batch-size 8 --lr 1e-4"
            " --out model_output/visual_vq.pt"
            " --captions data/explicit_media/captions.json --image-root data/explicit_media"
            + laion_tars_arg
        )
    run(vq_cmd)

    # 2B multimodal LM: hard config (100k steps, 80% image); optional LAION
    train_cmd = (
        "python scripts/train_multimodal.py"
        " --config configs/train_2b_multimodal_hard.yaml"
        " --data-dir data --output-dir output_2b_multimodal_hard"
        " --vq-checkpoint model_output/visual_vq.pt"
        + laion_path_arg
    )
    if multi_gpu and int(nproc) >= 2:
        train_cmd = (
            f"torchrun --nproc_per_node={nproc} scripts/train_multimodal.py"
            " --config configs/train_2b_multimodal_hard.yaml"
            " --data-dir data --output-dir output_2b_multimodal_hard"
            " --vq-checkpoint model_output/visual_vq.pt"
            + laion_path_arg
        )
    run(train_cmd)

    print("\n=== Done. Checkpoints in output_2b_multimodal_hard/ ===")

if __name__ == "__main__":
    main()
