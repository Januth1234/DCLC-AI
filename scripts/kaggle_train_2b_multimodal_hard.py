"""
Kaggle T4 x2 (or 1 GPU): full pipeline for HARD 2B multimodal (ultra-realistic images, incl. explicit).
  - VQ: 16384 codebook, 120 epochs (high-fidelity tokenizer)
  - LM: 100k steps, 80% image-caption mix, AMP + gradient checkpointing
  - Auto-uses all visible GPUs (e.g. T4 x2 on Kaggle).

One command on Kaggle:  python scripts/kaggle_train_2b_multimodal_hard.py
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
    # Use multi-GPU if --multi-gpu, or if we detect 2+ GPUs (e.g. Kaggle T4 x2)
    multi_gpu = "--multi-gpu" in sys.argv or os.environ.get("MULTI_GPU", "").lower() in ("1", "true", "yes")
    run("pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate requests beautifulsoup4")
    ngpu = num_gpus()
    if ngpu >= 2 and not multi_gpu:
        multi_gpu = True
        print(f"Detected {ngpu} GPUs — using DDP (torchrun --nproc_per_node={ngpu})")
    nproc = os.environ.get("WORLD_SIZE", str(ngpu)) if multi_gpu else "1"

    run("python scripts/fetch_explicit_sources.py")
    run("python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala")
    run("python scripts/merge_raw_data.py")
    run("python scripts/train_sinhala_tokenizer.py")

    # VQ: train hard — 16384 codebook, 120 epochs
    vq_cmd = (
        "python scripts/train_visual_vq.py"
        " --epochs 120 --codebook-size 16384 --batch-size 8"
        " --lr 1e-4 --commitment 0.25"
        " --captions data/explicit_media/captions.json"
        " --image-root data/explicit_media --out model_output/visual_vq.pt"
    )
    if multi_gpu and int(nproc) >= 2:
        vq_cmd = (
            f"torchrun --nproc_per_node={nproc} scripts/train_visual_vq.py"
            " --epochs 120 --codebook-size 16384 --batch-size 8 --lr 1e-4"
            " --out model_output/visual_vq.pt"
            " --captions data/explicit_media/captions.json --image-root data/explicit_media"
        )
    run(vq_cmd)

    # 2B multimodal LM: hard config (100k steps, 80% image)
    train_cmd = (
        "python scripts/train_multimodal.py"
        " --config configs/train_2b_multimodal_hard.yaml"
        " --data-dir data --output-dir output_2b_multimodal_hard"
        " --vq-checkpoint model_output/visual_vq.pt"
    )
    if multi_gpu and int(nproc) >= 2:
        train_cmd = (
            f"torchrun --nproc_per_node={nproc} scripts/train_multimodal.py"
            " --config configs/train_2b_multimodal_hard.yaml"
            " --data-dir data --output-dir output_2b_multimodal_hard"
            " --vq-checkpoint model_output/visual_vq.pt"
        )
    run(train_cmd)

    print("\n=== Done. Checkpoints in output_2b_multimodal_hard/ ===")

if __name__ == "__main__":
    main()
