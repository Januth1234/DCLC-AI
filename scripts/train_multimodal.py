"""
Train 2B multimodal (text + image in/out). Requires:
  1. VQ trained: python scripts/train_visual_vq.py  -> model_output/visual_vq.pt
  2. Text tokenizer: model_output/sinhala_tokenizer.json
  3. Data: data/sinhala/corpus.txt, data/explicit_media/captions.json + images

Usage:
  Single GPU (Kaggle):  python scripts/train_multimodal.py --config configs/train_2b_multimodal.yaml
  4x T4 (torchrun):    torchrun --nproc_per_node=4 scripts/train_multimodal.py --config configs/train_2b_multimodal_hard.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.models.dclc_transformer import DCLCTransformer
from src.training.dataset import MixedModalDataset, UnifiedMultimodalDataset
from src.training.trainer import Trainer
from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer
from src.tokenizers.visual_vq import VisualVQ


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    is_kaggle = os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    allow_local = os.environ.get("ALLOW_LOCAL_TRAIN", "").lower() in ("1", "true", "yes")
    if not is_kaggle and not allow_local:
        print("ERROR: Multimodal training is for Kaggle/Colab. Set ALLOW_LOCAL_TRAIN=1 to run locally.")
        sys.exit(1)

    rank, world_size, local_rank = setup_ddp()
    is_ddp = world_size > 1

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_2b_multimodal.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="output_2b_multimodal")
    p.add_argument("--vq-checkpoint", default="model_output/visual_vq.pt")
    p.add_argument("--resume", default=None)
    p.add_argument("--laion-path", default=None, help="Path to LAION webdataset tars; mixed with explicit+text (additive)")
    p.add_argument("--laion-prob", type=float, default=None, help="Fraction of steps from LAION (overrides config when set)")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_dir = Path(train_cfg.get("data_dir", args.data_dir))
    codebook_size = model_cfg.get("codebook_size", 8192)
    vocab_size = model_cfg.get("vocab_size", 50000 + codebook_size)
    max_seq_len = model_cfg.get("max_seq_len", 512)

    text_tok = SinhalaTokenizer()
    if not text_tok.load():
        if rank == 0:
            print("Tokenizer not found at model_output/sinhala_tokenizer.json. Run train_sinhala_tokenizer.py first.")
        sys.exit(1)
    text_vocab_size = text_tok.get_vocab_size() or 50000
    visual_start_id = text_vocab_size

    vq = VisualVQ(codebook_size=codebook_size, latent_size=16)
    if Path(args.vq_checkpoint).exists():
        state = torch.load(args.vq_checkpoint, map_location="cpu", weights_only=True)
        vq.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"Loaded VQ from {args.vq_checkpoint}")
    else:
        if rank == 0:
            print("VQ checkpoint not found. Run: python scripts/train_visual_vq.py")
        sys.exit(1)
    vq.eval()

    corpus_path = data_dir / "sinhala" / "corpus.txt"
    captions_json = data_dir / "explicit_media" / "captions.json"
    image_root = data_dir / "explicit_media"
    laion_path = train_cfg.get("laion_path") or args.laion_path
    laion_prob = args.laion_prob if args.laion_prob is not None else train_cfg.get("laion_prob", 0.0)

    mixed_ds = MixedModalDataset(
        str(corpus_path),
        str(captions_json),
        str(image_root),
        text_tok,
        vq,
        visual_start_id,
        max_seq_len=max_seq_len,
        text_only_prob=train_cfg.get("text_only_prob", 0.4),
    )
    if len(mixed_ds) == 0 and not laion_path:
        if rank == 0:
            print("No data. Add corpus.txt and/or captions.json + images, or --laion-path.")
        sys.exit(1)

    if laion_path and Path(laion_path).exists():
        ds = UnifiedMultimodalDataset(
            mixed_ds,
            laion_path,
            text_tok,
            vq,
            visual_start_id,
            laion_prob=laion_prob,
            max_seq_len=max_seq_len,
            max_caption_len=train_cfg.get("max_caption_len", 200),
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(f"Using LAION (additive): {laion_path} laion_prob={laion_prob}")
        use_iterable = True
    else:
        ds = mixed_ds
        use_iterable = False

    pad_id = text_tok.get_special_token_ids().get("[PAD]", 0)

    def collate(batch):
        ids_list = [b["input_ids"] for b in batch]
        max_len = min(max(len(x) for x in ids_list), max_seq_len)
        padded = []
        for ids in ids_list:
            ids = ids[:max_len]
            padded.append(ids + [pad_id] * (max_len - len(ids)))
        input_ids = torch.tensor(padded, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == pad_id] = -100  # do not train on padding
        return {"input_ids": input_ids, "labels": labels}

    batch_size = train_cfg.get("batch_size", 1)
    sampler = None if use_iterable else (DistributedSampler(ds, shuffle=True, num_replicas=world_size, rank=rank) if is_ddp else None)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and not use_iterable),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    model = DCLCTransformer(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 2048),
        n_layers=model_cfg.get("n_layers", 40),
        n_heads=model_cfg.get("n_heads", 16),
        ffn_dim=model_cfg.get("ffn_dim", 8192),
        max_seq_len=max_seq_len,
    )
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_ddp:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = model.to(device)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    train_cfg["output_dir"] = str(output_dir)
    if rank == 0:
        effective = batch_size * world_size * train_cfg.get("gradient_accumulation", 1)
        print(f"Training: {world_size} GPU(s) | effective batch {effective} | Output: {output_dir} | Vocab: {vocab_size}")
    trainer = Trainer(model, None, loader, train_cfg, rank=rank)
    trainer.train(resume_from=args.resume)
    if rank == 0:
        print("Done. Checkpoints in", output_dir)
    if is_ddp:
        dist.destroy_process_group()
