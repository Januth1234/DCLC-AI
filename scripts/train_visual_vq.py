"""
Train VQ-VAE on images (from captions.json). Required before multimodal 2B training.
Saves model_output/visual_vq.pt. Images are resized to 256x256.
Supports DDP for multi-GPU (e.g. 4x T4): torchrun --nproc_per_node=4 scripts/train_visual_vq.py --epochs 120 --codebook-size 16384
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

try:
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize
except ImportError:
    print("Install torchvision: pip install torchvision")
    sys.exit(1)

from src.tokenizers.visual_vq import VisualVQ
from src.training.dataset import CombinedImageIterableDataset, LAIONImageDataset, _parse_captions_json


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


class ImageListDataset(Dataset):
    def __init__(self, json_path: str, image_root: str, size: int = 256):
        import json
        self.image_root = Path(image_root)
        with open(json_path) as f:
            raw = json.load(f)
        self.samples = _parse_captions_json(raw)
        if not isinstance(self.samples, list):
            self.samples = list(self.samples) if self.samples else []
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        path = self.image_root / s.get("image_path", s.get("path", ""))
        if not path.exists():
            return None
        try:
            img = read_image(str(path)).float() / 255.0
            if img.dim() == 2:
                img = img.unsqueeze(0).expand(3, -1, -1)
            elif img.size(0) == 1:
                img = img.expand(3, -1, -1)
            img = resize(img.unsqueeze(0), [self.size, self.size]).squeeze(0)
            return img
        except Exception:
            return None


def main():
    rank, world_size, local_rank = setup_ddp()
    is_ddp = world_size > 1

    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/explicit_media/captions.json")
    p.add_argument("--image-root", default="data/explicit_media")
    p.add_argument("--out", default="model_output/visual_vq.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--commitment", type=float, default=0.25)
    p.add_argument("--laion-tars", default=None, help="Path to LAION webdataset tars; mixes with explicit images")
    p.add_argument("--laion-prob", type=float, default=0.5, help="Fraction of batches from LAION when --laion-tars set")
    p.add_argument("--max-steps", type=int, default=None, help="Max batches when using LAION (infinite stream); else use epochs")
    args = p.parse_args()

    captions_path = Path(args.captions)
    if not captions_path.exists():
        if rank == 0:
            print(f"Captions not found: {captions_path}. Run fetch_explicit_sources + merge first.")
        sys.exit(1)

    ds = ImageListDataset(args.captions, args.image_root)
    valid = [i for i in range(len(ds)) if ds[i] is not None]
    if len(valid) < 4 and not args.laion_tars:
        if rank == 0:
            print("Too few valid images. Need at least a few dozen for VQ training (or use --laion-tars).")
        sys.exit(1)

    class FilteredDataset(Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.parent[self.indices[i]]

    explicit_ds = FilteredDataset(ds, valid) if len(valid) >= 4 else None

    if args.laion_tars and Path(args.laion_tars).exists():
        from pathlib import Path as P
        tars = list(P(args.laion_tars).glob("*.tar"))
        if tars:
            if explicit_ds is not None and len(explicit_ds) > 0:
                ds = CombinedImageIterableDataset(
                    explicit_ds,
                    args.laion_tars,
                    laion_prob=args.laion_prob,
                    image_size=256,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                ds = LAIONImageDataset(args.laion_tars, image_size=256, rank=rank, world_size=world_size)
            if rank == 0:
                print(f"VQ training with LAION tars: {args.laion_tars} (laion_prob={args.laion_prob})")
            sampler = None
        else:
            ds = explicit_ds
            sampler = DistributedSampler(ds, shuffle=True, num_replicas=world_size, rank=rank) if is_ddp else None
    else:
        ds = explicit_ds
        sampler = DistributedSampler(ds, shuffle=True, num_replicas=world_size, rank=rank) if is_ddp else None

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    vq = VisualVQ(codebook_size=args.codebook_size, latent_size=16).to(device)
    if is_ddp:
        vq = torch.nn.parallel.DistributedDataParallel(vq, device_ids=[local_rank])
    opt = torch.optim.Adam(vq.parameters(), lr=args.lr)

    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    use_laion = args.laion_tars and Path(args.laion_tars).exists() and list(Path(args.laion_tars).glob("*.tar"))
    max_steps = args.max_steps if use_laion else None

    for epoch in range(args.epochs):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        vq.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if batch is None or (isinstance(batch, list) and None in batch):
                continue
            x = batch.to(device, non_blocking=True)
            mod = vq.module if is_ddp else vq
            z_q, idx = mod.encoder(x)
            recon = mod.decoder(idx)
            recon_loss = F.mse_loss(recon, x)
            z_q_hat = mod.encoder.codebook(idx.view(-1)).view(z_q.shape)
            commit_loss = F.mse_loss(z_q, z_q_hat.detach())
            loss = recon_loss + args.commitment * commit_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            if max_steps is not None and n_batches >= max_steps:
                break
        if n_batches and rank == 0:
            print(f"Epoch {epoch+1} loss {total_loss/n_batches:.4f}")
        if max_steps is not None:
            break
    if rank == 0:
        state = vq.module.state_dict() if is_ddp else vq.state_dict()
        torch.save(state, args.out)
        print(f"Saved VQ to {args.out}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
