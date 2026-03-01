"""Train VQ visual tokenizer (optional; prefer pretrained)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader, Dataset
from src.tokenizers.visual_vq import VisualVQ


class ImageDataset(Dataset):
    def __init__(self, root, size=256):
        self.root = Path(root)
        self.files = list(self.root.glob("**/*.jpg")) + list(self.root.glob("**/*.png"))[:10000]
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.files[i]).convert("RGB")
        import torchvision.transforms as T
        t = T.Compose([T.Resize((self.size, self.size)), T.ToTensor()])
        return t(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/images")
    p.add_argument("--output", default="model_output/visual_vq.pt")
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds = ImageDataset(args.data)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    model = VisualVQ(codebook_size=8192, latent_size=16).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        batch = batch.cuda()
        z_q, idx = model.encoder(batch)
        recon = model.decoder(idx)
        loss = torch.nn.functional.mse_loss(recon, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(step, loss.item())
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
