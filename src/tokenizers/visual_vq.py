"""VQ visual tokenizer - encoder/decoder for images."""
import torch
import torch.nn as nn


class VQEncoder(nn.Module):
    """Encoder: image -> latent indices."""

    def __init__(self, in_channels=3, hidden=256, codebook_size=8192, latent_size=16):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, hidden, 4, 2, 1),
            nn.ReLU(),
        )
        self.codebook = nn.Embedding(codebook_size, hidden)
        self.codebook_size = codebook_size
        self.hidden = hidden

    def forward(self, x):
        z = self.encoder(x)
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).reshape(-1, c)
        d = (z.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).norm(dim=2)
        idx = d.argmin(dim=1)
        z_q = self.codebook(idx)
        z_q = z_q.view(b, h, w, c).permute(0, 3, 1, 2)
        z_q = z + (z_q - z).detach()
        return z_q, idx.view(b, h, w)


class VQDecoder(nn.Module):
    """Decoder: latent indices -> image."""

    def __init__(self, hidden=256, codebook_size=8192, out_channels=3):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, idx, shape=None):
        b = idx.shape[0]
        z = self.codebook(idx.flatten())
        h = w = int((idx.numel() // b) ** 0.5)
        z = z.view(b, h, w, -1).permute(0, 3, 1, 2)
        return self.decoder(z)


class VisualVQ(nn.Module):
    """Full VQ-VAE for images."""

    def __init__(self, codebook_size=8192, latent_size=16):
        super().__init__()
        self.encoder = VQEncoder(codebook_size=codebook_size, latent_size=latent_size)
        self.decoder = VQDecoder(codebook_size=codebook_size)
        self.latent_size = latent_size

    def encode_to_ids(self, x):
        _, idx = self.encoder(x)
        return idx

    def decode_from_ids(self, idx):
        return self.decoder(idx)
