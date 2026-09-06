from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from fgpvdta.seed import seed_everything

DecoderVariant = Literal["rgb3", "rg_zero_blue"]


class ContactMapImageDataset(Dataset):
    def __init__(self, image_dir: str | Path, target_ids=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if target_ids is not None:
            self.image_paths = [self.image_dir / f"{tid}_stacked.png" for tid in sorted(target_ids)]
            if not all(path.is_file() for path in self.image_paths):
                raise FileNotFoundError("A requested training-target image is missing")
        if not self.image_paths:
            raise RuntimeError(f"No images in {self.image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


class MaskedMSELoss(nn.Module):
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, prediction, target):
        mask = (target > self.threshold).float()
        return (((prediction - target) ** 2) * mask).sum() / (mask.sum() + 1e-8)


class ContactMapAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, decoder_variant: DecoderVariant = "rg_zero_blue"):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_variant = decoder_variant
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        out_ch = 3 if decoder_variant == "rgb3" else 2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_ch, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.fc_encode(self.flatten(self.encoder(x)))

    def decode(self, z):
        d = self.decoder(self.fc_decode(z).view(-1, 256, 16, 16))
        return (
            d
            if self.decoder_variant == "rgb3"
            else torch.cat([d, torch.zeros_like(d[:, :1])], dim=1)
        )

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(
    image_dir: str | Path,
    output_dir: str | Path,
    decoder_variant: DecoderVariant = "rg_zero_blue",
    latent_dim: int = 128,
    threshold: float = 0.05,
    epochs: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    device: str | torch.device | None = None,
    target_ids=None,
) -> Path:
    seed_everything(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if epochs < 1 or batch_size < 1:
        raise ValueError("AE epochs and batch size must be positive")
    ds = ContactMapImageDataset(image_dir, target_ids)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=gen
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ContactMapAutoencoder(latent_dim, decoder_variant).to(device)
    criterion = MaskedMSELoss(threshold)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for images, _ in tqdm(loader, desc=f"AE epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        history.append({"epoch": epoch, "loss": total / max(n, 1)})
    path = out / "autoencoder_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "decoder_variant": decoder_variant,
            "threshold": threshold,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "training_target_ids": sorted(target_ids) if target_ids is not None else None,
        },
        path,
    )
    (out / "autoencoder_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


@torch.no_grad()
def export_autoencoder_latents(
    model: ContactMapAutoencoder,
    image_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    device: str | torch.device | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = ContactMapImageDataset(image_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    saved = {}
    for images, names in loader:
        z = model.encode(images.to(device)).cpu()
        for vec, name in zip(z, names, strict=False):
            stem = Path(name).stem
            pid = stem[:-8] if stem.endswith("_stacked") else stem
            path = out / f"{pid}.pt"
            torch.save(vec.contiguous(), path)
            saved[pid] = path
    return saved
