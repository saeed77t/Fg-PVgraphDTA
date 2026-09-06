from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torchvision import transforms

from fgpvdta.preprocessing.validation import (
    latent_candidates,
    load_torch_object,
    validate_ae_latent,
)

AEMissingPolicy = Literal["error", "ones"]


class PVImageTransform:
    def __init__(self, require_input_256: bool = True):
        self.require_input_256 = require_input_256
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, path: str | Path) -> torch.Tensor:
        path = Path(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.require_input_256 and image.size != (256, 256):
                raise ValueError(f"Expected 256x256 image: {path}, got {image.size}")
            return self.transform(image)


@dataclass
class InteractionItem:
    ligand: Data
    protein: Data
    target: float
    row_index: int
    drug_id: str
    target_id: str
    image: torch.Tensor | None = None
    ae_latent: torch.Tensor | None = None


@dataclass
class InteractionBatch:
    ligand: Batch
    protein: Batch
    targets: torch.Tensor
    row_indices: torch.Tensor
    drug_ids: list[str]
    target_ids: list[str]
    images: torch.Tensor | None
    ae_latents: torch.Tensor | None

    def to(self, device):
        return InteractionBatch(
            self.ligand.to(device),
            self.protein.to(device),
            self.targets.to(device),
            self.row_indices.to(device),
            self.drug_ids,
            self.target_ids,
            None if self.images is None else self.images.to(device),
            None if self.ae_latents is None else self.ae_latents.to(device),
        )


class InteractionDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        indices: Sequence[int],
        ligand_graph_dir: str | Path,
        protein_graph_dir: str | Path,
        image_dir: str | Path | None = None,
        ae_latent_dir: str | Path | None = None,
        require_vision: bool = False,
        require_ae: bool = False,
        ae_missing_policy: AEMissingPolicy = "error",
        ae_dim: int = 128,
    ):
        self.frame = interactions
        self.indices = list(map(int, indices))
        self.ligand_graph_dir = Path(ligand_graph_dir)
        self.protein_graph_dir = Path(protein_graph_dir)
        self.image_dir = None if image_dir is None else Path(image_dir)
        self.ae_latent_dir = None if ae_latent_dir is None else Path(ae_latent_dir)
        self.require_vision = require_vision
        self.require_ae = require_ae
        self.ae_missing_policy = ae_missing_policy
        self.ae_dim = ae_dim
        self.image_transform = PVImageTransform(True) if require_vision else None
        if require_vision and self.image_dir is None:
            raise ValueError("image_dir required")
        if require_ae and self.ae_latent_dir is None:
            raise ValueError("ae_latent_dir required")

    def __len__(self):
        return len(self.indices)

    def _graph(self, path):
        obj = load_torch_object(path)
        if isinstance(obj, Data):
            return obj
        if isinstance(obj, dict):
            return Data(**obj)
        raise TypeError(f"Unsupported graph object: {type(obj)}")

    def _ae(self, tid):
        for p in latent_candidates(self.ae_latent_dir, tid):
            if p.is_file():
                return validate_ae_latent(p, self.ae_dim)
        if self.ae_missing_policy == "ones":
            return torch.ones(self.ae_dim, dtype=torch.float32)
        raise FileNotFoundError(f"No AE latent for {tid}")

    def __getitem__(self, pos):
        idx = self.indices[pos]
        row = self.frame.iloc[idx]
        did, tid = str(row.Drug_ID), str(row.Target_ID)
        image = (
            self.image_transform(self.image_dir / f"{tid}_stacked.png")
            if self.require_vision
            else None
        )
        ae = self._ae(tid) if self.require_ae else None
        return InteractionItem(
            self._graph(self.ligand_graph_dir / f"{did}_graph.pt"),
            self._graph(self.protein_graph_dir / f"{tid}_graph.pt"),
            float(row.Y),
            idx,
            did,
            tid,
            image,
            ae,
        )


def collate_interactions(items: list[InteractionItem]) -> InteractionBatch:
    ligand = Batch.from_data_list([i.ligand for i in items])
    protein = Batch.from_data_list([i.protein for i in items])
    targets = torch.tensor([i.target for i in items], dtype=torch.float32).view(-1, 1)
    rows = torch.tensor([i.row_index for i in items], dtype=torch.long)
    images = None if items[0].image is None else torch.stack([i.image for i in items])
    ae = None if items[0].ae_latent is None else torch.stack([i.ae_latent for i in items])
    return InteractionBatch(
        ligand,
        protein,
        targets,
        rows,
        [i.drug_id for i in items],
        [i.target_id for i in items],
        images,
        ae,
    )


def build_interaction_loader(
    dataset: InteractionDataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0
) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_interactions,
        generator=gen,
        pin_memory=torch.cuda.is_available(),
    )
