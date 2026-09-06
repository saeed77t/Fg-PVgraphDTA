"""Validation helpers for reproducible preprocessed assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data


def load_torch_object(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_graph_file(path: str | Path, expected_feature_dim: int) -> Data:
    obj = load_torch_object(path)
    if not isinstance(obj, Data):
        raise TypeError(f"{path} is not a PyG Data")
    if obj.x.ndim != 2 or obj.x.shape[1] != expected_feature_dim:
        raise ValueError(f"Feature dimension mismatch in {path}")
    if obj.edge_index.ndim != 2 or obj.edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index in {path}")
    if not torch.isfinite(obj.x).all():
        raise ValueError(f"Non-finite graph features: {path}")
    return obj


def validate_historical_stacked_png(path: str | Path, blue_tolerance: int = 0) -> None:
    path = Path(path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB")
    if int(rgb[..., 2].max()) > blue_tolerance:
        raise ValueError(f"Historical stacked image has non-zero blue channel: {path}")


def latent_candidates(latent_dir: str | Path, target_id: str) -> list[Path]:
    root = Path(latent_dir)
    return [
        root / f"{target_id}.pt",
        root / f"{target_id}_latent.pt",
        root / f"{target_id}_stacked_latent.pt",
    ]


def validate_ae_latent(path: str | Path, expected_dim: int = 128) -> torch.Tensor:
    obj = load_torch_object(path)
    if isinstance(obj, dict):
        for key in ("latent", "z", "embedding", "ae"):
            if key in obj:
                obj = obj[key]
                break
    latent = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    latent = latent.float().view(-1)
    if latent.numel() != expected_dim or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid AE latent: {path}")
    return latent


def validate_interaction_assets(
    interactions_csv: str | Path,
    ligand_graph_dir: str | Path,
    protein_graph_dir: str | Path,
    ligand_feature_dim: int,
    protein_feature_dim: int,
    image_dir: str | Path | None = None,
    ae_latent_dir: str | Path | None = None,
) -> None:
    frame = pd.read_csv(interactions_csv)
    for did in sorted(frame.Drug_ID.astype(str).unique()):
        validate_graph_file(Path(ligand_graph_dir) / f"{did}_graph.pt", ligand_feature_dim)
    for tid in sorted(frame.Target_ID.astype(str).unique()):
        validate_graph_file(Path(protein_graph_dir) / f"{tid}_graph.pt", protein_feature_dim)
        if image_dir is not None:
            validate_historical_stacked_png(Path(image_dir) / f"{tid}_stacked.png")
        if ae_latent_dir is not None:
            existing = [p for p in latent_candidates(ae_latent_dir, tid) if p.is_file()]
            if not existing:
                raise FileNotFoundError(f"No AE latent for {tid}")
            validate_ae_latent(existing[0])
