from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .graph_encoder import GraphEncoder

FGVariant = Literal["base", "ligand_fg", "protein_fg", "full"]
_VARIANT_DIMS = {"base": (78, 54), "ligand_fg": (98, 54), "protein_fg": (78, 74), "full": (98, 74)}


class FGGraphDTA(nn.Module):
    def __init__(
        self,
        variant: FGVariant = "full",
        embedding_dim: int = 128,
        dropout: float = 0.2,
        n_output: int = 1,
        fg_count: int = 20,
    ):
        super().__init__()
        if variant not in _VARIANT_DIMS:
            raise ValueError(variant)
        if fg_count < 0:
            raise ValueError("fg_count must be nonnegative")
        ld = 78 + (fg_count if variant in {"ligand_fg", "full"} else 0)
        pd = 54 + (fg_count if variant in {"protein_fg", "full"} else 0)
        self.variant = variant
        self.ligand_encoder = GraphEncoder(ld, embedding_dim, dropout)
        self.protein_encoder = GraphEncoder(pd, embedding_dim, dropout)
        self.fc1 = nn.Linear(embedding_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, ligand: Data | Batch, protein: Data | Batch) -> torch.Tensor:
        x = torch.cat([self.ligand_encoder(ligand), self.protein_encoder(protein)], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x)

    @classmethod
    def expected_feature_dims(cls, variant: FGVariant) -> tuple[int, int]:
        return _VARIANT_DIMS[variant]
