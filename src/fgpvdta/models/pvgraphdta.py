from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .graph_encoder import GraphEncoder

ResNetVariant = Literal["resnet18", "resnet50", "resnet101", "resnet152"]


@dataclass
class PVOutput:
    prediction: torch.Tensor
    info_nce: torch.Tensor
    ligand_embedding: torch.Tensor
    protein_embedding: torch.Tensor
    image_embedding: torch.Tensor | None


def _make_resnet(variant, pretrained):
    mapping = {
        "resnet18": (models.resnet18, models.ResNet18_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
        "resnet101": (models.resnet101, models.ResNet101_Weights),
        "resnet152": (models.resnet152, models.ResNet152_Weights),
    }
    if variant not in mapping:
        raise ValueError(variant)
    fn, W = mapping[variant]
    backbone = fn(weights=W.DEFAULT if pretrained else None)
    dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, dim


class PVGraphDTA(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        use_vision: bool = True,
        use_ae_gating: bool = True,
        use_infonce: bool = True,
        resnet_variant: ResNetVariant = "resnet101",
        pretrained_resnet: bool = True,
        fine_tune_resnet: bool = False,
        temperature: float = 0.07,
        n_output: int = 1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if use_ae_gating and not use_vision:
            raise ValueError("AE gating requires vision")
        if use_infonce and not use_vision:
            raise ValueError("InfoNCE requires vision")
        self.use_vision = use_vision
        self.use_ae_gating = use_ae_gating
        self.use_infonce = use_infonce
        self.temperature = temperature
        self.ligand_encoder = GraphEncoder(78, embedding_dim, dropout)
        self.protein_encoder = GraphEncoder(54, embedding_dim, dropout)
        if use_vision:
            self.cnn_backbone, dim = _make_resnet(resnet_variant, pretrained_resnet)
            if not fine_tune_resnet:
                for p in self.cnn_backbone.parameters():
                    p.requires_grad = False
            self.cnn_projection = nn.Sequential(
                nn.Linear(dim, embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
            )
        else:
            self.cnn_backbone = self.cnn_projection = None
        if use_infonce:
            self.proj_ligand = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True)
            )
            self.proj_protein = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True)
            )
            self.proj_image = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True)
            )
        streams = 3 if use_vision else 2
        self.fc1 = nn.Linear(embedding_dim * streams, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _pair(self, a, b):
        logits = (a @ b.T) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

    def info_nce_triplet(self, ligand, protein, image):
        ligand = F.normalize(self.proj_ligand(ligand), dim=-1)
        protein = F.normalize(self.proj_protein(protein), dim=-1)
        image = F.normalize(self.proj_image(image), dim=-1)
        return self._pair(ligand, protein) + self._pair(ligand, image) + self._pair(protein, image)

    def forward(self, ligand, protein, images=None, ae_latents=None):
        ligand_embedding = self.ligand_encoder(ligand)
        p = self.protein_encoder(protein)
        streams = [ligand_embedding, p]
        image = None
        info = torch.zeros((), device=ligand_embedding.device)
        if self.use_vision:
            if images is None:
                raise ValueError("images required")
            image = self.cnn_projection(self.cnn_backbone(images))
            if self.use_ae_gating:
                if ae_latents is None or ae_latents.shape != image.shape:
                    raise ValueError("AE latent mismatch")
                image = image * ae_latents.detach()
            streams.append(image)
            if self.use_infonce and self.training:
                info = self.info_nce_triplet(ligand_embedding, p, image)
        x = torch.cat(streams, dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return PVOutput(self.out(x), info, ligand_embedding, p, image)
