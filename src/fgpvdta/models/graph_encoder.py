from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim * 2)
        self.conv3 = GCNConv(input_dim * 2, input_dim * 4)
        self.fc1 = nn.Linear(input_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.relu(self.conv2(x, data.edge_index))
        x = self.relu(self.conv3(x, data.edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.dropout(self.fc2(x))
