"""GNN-based Q-network.

A single forward pass takes one (state, action) graph and outputs a scalar
Q-value. Batching multiple (state, action) pairs is handled by PyG's
`Batch.from_data_list`. The agent calls this k times per state to get a
Q-vector of length k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNQNet(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h
        g = global_mean_pool(h, batch)
        return self.readout(g).squeeze(-1)
