"""State+action -> PyG Data encoder.

We treat each undirected link as a node in a *line graph* (two line-graph
nodes are connected if their original edges share an endpoint). Each line
node carries: [available_capacity_norm, betweenness, on_path, demand_bw_norm].
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class LineGraphCache:
    edge_list: list[tuple[int, int]]                 # canonical, sorted
    edge_to_idx: dict[tuple[int, int], int]
    line_edge_index: torch.Tensor                    # [2, E_line] long
    num_edges: int


def build_line_graph(g: nx.Graph) -> LineGraphCache:
    edge_list = sorted(tuple(sorted(e)) for e in g.edges())
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    src, dst = [], []
    for node in g.nodes():
        incident = [tuple(sorted((node, nb))) for nb in g.neighbors(node)]
        idxs = [edge_to_idx[e] for e in incident]
        for i in idxs:
            for j in idxs:
                if i != j:
                    src.append(i)
                    dst.append(j)
    if not src:
        line_edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        line_edge_index = torch.tensor([src, dst], dtype=torch.long)
    return LineGraphCache(edge_list, edge_to_idx, line_edge_index, len(edge_list))


def features_from_snapshot(
    capacities: np.ndarray,
    betweenness: np.ndarray,
    path_mask: np.ndarray,
    bw: float,
    link_capacity: int,
    max_bw: int,
) -> torch.Tensor:
    """Build the [num_edges, 4] feature matrix for a single (state, action)."""
    x = np.stack(
        [
            capacities / link_capacity,
            betweenness,
            path_mask,
            np.full_like(capacities, bw / max_bw, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return torch.from_numpy(x)


def encode(obs: dict, action_idx: int, line_edge_index: torch.Tensor) -> Data:
    """Build a single PyG Data object for (obs, action_idx). Used at acting time."""
    x = features_from_snapshot(
        capacities=obs["capacities"],
        betweenness=obs["betweenness"],
        path_mask=obs["path_masks"][action_idx],
        bw=float(obs["demand"].bw),
        link_capacity=obs["link_capacity"],
        max_bw=obs["max_bw"],
    )
    return Data(x=x, edge_index=line_edge_index)
