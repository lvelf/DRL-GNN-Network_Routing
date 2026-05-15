"""Minimal OTN-style routing environment (gymnasium-compatible stub).

State: an undirected graph with per-link `available_capacity` and `betweenness`,
plus the current traffic demand `(src, dst, bw)`.

Action: pick one of `k` precomputed shortest paths between `src` and `dst`.

Reward: `bw` if the demand fits on the chosen path, else 0 and the episode ends.

`obs` contains both a (mutable) NetworkX graph for the heuristic baselines and
canonical-edge-ordered NumPy arrays for the GNN encoder / replay buffer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from src.utils.encoding import LineGraphCache, build_line_graph


@dataclass
class Demand:
    src: int
    dst: int
    bw: int


class RoutingEnv:
    def __init__(
        self,
        graph: nx.Graph,
        link_capacity: int = 200,
        bw_choices: tuple[int, ...] = (8, 32, 64),
        k_paths: int = 4,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        self.base_graph = graph.copy()
        self.link_capacity = link_capacity
        self.bw_choices = bw_choices
        self.max_bw = max(bw_choices)
        self.k_paths = k_paths
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.line_cache: LineGraphCache = build_line_graph(self.base_graph)
        self._candidate_paths = self._precompute_paths()
        self._betweenness_arr = self._compute_betweenness_array()
        self.reset()

    # ---- topology bookkeeping ------------------------------------------------
    def _precompute_paths(self) -> dict[tuple[int, int], list[list[int]]]:
        cache: dict[tuple[int, int], list[list[int]]] = {}
        for s in self.base_graph.nodes():
            for t in self.base_graph.nodes():
                if s == t:
                    continue
                gen = nx.shortest_simple_paths(self.base_graph, s, t)
                paths = []
                for p in gen:
                    paths.append(p)
                    if len(paths) >= self.k_paths:
                        break
                cache[(s, t)] = paths
        return cache

    def _compute_betweenness_array(self) -> np.ndarray:
        counts = np.zeros(self.line_cache.num_edges, dtype=np.float32)
        total = 0
        for paths in self._candidate_paths.values():
            for p in paths:
                for u, v in zip(p[:-1], p[1:]):
                    idx = self.line_cache.edge_to_idx[tuple(sorted((u, v)))]
                    counts[idx] += 1
                total += 1
        return counts / max(1, total)

    def _path_to_mask(self, path: list[int]) -> np.ndarray:
        mask = np.zeros(self.line_cache.num_edges, dtype=np.float32)
        for u, v in zip(path[:-1], path[1:]):
            mask[self.line_cache.edge_to_idx[tuple(sorted((u, v)))]] = 1.0
        return mask

    def _capacity_array(self) -> np.ndarray:
        caps = np.empty(self.line_cache.num_edges, dtype=np.float32)
        for i, (u, v) in enumerate(self.line_cache.edge_list):
            caps[i] = self.graph[u][v]["capacity"]
        return caps

    # ---- gym-style API -------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.graph = self.base_graph.copy()
        for u, v in self.graph.edges():
            self.graph[u][v]["capacity"] = self.link_capacity
            self.graph[u][v]["betweenness"] = float(
                self._betweenness_arr[self.line_cache.edge_to_idx[tuple(sorted((u, v)))]]
            )
        self.steps = 0
        self.done = False
        self.current_demand = self._sample_demand()
        return self._observe()

    def step(self, action_idx: int) -> tuple[dict, float, bool, dict]:
        assert not self.done, "step() after done — call reset()"
        d = self.current_demand
        paths = self._candidate_paths[(d.src, d.dst)]
        if action_idx < 0 or action_idx >= len(paths):
            raise ValueError(f"action {action_idx} out of range for {len(paths)} paths")
        path = paths[action_idx]

        fits = all(self.graph[u][v]["capacity"] >= d.bw for u, v in zip(path[:-1], path[1:]))
        if fits:
            for u, v in zip(path[:-1], path[1:]):
                self.graph[u][v]["capacity"] -= d.bw
            reward = float(d.bw)
        else:
            reward = 0.0
            self.done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        if not self.done:
            self.current_demand = self._sample_demand()

        info = {"path": path, "fit": fits, "demand": d}
        return self._observe(), reward, self.done, info

    # ---- helpers -------------------------------------------------------------
    def _sample_demand(self) -> Demand:
        nodes = list(self.graph.nodes())
        src, dst = self.rng.choice(nodes, size=2, replace=False)
        bw = int(self.rng.choice(self.bw_choices))
        return Demand(int(src), int(dst), bw)

    def candidate_paths(self) -> list[list[int]]:
        d = self.current_demand
        return self._candidate_paths[(d.src, d.dst)]

    def _observe(self) -> dict:
        if self.done:
            paths: list[list[int]] = []
            path_masks = np.zeros((0, self.line_cache.num_edges), dtype=np.float32)
        else:
            paths = self.candidate_paths()
            path_masks = np.stack([self._path_to_mask(p) for p in paths], axis=0)
        return {
            "graph": self.graph,                          # for heuristic baselines
            "demand": self.current_demand,
            "candidate_paths": paths,
            "capacities": self._capacity_array(),         # canonical edge order
            "betweenness": self._betweenness_arr,
            "path_masks": path_masks,                     # [k, num_edges]
            "link_capacity": self.link_capacity,
            "max_bw": self.max_bw,
        }
