"""Minimal OTN-style routing environment (gymnasium-compatible stub).

State: an undirected graph with per-link `available_capacity` and `betweenness`,
plus the current traffic demand `(src, dst, bw)`.

Action: pick one of `k` precomputed shortest paths between `src` and `dst`.

Reward: `bw` if the demand fits on the chosen path, else 0 and the episode ends.

This file is intentionally small — the agent/model wiring lives elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


@dataclass
class Demand:
    src: int
    dst: int
    bw: int


class RoutingEnv:
    """Gymnasium-style env. We avoid inheriting from `gym.Env` to keep deps light;
    the API (`reset`, `step`) is identical."""

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
        self.k_paths = k_paths
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self._candidate_paths = self._precompute_paths()
        self._betweenness = self._compute_link_betweenness()
        self.reset()

    # ---- topology bookkeeping ------------------------------------------------
    def _precompute_paths(self) -> dict[tuple[int, int], list[list[int]]]:
        """Cache k shortest (by hops) simple paths for every (src, dst)."""
        cache: dict[tuple[int, int], list[list[int]]] = {}
        nodes = list(self.base_graph.nodes())
        for s in nodes:
            for t in nodes:
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

    def _compute_link_betweenness(self) -> dict[tuple[int, int], float]:
        """Fraction of cached k-shortest paths that traverse each link."""
        counts: dict[tuple[int, int], int] = {tuple(sorted(e)): 0 for e in self.base_graph.edges()}
        total = 0
        for paths in self._candidate_paths.values():
            for p in paths:
                for u, v in zip(p[:-1], p[1:]):
                    counts[tuple(sorted((u, v)))] += 1
                total += 1
        return {e: c / max(1, total) for e, c in counts.items()}

    # ---- gym-style API -------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.graph = self.base_graph.copy()
        for u, v in self.graph.edges():
            self.graph[u][v]["capacity"] = self.link_capacity
            self.graph[u][v]["betweenness"] = self._betweenness[tuple(sorted((u, v)))]
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
        """Return a plain-dict observation. The agent decides how to featurize it."""
        edges = list(self.graph.edges(data=True))
        return {
            "graph": self.graph,
            "demand": self.current_demand,
            "candidate_paths": self.candidate_paths() if not self.done else [],
            "edges": edges,
        }
