"""Non-learned baselines used to bracket the DRL agent's performance."""
from __future__ import annotations

import numpy as np


def shortest_path_policy(obs: dict, rng: np.random.Generator | None = None) -> int:
    """Always pick the path with the fewest hops (index 0 after k-shortest sort)."""
    return 0


def random_policy(obs: dict, rng: np.random.Generator) -> int:
    paths = obs["candidate_paths"]
    return int(rng.integers(0, len(paths)))


def load_balancing_policy(obs: dict, rng: np.random.Generator | None = None) -> int:
    """Pick the path whose bottleneck link has the most remaining capacity."""
    graph = obs["graph"]
    paths = obs["candidate_paths"]
    best_idx, best_score = 0, -1.0
    for i, p in enumerate(paths):
        bottleneck = min(graph[u][v]["capacity"] for u, v in zip(p[:-1], p[1:]))
        if bottleneck > best_score:
            best_score = bottleneck
            best_idx = i
    return best_idx


POLICIES = {
    "shortest": shortest_path_policy,
    "random": random_policy,
    "load_balance": load_balancing_policy,
}
