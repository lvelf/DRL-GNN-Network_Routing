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


def run_fluid_episode(env, seed: int | None = None) -> float:
    """Theoretical-fluid reference (Almasan et al. 2022, Sec. V-B).

    For each demand, split the requested bw across the k candidate paths
    in proportion to their bottleneck available capacity. The split is
    non-realizable (ODU demands can't actually be fractioned) but gives
    a fast upper-reference for what an idealized splitting policy could
    achieve.

    Allocation succeeds iff sum of bottlenecks across candidate paths
    >= demand bw (then a feasible proportional split exists). Otherwise
    the episode ends, mirroring the env's failure condition.
    """
    obs = env.reset(seed=seed)
    total = 0.0
    while not env.done:
        d = env.current_demand
        paths = env.candidate_paths()

        bottlenecks = []
        for p in paths:
            b = min(env.graph[u][v]["capacity"] for u, v in zip(p[:-1], p[1:]))
            bottlenecks.append(b)
        total_b = sum(bottlenecks)

        if total_b < d.bw:
            env.done = True
            break

        shares = [d.bw * (b / total_b) for b in bottlenecks]
        for p, s in zip(paths, shares):
            for u, v in zip(p[:-1], p[1:]):
                env.graph[u][v]["capacity"] -= s
        total += d.bw

        env.steps += 1
        if env.steps >= env.max_steps:
            env.done = True
        else:
            env.current_demand = env._sample_demand()
    return total
