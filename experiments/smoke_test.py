"""End-to-end sanity check: load topology -> build env -> run baselines.

Run with the repo root on PYTHONPATH:

    python -m experiments.smoke_test
"""
from __future__ import annotations

import numpy as np

from src.baselines.heuristics import POLICIES
from src.env.routing_env import RoutingEnv
from src.utils.topology import load_named, summarize


def run_episode(env: RoutingEnv, policy, rng) -> tuple[float, int]:
    obs = env.reset()
    total, steps = 0.0, 0
    done = False
    while not done:
        a = policy(obs, rng)
        obs, r, done, _ = env.step(a)
        total += r
        steps += 1
    return total, steps


def main():
    for name in ["nsfnet", "geant2"]:
        g = load_named(name)
        print(f"\n=== {name.upper()} ===  {summarize(g)}")
        env = RoutingEnv(g, seed=0)
        rng = np.random.default_rng(0)
        for pname, pol in POLICIES.items():
            scores, lengths = [], []
            for ep in range(20):
                env.reset(seed=ep)
                rng = np.random.default_rng(ep)
                s, t = run_episode(env, pol, rng)
                scores.append(s)
                lengths.append(t)
            print(f"  {pname:13s}  mean bw allocated = {np.mean(scores):8.1f}   "
                  f"mean steps = {np.mean(lengths):5.1f}")


if __name__ == "__main__":
    main()
