"""Sweep generalization across random topologies of growing size.

Replicates the paper's Fig. 8a: train on a small topology (NSFNet, 14 nodes),
evaluate zero-shot on synthetic topologies of increasing size, and report
how performance scales (typically degrades gracefully).

For each requested topology size:
  - Generate `per_size` random graphs (BA / ER / WS), seeded deterministically
  - For each, run `episodes_per_topo` eval episodes for agent + 3 baselines
    on a shared, deterministic demand sequence
  - Aggregate to per-(size, seed) mean scores

Outputs in results/eval/<tag>/:
  raw.json            per-method per-episode scores, keyed by (size, topo_seed)
  summary.csv         flat CSV: size, topo_seed, method, mean, std
  absolute_by_size.png   mean score vs topology size
  relative_by_size.png   ratio (agent / load_balance) vs size — paper Fig 8a style
  boxplot_by_size.png    distribution of per-topology means at each size

Usage:
    python -m experiments.eval_sweep \\
        --ckpt results/dqn_nsfnet_double_normd_20260515_051929/best.pt \\
        --sizes 20 30 40 60 80 100 \\
        --per-size 4 \\
        --episodes-per-topo 30 \\
        --kind ba \\
        --tag size_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.agents.dqn import DQNAgent
from src.baselines.heuristics import POLICIES, run_fluid_episode
from src.env.routing_env import RoutingEnv
from src.utils.topology import random_topology, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_agent(env, agent):
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        a = agent.act(obs, greedy=True)
        obs, r, done, _ = env.step(a)
        total += r
    return total


def _run_policy(env, pol, rng):
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        a = pol(obs, rng)
        obs, r, done, _ = env.step(a)
        total += r
    return total


def build_agent_for_env(env: RoutingEnv, agent_cfg: dict, state_dict, device: str) -> DQNAgent:
    """Construct a thin eval-only agent; load weights from a shared state_dict."""
    agent = DQNAgent(
        line_cache=env.line_cache,
        hidden=agent_cfg["hidden"],
        n_layers=agent_cfg["n_layers"],
        lr=agent_cfg["lr"],
        gamma=agent_cfg["gamma"],
        buffer_size=64,
        batch_size=agent_cfg["batch_size"],
        target_sync=agent_cfg["target_sync"],
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1,
        warmup_steps=0,
        reward_scale=agent_cfg.get("reward_scale", 1.0),
        double_dqn=agent_cfg.get("double_dqn", True),
        device=device,
        seed=0,
    )
    agent.qnet.load_state_dict(state_dict)
    agent.qnet.eval()
    return agent


def eval_one_topology(graph, agent_state, agent_cfg, env_cfg, n_eps, episode_seed_base, device):
    env = RoutingEnv(
        graph=graph,
        link_capacity=env_cfg["link_capacity"],
        bw_choices=tuple(env_cfg["bw_choices"]),
        k_paths=env_cfg["k_paths"],
        max_steps=env_cfg["max_steps"],
        seed=0,
    )
    agent = build_agent_for_env(env, agent_cfg, agent_state, device)

    seeds = list(range(episode_seed_base, episode_seed_base + n_eps))
    methods: dict[str, list[float]] = {"agent": []}
    for s in seeds:
        env.reset(seed=s)
        methods["agent"].append(_run_agent(env, agent))
    for name, pol in POLICIES.items():
        methods[name] = []
        for s in seeds:
            env.reset(seed=s)
            rng = np.random.default_rng(s)
            methods[name].append(_run_policy(env, pol, rng))
    methods["fluid"] = [run_fluid_episode(env, seed=s) for s in seeds]
    return methods


PALETTE = {"agent": "#4c78a8", "shortest": "#f58518",
           "random": "#54a24b", "load_balance": "#e45756", "fluid": "#9467bd"}
ABSOLUTE_ORDER = ["agent", "fluid", "load_balance", "shortest", "random"]


def plot_absolute(by_size: dict, out: Path, kind: str):
    sizes = sorted(by_size.keys())
    plt.figure(figsize=(8, 4.5))
    for m in ABSOLUTE_ORDER:
        means = [np.mean([np.mean(by_size[s][topo][m]) for topo in by_size[s]]) for s in sizes]
        stds  = [np.std ([np.mean(by_size[s][topo][m]) for topo in by_size[s]]) for s in sizes]
        plt.errorbar(sizes, means, yerr=stds, marker="o", lw=2, capsize=4,
                     label=m, color=PALETTE[m])
    plt.xlabel("topology size (nodes)")
    plt.ylabel("mean bandwidth allocated per episode")
    plt.title(f"Zero-shot generalization to random {kind.upper()} topologies")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def plot_relative(by_size: dict, out: Path, kind: str, denominator: str = "fluid"):
    """Paper Fig. 8a style: x-axis = topology size, y-axis = score / denominator."""
    sizes = sorted(by_size.keys())
    others = [m for m in ABSOLUTE_ORDER if m != denominator]

    plt.figure(figsize=(8, 4.5))
    plt.axhline(1.0, color=PALETTE[denominator], ls="--", lw=2,
                label=f"{denominator} (=1.0)")
    for m in others:
        means, stds = [], []
        for s in sizes:
            ratios = []
            for topo in by_size[s]:
                d = np.mean(by_size[s][topo][denominator])
                if d == 0:
                    continue
                ratios.append(np.mean(by_size[s][topo][m]) / d)
            means.append(np.mean(ratios))
            stds.append(np.std(ratios))
        marker = "o" if m == "agent" else "s"
        plt.errorbar(sizes, means, yerr=stds, marker=marker, lw=2, capsize=4,
                     color=PALETTE[m], label=m)
    plt.xlabel("topology size (nodes)")
    plt.ylabel(f"score / {denominator} score")
    plt.title(f"Performance vs topology size (relative to {denominator})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def plot_boxplot(by_size: dict, out: Path):
    sizes = sorted(by_size.keys())
    data = []
    for s in sizes:
        agent_means = [np.mean(by_size[s][topo]["agent"]) for topo in by_size[s]]
        data.append(agent_means)
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, tick_labels=[str(s) for s in sizes], patch_artist=True,
                boxprops=dict(facecolor="#4c78a8", alpha=0.6))
    plt.xlabel("topology size (nodes)")
    plt.ylabel("mean bandwidth allocated (DRL+GNN)")
    plt.title("Per-topology agent score distribution by size")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--per-size", type=int, default=4)
    ap.add_argument("--episodes-per-topo", type=int, default=30)
    ap.add_argument("--kind", choices=["ba", "er", "ws"], default="ba")
    ap.add_argument("--avg-degree", type=int, default=4,
                    help="target average degree (BA needs >=4 to give multiple paths)")
    ap.add_argument("--tag", type=str, default="size_sweep")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    cfg_path = ckpt.parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    print(f"[ckpt] {ckpt}")
    print(f"[sizes] {args.sizes}  per-size={args.per_size}  eps/topo={args.episodes_per_topo}")

    out_dir = REPO_ROOT / "results" / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    by_size: dict[int, dict[int, dict[str, list[float]]]] = {}
    rows = []

    for size in args.sizes:
        by_size[size] = {}
        for k in range(args.per_size):
            topo_seed = 10_000 * size + k
            g = random_topology(n=size, kind=args.kind, seed=topo_seed,
                                avg_degree=args.avg_degree)
            stats = summarize(g)
            print(f"\n[size={size:3d}  topo_seed={topo_seed}]  {stats}")

            methods = eval_one_topology(
                graph=g, agent_state=state, agent_cfg=cfg["agent"],
                env_cfg=cfg["env"], n_eps=args.episodes_per_topo,
                episode_seed_base=topo_seed * 7 + 1,
                device=args.device,
            )
            by_size[size][topo_seed] = methods
            for m, scores in methods.items():
                arr = np.array(scores)
                rows.append({
                    "size": size, "topo_seed": topo_seed, "method": m,
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "median": float(np.median(arr)),
                })
                print(f"   {m:<13s}  mean={arr.mean():8.1f}  std={arr.std():6.1f}")

    # ---- persist ----
    (out_dir / "raw.json").write_text(json.dumps(
        {str(s): {str(t): {m: list(map(float, v)) for m, v in methods.items()}
                  for t, methods in topos.items()}
         for s, topos in by_size.items()},
        indent=2))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "topo_seed", "method", "mean", "std", "median"])
        w.writeheader()
        w.writerows(rows)

    plot_absolute(by_size, out_dir / "absolute_by_size.png", args.kind)
    plot_relative(by_size, out_dir / "relative_vs_fluid.png", args.kind, denominator="fluid")
    plot_relative(by_size, out_dir / "relative_vs_load_balance.png", args.kind,
                  denominator="load_balance")
    plot_boxplot (by_size, out_dir / "boxplot_by_size.png")

    print(f"\n[done] artifacts in {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
