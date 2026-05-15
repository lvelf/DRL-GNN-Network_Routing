"""Evaluate a trained DQN+GNN agent on an arbitrary topology.

Loads a checkpoint trained on one topology (e.g. NSFNet) and runs it greedily
on the requested topology (e.g. GEANT2) against the three heuristic baselines
on an *identical* demand sequence (same seed per episode).

Outputs:
    results/<eval_tag>/
        results.json          — per-method per-episode scores
        summary.txt           — mean / std / median table
        boxplot.png           — paper Fig. 4-style distribution
        cdf.png               — CDF of bandwidth allocated

Usage:
    python -m experiments.evaluate \\
        --ckpt results/dqn_nsfnet_double_normd_20260515_051929/best.pt \\
        --topology geant2 \\
        --n-episodes 200 \\
        --tag nsfnet_to_geant2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.agents.dqn import DQNAgent
from src.baselines.heuristics import POLICIES, run_fluid_episode
from src.env.routing_env import RoutingEnv
from src.utils.topology import load_named, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_agent(env: RoutingEnv, agent_cfg: dict, ckpt_path: Path, device: str = "cuda") -> DQNAgent:
    """Construct an agent and load checkpoint weights."""
    agent = DQNAgent(
        line_cache=env.line_cache,
        hidden=agent_cfg["hidden"],
        n_layers=agent_cfg["n_layers"],
        lr=agent_cfg["lr"],
        gamma=agent_cfg["gamma"],
        buffer_size=64,                       # irrelevant for greedy eval
        batch_size=agent_cfg["batch_size"],
        target_sync=agent_cfg["target_sync"],
        eps_start=0.0, eps_end=0.0, eps_decay_steps=1,
        warmup_steps=0,
        reward_scale=agent_cfg.get("reward_scale", 1.0),
        double_dqn=agent_cfg.get("double_dqn", True),
        device=device,
        seed=0,
    )
    state = torch.load(ckpt_path, map_location=agent.device)
    agent.qnet.load_state_dict(state)
    agent.qnet.eval()
    agent.target.load_state_dict(state)
    return agent


def run_agent_episode(env: RoutingEnv, agent: DQNAgent) -> float:
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        a = agent.act(obs, greedy=True)
        obs, r, done, _ = env.step(a)
        total += r
    return total


def run_policy_episode(env: RoutingEnv, policy, rng: np.random.Generator) -> float:
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        a = policy(obs, rng)
        obs, r, done, _ = env.step(a)
        total += r
    return total


def evaluate_all(env: RoutingEnv, agent: DQNAgent, n_eps: int, seed_offset: int) -> dict:
    seeds = list(range(seed_offset, seed_offset + n_eps))
    methods: dict[str, list[float]] = {}

    print(f"  running agent  ({n_eps} eps)")
    methods["agent"] = []
    for s in seeds:
        env.reset(seed=s)
        methods["agent"].append(run_agent_episode(env, agent))

    for name, pol in POLICIES.items():
        print(f"  running {name}")
        scores = []
        for s in seeds:
            env.reset(seed=s)
            rng = np.random.default_rng(s)
            scores.append(run_policy_episode(env, pol, rng))
        methods[name] = scores

    print("  running fluid")
    methods["fluid"] = [run_fluid_episode(env, seed=s) for s in seeds]

    return methods


def summarize_results(methods: dict[str, list[float]]) -> str:
    lines = [f"{'method':<15s} {'mean':>8s} {'std':>8s} {'median':>8s} {'min':>6s} {'max':>6s}"]
    for name, scores in methods.items():
        arr = np.array(scores)
        lines.append(f"{name:<15s} {arr.mean():8.1f} {arr.std():8.1f} "
                     f"{np.median(arr):8.1f} {arr.min():6.0f} {arr.max():6.0f}")
    return "\n".join(lines)


PALETTE = {
    "agent": "#4c78a8", "shortest": "#f58518", "random": "#54a24b",
    "load_balance": "#e45756", "fluid": "#9467bd",
}
ORDER = ["agent", "fluid", "load_balance", "shortest", "random"]
LABELS = {"agent": "DRL+GNN", "fluid": "Theoretical Fluid",
          "load_balance": "load_balance", "shortest": "shortest", "random": "random"}


def plot_boxplot(methods: dict[str, list[float]], out: Path, title: str) -> None:
    data = [methods[k] for k in ORDER]
    plt.figure(figsize=(8, 4.5))
    bp = plt.boxplot(data, tick_labels=[LABELS[k] for k in ORDER],
                     showmeans=True, patch_artist=True)
    for patch, k in zip(bp["boxes"], ORDER):
        patch.set_facecolor(PALETTE[k]); patch.set_alpha(0.6)
    plt.ylabel("bandwidth allocated per episode")
    plt.title(title); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def plot_cdf(methods: dict[str, list[float]], out: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in ORDER:
        x = np.sort(methods[name])
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, lw=2, label=LABELS[name], color=PALETTE[name])
    plt.xlabel("bandwidth allocated per episode")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def load_train_config(ckpt: Path) -> dict:
    """Find config.yaml next to the checkpoint to recover the agent hyperparams."""
    cfg_path = ckpt.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml beside {ckpt}; cannot recover agent hyperparams")
    return yaml.safe_load(cfg_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="path to best.pt / last.pt")
    ap.add_argument("--topology", required=True, type=str,
                    help="evaluation topology name (nsfnet / geant2 / ...)")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=100_000,
                    help="eval seeds = [seed_offset, seed_offset + n_episodes)")
    ap.add_argument("--tag", type=str, default=None,
                    help="results subdir name (default: derived from ckpt + topology)")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    cfg = load_train_config(ckpt)
    print(f"[ckpt] {ckpt}")
    print(f"[train cfg] {cfg['experiment_name']}")

    g = load_named(args.topology)
    print(f"[eval topology] {args.topology}: {summarize(g)}")
    env = RoutingEnv(
        graph=g,
        link_capacity=cfg["env"]["link_capacity"],
        bw_choices=tuple(cfg["env"]["bw_choices"]),
        k_paths=cfg["env"]["k_paths"],
        max_steps=cfg["env"]["max_steps"],
        seed=0,
    )

    agent = build_agent(env, cfg["agent"], ckpt, device=args.device)
    print(f"[agent] device={agent.device}  params={sum(p.numel() for p in agent.qnet.parameters()):,}")

    methods = evaluate_all(env, agent, args.n_episodes, args.seed_offset)

    tag = args.tag or f"{ckpt.parent.name}__on_{args.topology}"
    out_dir = REPO_ROOT / "results" / "eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps({k: list(map(float, v)) for k, v in methods.items()}, indent=2))
    summary = summarize_results(methods)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)

    title = f"{cfg['env']['topology'].upper()} ckpt  →  {args.topology.upper()}   ({args.n_episodes} episodes)"
    plot_boxplot(methods, out_dir / "boxplot.png", title)
    plot_cdf(methods, out_dir / "cdf.png", title)
    print(f"\n[done] artifacts in {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
