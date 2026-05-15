"""Read results/<run>/log.jsonl and produce PNGs.

Usage:
    python -m experiments.plot results/dqn_nsfnet_long_xxx
    python -m experiments.plot results/dqn_nsfnet_long_xxx --smooth 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(run_dir: Path) -> list[dict]:
    path = run_dir / "log.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No log.jsonl in {run_dir}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_train_reward(rows: list[dict], out: Path, smooth: int) -> None:
    eps = np.array([r["episode"] for r in rows])
    rew = np.array([r["train_reward"] for r in rows], dtype=float)

    plt.figure(figsize=(8, 4))
    plt.plot(eps, rew, alpha=0.3, label="per-episode")
    if smooth > 1:
        smoothed = moving_avg(rew, smooth)
        plt.plot(eps[smooth - 1:], smoothed, lw=2, label=f"moving avg (w={smooth})")
    plt.xlabel("episode")
    plt.ylabel("training reward (bw allocated)")
    plt.title("Training reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_loss(rows: list[dict], out: Path, smooth: int) -> None:
    eps, losses = [], []
    for r in rows:
        if r.get("mean_loss") is not None:
            eps.append(r["episode"])
            losses.append(r["mean_loss"])
    if not eps:
        print("  (no loss values logged; skipping loss curve)")
        return
    eps = np.array(eps); losses = np.array(losses)

    plt.figure(figsize=(8, 4))
    plt.plot(eps, losses, alpha=0.3, label="per-episode")
    if smooth > 1:
        sm = moving_avg(losses, smooth)
        plt.plot(eps[smooth - 1:], sm, lw=2, label=f"moving avg (w={smooth})")
    plt.xlabel("episode")
    plt.ylabel("mean smooth-L1 loss")
    plt.title("DQN training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def plot_eval_vs_baselines(rows: list[dict], out: Path) -> None:
    eval_pts = [r for r in rows if "eval" in r]
    if not eval_pts:
        print("  (no eval entries; skipping eval plot)")
        return
    eps = np.array([r["episode"] for r in eval_pts])
    agent = np.array([r["eval"]["agent_mean"] for r in eval_pts])
    shortest = np.array([r["eval"]["shortest_mean"] for r in eval_pts])
    rand = np.array([r["eval"]["random_mean"] for r in eval_pts])
    lb = np.array([r["eval"]["load_balance_mean"] for r in eval_pts])

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, agent, "o-", lw=2, label="DRL+GNN (agent)")
    plt.axhline(shortest.mean(), ls="--", c="tab:orange", label=f"shortest path ({shortest.mean():.0f})")
    plt.axhline(rand.mean(), ls=":",  c="tab:gray",   label=f"random ({rand.mean():.0f})")
    plt.axhline(lb.mean(), ls="--", c="tab:green",  label=f"load balance ({lb.mean():.0f})")
    plt.xlabel("episode (eval checkpoints)")
    plt.ylabel("mean bandwidth allocated over eval episodes")
    plt.title("Agent vs baselines")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def plot_epsilon(rows: list[dict], out: Path) -> None:
    eps_x = np.array([r["episode"] for r in rows])
    eps_y = np.array([r["eps"] for r in rows])
    plt.figure(figsize=(8, 3))
    plt.plot(eps_x, eps_y, lw=2)
    plt.xlabel("episode"); plt.ylabel("epsilon")
    plt.title("Exploration schedule")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="path to results/<run_name>")
    ap.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rows = load_log(run)
    print(f"loaded {len(rows)} log rows from {run}")

    plot_train_reward(rows, run / "train_reward.png", args.smooth)
    plot_loss(rows, run / "loss_curve.png", args.smooth)
    plot_eval_vs_baselines(rows, run / "eval_vs_baselines.png")
    plot_epsilon(rows, run / "epsilon.png")
    print(f"figures written to {run}")
    for p in sorted(run.glob("*.png")):
        print(f"  - {p.relative_to(run.parent.parent)}")


if __name__ == "__main__":
    main()
