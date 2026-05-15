"""Produce all six figures the paper needs, as PDFs, in results/figures/.

Run once after training and evaluation are complete:

    python -m experiments.make_paper_figures \\
        --improved-run results/dqn_nsfnet_double_normd_20260515_051929 \\
        --vanilla-run  results/dqn_nsfnet_long_20260515_050410 \\
        --geant2-eval  results/eval/nsfnet_to_geant2_with_fluid \\
        --sweep-eval   results/eval/size_sweep_with_fluid

Figures produced:
    figures/topology_nsfnet.pdf        — Problem Formulation Fig 1
    figures/training_curve.pdf         — Results Fig 2 (improved DQN learning)
    figures/vanilla_vs_improved.pdf    — Results Fig 3 (stability ablation)
    figures/geant2_boxplot.pdf         — Results Fig 4 (left)
    figures/geant2_cdf.pdf             — Results Fig 4 (right)
    figures/sweep_relative_fluid.pdf   — Results Fig 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml

from src.agents.dqn import DQNAgent
from src.baselines.heuristics import POLICIES, run_fluid_episode
from src.env.routing_env import RoutingEnv
from src.utils.topology import load_named

REPO_ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "agent": "#4c78a8", "fluid": "#9467bd",
    "shortest": "#f58518", "random": "#54a24b",
    "vanilla": "#f58518", "improved": "#4c78a8",
}
LABELS = {
    "agent": "DRL+GNN", "fluid": "Theoretical Fluid",
    "shortest": "shortest path", "random": "random",
}


# ---------------------------------------------------------------------------
# Figure 1 — NSFNet topology with the 4 candidate paths 0 -> 9
# ---------------------------------------------------------------------------
def fig_topology(out_path: Path):
    g = load_named("nsfnet")
    pos = nx.kamada_kawai_layout(g)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(g, pos, ax=ax, width=1.0, alpha=0.4, edge_color="gray")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=420, node_color="#cfe2ff",
                           edgecolors="#1f3a68", linewidths=1.2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=10, font_weight="bold")
    src, dst, k = 0, 9, 4
    paths = []
    for p in nx.shortest_simple_paths(g, src, dst):
        paths.append(p)
        if len(paths) >= k:
            break
    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, p in enumerate(paths):
        edges = list(zip(p[:-1], p[1:]))
        nx.draw_networkx_edges(g, pos, edgelist=edges, ax=ax,
                               width=3.0 - 0.4 * i, edge_color=colors[i],
                               alpha=0.85, label=f"path {i}: {len(p)-1} hops")
    nx.draw_networkx_nodes(g, pos, nodelist=[src, dst], ax=ax,
                           node_size=620, node_color="#fff3b0",
                           edgecolors="#c44b00", linewidths=2.5)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Compute reference baselines on NSFNet (incl. fluid).
# ---------------------------------------------------------------------------
def nsfnet_baselines(improved_run: Path, n_eps: int = 100) -> dict[str, float]:
    cfg = yaml.safe_load((improved_run / "config.yaml").read_text())
    g = load_named("nsfnet")
    env = RoutingEnv(graph=g,
                     link_capacity=cfg["env"]["link_capacity"],
                     bw_choices=tuple(cfg["env"]["bw_choices"]),
                     k_paths=cfg["env"]["k_paths"],
                     max_steps=cfg["env"]["max_steps"],
                     seed=0)
    seeds = list(range(50_000, 50_000 + n_eps))

    out: dict[str, list[float]] = {"shortest": [], "random": [], "fluid": []}
    for s in seeds:
        env.reset(seed=s)
        rng = np.random.default_rng(s)
        obs = env._observe()
        total, done = 0.0, False
        while not done:
            a = POLICIES["shortest"](obs, rng)
            obs, r, done, _ = env.step(a)
            total += r
        out["shortest"].append(total)

        env.reset(seed=s)
        rng = np.random.default_rng(s)
        obs = env._observe()
        total, done = 0.0, False
        while not done:
            a = POLICIES["random"](obs, rng)
            obs, r, done, _ = env.step(a)
            total += r
        out["random"].append(total)

        out["fluid"].append(run_fluid_episode(env, seed=s))

    means = {k: float(np.mean(v)) for k, v in out.items()}
    print(f"NSFNet baselines over {n_eps} eps: {means}")
    return means


# ---------------------------------------------------------------------------
# Figure 2 — improved DQN training curve vs baselines (no load_balance)
# ---------------------------------------------------------------------------
def fig_training_curve(improved_run: Path, baselines: dict[str, float], out_path: Path):
    rows = [json.loads(l) for l in (improved_run / "log.jsonl").read_text().splitlines() if l.strip()]
    eval_rows = [r for r in rows if "eval" in r]
    eps = np.array([r["episode"] for r in eval_rows])
    agent = np.array([r["eval"]["agent_mean"] for r in eval_rows])

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, agent, "o-", lw=2, color=PALETTE["agent"],
             label=LABELS["agent"])
    plt.axhline(baselines["fluid"], ls="--", lw=2, color=PALETTE["fluid"],
                label=f"{LABELS['fluid']} ({baselines['fluid']:.0f})")
    plt.axhline(baselines["shortest"], ls="--", lw=2, color=PALETTE["shortest"],
                label=f"{LABELS['shortest']} ({baselines['shortest']:.0f})")
    plt.axhline(baselines["random"], ls=":", lw=2, color=PALETTE["random"],
                label=f"{LABELS['random']} ({baselines['random']:.0f})")
    plt.xlabel("training episode")
    plt.ylabel("mean bandwidth allocated (eval)")
    plt.title("DRL+GNN training on NSFNet")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — vanilla DQN vs improved DQN (eval + loss, 2 panels)
# ---------------------------------------------------------------------------
def fig_vanilla_vs_improved(vanilla_run: Path, improved_run: Path, out_path: Path):
    def load(run):
        return [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]

    v_rows = load(vanilla_run)
    i_rows = load(improved_run)

    v_eval = [r for r in v_rows if "eval" in r]
    i_eval = [r for r in i_rows if "eval" in r]
    v_eps  = np.array([r["episode"]                  for r in v_eval])
    v_agent = np.array([r["eval"]["agent_mean"]      for r in v_eval])
    i_eps  = np.array([r["episode"]                  for r in i_eval])
    i_agent = np.array([r["eval"]["agent_mean"]      for r in i_eval])

    v_loss_eps    = np.array([r["episode"]   for r in v_rows if r.get("mean_loss") is not None])
    v_loss_vals   = np.array([r["mean_loss"] for r in v_rows if r.get("mean_loss") is not None])
    i_loss_eps    = np.array([r["episode"]   for r in i_rows if r.get("mean_loss") is not None])
    i_loss_vals   = np.array([r["mean_loss"] for r in i_rows if r.get("mean_loss") is not None])

    def moving_avg(x, w=20):
        if len(x) < w: return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- left: eval reward ----
    ax = axes[0]
    ax.plot(v_eps, v_agent, "s-", lw=2, color=PALETTE["vanilla"], label="vanilla DQN")
    ax.plot(i_eps, i_agent, "o-", lw=2, color=PALETTE["improved"], label="reward-norm. + Double DQN")
    ax.set_xlabel("training episode")
    ax.set_ylabel("mean bandwidth allocated (eval)")
    ax.set_title("(a) Evaluation reward")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- right: loss curves with TWO y-axes (very different scales) ----
    ax_v = axes[1]
    ax_i = ax_v.twinx()
    sm = 20
    ax_v.plot(v_loss_eps[sm-1:], moving_avg(v_loss_vals, sm),
              lw=2, color=PALETTE["vanilla"], label="vanilla DQN")
    ax_i.plot(i_loss_eps[sm-1:], moving_avg(i_loss_vals, sm),
              lw=2, color=PALETTE["improved"], label="reward-norm. + Double DQN")
    ax_v.set_xlabel("training episode")
    ax_v.set_ylabel("vanilla DQN loss (raw reward scale)", color=PALETTE["vanilla"])
    ax_i.set_ylabel("improved DQN loss (normalized scale)", color=PALETTE["improved"])
    ax_v.tick_params(axis="y", colors=PALETTE["vanilla"])
    ax_i.tick_params(axis="y", colors=PALETTE["improved"])
    ax_v.set_title("(b) Training loss (moving avg, w=20)")
    ax_v.grid(alpha=0.3)
    # combined legend
    lines_v, lbls_v = ax_v.get_legend_handles_labels()
    lines_i, lbls_i = ax_i.get_legend_handles_labels()
    ax_v.legend(lines_v + lines_i, lbls_v + lbls_i, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figures 4a, 4b — GEANT2 boxplot + CDF (re-render from results.json as PDF)
# ---------------------------------------------------------------------------
def _eval_data(run: Path) -> dict[str, list[float]]:
    return json.loads((run / "results.json").read_text())


def fig_geant2_boxplot(eval_run: Path, out_path: Path):
    data = _eval_data(eval_run)
    order = ["agent", "fluid", "shortest", "random"]
    data_arrs = [data[k] for k in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data_arrs,
                    tick_labels=[LABELS[k] for k in order],
                    showmeans=True, patch_artist=True)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[k]); patch.set_alpha(0.6)
    ax.set_ylabel("bandwidth allocated per episode")
    ax.set_title("NSFNet ckpt $\\rightarrow$ GEANT2 (200 episodes)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def fig_geant2_cdf(eval_run: Path, out_path: Path):
    data = _eval_data(eval_run)
    order = ["agent", "fluid", "shortest", "random"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in order:
        x = np.sort(data[k])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=2, label=LABELS[k], color=PALETTE[k])
    ax.set_xlabel("bandwidth allocated per episode")
    ax.set_ylabel("CDF")
    ax.set_title("NSFNet ckpt $\\rightarrow$ GEANT2 (200 episodes)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — sweep relative-to-fluid plot (PDF)
# ---------------------------------------------------------------------------
def fig_sweep_relative_fluid(sweep_run: Path, out_path: Path):
    raw = json.loads((sweep_run / "raw.json").read_text())
    by_size = {int(s): {int(t): m for t, m in topos.items()} for s, topos in raw.items()}
    sizes = sorted(by_size.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1.0, color=PALETTE["fluid"], ls="--", lw=2,
               label=f"{LABELS['fluid']} (=1.0)")
    for m in ["agent", "shortest", "random"]:
        means, stds = [], []
        for s in sizes:
            ratios = []
            for t in by_size[s]:
                f = np.mean(by_size[s][t]["fluid"])
                if f == 0: continue
                ratios.append(np.mean(by_size[s][t][m]) / f)
            means.append(np.mean(ratios)); stds.append(np.std(ratios))
        marker = "o" if m == "agent" else "s"
        ax.errorbar(sizes, means, yerr=stds, marker=marker, lw=2, capsize=4,
                    color=PALETTE[m], label=LABELS[m])
    ax.set_xlabel("topology size (nodes)")
    ax.set_ylabel(f"score / {LABELS['fluid']} score")
    ax.set_title(f"Performance vs topology size (relative to {LABELS['fluid']})")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--improved-run", required=True, type=str)
    ap.add_argument("--vanilla-run",  required=True, type=str)
    ap.add_argument("--geant2-eval",  required=True, type=str)
    ap.add_argument("--sweep-eval",   required=True, type=str)
    ap.add_argument("--out-dir", type=str, default="results/figures")
    ap.add_argument("--baseline-eps", type=int, default=100)
    args = ap.parse_args()

    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig_topology(out / "topology_nsfnet.pdf")

    baselines = nsfnet_baselines(Path(args.improved_run), n_eps=args.baseline_eps)

    fig_training_curve(Path(args.improved_run), baselines,
                       out / "training_curve.pdf")
    fig_vanilla_vs_improved(Path(args.vanilla_run), Path(args.improved_run),
                            out / "vanilla_vs_improved.pdf")
    fig_geant2_boxplot(Path(args.geant2_eval), out / "geant2_boxplot.pdf")
    fig_geant2_cdf    (Path(args.geant2_eval), out / "geant2_cdf.pdf")
    fig_sweep_relative_fluid(Path(args.sweep_eval), out / "sweep_relative_fluid.pdf")

    print(f"\n[done] all PDFs in {out.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
