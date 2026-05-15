"""Regenerate plots from existing raw.json / results.json without re-running episodes.

Used to drop methods (e.g. our custom `load_balance` heuristic) so the
comparison only contains paper-aligned baselines.

Usage:
    # Regenerate single-topology evaluation plots
    python -m experiments.replot eval results/eval/nsfnet_to_geant2_with_fluid \\
        --exclude load_balance --title "NSFNet ckpt -> GEANT2"

    # Regenerate size-sweep plots
    python -m experiments.replot sweep results/eval/size_sweep_with_fluid \\
        --exclude load_balance --kind ba
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "agent": "#4c78a8", "shortest": "#f58518", "random": "#54a24b",
    "load_balance": "#e45756", "fluid": "#9467bd",
}
LABELS = {
    "agent": "DRL+GNN", "fluid": "Theoretical Fluid",
    "load_balance": "load_balance", "shortest": "shortest", "random": "random",
}


def _plot_boxplot(methods: dict[str, list[float]], order: list[str], out: Path, title: str):
    data = [methods[k] for k in order]
    plt.figure(figsize=(8, 4.5))
    bp = plt.boxplot(data, tick_labels=[LABELS[k] for k in order],
                     showmeans=True, patch_artist=True)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[k]); patch.set_alpha(0.6)
    plt.ylabel("bandwidth allocated per episode")
    plt.title(title); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def _plot_cdf(methods: dict[str, list[float]], order: list[str], out: Path, title: str):
    plt.figure(figsize=(8, 4.5))
    for name in order:
        x = np.sort(methods[name])
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, lw=2, label=LABELS[name], color=PALETTE[name])
    plt.xlabel("bandwidth allocated per episode")
    plt.ylabel("CDF")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


def replot_eval(run_dir: Path, exclude: list[str], title: str | None):
    data = json.loads((run_dir / "results.json").read_text())
    desired = ["agent", "fluid", "shortest", "random"]
    order = [m for m in desired if m in data and m not in exclude]
    title = title or run_dir.name
    _plot_boxplot(data, order, run_dir / "boxplot_clean.png", title)
    _plot_cdf(data, order, run_dir / "cdf_clean.png", title)
    print(f"wrote {run_dir/'boxplot_clean.png'}")
    print(f"wrote {run_dir/'cdf_clean.png'}")


def replot_sweep(run_dir: Path, exclude: list[str], kind: str):
    raw = json.loads((run_dir / "raw.json").read_text())
    # raw is keyed by str(size) -> str(topo_seed) -> method -> list[float]
    by_size: dict[int, dict[int, dict[str, list[float]]]] = {
        int(s): {int(t): m for t, m in topos.items()} for s, topos in raw.items()
    }
    sizes = sorted(by_size.keys())
    methods_full = ["agent", "fluid", "shortest", "random"]
    order = [m for m in methods_full if m not in exclude]

    # absolute
    plt.figure(figsize=(8, 4.5))
    for m in order:
        means = [np.mean([np.mean(by_size[s][t][m]) for t in by_size[s]]) for s in sizes]
        stds  = [np.std ([np.mean(by_size[s][t][m]) for t in by_size[s]]) for s in sizes]
        plt.errorbar(sizes, means, yerr=stds, marker="o", lw=2, capsize=4,
                     label=LABELS[m], color=PALETTE[m])
    plt.xlabel("topology size (nodes)")
    plt.ylabel("mean bandwidth allocated per episode")
    plt.title(f"Zero-shot generalization to random {kind.upper()} topologies")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir / "absolute_clean.png", dpi=150); plt.close()
    print(f"wrote {run_dir/'absolute_clean.png'}")

    # relative vs fluid (paper Fig 8a style)
    if "fluid" not in exclude:
        plt.figure(figsize=(8, 4.5))
        plt.axhline(1.0, color=PALETTE["fluid"], ls="--", lw=2,
                    label="Theoretical Fluid (=1.0)")
        for m in [x for x in order if x != "fluid"]:
            means, stds = [], []
            for s in sizes:
                ratios = []
                for t in by_size[s]:
                    d = np.mean(by_size[s][t]["fluid"])
                    if d == 0:
                        continue
                    ratios.append(np.mean(by_size[s][t][m]) / d)
                means.append(np.mean(ratios))
                stds.append(np.std(ratios))
            marker = "o" if m == "agent" else "s"
            plt.errorbar(sizes, means, yerr=stds, marker=marker, lw=2, capsize=4,
                         color=PALETTE[m], label=LABELS[m])
        plt.xlabel("topology size (nodes)")
        plt.ylabel("score / Theoretical Fluid score")
        plt.title("Performance vs topology size (relative to Theoretical Fluid)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir / "relative_vs_fluid_clean.png", dpi=150); plt.close()
        print(f"wrote {run_dir/'relative_vs_fluid_clean.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "sweep"])
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--exclude", nargs="*", default=["load_balance"])
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--kind", type=str, default="ba")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if args.mode == "eval":
        replot_eval(rd, args.exclude, args.title)
    else:
        replot_sweep(rd, args.exclude, args.kind)


if __name__ == "__main__":
    main()
