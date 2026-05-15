"""Visualize a named topology, optionally highlighting the k candidate paths
between a source-destination pair.

Usage:
    python -m experiments.plot_topology --topology nsfnet
    python -m experiments.plot_topology --topology nsfnet --src 0 --dst 9 --k 4
    python -m experiments.plot_topology --topology geant2 --out results/geant2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.utils.topology import load_named, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]


def k_shortest_paths(g: nx.Graph, src: int, dst: int, k: int) -> list[list[int]]:
    gen = nx.shortest_simple_paths(g, src, dst)
    out = []
    for p in gen:
        out.append(p)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", type=str, default="nsfnet")
    ap.add_argument("--src", type=int, default=None)
    ap.add_argument("--dst", type=int, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--layout", choices=["spring", "kamada_kawai", "spectral"], default="kamada_kawai")
    args = ap.parse_args()

    g = load_named(args.topology)
    print(f"[{args.topology}] {summarize(g)}")

    if args.layout == "spring":
        pos = nx.spring_layout(g, seed=42)
    elif args.layout == "spectral":
        pos = nx.spectral_layout(g)
    else:
        pos = nx.kamada_kawai_layout(g)

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, width=1.0, alpha=0.4, edge_color="gray")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=420, node_color="#cfe2ff",
                           edgecolors="#1f3a68", linewidths=1.2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=10, font_weight="bold")

    title = f"{args.topology.upper()}  ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)"

    if args.src is not None and args.dst is not None:
        paths = k_shortest_paths(g, args.src, args.dst, args.k)
        colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        for i, p in enumerate(paths):
            edges = list(zip(p[:-1], p[1:]))
            nx.draw_networkx_edges(g, pos, edgelist=edges, ax=ax,
                                   width=3.0 - 0.4 * i,
                                   edge_color=colors[i % len(colors)],
                                   alpha=0.85,
                                   label=f"path {i}: {len(p)-1} hops")
        # highlight src/dst
        nx.draw_networkx_nodes(g, pos, nodelist=[args.src, args.dst], ax=ax,
                               node_size=620, node_color="#fff3b0",
                               edgecolors="#c44b00", linewidths=2.5)
        ax.legend(loc="lower left", fontsize=9)
        title += f"   |   {len(paths)} shortest paths  {args.src} -> {args.dst}"

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    out = args.out or str(REPO_ROOT / "results" / f"topology_{args.topology}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
