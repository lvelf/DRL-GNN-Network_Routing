"""Topology loading and synthetic-graph generation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "topologies"


def load_edge_list(path: str | Path) -> nx.Graph:
    """Load an undirected graph from an edge-list file (lines `u v`, `#` comments)."""
    g = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = line.split()[:2]
            g.add_edge(int(u), int(v))
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return g


def load_named(name: str) -> nx.Graph:
    """Load a named built-in topology (`nsfnet`, `geant2`)."""
    name = name.lower()
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Unknown topology '{name}' (expected {path})")
    return load_edge_list(path)


def random_topology(
    n: int,
    kind: str = "ba",
    seed: Optional[int] = None,
    avg_degree: int = 3,
) -> nx.Graph:
    """Generate a synthetic topology for generalization experiments."""
    if kind == "ba":
        m = max(1, avg_degree // 2)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "er":
        p = avg_degree / max(1, n - 1)
        g = nx.erdos_renyi_graph(n, p, seed=seed)
    elif kind == "ws":
        g = nx.watts_strogatz_graph(n, k=avg_degree, p=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown random topology kind: {kind}")
    # Ensure connected — fall back by adding edges to giant component.
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        main = comps[0]
        for other in comps[1:]:
            u = next(iter(main))
            v = next(iter(other))
            g.add_edge(u, v)
            main = main | other
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def summarize(g: nx.Graph) -> dict:
    """Return basic structural stats — useful for sanity-checking loaders."""
    degs = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "avg_degree": sum(degs) / len(degs),
        "connected": nx.is_connected(g),
        "diameter": nx.diameter(g) if nx.is_connected(g) else None,
    }
