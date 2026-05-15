"""Replay buffer of immutable per-step snapshots (no shared NetworkX state)."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Snapshot:
    """Just the numerics the encoder needs — no mutable graph reference."""
    capacities: np.ndarray          # [num_edges]
    betweenness: np.ndarray         # [num_edges]
    path_masks: np.ndarray          # [k, num_edges] (empty if terminal)
    demand_bw: int
    link_capacity: int
    max_bw: int


def snapshot_from_obs(obs: dict) -> Snapshot:
    return Snapshot(
        capacities=obs["capacities"].copy(),
        betweenness=obs["betweenness"],   # constant, safe to share
        path_masks=obs["path_masks"].copy(),
        demand_bw=int(obs["demand"].bw),
        link_capacity=int(obs["link_capacity"]),
        max_bw=int(obs["max_bw"]),
    )


@dataclass
class Transition:
    state: Snapshot
    action: int
    reward: float
    next_state: Optional[Snapshot]   # None if terminal
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.buf: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)
