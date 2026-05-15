"""DQN agent with a GNN Q-network.

For each environment step:
  - Build k=|candidate_paths| PyG `Data` objects (one per action).
  - Batch them, run the Q-net once -> Q-vector of length k.
  - ε-greedy over the Q-vector.

For each learn step:
  - Sample a batch of transitions.
  - For each transition build the (state, chosen_action) graph; batch all.
  - For each non-terminal next_state build k graphs and take the max Q under the
    target net.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.agents.replay import ReplayBuffer, Snapshot, Transition
from src.models.gnn_qnet import GNNQNet
from src.utils.encoding import LineGraphCache, features_from_snapshot


def _data_from_snapshot(snap: Snapshot, action_idx: int, line_edge_index: torch.Tensor) -> Data:
    x = features_from_snapshot(
        capacities=snap.capacities,
        betweenness=snap.betweenness,
        path_mask=snap.path_masks[action_idx],
        bw=float(snap.demand_bw),
        link_capacity=snap.link_capacity,
        max_bw=snap.max_bw,
    )
    return Data(x=x, edge_index=line_edge_index)


class DQNAgent:
    def __init__(
        self,
        line_cache: LineGraphCache,
        hidden: int = 64,
        n_layers: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.95,
        buffer_size: int = 4000,
        batch_size: int = 32,
        target_sync: int = 200,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 5000,
        warmup_steps: int = 200,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.cache = line_cache
        self.line_edge_index = line_cache.line_edge_index
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.warmup_steps = warmup_steps
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.qnet = GNNQNet(in_dim=4, hidden=hidden, n_layers=n_layers).to(self.device)
        self.target = copy.deepcopy(self.qnet).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.buf = ReplayBuffer(buffer_size, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0
        self.acted_steps = 0

    # ---- ε-greedy schedule ---------------------------------------------------
    def _epsilon(self) -> float:
        frac = min(1.0, self.acted_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ---- forward helpers -----------------------------------------------------
    def _q_for_actions(self, obs_or_snap, k_actions: int, net: GNNQNet) -> torch.Tensor:
        """Return Q-values for all k candidate actions at this state."""
        if isinstance(obs_or_snap, Snapshot):
            datas = [_data_from_snapshot(obs_or_snap, a, self.line_edge_index) for a in range(k_actions)]
        else:
            x_caps = obs_or_snap["capacities"]
            x_bw = obs_or_snap["betweenness"]
            masks = obs_or_snap["path_masks"]
            datas = []
            for a in range(k_actions):
                x = features_from_snapshot(
                    capacities=x_caps,
                    betweenness=x_bw,
                    path_mask=masks[a],
                    bw=float(obs_or_snap["demand"].bw),
                    link_capacity=obs_or_snap["link_capacity"],
                    max_bw=obs_or_snap["max_bw"],
                )
                datas.append(Data(x=x, edge_index=self.line_edge_index))
        batch = Batch.from_data_list(datas).to(self.device)
        return net(batch.x, batch.edge_index, batch.batch)

    # ---- acting --------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: dict, greedy: bool = False) -> int:
        k = len(obs["candidate_paths"])
        eps = 0.0 if greedy else self._epsilon()
        self.acted_steps += 1 if not greedy else 0
        if not greedy and self.rng.random() < eps:
            return int(self.rng.integers(0, k))
        self.qnet.eval()
        q = self._q_for_actions(obs, k, self.qnet)
        return int(q.argmax().item())

    def remember(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool) -> None:
        from src.agents.replay import snapshot_from_obs
        s = snapshot_from_obs(obs)
        s_next = None if done else snapshot_from_obs(next_obs)
        self.buf.push(Transition(state=s, action=action, reward=reward, next_state=s_next, done=done))

    # ---- learning ------------------------------------------------------------
    def learn_step(self) -> Optional[float]:
        if len(self.buf) < max(self.batch_size, self.warmup_steps):
            return None
        batch = self.buf.sample(self.batch_size)

        # Q(s, a_taken)
        cur_data = [_data_from_snapshot(tr.state, tr.action, self.line_edge_index) for tr in batch]
        cur_batch = Batch.from_data_list(cur_data).to(self.device)
        self.qnet.train()
        q_sa = self.qnet(cur_batch.x, cur_batch.edge_index, cur_batch.batch)

        # max_a' Q_target(s', a') for non-terminal next states
        with torch.no_grad():
            target_vals = torch.zeros(self.batch_size, device=self.device)
            non_term = [(i, tr) for i, tr in enumerate(batch) if not tr.done and tr.next_state is not None]
            if non_term:
                all_next_data = []
                k_per = []
                for _, tr in non_term:
                    k = tr.next_state.path_masks.shape[0]
                    k_per.append(k)
                    for a in range(k):
                        all_next_data.append(_data_from_snapshot(tr.next_state, a, self.line_edge_index))
                nb = Batch.from_data_list(all_next_data).to(self.device)
                q_all = self.target(nb.x, nb.edge_index, nb.batch)  # [sum_k]
                # split into per-transition chunks and take max
                offset = 0
                for (i, _), k in zip(non_term, k_per):
                    target_vals[i] = q_all[offset:offset + k].max()
                    offset += k

        rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32, device=self.device)
        y = rewards + self.gamma * target_vals
        loss = F.smooth_l1_loss(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.opt.step()
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        return float(loss.item())
