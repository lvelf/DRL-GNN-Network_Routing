"""Train a DQN+GNN routing agent and compare it against heuristic baselines.

Usage:
    python -m experiments.train --config configs/dqn_nsfnet.yaml
    python -m experiments.train --config configs/dqn_nsfnet.yaml --episodes 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.agents.dqn import DQNAgent
from src.baselines.heuristics import POLICIES
from src.env.routing_env import RoutingEnv
from src.utils.topology import load_named, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_episode_with_policy(env: RoutingEnv, policy, rng) -> tuple[float, int]:
    obs = env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        a = policy(obs, rng)
        obs, r, done, _ = env.step(a)
        total += r
        steps += 1
    return total, steps


def run_episode_with_agent(env: RoutingEnv, agent: DQNAgent, greedy: bool = True) -> tuple[float, int]:
    obs = env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        a = agent.act(obs, greedy=greedy)
        obs, r, done, _ = env.step(a)
        total += r
        steps += 1
    return total, steps


def evaluate(env: RoutingEnv, agent: DQNAgent, n_eps: int, seed_offset: int = 10_000) -> dict:
    rng = np.random.default_rng(0)
    scores, lens = [], []
    for ep in range(n_eps):
        env.reset(seed=seed_offset + ep)
        s, t = run_episode_with_agent(env, agent, greedy=True)
        scores.append(s); lens.append(t)
    out = {"agent_mean": float(np.mean(scores)), "agent_std": float(np.std(scores)),
           "agent_steps": float(np.mean(lens))}
    for pname, pol in POLICIES.items():
        sc, ln = [], []
        for ep in range(n_eps):
            env.reset(seed=seed_offset + ep)
            rng = np.random.default_rng(seed_offset + ep)
            s, t = run_episode_with_policy(env, pol, rng)
            sc.append(s); ln.append(t)
        out[f"{pname}_mean"] = float(np.mean(sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=None, help="override train.episodes")
    ap.add_argument("--tag", type=str, default="", help="extra suffix for results dir")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.episodes is not None:
        cfg["train"]["episodes"] = args.episodes

    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- env ----
    g = load_named(cfg["env"]["topology"])
    print(f"[topology] {cfg['env']['topology']}: {summarize(g)}")
    env = RoutingEnv(
        graph=g,
        link_capacity=cfg["env"]["link_capacity"],
        bw_choices=tuple(cfg["env"]["bw_choices"]),
        k_paths=cfg["env"]["k_paths"],
        max_steps=cfg["env"]["max_steps"],
        seed=seed,
    )

    # ---- agent ----
    a_cfg = cfg["agent"]
    agent = DQNAgent(
        line_cache=env.line_cache,
        hidden=a_cfg["hidden"],
        n_layers=a_cfg["n_layers"],
        lr=a_cfg["lr"],
        gamma=a_cfg["gamma"],
        buffer_size=a_cfg["buffer_size"],
        batch_size=a_cfg["batch_size"],
        target_sync=a_cfg["target_sync"],
        eps_start=a_cfg["eps_start"],
        eps_end=a_cfg["eps_end"],
        eps_decay_steps=a_cfg["eps_decay_steps"],
        warmup_steps=a_cfg["warmup_steps"],
        device=cfg["device"],
        seed=seed,
    )
    print(f"[agent] device={agent.device}  params="
          f"{sum(p.numel() for p in agent.qnet.parameters()):,}")

    # ---- output dir ----
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{cfg['experiment_name']}{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = out_dir / "log.jsonl"
    print(f"[run] writing results to {out_dir.relative_to(REPO_ROOT)}")

    # ---- training loop ----
    tcfg = cfg["train"]
    best_eval = -1.0
    for ep in range(1, tcfg["episodes"] + 1):
        obs = env.reset(seed=seed + ep)
        ep_reward, ep_steps, losses = 0.0, 0, []
        done = False
        while not done:
            a = agent.act(obs, greedy=False)
            next_obs, r, done, _ = env.step(a)
            agent.remember(obs, a, r, next_obs, done)
            obs = next_obs
            ep_reward += r; ep_steps += 1
            if ep_steps % tcfg["learn_every"] == 0:
                loss = agent.learn_step()
                if loss is not None:
                    losses.append(loss)

        log = {
            "episode": ep,
            "train_reward": ep_reward,
            "train_steps": ep_steps,
            "eps": agent._epsilon(),
            "mean_loss": float(np.mean(losses)) if losses else None,
        }

        if ep % tcfg["log_every"] == 0:
            print(f"[ep {ep:4d}]  reward={ep_reward:7.1f}  steps={ep_steps:4d}  "
                  f"eps={agent._epsilon():.3f}  loss={log['mean_loss']}")

        if ep % tcfg["eval_every"] == 0 or ep == tcfg["episodes"]:
            ev = evaluate(env, agent, tcfg["eval_episodes"])
            log["eval"] = ev
            print(f"   eval  agent={ev['agent_mean']:7.1f}   "
                  f"shortest={ev['shortest_mean']:7.1f}   "
                  f"random={ev['random_mean']:7.1f}   "
                  f"load_balance={ev['load_balance_mean']:7.1f}")
            if ev["agent_mean"] > best_eval:
                best_eval = ev["agent_mean"]
                torch.save(agent.qnet.state_dict(), out_dir / "best.pt")

        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

    torch.save(agent.qnet.state_dict(), out_dir / "last.pt")
    print(f"[done] best eval agent score={best_eval:.1f}")
    print(f"[done] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
