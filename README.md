# DRL + GNN for Network Routing

an RL agent that uses a GNN to encode network state and routes traffic demands over a fixed topology.

## Environment

```bash
conda activate routing
```

Python 3.10, PyTorch 2.4 (CUDA 12.1), PyTorch Geometric 2.7, NetworkX, gymnasium.

## Layout

```
src/
  env/         OTN-style routing environment (gymnasium)
  models/      GNN encoder + Q-head
  agents/      DQN agent (replay buffer, target net, training loop)
  baselines/   shortest path / random / load-balancing
  utils/       topology loaders, seeding, logging
experiments/   training & evaluation entry points
configs/       YAML hyperparameter files
data/topologies/  NSFNet, GEANT2, optional Topology Zoo graphs
results/       checkpoints, plots, metrics
```

## Quickstart (after stubs are filled in)

```bash
python -m experiments.smoke_test         # sanity check env + baselines
python -m experiments.train --config configs/dqn_nsfnet.yaml
python -m experiments.evaluate --ckpt results/.../best.pt --topology geant2
```
