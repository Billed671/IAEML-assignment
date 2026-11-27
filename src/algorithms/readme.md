# Reinforcement learning algorithms

The first implemented agent is PPO (Proximal Policy Optimization) for the
continuous-action `RobotaxiEnv`.

- `networks.py` contains the Flax actor-critic MLP.
- `ppo_utils.py` scales observations, converts normalized policy actions to
  acceleration/steering commands, and calculates the squashed-Gaussian log
  probability.
- `ppo.py` contains parallel rollout collection, GAE, PPO updates, logging and
  checkpoint saving.
- `policy.py` restores a checkpoint and creates the deterministic policy used
  by the Pygame evaluator.

Run training from the repository root with:

```bash
python src/main.py --mode train --total-timesteps 500000
```

Training outputs are stored in a unique `artifacts/ppo_map<id>_<timestamp>/`
folder. It contains the Flax checkpoint, checkpoint metadata, `run_config.json`,
and `training_metrics.csv`. The Pygame simulator records manual and policy
runs to `artifacts/simulator/<timestamp>/episode.mp4` by default. Pass
`--no-record` to disable video capture.
