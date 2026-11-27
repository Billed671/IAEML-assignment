# IAEML — Assignment

This repository is designed to practice concepts from the IAEML course at
Innopolis University through a small 2D autonomous-driving environment.
Vehicles, goals and obstacles are represented in a planar world. Static and
moving obstacles are supported.

## Environment and vehicle physics

The interactive demo uses `RobotaxiEnv`, which extends `BaseEnv` with a
continuous-control vehicle model. It includes:

- state: position, heading, signed speed and steering angle;
- continuous actions: longitudinal acceleration (m/s²) and steering angle
  (radians);
- a kinematic bicycle model with a finite steering response rate;
- drive/brake force, rolling resistance and aerodynamic drag;
- tyre friction limits, including a friction circle shared by longitudinal and
  lateral acceleration;
- observations with route information, range sensors and vehicle state;
- a dense reward for progress, route following and smooth control, with
  terminal rewards/penalties for goal completion and collisions.

The vehicle model is simplified and operates in a 2D plane. It does not model
suspension, weight transfer or detailed tyre slip dynamics. More details and
the equations are in [`docs/stage_1_model.md`](docs/stage_1_model.md).

## Reinforcement learning (PPO)

The repository also includes a continuous-action PPO agent. Its actor outputs
acceleration and steering commands, while its critic estimates the state
value. PPO uses clipped policy updates, Generalized Advantage Estimation (GAE),
parallel environment rollouts and a tanh-squashed Gaussian action distribution.
The neural network is implemented with Flax and optimization uses Optax.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train an agent (no graphical window is opened during training):

```bash
python src/main.py --mode train --total-timesteps 500000
```

Useful training options:

```bash
python src/main.py --mode train \
    --map-id 1 \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --rollout-steps 128 \
    --episode-horizon 1200 \
    --checkpoint models/robotaxi_ppo.msgpack
```

The checkpoint and a small JSON metadata file are saved under `models/`.
The `--map-id` option selects one of the three maps in `src/env/tasks.py`.
Training currently uses the selected map for all parallel workers; evaluating
on another map is a useful generalization test, but is not guaranteed to work
without additional training.

Watch a trained policy:

```bash
python src/main.py --mode eval --checkpoint models/robotaxi_ppo.msgpack
```

The evaluation map should match the map used for training:

```bash
python src/main.py --mode eval --map-id 1 \
    --checkpoint models/robotaxi_ppo.msgpack
```

The environment itself is created with `max_steps=0`, so it has no time limit.
For RL only, `--episode-horizon` applies a trainer-side truncation and reset.
A truncated transition bootstraps the critic from its final observation; this
keeps learning batches finite without changing the environment's dynamics or
its unlimited-step setting. Goal completion and collisions remain true terminal
states.

## Manual simulator

```bash
python src/main.py
```

The demo uses `max_steps=0`, which disables the time limit. Episodes still end
when the car reaches the goal or collides with an obstacle; the demo then
resets the environment automatically. A positive `max_steps` can be used for
finite-horizon experiments.

## Controls

| Keys | Action |
| --- | --- |
| W / Up | Accelerate forward |
| S / Down | Brake, then reverse while held |
| A / Left | Steer left |
| D / Right | Steer right |
| Space | Pause or resume |
| R | Reset the current episode |
| V | Show or hide sensor rays |
| Esc | Exit |

## Original environment components

`src/env/base.py` provides map generation, collision geometry, perception rays
and dynamic-programming path planning. `src/env/tasks.py` contains the sample
maps. The original `BaseEnv` remains available for experiments that need the
simpler direction-control model; the interactive entry point now launches
`RobotaxiEnv`.

## Reinforcement learning (PPO)

The project includes an optional PPO agent for continuous vehicle control. Install
all dependencies (including the FFmpeg-enabled ImageIO package) and run training:

```bash
python -m pip install -r requirements.txt
python src/main.py --mode train --total-timesteps 500000 --map-id 1
```

Each training run creates its own timestamped folder under `artifacts/` with:

- `robotaxi_ppo.msgpack` and its JSON metadata (trained policy checkpoint);
- `training_metrics.csv` (returns, goal/collision counts and PPO losses per update);
- `run_config.json` (training configuration and total environment steps).

After training, launch the trained policy and automatically record its run:

```bash
python src/main.py --mode eval --map-id 1 \
  --checkpoint artifacts/ppo_map<TIMESTAMP>/robotaxi_ppo.msgpack
```

Manual simulator runs are recorded as well:

```bash
python src/main.py
```

Videos are saved as MP4 files under `artifacts/simulator/<timestamp>/episode.mp4`.
Use `--no-record` to disable video recording, or `--artifacts-dir PATH` to
change the output directory. Generated artifacts are excluded from Git.
