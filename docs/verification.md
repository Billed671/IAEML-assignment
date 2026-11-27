# Robotaxi verification

The model and renderer build on [GitHub revision 7f75060](https://github.com/Billed671/IAEML-assignment/commit/7f75060e2413eedc14c3b29e6cb4588af0ba2040).

## Vehicle and simulator

- `RobotaxiEnv` inherits from `BaseEnv`. A positive acceleration command
  increases speed. Steering changes both the front-wheel angle and the heading.
- Rolling resistance and aerodynamic drag slow the car while coasting.
  With 32 rays, the observation has 73 finite values.
- Reaching the goal region gives a positive terminal reward. A static
  obstacle collision gives a negative reward and ends the episode.
- Manual and policy modes were run with Python 3.11, JAX on CPU, and a virtual
  Pygame display. Manual driving uses an unlimited episode by default.

## Pretraining and PPO

The six earlier checkpoints each failed to reach the goal in a deterministic
evaluation on map 1. The earlier training logs also contained no goal arrivals.

For the new run, a controller first demonstrated a successful trajectory in
448 steps. A neural actor learned from that demonstration and from controller
labels on states visited by its own attempts. This actor reached the goal in
463 steps. PPO then ran eight updates over 8,192 environment steps. Across
the completed training episodes, it recorded 16 goal arrivals and zero
collisions.

In a separate deterministic evaluation, the final PPO checkpoint reached the
goal in **459 steps**, or **7.65 seconds** at 60 simulation steps per second.
It finished 0.974 m from the goal center, within the 1 m success region.

| Evaluation field | Result |
| --- | --- |
| Map and seed | 1 and 0 |
| Goal reached | Yes |
| Collision | No |
| Timeout | No |
| Episode return | 24,999.665 |

The result is for the training map and its standard initial state.

- [Recorded evaluation](robotaxi-policy-run.mp4)
- [Evaluation JSON](ppo-evaluation.json)
- [PPO configuration](ppo-training-config.json)
- [PPO metrics](ppo-training-metrics.csv)
