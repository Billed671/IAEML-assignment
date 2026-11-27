# IAEML — Assignment

This repository is designed to practice concepts from the IAEML course at
Innopolis University through a small 2D autonomous-driving environment.
Vehicles, goals and obstacles are represented in a planar world. Static and
moving obstacles are supported.

## Implemented environment

The interactive demo in this repository uses `RobotaxiEnv`, which extends
`BaseEnv` with a continuous-control vehicle model. It includes:

- a state space with position, heading, signed speed and steering angle;
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

## Installation and launch

```bash
python -m pip install -r requirements.txt
python src/main.py
```

The demo uses `max_steps=0`, which disables the time limit. Episodes still end
when the car reaches the goal or collides with an obstacle; the demo then
resets the environment automatically. A positive `max_steps` can be used for
finite-horizon experiments.

There are sometimes JAX platform issues on macOS. To force the CPU backend:

```bash
JAX_PLATFORM_NAME=cpu python src/main.py
```

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
