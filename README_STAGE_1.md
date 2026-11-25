# Stage 1 implementation notes

The Stage 1 solution is implemented in `src/env/robotaxi.py`.

## Run

Install the packages from `requirements.txt`, then run:

```bash
python src/main.py
```

The main entry point now launches `RobotaxiEnv` with the Pygame renderer.

## Manual controls

- `W` / `Up`: accelerate
- `S` / `Down`: brake
- `A` / `Left`: steer left
- `D` / `Right`: steer right

## Files changed

- `src/env/robotaxi.py` — vehicle parameters, state, observation, bicycle dynamics, collision handling and reward.
- `src/main.py` — launches the new environment.
- `src/utils/renderer.py` — maps keyboard input to throttle and steering for `RobotaxiEnv` while preserving the original controls for `BaseEnv`.
- `report_stage_1.md` — model description, equations, observation, reward and validation checklist.

See `report_stage_1.md` for the full Stage 1 description.
