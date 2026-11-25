# IAEML Assignment — Stage 1: Robotaxi Modeling

## 1. Goal

The goal of Stage 1 is to replace the point-mass movement in `BaseEnv` with a simple vehicle model. The new `RobotaxiEnv` inherits from `BaseEnv` and reuses its maps, obstacle representation, ray sensors, collision helpers, and path planner.

The vehicle is represented as a circle with the same radius as the original agent. This keeps collision checking and visualization compatible with the supplied project.

## 2. Vehicle state

The original state is extended with two dynamic variables:

- `speed` — current forward speed in map units per second;
- `steering_angle` — current front-wheel steering angle in radians.

The inherited `agent_pos` is the vehicle position, and `agent_forward_dir` is its normalized heading vector. Reverse driving is not modeled: a negative throttle command brakes the vehicle until its speed reaches zero.

## 3. Action space

The action is a two-dimensional continuous vector:


$a_t = [u_t, s_t], \qquad u_t,s_t \in [-1,1].$

- `u_t` is throttle. Positive values accelerate; negative values brake.
- `s_t` is steering. Negative values steer right; positive values steer left.

The environment clips both values to the interval `[-1, 1]`. The steering actuator has a finite steering rate, so the steering angle cannot change instantly.

## 4. Vehicle dynamics and forces

The implementation uses a kinematic single-track (bicycle) model. The simulation time step is inherited from `BaseEnv`:


$\Delta t = 1 / \text{FPS}.$

### 4.1 Longitudinal dynamics

The commanded acceleration depends on the sign of throttle:

\[
 a_{cmd} =
 \begin{cases}
 u_t a_{max}, & u_t \ge 0,\\
 u_t a_{brake}, & u_t < 0.
 \end{cases}
\]

Rolling resistance and aerodynamic drag are subtracted:


$a_{raw} = a_{cmd} - c_r\,\mathrm{sign}(v_t) - c_d v_t |v_t|.$

The tire-road friction coefficient limits the available longitudinal acceleration:


$|a_t| \leq \mu g.$

Speed is then integrated and clipped to the physical interval:


$v_{t+1} = \mathrm{clip}(v_t + a_t\Delta t, 0, v_{max}).$

### 4.2 Steering and yaw rate

The requested steering angle is `s_t * max_steering_angle`. A rate limiter models the steering actuator:


$|\delta_{t+1}-\delta_t| \leq \dot\delta_{max}\Delta t.$

The bicycle model computes yaw rate as:


$\dot\psi = \frac{\bar v}{L}\tan(\delta),$

where `L` is the wheelbase and `v_bar` is the average speed during the step. The yaw rate is also limited so lateral acceleration does not exceed the available tire grip:


$\bar v |\dot\psi| \leq \mu g.$

The new heading is obtained by rotating the old heading by `yaw_rate * dt`. Position is integrated using the average speed and midpoint heading. This gives a more stable update while the car is turning.

### 4.3 Modeled forces and assumptions

The model includes:

- motor acceleration;
- braking;
- rolling resistance;
- aerodynamic drag;
- tire-road friction limits for longitudinal and lateral acceleration.

It is a kinematic model, not a full vehicle simulator. It does not model suspension, load transfer, tire slip angles, steering geometry details, or a powertrain. The friction limit is a simplified approximation.

## 5. Observation space

The observation reuses all `BaseEnv` perception features and adds vehicle information:

| Feature | Meaning |
|---|---|
| `distance_to_path` | Distance from the vehicle to the current planned path |
| `direction_of_path` | Unit vector pointing along the local path direction |
| `collision_rays` | Ray sensor vectors to nearby obstacles |
| `speed` | Current forward speed |
| `steering_angle` | Current steering angle in radians |
| `distance_to_goal` | Euclidean distance to the destination |
| `heading_alignment` | Dot product of vehicle heading and local path direction |

`RobotaxiObservation.as_vector()` flattens these values into one vector for a policy network. With `N` ray sensors, its size is `2N + 7` values: one path-distance value, two path-direction values, `2N` ray coordinates, and four vehicle/goal values.

## 6. Reward function

The reward is dense and combines progress with safety and control costs:

\[
\begin{aligned}
 r_t ={}& w_p(d_t-d_{t+1})
 - w_{path}\,D_{path}
 - w_{steer}\frac{|\delta|}{\delta_{max}}
 - c_{step} \\
 &+ R_{goal}\mathbb{1}_{goal}
 - C_{collision}\mathbb{1}_{collision}
 - C_{timeout}\mathbb{1}_{timeout}.
\end{aligned}
\]

Here `d_t` and `d_(t+1)` are the distances to the goal before and after the transition. Positive progress increases reward. Staying close to the planned path and avoiding unnecessary steering are encouraged. Reaching the goal gives a positive terminal bonus, while collision and timeout produce penalties.

The default weights are defined in `RobotaxiEnv.init_params()` and can be changed without editing the transition function.

## 7. Termination conditions

An episode ends when at least one of these conditions is true:

1. The vehicle reaches the goal region.
2. The vehicle collides with an obstacle. Both the middle and end points of the movement are checked to reduce tunnelling.
3. The maximum number of steps is reached.

Kinematic obstacles continue to move using the original environment's motion rule. The path is recalculated once per simulated second to account for moving obstacles.

## 8. Renderer controls

Run the project from the repository root:

```bash
python src/main.py
```

Controls:

- `W` / `Up` — accelerate;
- `S` / `Down` — brake;
- `A` / `Left` — steer left;
- `D` / `Right` — steer right.

The renderer keeps the original point-mass controls when used with `BaseEnv`; the new control mapping is enabled only for `RobotaxiEnv`.

## 9. Validation checklist

When running in an environment with the project dependencies installed, check the following:

- the renderer opens and displays the selected map;
- pressing `W` increases speed and moves the vehicle forward;
- pressing `S` reduces speed;
- pressing `A` and `D` change the heading in opposite directions;
- speed remains in `[0, max_speed]`;
- steering remains within the configured maximum angle;
- collision, goal and time-limit conditions set `done=True`;
- observations and reward values remain finite during normal driving.

The current execution container did not have `equinox`, `chex`, or `pygame` installed, and package installation was unavailable because network access to the package index failed. Therefore, the implementation could be syntax-checked here, but the interactive renderer and JAX runtime could not be executed in this session. The run command above is the required final runtime check on a machine with the dependencies installed.
