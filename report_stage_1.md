# IAEML Assignment — Stage 1: Robotaxi Modeling

## 1. Goal

The goal of Stage 1 is to replace the point-mass movement in `BaseEnv` with a simple vehicle model. The new `RobotaxiEnv` inherits from `BaseEnv` and reuses its maps, obstacle representation, ray sensors, collision helpers, and path planner.

Collision detection checks the vehicle position against the same obstacle radius convention as `BaseEnv`.

## 2. Vehicle state

The original state is extended with two dynamic variables:

- `speed` — current signed speed in meters per second, with negative values for reverse;
- `steering_angle` — current front-wheel steering angle in radians.

The inherited `agent_pos` is the vehicle position, and `agent_forward_dir` is its normalized heading vector. A negative acceleration request brakes the vehicle while it is moving forward and starts reverse motion after it stops.

## 3. Action space

The action is a two-dimensional continuous vector:

\[
 a_t = [a_{\mathrm{req}}, \delta_{\mathrm{req}}].
\]

- `a_req` is requested longitudinal acceleration in m/s².
- `delta_req` is requested steering angle in radians. Positive values steer left and negative values steer right.

The environment clips requests to the configured actuator limits. The steering actuator also applies a finite steering rate.

## 4. Vehicle dynamics and forces

The implementation uses a kinematic single-track (bicycle) model. The simulation time step is inherited from `BaseEnv`:

\[
\Delta t = 1 / \text{FPS}.
\]

### 4.1 Longitudinal dynamics

The commanded acceleration clips the requested acceleration to the limits for the current direction of motion:

\[
 a_{\mathrm{cmd}} = \mathrm{clip}(a_{\mathrm{req}}, a_{\min}(v_t), a_{\max}(v_t)).
\]

The tyre force includes drive or braking force and rolling resistance. Aerodynamic drag is applied after limiting the tyre force:

\[
 F_{\mathrm{tyre}} = \mathrm{clip}(m a_{\mathrm{cmd}} - F_{\mathrm{rolling}}, -\mu m g, \mu m g),
 \qquad a_{\mathrm{net}} = (F_{\mathrm{tyre}} - F_{\mathrm{drag}})/m.
\]

Speed is integrated and clipped to the configured forward and reverse limits. If braking or resistance would cross zero in one step, the vehicle stops at zero for that step.

\[
 v_{t+1} = \mathrm{clip}(v_t + a_{\mathrm{net}}\Delta t, -v_{\mathrm{reverse,max}}, v_{\max}).
\]

### 4.2 Steering and yaw rate

The requested steering angle is clipped to the steering range. A rate limiter models the steering actuator:

\[
 |\delta_{t+1}-\delta_t| \leq \dot\delta_{max}\Delta t.
\]

The bicycle model computes yaw rate as:

\[
 \dot\psi = \frac{\bar v}{L}\tan(\delta),
\]

where `L` is the wheelbase and `v_bar` is the average speed during the step. The yaw rate is also limited so lateral acceleration does not exceed the available tire grip:

\[
 |\bar v \dot\psi| \leq \sqrt{\max(0, (\mu g)^2-(F_{\mathrm{tyre}}/m)^2)}.
\]

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
| `speed` | Current signed speed |
| `steering_angle` | Current steering angle in radians |
| `distance_to_goal` | Euclidean distance to the destination |
| `heading_alignment` | Dot product of vehicle heading and local path direction |
| `heading_direction` | Two components of the vehicle heading vector |

`RobotaxiObservation.as_vector()` flattens these values into `2N + 9` numbers for `N` ray sensors. With 32 sensors, the vector has 73 values.

## 6. Reward function

The reward is dense and combines progress with safety and control costs:

\[
\begin{aligned}
 r_t ={}& w_p(d_t-d_{t+1})
 - w_{path}\,D_{path}
 - w_{steer}\frac{|\delta|}{\delta_{max}}
 - w_{acc}a_{\mathrm{cmd}}^2
 - c_{step} \\
 &+ R_{goal}\mathbb{1}_{goal}
 - C_{collision}\mathbb{1}_{collision}
 - C_{timeout}\mathbb{1}_{timeout}.
\end{aligned}
\]

Here `d_t` and `d_(t+1)` are the distances to the goal before and after the transition. Positive progress increases reward. Staying close to the planned path and avoiding unnecessary steering are encouraged. Reaching the goal gives a positive terminal bonus, while collision and timeout produce penalties.

The default weights are defined in `RobotaxiEnv.init_params()`. The timeout penalty applies when the time limit is reached without reaching the goal.

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
- speed remains in `[-max_reverse_speed, max_speed]`;
- steering remains within the configured maximum angle;
- collision, goal and time-limit conditions set `done=True`;
- observations and reward values remain finite during normal driving.

The model and renderer were tested with JAX on CPU and a virtual Pygame display. [Verification results](docs/verification.md) record the tested dynamics, reward, rendering and policy outcomes.
