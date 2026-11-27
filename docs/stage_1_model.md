# Stage 1 — Robotaxi dynamics, observations and reward

This implementation follows the Stage 1 requirements in `instructions/stage_1.md`.
It inherits the map, obstacles, route planner and range sensors from `BaseEnv`,
and adds a continuous-control vehicle model.

## State space

The vehicle state extends `BaseEnvState` with:

- `speed` — signed longitudinal speed in m/s (negative means reversing);
- `steering_angle` — current front-wheel steering angle in radians.

The inherited state also contains the vehicle position, heading direction,
goal, static and moving obstacles, and the planned route.

## Action space

The action is a two-element continuous vector:

- `action[0]`: requested longitudinal acceleration in m/s²;
- `action[1]`: requested steering-wheel/front-wheel angle in radians.

The environment clips acceleration and steering to configured limits. Steering
is rate-limited to represent the actuator. A negative acceleration brakes a
vehicle moving forward; once it stops, a continued negative command makes it
reverse. The same rule works in the opposite direction.

## Vehicle physics

The model uses a simplified planar single-track (kinematic bicycle) model.
Longitudinal forces are:

- drive/brake force: `F_drive = mass * commanded_acceleration`;
- rolling resistance: `F_roll = C_rr * mass * gravity * sign(speed)`;
- aerodynamic drag: `F_drag = 0.5 * air_density * Cd * frontal_area * speed * abs(speed)`.

The drive/brake and rolling-resistance terms define the longitudinal tyre
force. It is limited by the available tyre-road friction:

`abs(F_tyre_longitudinal) <= mu * mass * gravity`.

Aerodynamic drag is then applied as an external force when calculating net
longitudinal acceleration. Lateral tyre grip is computed from the remaining
friction-circle budget after longitudinal tyre force is accounted for.

The bicycle model uses `yaw_rate = speed / wheelbase * tan(steering_angle)`.
Lateral acceleration consumes part of the same tyre grip as longitudinal
acceleration. The implementation therefore uses a friction circle to reduce
the available steering/yaw rate when the car is braking or accelerating hard.
Position and heading are integrated using a midpoint update.

This is intentionally a simplified 2D model: it includes acceleration,
braking, reverse motion, steering response, rolling resistance, aerodynamic
drag and tyre friction, but does not model suspension, weight transfer, tyre
slip angles or a full multi-body vehicle.

## Observation space

The policy receives the inherited route distance, route direction and range
sensor rays, together with speed, steering angle, distance to goal, heading
direction and alignment between the vehicle heading and the planned route.
`RobotaxiObservation.as_vector()` concatenates these values into one vector.

## Reward function

The dense reward consists of:

- positive reward for reducing distance to the goal;
- penalties for distance from the planned route, steering magnitude,
  acceleration magnitude and each simulation step;
- a terminal bonus for reaching the goal;
- a terminal penalty for collision;
- an optional timeout penalty when a positive time limit is configured.

## Episode length

`max_steps=0` means that the episode has no time limit. Goal and collision
termination remain enabled. A positive `max_steps` can still be passed when a
finite horizon is needed for experiments.

## Renderer controls

- `W` / Up: accelerate forward;
- `S` / Down: brake, then reverse while held;
- `A` / Left: steer left;
- `D` / Right: steer right;
- `Space`: pause/resume;
- `R`: reset;
- `V`: show/hide sensor rays;
- `Esc`: exit.
