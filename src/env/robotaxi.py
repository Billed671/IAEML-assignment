"""Physics-based robotaxi environment built on top of :class:`BaseEnv`.

The map, path planner, range sensors and obstacle handling are inherited from
``BaseEnv``.  The vehicle is controlled by a longitudinal acceleration and a
front-wheel steering angle, and its motion follows a kinematic bicycle model.

World coordinates use metres and seconds.  The map is intentionally small,
so this is a simplified planar vehicle model rather than a full automotive
simulator.
"""

from functools import partial
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvParams, BaseEnvState, BaseEnvObservation
from utils import utils


class RobotaxiEnvParams(BaseEnvParams):
    """Static map, vehicle, physics and reward parameters."""

    # Vehicle and actuator limits (SI units).
    max_speed: float
    max_reverse_speed: float
    max_acceleration: float
    max_reverse_acceleration: float
    max_braking: float
    max_steering_angle: float
    max_steering_rate: float
    wheelbase: float

    # Physical properties and resistance coefficients.
    vehicle_mass: float
    rolling_resistance: float
    aerodynamic_drag: float
    frontal_area: float
    air_density: float
    tire_friction_coefficient: float
    gravity: float

    # Reward weights.
    progress_reward_scale: float
    path_deviation_penalty: float
    steering_penalty: float
    acceleration_penalty: float
    step_penalty: float
    goal_reward: float
    collision_penalty: float
    timeout_penalty: float


class RobotaxiState(BaseEnvState):
    """Base environment state plus longitudinal speed and steering angle."""

    speed: chex.Array
    steering_angle: chex.Array


class RobotaxiObservation(BaseEnvObservation):
    """Navigation/perception observations and vehicle-specific measurements."""

    speed: chex.Array
    steering_angle: chex.Array
    distance_to_goal: chex.Array
    heading_alignment: chex.Array
    heading_direction: chex.Array

    def as_vector(self) -> chex.Array:
        """Flatten all observations into a vector for a policy network."""
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.distance_to_path),
                jnp.ravel(self.direction_of_path),
                jnp.ravel(self.collision_rays),
                jnp.atleast_1d(self.speed),
                jnp.atleast_1d(self.steering_angle),
                jnp.atleast_1d(self.distance_to_goal),
                jnp.atleast_1d(self.heading_alignment),
                jnp.ravel(self.heading_direction),
            ],
            axis=0,
        )


class RobotaxiEnv(BaseEnv):
    """Continuous-control robotaxi environment with a simple vehicle model.

    Action space (shape ``(2,)``):
        ``action[0]`` is the requested longitudinal acceleration in m/s^2.
        Positive values accelerate forward; negative values brake or drive in
        reverse after the car stops.
        ``action[1]`` is the requested front-wheel steering angle in radians.
        Positive angles steer left and negative angles steer right.

    The action is clipped to the physical actuator limits.  A finite steering
    rate models actuator response.  Longitudinal and lateral tyre forces share
    a friction circle, so the car cannot brake/accelerate and turn with
    unlimited grip at the same time.

    A ``max_steps`` value of zero disables the time limit.  Reaching the goal
    or colliding with an obstacle still terminates the current episode.
    """

    uses_vehicle_controls = True

    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int = 0,
        path_length: int = 100,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: int = 32,
        fov: float = jnp.pi,
        fps: int = 60,
        # Vehicle limits.
        max_speed: float = 7.0,
        max_reverse_speed: float = 3.5,
        max_acceleration: float = 3.0,
        max_reverse_acceleration: float = 2.0,
        max_braking: float = 6.0,
        max_steering_angle: float = 1.10,  # 2x original steering angle
        max_steering_rate: float = 3.0,  # 2x original steering rate
        wheelbase: float = 2.5,
        # Simplified SI physics parameters.
        vehicle_mass: float = 1200.0,
        rolling_resistance: float = 0.015,
        aerodynamic_drag: float = 0.8,
        frontal_area: float = 2.2,
        air_density: float = 1.225,
        tire_friction_coefficient: float = 0.8,
        gravity: float = 9.81,
        # Reward parameters.
        progress_reward_scale: float = 2.0,
        path_deviation_penalty: float = 0.05,
        steering_penalty: float = 0.01,
        acceleration_penalty: float = 0.002,
        step_penalty: float = 0.005,
        goal_reward: float = 25.0,
        collision_penalty: float = 15.0,
        timeout_penalty: float = 3.0,
    ) -> Tuple[RobotaxiEnvParams, RobotaxiState]:
        """Create environment parameters and the initial vehicle state."""
        if max_steps < 0:
            raise ValueError("max_steps must be zero (unlimited) or positive")
        if fps <= 0 or wheelbase <= 0 or vehicle_mass <= 0:
            raise ValueError("fps, wheelbase and vehicle_mass must be positive")

        base_params, base_state = BaseEnv.init_params(
            key=key,
            map_id=map_id,
            max_steps=max_steps,
            path_length=path_length,
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            num_ray_sensors=num_ray_sensors,
            fov=fov,
            fps=fps,
        )

        env_params = RobotaxiEnvParams(
            max_steps_in_episode=base_params.max_steps_in_episode,
            fps=base_params.fps,
            step_size=base_params.step_size,
            map_height_width=base_params.map_height_width,
            nav_grid_shape_dtype=base_params.nav_grid_shape_dtype,
            path_array_shape_dtype=base_params.path_array_shape_dtype,
            discretization_scale=base_params.discretization_scale,
            num_ray_sensors=base_params.num_ray_sensors,
            agent_FOV=base_params.agent_FOV,
            perception_radius=base_params.perception_radius,
            max_speed=max_speed,
            max_reverse_speed=max_reverse_speed,
            max_acceleration=max_acceleration,
            max_reverse_acceleration=max_reverse_acceleration,
            max_braking=max_braking,
            max_steering_angle=max_steering_angle,
            max_steering_rate=max_steering_rate,
            wheelbase=wheelbase,
            vehicle_mass=vehicle_mass,
            rolling_resistance=rolling_resistance,
            aerodynamic_drag=aerodynamic_drag,
            frontal_area=frontal_area,
            air_density=air_density,
            tire_friction_coefficient=tire_friction_coefficient,
            gravity=gravity,
            progress_reward_scale=progress_reward_scale,
            path_deviation_penalty=path_deviation_penalty,
            steering_penalty=steering_penalty,
            acceleration_penalty=acceleration_penalty,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            collision_penalty=collision_penalty,
            timeout_penalty=timeout_penalty,
        )

        # Point the vehicle along the first segment of the planned route.
        initial_heading = base_state.path_array[0] - base_state.agent_pos
        heading_norm = jnp.linalg.norm(initial_heading)
        goal_heading = base_state.goal_pos - base_state.agent_pos
        fallback_heading = goal_heading / (jnp.linalg.norm(goal_heading) + 1e-8)
        initial_heading = jnp.where(
            heading_norm > 1e-6,
            initial_heading / (heading_norm + 1e-8),
            fallback_heading,
        )

        init_state = RobotaxiState(
            time=base_state.time,
            goal_pos=base_state.goal_pos,
            agent_pos=base_state.agent_pos,
            agent_forward_dir=initial_heading.astype(jnp.float32),
            static_obstacles=base_state.static_obstacles,
            kinematic_obstacles=base_state.kinematic_obstacles,
            kinematic_obst_velocities=base_state.kinematic_obst_velocities,
            path_array=base_state.path_array,
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            steering_angle=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return env_params, init_state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey,
        env_params: RobotaxiEnvParams,
        init_state: RobotaxiState | None,
    ) -> Tuple[RobotaxiObservation, RobotaxiState]:
        """Reset to the supplied initial state."""
        del key  # The map/task is fixed for one environment instance.
        if init_state is None:
            raise ValueError("init_state must be provided for RobotaxiEnv")
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def get_observation(
        env_state: RobotaxiState, env_params: RobotaxiEnvParams
    ) -> RobotaxiObservation:
        """Return route, sensor and vehicle state for the agent."""
        base_obs = BaseEnv.get_observation(env_state, env_params)
        to_goal = env_state.goal_pos - env_state.agent_pos
        distance_to_goal = jnp.linalg.norm(to_goal)
        heading_alignment = jnp.dot(
            env_state.agent_forward_dir, base_obs.direction_of_path
        )
        return RobotaxiObservation(
            distance_to_path=base_obs.distance_to_path,
            direction_of_path=base_obs.direction_of_path,
            collision_rays=base_obs.collision_rays,
            speed=env_state.speed,
            steering_angle=env_state.steering_angle,
            distance_to_goal=distance_to_goal,
            heading_alignment=heading_alignment,
            heading_direction=env_state.agent_forward_dir,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RobotaxiState,
        action: chex.Array,
        env_params: RobotaxiEnvParams,
    ) -> Tuple[
        RobotaxiObservation,
        RobotaxiState,
        chex.Scalar,
        chex.Array,
        Dict[Any, Any],
    ]:
        """Apply one physics step using acceleration and steering angle."""
        del key  # This environment has deterministic transitions.

        action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.reshape(action, (2,))
        requested_acceleration = action[0]
        requested_steering = jnp.clip(
            action[1],
            -env_params.max_steering_angle,
            env_params.max_steering_angle,
        )

        dt = env_params.step_size
        old_speed = env_state.speed
        speed_epsilon = 1e-4

        # The braking limit is stronger than the normal reverse acceleration.
        # At zero speed the negative action starts reversing the vehicle.
        max_positive_accel = jnp.where(
            old_speed < -speed_epsilon,
            env_params.max_braking,
            env_params.max_acceleration,
        )
        max_negative_accel = jnp.where(
            old_speed > speed_epsilon,
            env_params.max_braking,
            env_params.max_reverse_acceleration,
        )
        commanded_acceleration = jnp.clip(
            requested_acceleration,
            -max_negative_accel,
            max_positive_accel,
        )

        # Steering is rate-limited to represent the steering actuator.
        max_steering_change = env_params.max_steering_rate * dt
        steering_change = jnp.clip(
            requested_steering - env_state.steering_angle,
            -max_steering_change,
            max_steering_change,
        )
        new_steering = env_state.steering_angle + steering_change

        # Longitudinal forces: drive/brake, rolling resistance and air drag.
        # F_drive = m*a_command; both resistance forces oppose current motion.
        mass = env_params.vehicle_mass
        rolling_force = (
            env_params.rolling_resistance
            * mass
            * env_params.gravity
            * jnp.sign(old_speed)
        )
        aerodynamic_force = (
            0.5
            * env_params.air_density
            * env_params.aerodynamic_drag
            * env_params.frontal_area
            * old_speed
            * jnp.abs(old_speed)
        )
        drive_force = mass * commanded_acceleration
        raw_tire_force = drive_force - rolling_force

        # The tyre-road friction coefficient limits the force transmitted
        # through the tyres. Aerodynamic drag is an external force and is
        # applied after this tyre-force limit.
        max_tire_force = (
            env_params.tire_friction_coefficient * mass * env_params.gravity
        )
        longitudinal_force = jnp.clip(
            raw_tire_force, -max_tire_force, max_tire_force
        )
        net_longitudinal_force = longitudinal_force - aerodynamic_force
        longitudinal_acceleration = net_longitudinal_force / mass
        tire_longitudinal_acceleration = longitudinal_force / mass
        candidate_speed = old_speed + longitudinal_acceleration * dt

        # Do not let resistance or braking numerically push the vehicle through
        # zero speed. A continued reverse/forward command moves it next frame.
        crossed_zero = old_speed * candidate_speed < 0.0
        candidate_speed = jnp.where(crossed_zero, 0.0, candidate_speed)
        new_speed = jnp.clip(
            candidate_speed,
            -env_params.max_reverse_speed,
            env_params.max_speed,
        )
        average_speed = 0.5 * (old_speed + new_speed)

        # Kinematic bicycle model: yaw_rate = v/L * tan(delta).
        requested_yaw_rate = (
            average_speed / env_params.wheelbase * jnp.tan(new_steering)
        )

        # Friction circle: longitudinal and lateral tyre forces share grip.
        # a_lat = v*yaw_rate; reduce the yaw rate when grip is insufficient.
        friction_acceleration = (
            env_params.tire_friction_coefficient * env_params.gravity
        )
        available_lateral_acceleration = jnp.sqrt(
            jnp.maximum(
                friction_acceleration**2 - tire_longitudinal_acceleration**2,
                0.0,
            )
        )
        requested_lateral_acceleration = jnp.abs(
            average_speed * requested_yaw_rate
        )
        grip_scale = jnp.minimum(
            1.0,
            available_lateral_acceleration
            / (requested_lateral_acceleration + 1e-8),
        )
        yaw_rate = requested_yaw_rate * grip_scale
        yaw_change = yaw_rate * dt

        # Midpoint integration reduces position error while cornering.
        midpoint_heading = utils.rotate_2d(
            env_state.agent_forward_dir, 0.5 * yaw_change
        )
        displacement = average_speed * dt * midpoint_heading
        candidate_position = env_state.agent_pos + displacement
        new_heading = utils.rotate_2d(env_state.agent_forward_dir, yaw_change)
        new_heading = new_heading / (jnp.linalg.norm(new_heading) + 1e-8)

        # Advance moving obstacles before testing the car's new position.
        new_kinematic_obstacles = BaseEnv._move_kinematic_obstacles(
            env_state, env_params
        )
        new_obstacles = jnp.concatenate(
            [env_state.static_obstacles, new_kinematic_obstacles], axis=0
        )

        # Check an intermediate point as well as the endpoint to reduce
        # tunnelling through obstacles at higher speeds.
        midpoint_position = env_state.agent_pos + 0.5 * displacement
        collision_midpoint = BaseEnv._check_collisions(
            midpoint_position, new_obstacles
        )
        collision_endpoint = BaseEnv._check_collisions(
            candidate_position, new_obstacles
        )
        collision_done = jnp.logical_or(
            collision_midpoint, collision_endpoint
        )
        goal_done = BaseEnv._check_goal(candidate_position, env_state.goal_pos)

        new_time = env_state.time + 1
        # max_steps == 0 means an unlimited episode.
        time_done = jnp.logical_and(
            env_params.max_steps_in_episode > 0,
            new_time >= env_params.max_steps_in_episode,
        )
        done = jnp.logical_or(
            goal_done, jnp.logical_or(collision_done, time_done)
        )

        # Re-plan once per simulated second to account for moving obstacles.
        should_replan = jnp.equal(jnp.mod(new_time, env_params.fps), 0)
        path_array = jax.lax.cond(
            should_replan,
            lambda _: BaseEnv._find_path(
                candidate_position,
                env_state.goal_pos,
                new_obstacles,
                env_params,
            ),
            lambda _: env_state.path_array,
            operand=None,
        )

        new_state = RobotaxiState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=candidate_position,
            agent_forward_dir=new_heading,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=new_kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
            speed=new_speed,
            steering_angle=new_steering,
        )
        obs = RobotaxiEnv.get_observation(new_state, env_params)

        # Dense reward: make progress, follow the route and avoid harsh inputs.
        old_goal_distance = jnp.linalg.norm(
            env_state.goal_pos - env_state.agent_pos
        )
        progress = old_goal_distance - obs.distance_to_goal
        progress_reward = env_params.progress_reward_scale * progress
        path_penalty = (
            env_params.path_deviation_penalty * obs.distance_to_path
        )
        steering_penalty = (
            env_params.steering_penalty
            * jnp.abs(new_steering)
            / (env_params.max_steering_angle + 1e-8)
        )
        acceleration_penalty = (
            env_params.acceleration_penalty
            * jnp.square(commanded_acceleration)
        )

        goal_bonus = jnp.where(goal_done, env_params.goal_reward, 0.0)
        collision_cost = jnp.where(
            collision_done, env_params.collision_penalty, 0.0
        )
        timeout_cost = jnp.where(
            jnp.logical_and(time_done, jnp.logical_not(goal_done)),
            env_params.timeout_penalty,
            0.0,
        )
        reward = (
            progress_reward
            - path_penalty
            - steering_penalty
            - acceleration_penalty
            - env_params.step_penalty
            + goal_bonus
            - collision_cost
            - timeout_cost
        )

        info = {
            "time": new_time,
            "speed": new_speed,
            "acceleration": longitudinal_acceleration,
            "commanded_acceleration": commanded_acceleration,
            "steering_angle": new_steering,
            "yaw_rate": yaw_rate,
            "lateral_acceleration": average_speed * yaw_rate,
            "available_lateral_acceleration": available_lateral_acceleration,
            "longitudinal_force": longitudinal_force,
            "net_longitudinal_force": net_longitudinal_force,
            "rolling_resistance_force": rolling_force,
            "aerodynamic_drag_force": aerodynamic_force,
            "distance_to_goal": obs.distance_to_goal,
            "distance_to_path": obs.distance_to_path,
            "heading_alignment": obs.heading_alignment,
            "progress_reward": progress_reward,
            "goal_reached": goal_done,
            "collision": collision_done,
            "timeout": time_done,
        }
        return obs, new_state, reward, done, info
