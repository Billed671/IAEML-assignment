"""Robotaxi environment with continuous acceleration and steering controls.

The environment reuses the maps, perception sensors, collision geometry and
path planner from BaseEnv. The vehicle itself is represented by a circular
collider and follows a kinematic single-track (bicycle) model.
"""

from functools import partial
from typing import Tuple, Dict, Any

import chex
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvParams, BaseEnvState, BaseEnvObservation
from utils import utils


class RobotaxiEnvParams(BaseEnvParams):
    """Static environment and vehicle parameters."""

    # Vehicle dynamics
    max_speed: float
    max_reverse_speed: float
    max_acceleration: float
    max_reverse_acceleration: float
    max_braking: float
    max_steering_angle: float
    max_steering_rate: float
    wheelbase: float
    rolling_resistance: float
    aerodynamic_drag: float
    tire_friction_coefficient: float
    gravity: float

    # Reward weights
    progress_reward_scale: float
    path_deviation_penalty: float
    steering_penalty: float
    step_penalty: float
    goal_reward: float
    collision_penalty: float
    timeout_penalty: float


class RobotaxiState(BaseEnvState):
    """Base environment state plus the vehicle's dynamic state."""

    speed: chex.Array
    steering_angle: chex.Array


class RobotaxiObservation(BaseEnvObservation):
    """Navigation and perception data plus vehicle-specific observations."""

    speed: chex.Array
    steering_angle: chex.Array
    distance_to_goal: chex.Array
    heading_alignment: chex.Array
    heading_direction: chex.Array

    def as_vector(self) -> chex.Array:
        """Return a flat vector suitable for a policy network."""
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
    """A continuous-control robotaxi environment.

    Action space:
        action[0]: signed drive command in [-1, 1]. Positive requests forward
                   motion; negative requests reverse. If the vehicle is moving
                   in the opposite direction, the command first brakes it.
        action[1]: steering command in [-1, 1]. Negative means right and
                   positive means left.

    Observation:
        BaseEnv path direction, distance to path and collision rays, together
        with speed, steering angle, distance to the goal and heading alignment.

    The agent remains a circle of the same radius as in BaseEnv, so the
    existing collision checks and renderer can be reused.
    """

    uses_vehicle_controls = True

    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: int = 32,
        fov: float = jnp.pi,
        fps: float = 60,
        # Vehicle parameters are expressed in map units and seconds.
        max_speed: float = 7.0,
        max_reverse_speed: float = 3.5,
        max_acceleration: float = 5.0,
        max_reverse_acceleration: float = 3.0,
        max_braking: float = 7.0,
        max_steering_angle: float = 0.55,
        max_steering_rate: float = 1.5,
        wheelbase: float = 2.5,
        rolling_resistance: float = 0.12,
        aerodynamic_drag: float = 0.025,
        tire_friction_coefficient: float = 0.8,
        gravity: float = 9.81,
        # Reward parameters
        progress_reward_scale: float = 2.0,
        path_deviation_penalty: float = 0.05,
        steering_penalty: float = 0.01,
        step_penalty: float = 0.005,
        goal_reward: float = 25.0,
        collision_penalty: float = 15.0,
        timeout_penalty: float = 3.0,
    ) -> Tuple[RobotaxiEnvParams, RobotaxiState]:
        """Create parameters and the initial state for a selected map."""
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
            rolling_resistance=rolling_resistance,
            aerodynamic_drag=aerodynamic_drag,
            tire_friction_coefficient=tire_friction_coefficient,
            gravity=gravity,
            progress_reward_scale=progress_reward_scale,
            path_deviation_penalty=path_deviation_penalty,
            steering_penalty=steering_penalty,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            collision_penalty=collision_penalty,
            timeout_penalty=timeout_penalty,
        )

        # Start by facing the destination. The policy can change this direction
        # later through the steering action.
        initial_heading = base_state.path_array[0] - base_state.agent_pos
        heading_norm = jnp.linalg.norm(initial_heading)
        goal_heading = base_state.goal_pos - base_state.agent_pos
        initial_heading = jnp.where(
            heading_norm > 1e-6,
            initial_heading / (heading_norm + 1e-8),
            goal_heading / (jnp.linalg.norm(goal_heading) + 1e-8),
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
        """Reset the environment to its supplied initial state."""
        del key  # The current maps are deterministic and do not need randomness.
        if init_state is None:
            raise ValueError("init_state must be provided for RobotaxiEnv")
        obs = RobotaxiEnv.get_observation(init_state, env_params)
        return obs, init_state

    @staticmethod
    def get_observation(
        env_state: RobotaxiState, env_params: RobotaxiEnvParams
    ) -> RobotaxiObservation:
        """Build the policy observation from the current state."""
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
    ) -> Tuple[RobotaxiObservation, RobotaxiState, chex.Scalar, chex.Array, Dict[Any, Any]]:
        """Advance the car by one time step using a bicycle model."""
        del key  # No stochastic transition is used in this environment.

        # Action is [signed drive command, steering], both in [-1, 1].
        action = jnp.asarray(action, dtype=jnp.float32)
        throttle = jnp.clip(action[0], -1.0, 1.0)
        steering_command = jnp.clip(action[1], -1.0, 1.0)

        dt = env_params.step_size
        old_speed = env_state.speed

        # Steering has a finite response rate instead of changing instantly.
        target_steering = steering_command * env_params.max_steering_angle
        max_steering_change = env_params.max_steering_rate * dt
        steering_change = jnp.clip(
            target_steering - env_state.steering_angle,
            -max_steering_change,
            max_steering_change,
        )
        new_steering = env_state.steering_angle + steering_change

        # A signed drive command selects forward (positive) or reverse (negative).
        # If the command opposes the current motion, apply the stronger brake
        # first. Once the speed reaches zero, a held command starts the other
        # direction on the next step.
        moving_against_command = old_speed * throttle < 0.0
        forward_or_reverse_acceleration = jnp.where(
            throttle >= 0.0,
            throttle * env_params.max_acceleration,
            throttle * env_params.max_reverse_acceleration,
        )
        braking_acceleration = -jnp.sign(old_speed) * (
            jnp.abs(throttle) * env_params.max_braking
        )
        commanded_acceleration = jnp.where(
            moving_against_command,
            braking_acceleration,
            forward_or_reverse_acceleration,
        )

        # Rolling resistance and aerodynamic drag always oppose the motion.
        rolling_force = env_params.rolling_resistance * jnp.sign(old_speed)
        aerodynamic_force = (
            env_params.aerodynamic_drag * old_speed * jnp.abs(old_speed)
        )
        raw_acceleration = commanded_acceleration - rolling_force - aerodynamic_force

        # The tire-road friction coefficient limits the available acceleration.
        friction_limit = (
            env_params.tire_friction_coefficient * env_params.gravity
        )
        longitudinal_acceleration = jnp.clip(
            raw_acceleration, -friction_limit, friction_limit
        )
        candidate_speed = old_speed + longitudinal_acceleration * dt

        # Do not jump from forward to reverse (or vice versa) in one braking
        # step. A continued opposite command starts the new direction next step.
        crossed_zero_while_braking = jnp.logical_and(
            moving_against_command,
            jnp.logical_or(
                jnp.logical_and(old_speed > 0.0, candidate_speed < 0.0),
                jnp.logical_and(old_speed < 0.0, candidate_speed > 0.0),
            ),
        )
        candidate_speed = jnp.where(
            crossed_zero_while_braking, 0.0, candidate_speed
        )
        new_speed = jnp.clip(
            candidate_speed,
            -env_params.max_reverse_speed,
            env_params.max_speed,
        )
        average_speed = 0.5 * (old_speed + new_speed)

        # Kinematic bicycle model: yaw_rate = v / L * tan(steering_angle).
        requested_yaw_rate = (
            average_speed
            / env_params.wheelbase
            * jnp.tan(new_steering)
        )

        # Limit lateral acceleration to the tire grip available on the road.
        max_yaw_rate = friction_limit / (jnp.abs(average_speed) + 1e-6)
        yaw_rate = jnp.clip(
            requested_yaw_rate, -max_yaw_rate, max_yaw_rate
        )
        yaw_change = yaw_rate * dt

        # Midpoint integration gives a more accurate position update at turns.
        midpoint_heading = utils.rotate_2d(
            env_state.agent_forward_dir, 0.5 * yaw_change
        )
        displacement = average_speed * dt * midpoint_heading
        candidate_position = env_state.agent_pos + displacement
        new_heading = utils.rotate_2d(
            env_state.agent_forward_dir, yaw_change
        )
        new_heading = new_heading / (jnp.linalg.norm(new_heading) + 1e-8)

        # Move dynamic obstacles using the original environment's motion rule.
        new_kinematic_obstacles = BaseEnv._move_kinematic_obstacles(
            env_state, env_params
        )
        new_obstacles = jnp.concatenate(
            [env_state.static_obstacles, new_kinematic_obstacles], axis=0
        )

        # Check the middle and end points to reduce the chance of tunnelling.
        midpoint_position = env_state.agent_pos + 0.5 * displacement
        collision_midpoint = BaseEnv._check_collisions(
            midpoint_position, new_obstacles
        )
        collision_endpoint = BaseEnv._check_collisions(
            candidate_position, new_obstacles
        )
        collision_done = jnp.logical_or(collision_midpoint, collision_endpoint)
        goal_done = BaseEnv._check_goal(candidate_position, env_state.goal_pos)

        new_time = env_state.time + 1
        time_done = new_time >= env_params.max_steps_in_episode
        done = jnp.logical_or(
            goal_done, jnp.logical_or(collision_done, time_done)
        )

        # Re-plan once per simulated second so moving obstacles can be considered.
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

        # Dense reward encourages progress and staying close to the planned path.
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

        goal_bonus = jnp.where(goal_done, env_params.goal_reward, 0.0)
        crash_penalty = jnp.where(
            collision_done, env_params.collision_penalty, 0.0
        )
        timeout_penalty = jnp.where(
            jnp.logical_and(time_done, jnp.logical_not(goal_done)),
            env_params.timeout_penalty,
            0.0,
        )

        reward = (
            progress_reward
            - path_penalty
            - steering_penalty
            - env_params.step_penalty
            + goal_bonus
            - crash_penalty
            - timeout_penalty
        )

        info = {
            "time": new_time,
            "speed": new_speed,
            "reverse": new_speed < 0.0,
            "steering_angle": new_steering,
            "distance_to_goal": obs.distance_to_goal,
            "distance_to_path": obs.distance_to_path,
            "heading_alignment": obs.heading_alignment,
            "progress_reward": progress_reward,
            "goal_reached": goal_done,
            "collision": collision_done,
            "timeout": time_done,
        }

        return obs, new_state, reward, done, info
