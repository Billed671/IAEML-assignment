"""Shared observation scaling and continuous-action helpers for PPO."""

import jax.numpy as jnp


def observation_to_vector(observation, env_params):
    """Convert a RobotaxiObservation into a compact, scaled float vector.

    Distances and speed are scaled to roughly unit range. Direction vectors,
    heading alignment and steering are already naturally bounded.
    This function supports both a single observation and a batch of them.
    """
    perception_scale = max(float(env_params.perception_radius), 1e-6)
    speed_scale = max(float(env_params.max_speed), 1e-6)
    steering_scale = max(float(env_params.max_steering_angle), 1e-6)
    map_height, map_width = env_params.map_height_width
    map_diagonal = max((float(map_height) ** 2 + float(map_width) ** 2) ** 0.5, 1e-6)

    rays = observation.collision_rays / perception_scale
    rays = jnp.reshape(rays, rays.shape[:-2] + (-1,))

    vector = jnp.concatenate(
        [
            jnp.expand_dims(observation.distance_to_path / perception_scale, -1),
            observation.direction_of_path,
            rays,
            jnp.expand_dims(observation.speed / speed_scale, -1),
            jnp.expand_dims(observation.steering_angle / steering_scale, -1),
            jnp.expand_dims(observation.distance_to_goal / map_diagonal, -1),
            jnp.expand_dims(observation.heading_alignment, -1),
            observation.heading_direction,
        ],
        axis=-1,
    )
    return jnp.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)


def normalized_to_env_action(normalized_action, env_params, speed):
    """Map tanh-bounded policy actions to acceleration and steering in SI units.

    Braking has a different limit from forward acceleration, so the first
    action component uses the corresponding scale based on its sign.
    """
    normalized_acceleration = normalized_action[..., 0]
    speed = jnp.asarray(speed)
    positive_scale = jnp.where(
        speed < -1e-4, env_params.max_braking, env_params.max_acceleration
    )
    negative_scale = jnp.where(
        speed > 1e-4, env_params.max_braking, env_params.max_reverse_acceleration
    )
    acceleration_scale = jnp.where(
        normalized_acceleration >= 0.0, positive_scale, negative_scale
    )
    acceleration = normalized_acceleration * acceleration_scale
    steering = normalized_action[..., 1] * env_params.max_steering_angle
    return jnp.stack([acceleration, steering], axis=-1)


def squashed_gaussian_log_prob(mean, log_std, raw_action):
    """Log probability of tanh-transformed diagonal Gaussian actions."""
    std = jnp.exp(log_std)
    gaussian_log_prob = -0.5 * (
        jnp.square((raw_action - mean) / std)
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    gaussian_log_prob = jnp.sum(gaussian_log_prob, axis=-1)

    squashed_action = jnp.tanh(raw_action)
    log_tanh_jacobian = jnp.sum(
        jnp.log(1.0 - jnp.square(squashed_action) + 1e-6), axis=-1
    )
    return gaussian_log_prob - log_tanh_jacobian
