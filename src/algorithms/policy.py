"""Load a trained PPO checkpoint and build a simulator policy."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from algorithms.networks import ActorCritic
from algorithms.ppo_utils import observation_to_vector, normalized_to_env_action
from env.robotaxi import RobotaxiEnv


def load_policy(checkpoint_path, env_params, init_state, map_id=None, hidden_size=128):
    """Restore a saved policy and return ``policy(state) -> action``.

    The returned function has the signature expected by ``PygameFrontend``.
    Checkpoints contain model parameters only; the network architecture is
    defined by ``ActorCritic`` and the observation scaling by this environment.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train an agent first."
        )

    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_map_id = int(metadata.get("map_id", -1))
        if checkpoint_map_id < 0:
            raise ValueError("Checkpoint metadata does not contain a valid map_id")
        if map_id is not None and checkpoint_map_id != int(map_id):
            raise ValueError(
                f"This checkpoint was trained on map {checkpoint_map_id}, "
                f"but map {map_id} was selected. Use --map-id {checkpoint_map_id} "
                "or train a checkpoint for the selected map."
            )

    initial_observation = RobotaxiEnv.get_observation(init_state, env_params)
    dummy_vector = observation_to_vector(initial_observation, env_params)

    model = ActorCritic(action_dim=2, hidden_size=hidden_size)
    variables = model.init(jax.random.PRNGKey(0), dummy_vector)
    params = serialization.from_bytes(
        variables["params"], checkpoint_path.read_bytes()
    )

    @jax.jit
    def policy(state):
        observation = RobotaxiEnv.get_observation(state, env_params)
        observation_vector = observation_to_vector(observation, env_params)
        mean, _, _ = model.apply({"params": params}, observation_vector)
        normalized_action = jnp.tanh(mean)
        return normalized_to_env_action(
            normalized_action, env_params, observation.speed
        )

    return policy
