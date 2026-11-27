"""PPO training for the continuous-control Robotaxi environment.

Run from the repository root:
    python src/main.py --mode train --total-timesteps 500000

The environment itself is created with max_steps=0 (unlimited). The trainer
uses a separate episode horizon and resets an episode at that horizon so that
an agent which gets stuck cannot remain in the same training episode forever.
"""

import argparse
import json
import math
import csv
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from algorithms.networks import ActorCritic
from algorithms.ppo_utils import (
    normalized_to_env_action,
    observation_to_vector,
    squashed_gaussian_log_prob,
)
from env.robotaxi import RobotaxiEnv


def _batch_tree_from_single(single_state, batch_size):
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
        single_state,
    )


def _select_reset_states(episode_end, stepped_states, initial_states):
    def select(new_value, initial_value):
        mask_shape = (episode_end.shape[0],) + (1,) * (new_value.ndim - 1)
        mask = jnp.reshape(episode_end, mask_shape)
        return jnp.where(mask, initial_value, new_value)

    return jax.tree_util.tree_map(select, stepped_states, initial_states)


def _make_gae_function(gamma, gae_lambda):
    def calculate_gae(rewards, terminated, episode_ends, values, next_values):
        # Scan backward. True terminations do not bootstrap; training-only
        # truncations bootstrap from the final observation but stop recursion.
        inputs = (
            rewards[::-1],
            terminated[::-1],
            episode_ends[::-1],
            values[::-1],
            next_values[::-1],
        )

        def gae_step(carry, transition):
            reward, terminal, episode_end, value, next_value = transition
            delta = reward + gamma * next_value * (1.0 - terminal) - value
            advantage = delta + gamma * gae_lambda * (1.0 - episode_end) * carry
            return advantage, advantage

        initial_advantage = jnp.zeros_like(rewards[0])
        _, reversed_advantages = jax.lax.scan(
            gae_step, initial_advantage, inputs
        )
        advantages = reversed_advantages[::-1]
        return advantages, advantages + values

    return jax.jit(calculate_gae)


def _make_update_function(
    model,
    num_samples,
    num_minibatches,
    update_epochs,
    clip_epsilon,
    value_coefficient,
    entropy_coefficient,
):
    minibatch_size = num_samples // num_minibatches

    def loss_function(params, batch):
        mean, log_std, values = model.apply({"params": params}, batch["observations"])
        new_log_probs = squashed_gaussian_log_prob(
            mean, log_std, batch["raw_actions"]
        )
        log_ratio = new_log_probs - batch["old_log_probs"]
        ratio = jnp.exp(log_ratio)

        unclipped_policy = -batch["advantages"] * ratio
        clipped_policy = -batch["advantages"] * jnp.clip(
            ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
        )
        policy_loss = jnp.mean(jnp.maximum(unclipped_policy, clipped_policy))

        clipped_values = batch["old_values"] + jnp.clip(
            values - batch["old_values"], -clip_epsilon, clip_epsilon
        )
        value_error = jnp.square(values - batch["returns"])
        clipped_value_error = jnp.square(clipped_values - batch["returns"])
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_error, clipped_value_error))

        # Gaussian entropy is an inexpensive approximation for tanh-Gaussian
        # entropy and is commonly used as the PPO exploration bonus.
        entropy = jnp.mean(
            jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
        )
        total_loss = (
            policy_loss
            + value_coefficient * value_loss
            - entropy_coefficient * entropy
        )
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        clip_fraction = jnp.mean(
            (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
        )
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        return total_loss, metrics

    def update(train_state, key, batch):
        def minibatch_update(state, indices):
            minibatch = jax.tree_util.tree_map(
                lambda value: value[indices], batch
            )
            (_, metrics), gradients = jax.value_and_grad(
                loss_function, has_aux=True
            )(state.params, minibatch)
            state = state.apply_gradients(grads=gradients)
            return state, metrics

        def epoch_update(carry, _):
            state, epoch_key = carry
            epoch_key, permutation_key = random.split(epoch_key)
            permutation = random.permutation(permutation_key, num_samples)
            minibatch_indices = jnp.reshape(
                permutation, (num_minibatches, minibatch_size)
            )
            state, metrics = jax.lax.scan(
                minibatch_update, state, minibatch_indices
            )
            return (state, epoch_key), metrics

        (train_state, key), epoch_metrics = jax.lax.scan(
            epoch_update, (train_state, key), None, length=update_epochs
        )
        mean_metrics = jax.tree_util.tree_map(
            lambda value: jnp.mean(value), epoch_metrics
        )
        return train_state, key, mean_metrics

    return jax.jit(update)


def train_ppo(
    map_id=1,
    total_timesteps=500_000,
    num_envs=8,
    rollout_steps=128,
    episode_horizon=1200,
    learning_rate=3e-4,
    update_epochs=4,
    num_minibatches=8,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coefficient=0.01,
    value_coefficient=0.5,
    max_grad_norm=0.5,
    seed=0,
    checkpoint_path="models/robotaxi_ppo.msgpack",
    save_every=10,
    artifacts_dir="artifacts",
    initial_checkpoint=None,
):
    """Train a PPO agent and save its parameters to a Flax msgpack checkpoint."""
    if total_timesteps <= 0 or num_envs <= 0 or rollout_steps <= 0:
        raise ValueError("timesteps, num_envs and rollout_steps must be positive")
    if episode_horizon <= 0:
        raise ValueError("episode_horizon must be positive")
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    if map_id not in (0, 1, 2):
        raise ValueError("map_id must be 0, 1 or 2")

    rollout_batch_size = num_envs * rollout_steps
    if rollout_batch_size % num_minibatches != 0:
        raise ValueError(
            "num_envs * rollout_steps must be divisible by num_minibatches"
        )
    num_updates = max(1, math.ceil(total_timesteps / rollout_batch_size))

    key = random.PRNGKey(seed)
    key, env_key, model_key = random.split(key, 3)
    env_params, init_state = RobotaxiEnv.init_params(
        key=env_key,
        map_id=map_id,
        max_steps=0,  # Keep the simulation's own episode length unlimited.
        path_length=100,
        fps=60,
        perception_radius=5.0,
        num_ray_sensors=32,
    )

    reset_states = _batch_tree_from_single(init_state, num_envs)
    single_observation = RobotaxiEnv.get_observation(init_state, env_params)
    observation_vector = observation_to_vector(single_observation, env_params)
    observation_size = int(observation_vector.shape[-1])
    model = ActorCritic(action_dim=2, hidden_size=128)
    model_variables = model.init(model_key, observation_vector)
    model_parameters = model_variables["params"]
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        metadata_path = initial_checkpoint.with_suffix(initial_checkpoint.suffix + ".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("map_id", map_id) != map_id:
                raise ValueError("Initial checkpoint was trained on a different map")
        model_parameters = serialization.from_bytes(model_parameters, initial_checkpoint.read_bytes())
        print(f"Initialized PPO from {initial_checkpoint}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=model_parameters,
        tx=optimizer,
    )

    current_states = reset_states
    initial_observations = jax.vmap(
        lambda state: RobotaxiEnv.get_observation(state, env_params)
    )(current_states)
    episode_returns = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_lengths = jnp.zeros((num_envs,), dtype=jnp.int32)

    # The environment is deterministic for a fixed action sequence. Randomness
    # comes from the PPO policy; every parallel worker starts at the same map.
    step_key = random.PRNGKey(seed + 1)

    def collect_rollout(policy_params, states, observations, rng, returns, lengths):
        def collect_one(carry, _):
            states, observations, rng, returns, lengths = carry
            obs_vectors = observation_to_vector(observations, env_params)
            means, log_stds, values = model.apply(
                {"params": policy_params}, obs_vectors
            )

            rng, action_key = random.split(rng)
            noise = random.normal(action_key, means.shape)
            raw_actions = means + jnp.exp(log_stds) * noise
            normalized_actions = jnp.tanh(raw_actions)
            environment_actions = normalized_to_env_action(
                normalized_actions, env_params, observations.speed
            )
            log_probs = squashed_gaussian_log_prob(
                means, log_stds, raw_actions
            )

            stepped_observations, stepped_states, rewards, terminated, info = jax.vmap(
                lambda state, action: RobotaxiEnv.step(
                    step_key, state, action, env_params
                )
            )(states, environment_actions)

            # Bootstrap a training-horizon truncation from its final observation.
            final_vectors = observation_to_vector(stepped_observations, env_params)
            _, _, next_values = model.apply(
                {"params": policy_params}, final_vectors
            )

            next_lengths = lengths + 1
            truncated = jnp.logical_and(
                jnp.logical_not(terminated), next_lengths >= episode_horizon
            )
            episode_ends = jnp.logical_or(terminated, truncated)
            completed_returns = returns + rewards
            reported_returns = jnp.where(episode_ends, completed_returns, 0.0)
            reported_goals = jnp.where(
                episode_ends, info["goal_reached"].astype(jnp.float32), 0.0
            )
            reported_collisions = jnp.where(
                episode_ends, info["collision"].astype(jnp.float32), 0.0
            )

            next_states = _select_reset_states(
                episode_ends, stepped_states, reset_states
            )
            next_observations = jax.vmap(
                lambda state: RobotaxiEnv.get_observation(state, env_params)
            )(next_states)
            returns = jnp.where(episode_ends, 0.0, completed_returns)
            lengths = jnp.where(episode_ends, 0, next_lengths)

            transition = {
                "observations": obs_vectors,
                "raw_actions": raw_actions,
                "old_log_probs": log_probs,
                "rewards": rewards,
                "terminated": terminated.astype(jnp.float32),
                "episode_ends": episode_ends.astype(jnp.float32),
                "values": values,
                "next_values": next_values,
                "completed_returns": reported_returns,
                "completed_goals": reported_goals,
                "completed_collisions": reported_collisions,
            }
            next_carry = (
                next_states,
                next_observations,
                rng,
                returns,
                lengths,
            )
            return next_carry, transition

        carry = (states, observations, rng, returns, lengths)
        return jax.lax.scan(collect_one, carry, None, length=rollout_steps)

    collect_rollout_jit = jax.jit(collect_rollout)
    calculate_gae = _make_gae_function(gamma, gae_lambda)
    update_policy = _make_update_function(
        model=model,
        num_samples=rollout_batch_size,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(artifacts_dir) / f"ppo_map{map_id}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_checkpoint = Path(checkpoint_path)
    checkpoint_path = run_dir / requested_checkpoint.name
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    metrics_path = run_dir / "training_metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=[
        "update", "environment_steps", "mean_episode_return", "completed_episodes",
        "goals", "collisions", "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_fraction",
    ])
    metrics_writer.writeheader()
    total_environment_steps = 0

    print("Starting PPO training")
    print(
        f"Map: {map_id} | parallel environments: {num_envs} | "
        f"rollout: {rollout_steps} | planned steps: {num_updates * rollout_batch_size:,}"
    )
    print(
        f"Environment max_steps=0 (unlimited); trainer episode horizon="
        f"{episode_horizon} steps"
    )

    for update_index in range(1, num_updates + 1):
        (current_states, initial_observations, key, episode_returns, episode_lengths), rollout = (
            collect_rollout_jit(
                train_state.params,
                current_states,
                initial_observations,
                key,
                episode_returns,
                episode_lengths,
            )
        )

        advantages, returns = calculate_gae(
            rollout["rewards"],
            rollout["terminated"],
            rollout["episode_ends"],
            rollout["values"],
            rollout["next_values"],
        )
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + 1e-8
        )

        batch = {
            "observations": rollout["observations"].reshape(
                (rollout_batch_size, observation_size)
            ),
            "raw_actions": rollout["raw_actions"].reshape(
                (rollout_batch_size, 2)
            ),
            "old_log_probs": rollout["old_log_probs"].reshape(
                (rollout_batch_size,)
            ),
            "advantages": advantages.reshape((rollout_batch_size,)),
            "returns": returns.reshape((rollout_batch_size,)),
            "old_values": rollout["values"].reshape((rollout_batch_size,)),
        }
        key, update_key = random.split(key)
        train_state, key, train_metrics = update_policy(
            train_state, update_key, batch
        )

        total_environment_steps += rollout_batch_size
        completed_returns = np.asarray(rollout["completed_returns"])
        finished_mask = np.asarray(rollout["episode_ends"], dtype=bool)
        completed_returns = completed_returns[finished_mask]
        completed_goals = float(jnp.sum(rollout["completed_goals"]))
        completed_collisions = float(jnp.sum(rollout["completed_collisions"]))
        mean_episode_return = (
            float(np.mean(completed_returns)) if completed_returns.size else float("nan")
        )
        metric_values = {
            name: float(value) for name, value in jax.device_get(train_metrics).items()
        }

        metrics_writer.writerow({
            "update": update_index,
            "environment_steps": total_environment_steps,
            "mean_episode_return": mean_episode_return,
            "completed_episodes": int(completed_returns.size),
            "goals": completed_goals,
            "collisions": completed_collisions,
            **metric_values,
        })
        metrics_file.flush()

        print(
            f"Update {update_index:>4}/{num_updates} | "
            f"steps {total_environment_steps:>9,} | "
            f"mean return {mean_episode_return:>8.3f} | "
            f"goals {completed_goals:.0f} | collisions {completed_collisions:.0f} | "
            f"policy loss {metric_values['policy_loss']:.4f} | "
            f"value loss {metric_values['value_loss']:.4f}"
        )

        if update_index % save_every == 0 or update_index == num_updates:
            checkpoint_path.write_bytes(serialization.to_bytes(train_state.params))
            metadata = {
                "algorithm": "PPO",
                "map_id": int(map_id),
                "observation_size": observation_size,
                "action_dim": 2,
                "hidden_size": 128,
                "environment_steps": total_environment_steps,
                "episode_horizon": int(episode_horizon),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved checkpoint: {checkpoint_path}")

    metrics_file.close()
    run_metadata = {
        "algorithm": "PPO", "map_id": int(map_id), "seed": int(seed),
        "total_timesteps_requested": int(total_timesteps),
        "total_environment_steps": int(total_environment_steps),
        "num_envs": int(num_envs), "rollout_steps": int(rollout_steps),
        "episode_horizon": int(episode_horizon), "learning_rate": float(learning_rate),
        "checkpoint": checkpoint_path.name,
        "initial_checkpoint": initial_checkpoint.name if initial_checkpoint else None,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(f"Training finished. Artifacts: {run_dir}")
    return checkpoint_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PPO for RobotaxiEnv")
    parser.add_argument("--map-id", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--episode-horizon", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument(
        "--checkpoint", type=str, default="models/robotaxi_ppo.msgpack"
    )
    parser.add_argument("--init-checkpoint", type=str)
    return parser


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    train_ppo(
        map_id=arguments.map_id,
        total_timesteps=arguments.total_timesteps,
        num_envs=arguments.num_envs,
        rollout_steps=arguments.rollout_steps,
        episode_horizon=arguments.episode_horizon,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        checkpoint_path=arguments.checkpoint,
        artifacts_dir=arguments.artifacts_dir,
        initial_checkpoint=arguments.init_checkpoint,
    )
