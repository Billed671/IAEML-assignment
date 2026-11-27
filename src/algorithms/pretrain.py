"""Pretrain the PPO actor on feedback-controller demonstrations for map 1."""

import argparse
import json
from pathlib import Path

from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.interpolate import CubicSpline

from algorithms.networks import ActorCritic
from algorithms.ppo_utils import normalized_to_env_action, observation_to_vector
from env.robotaxi import RobotaxiEnv


WAYPOINTS = np.array([
    (2.0, 1.0), (2.65, 1.8), (2.7, 3.2), (2.75, 4.7), (3.18, 5.73),
    (4.2, 6.32), (6.0, 6.38), (7.4, 6.15), (8.45, 5.40), (8.95, 4.21),
    (10.0, 3.57), (11.5, 3.44), (13.0, 3.45), (14.45, 2.53),
    (15.65, 1.46), (16.4, 1.24),
])
ROUTE_SPACING = 0.05
ROLLOUT_LIMIT = 1024


def route_samples():
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1))]
    spline = CubicSpline(distance, WAYPOINTS, bc_type=((1, [0.7071, 0.7071]), (1, [1.0, -0.1])))
    positions = spline(np.linspace(0.0, distance[-1], 500))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    uniform = np.arange(0, arc[-1], ROUTE_SPACING)
    return jnp.asarray(np.column_stack([np.interp(uniform, arc, axis) for axis in positions.T]), dtype=jnp.float32)


def guide_action(state, route, params, cruise=3.2):
    delta = route - state.agent_pos
    nearest = jnp.argmin(jnp.sum(delta**2, axis=1))
    lookahead = (0.64 + 0.16 * state.speed) / ROUTE_SPACING
    index = jnp.minimum(nearest + lookahead.astype(jnp.int32), len(route) - 1)
    target = delta[index]
    forward = state.agent_forward_dir
    cross = forward[0] * target[1] - forward[1] * target[0]
    squared_distance = jnp.maximum(jnp.sum(target**2), 0.01)
    steering = jnp.arctan2(2.0 * params.wheelbase * cross, squared_distance)
    steering = jnp.clip(steering, -params.max_steering_angle, params.max_steering_angle)
    speed = cruise * (1.0 - 0.22 * jnp.abs(steering) / params.max_steering_angle)
    acceleration = jnp.clip((speed - state.speed) * 3.0 + 0.15, -params.max_braking, params.max_acceleration)
    return jnp.stack([acceleration, steering])


def _rollout(initial, params, key, action_fn):
    def transition(carry, _):
        state, done, goal, collision, total_reward = carry

        def advance():
            action = action_fn(state)
            _, next_state, reward, ended, info = RobotaxiEnv.step(key, state, action, params)
            return (next_state, ended, info["goal_reached"], info["collision"], total_reward + reward), (state, True)

        return jax.lax.cond(done, lambda: (carry, (state, False)), advance)

    return jax.lax.scan(transition, (initial, False, False, False, 0.0), None, length=ROLLOUT_LIMIT)


def _generate_examples(states, route, params, rng, copies):
    def duplicate(value):
        return jnp.repeat(value, copies, axis=0)

    expanded = jax.tree.map(duplicate, states)
    noise = jax.random.normal(rng, (len(expanded.time), 5))
    clean_index = jnp.arange(len(noise)) % copies == 0
    noise = jnp.where(clean_index[:, None], 0.0, noise)
    rotations = 0.12 * noise[:, 2]
    vectors = expanded.agent_forward_dir
    angles = jnp.arctan2(vectors[:, 1], vectors[:, 0]) + rotations
    expanded = type(expanded)(
        time=expanded.time,
        goal_pos=expanded.goal_pos,
        agent_pos=expanded.agent_pos + 0.10 * noise[:, :2],
        agent_forward_dir=jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1),
        static_obstacles=expanded.static_obstacles,
        kinematic_obstacles=expanded.kinematic_obstacles,
        kinematic_obst_velocities=expanded.kinematic_obst_velocities,
        path_array=expanded.path_array,
        speed=jnp.maximum(0.0, expanded.speed + 0.24 * noise[:, 3]),
        steering_angle=jnp.clip(expanded.steering_angle + 0.13 * noise[:, 4], -params.max_steering_angle, params.max_steering_angle),
    )

    def label(state):
        observation = RobotaxiEnv.get_observation(state, params)
        acceleration, steering = guide_action(state, route, params)
        acceleration_scale = jnp.where(
            acceleration >= 0.0, params.max_acceleration,
            jnp.where(state.speed > 1e-4, params.max_braking, params.max_reverse_acceleration),
        )
        action = jnp.clip(jnp.stack([acceleration / acceleration_scale, steering / params.max_steering_angle]), -1.0, 1.0)
        return observation_to_vector(observation, params), action

    label_batch = jax.jit(jax.vmap(label))
    parts = [label_batch(jax.tree.map(lambda value: value[offset:offset + 256], expanded)) for offset in range(0, len(noise), 256)]
    return tuple(jnp.concatenate(items, axis=0) for items in zip(*parts))


def pretrain(output_path, seed=0, rounds=8, updates=500):
    key = jax.random.PRNGKey(seed)
    params, initial = RobotaxiEnv.init_params(key, map_id=1, fps=60, max_steps=0)
    route = route_samples()
    model = ActorCritic()

    @jax.jit
    def guide_rollout():
        return _rollout(initial, params, key, lambda state: guide_action(state, route, params))

    expert_end, (expert_states, expert_valid) = guide_rollout()
    if not bool(expert_end[2]) or bool(expert_end[3]):
        raise RuntimeError("The demonstration failed to reach the goal safely.")
    print(f"Expert reached the goal in {int(expert_end[0].time)} steps.", flush=True)
    expert_states = jax.tree.map(lambda value: value[expert_valid], expert_states)
    key, dataset_key = jax.random.split(key)
    inputs, targets = _generate_examples(expert_states, route, params, dataset_key, copies=22)
    variables = model.init(key, inputs[0])
    weights = dict(variables["params"])
    weights["log_std"] = jnp.full((2,), -2.7, dtype=jnp.float32)
    learner = TrainState.create(apply_fn=model.apply, params=weights, tx=optax.adam(1e-3))

    @jax.jit
    def fit(learner, observations, actions):
        def loss(weights):
            mean, _, _ = model.apply({"params": weights}, observations)
            error = jnp.tanh(mean) - actions
            return jnp.mean(error[:, 0] ** 2 + 2 * error[:, 1] ** 2)

        loss_value, gradients = jax.value_and_grad(loss)(learner.params)
        return learner.apply_gradients(grads=gradients), loss_value

    @jax.jit
    def evaluate(weights):
        def policy(state):
            observation = RobotaxiEnv.get_observation(state, params)
            features = observation_to_vector(observation, params)
            mean, _, _ = model.apply({"params": weights}, features)
            return normalized_to_env_action(jnp.tanh(mean), params, state.speed)
        return _rollout(initial, params, key, policy)

    for iteration in range(rounds):
        for _update in range(updates):
            key, batch_key = jax.random.split(key)
            indices = jax.random.randint(batch_key, (256,), 0, len(inputs))
            learner, loss = fit(learner, inputs[indices], targets[indices])

        end, (states, valid) = evaluate(learner.params)
        end_state, _, goal, collision, reward = end
        distance = jnp.linalg.norm(end_state.agent_pos - end_state.goal_pos)
        print(
            f"Round {iteration + 1}: loss={float(loss):.4f}, steps={int(end_state.time)}, "
            f"goal={bool(goal)}, collision={bool(collision)}, distance={float(distance):.2f} m",
            flush=True,
        )
        if bool(goal) and not bool(collision):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(serialization.to_bytes(learner.params))
            metadata = {
                "algorithm": "Behavior cloning",
                "map_id": 1,
                "observation_size": int(inputs.shape[-1]),
                "action_dim": 2,
                "hidden_size": 128,
                "training_rounds": iteration + 1,
                "seed": seed,
                "evaluation_steps": int(end_state.time),
                "evaluation_return": float(reward),
            }
            output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return output_path

        key, dataset_key = jax.random.split(key)
        states = jax.tree.map(lambda value: value[valid], states)
        new_inputs, new_targets = _generate_examples(states, route, params, dataset_key, copies=10)
        inputs = jnp.concatenate([inputs, new_inputs])
        targets = jnp.concatenate([targets, new_targets])

    raise RuntimeError("The learned actor did not finish the route within the pretraining budget.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/robotaxi_pretrained.msgpack"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--updates", type=int, default=500)
    arguments = parser.parse_args()
    print(pretrain(arguments.output, arguments.seed, arguments.rounds, arguments.updates))
