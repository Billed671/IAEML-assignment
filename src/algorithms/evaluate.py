"""Evaluate one deterministic Robotaxi episode and optionally record it."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from algorithms.policy import load_policy
from env.robotaxi import RobotaxiEnv


def evaluate(checkpoint, map_id=1, max_steps=1024, seed=0, video=None):
    if max_steps <= 0:
        raise ValueError("Evaluation requires a positive step limit")

    key = jax.random.PRNGKey(seed)
    params, initial = RobotaxiEnv.init_params(key, map_id=map_id, fps=60, max_steps=max_steps)
    policy = load_policy(checkpoint, params, initial, map_id=map_id)
    _, state = RobotaxiEnv.reset(key, params, initial)
    reward_sum = 0.0
    minimum_distance = float(np.linalg.norm(initial.goal_pos - initial.agent_pos))
    frontend = None
    if video is not None:
        import pygame
        from utils.renderer import PygameFrontend

        frontend = PygameFrontend(
            RobotaxiEnv, params, initial, eval_mode=True, agent_fn=policy,
            video_path=video, record_video=True,
        )

    @jax.jit
    def transition(state):
        return RobotaxiEnv.step(key, state, policy(state), params)

    try:
        for _step in range(max_steps):
            observation, state, reward, done, info = transition(state)
            distance = float(info["distance_to_goal"])
            minimum_distance = min(minimum_distance, distance)
            reward_sum += float(reward)
            if frontend is not None:
                pygame.event.pump()
                frontend.obs, frontend.state, frontend.reward, frontend.info = observation, state, reward, info
                frontend.draw()
            if bool(done):
                break

        if frontend is not None:
            for _frame in range(params.fps // 2):
                frontend.draw()
    finally:
        if frontend is not None:
            frontend.video_writer.close()
            pygame.quit()

    return {
        "map_id": map_id,
        "seed": seed,
        "goal_reached": bool(info["goal_reached"]),
        "collision": bool(info["collision"]),
        "timeout": bool(info["timeout"]),
        "steps": int(state.time),
        "simulated_seconds": int(state.time) / params.fps,
        "distance_to_goal": distance,
        "closest_distance": minimum_distance,
        "episode_return": reward_sum,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/robotaxi_ppo.msgpack"))
    parser.add_argument("--map-id", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    outcome = evaluate(args.checkpoint, args.map_id, args.max_steps, args.seed, args.video)
    summary = json.dumps(outcome, indent=2)
    print(summary)
    if args.results is not None:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(summary + "\n", encoding="utf-8")
    raise SystemExit(0 if outcome["goal_reached"] and not outcome["collision"] else 1)
