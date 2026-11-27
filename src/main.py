"""Entry point for manual driving, PPO training and policy evaluation."""

import argparse
from datetime import datetime
from pathlib import Path

import jax

from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="IAEML robotaxi simulator")
    parser.add_argument(
        "--mode",
        choices=("play", "train", "eval"),
        default="play",
        help="manual simulator, PPO training, or trained-policy evaluation",
    )
    parser.add_argument("--map-id", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--episode-horizon", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument(
        "--no-record", action="store_true",
        help="disable simulator video recording",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/robotaxi_ppo.msgpack"
    )
    return parser.parse_args()


def run_simulator(args):
    from utils.renderer import PygameFrontend

    key = jax.random.PRNGKey(args.seed)
    env_params, init_state = RobotaxiEnv.init_params(
        key=key,
        map_id=args.map_id,
        max_steps=1024,
        discretization_scale=1,
        path_length=100,
        fps=60,
        perception_radius=5.0,
        num_ray_sensors=32,
    )

    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)
    agent_fn = None
    if args.mode == "eval":
        from algorithms.policy import load_policy

        agent_fn = load_policy(
            args.checkpoint, env_params, init_state, map_id=args.map_id
        )

    frontend = PygameFrontend(
        env,
        env_params,
        init_state,
        eval_mode=(args.mode == "eval"),
        agent_fn=agent_fn,
        video_path=(
            None if args.no_record else
            Path(args.artifacts_dir) / "simulator" /
            datetime.now().strftime("%Y%m%d_%H%M%S") / "episode.mp4"
        ),
        record_video=not args.no_record,
    )
    frontend.run()


def main():
    args = parse_args()
    if args.mode == "train":
        from algorithms.ppo import train_ppo

        train_ppo(
            map_id=args.map_id,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            episode_horizon=args.episode_horizon,
            learning_rate=args.learning_rate,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            artifacts_dir=args.artifacts_dir,
        )
        return

    run_simulator(args)


if __name__ == "__main__":
    main()
