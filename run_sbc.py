import argparse
import os

import numpy as np

import deep_control as dc


from run_afbc import load_env_make_buffer


def main(args):
    env, buffer = load_env_make_buffer(args, per=False)

    for _ in range(args.seeds):
        # create agent
        agent = dc.sbc.SBCAgent(
            obs_space_size=env.observation_space.shape[0],
            act_space_size=env.action_space.shape[0],
            log_std_low=-10.0,
            log_std_high=2.0,
            ensemble_size=1,
            hidden_size=args.hidden_size,
            beta_dist=False,
        )

        # run afbc
        dc.sbc.sbc(
            agent=agent,
            buffer=buffer,
            test_env=env,
            num_steps_offline=args.steps,
            verbosity=1,
            name=args.name,
            max_episode_steps=args.max_episode_steps,
            log_prob_clip=args.log_prob_clip,
            actor_lr=args.actor_lr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="walker_walk")
    parser.add_argument("--d4rl", action="store_true")
    parser.add_argument("--neorl", action="store_true")
    parser.add_argument("--noisy_dmc", action="store_true")
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--quality", type=str, default="high")
    parser.add_argument("--size", type=int, choices=[99, 999, 9999], default=9999)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--name", type=str, default="afbc")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--vb", type=int, default=0)
    parser.add_argument("--b", type=int, default=0)
    parser.add_argument("--ok", type=int, default=0)
    parser.add_argument("--g", type=int, default=200_000)
    parser.add_argument("--e", type=int, default=200_000)
    parser.add_argument("--log_prob_clip", type=float, default=1000.0)
    args = parser.parse_args()
    main(args)
