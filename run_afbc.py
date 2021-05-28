import argparse
import os

import numpy as np

import rl_utils as dc

from agent import AFBCAgent
from estimator import AdvantageEstimator
from filters import AdvEstimatorFilter, AdvClassifierFilter
from afbc import afbc


def load_env_make_buffer(args, per=True):
    if args.d4rl:
        import gym
        import d4rl

        # load env
        env = gym.make(args.dset)
        # get offline datset
        dset = d4rl.qlearning_dataset(env)
        s, a, r, s1, d = (
            dset["observations"],
            dset["actions"],
            dset["rewards"],
            dset["next_observations"],
            dset["terminals"],
        )
        args.max_episode_steps = 1000

    elif args.neorl:
        import gym
        import neorl

        os.environ["MC1D"] = "False"
        # load env
        env = neorl.make(args.dset)
        dset, _ = env.get_dataset(
            data_type=args.quality, need_val=False, train_num=args.size
        )

        # First, we normalize rewards in [0, 1]
        rew = dset["reward"]
        dset["reward"] = (rew - rew.min()) / (rew.max() - rew.min())

        assert dset["action"].max() <= 1.0
        assert dset["action"].min() >= -1.0
        """
        Normalizing the action space is challenging in this benchmark because
        the datasets contain large outliers.

        env.action_space.high = dset["action"].max(0)
        env.action_space.low = dset["action"].min(0)
        env = dc.envs.NormalizeContinuousActionSpace(env)
        act = dset["action"]
        dset["action"] = 2.0 * ((act - act.min(0)) / (act.max(0) - act.min(0))) - 1.
        """

        # Finally, we normalize the state space
        obs_mean = dset["obs"].mean(0)
        obs_std = dset["obs"].std(0)
        dset["obs"] = (dset["obs"] - obs_mean) / (obs_std + 1e-5)
        dset["next_obs"] = (dset["next_obs"] - obs_mean) / (obs_std + 1e-5)
        env = dc.envs.NormalizeObservationSpace(env, obs_mean, obs_std)

        s, a, r, s1, d = (
            dset["obs"],
            dset["action"],
            dset["reward"],
            dset["next_obs"],
            dset["done"],
        )

        # set custom episode lengths
        if args.dset in ["finance"]:
            args.max_episode_steps = 1_000_000
        elif args.dset in ["citylearn"]:
            args.max_episode_steps = 8759
        elif args.dset in ["ib"]:
            args.max_episode_steps = 1000

    elif args.noisy_dmc:
        from noisy_dmc_loader import load

        os.environ["MC1D"] = "False"
        env, (s, a, r, s1, d) = load(
            args.dset,
            args.vb,
            args.b,
            args.ok,
            args.g,
            args.e,
            base_path=args.noisy_dmc_base_path,
        )
        args.max_episode_steps = 1000

    else:
        raise ValueError("set --noisy_dmc, --neorl or --d4rl flags")

    # create replay buffer
    if per:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        size=s.shape[0],
        state_shape=env.observation_space.shape,
        state_dtype=float,
        action_shape=env.action_space.shape,
    )
    buffer.load_experience(s, a, r, s1, d)

    return env, buffer


def main(args):
    env, buffer = load_env_make_buffer(args)

    for _ in range(args.seeds):
        # create agent
        agent = AFBCAgent(
            obs_space_size=env.observation_space.shape[0],
            act_space_size=env.action_space.shape[0],
            log_std_low=-10.0,
            log_std_high=2.0,
            critics=args.critics,
            beta_dist=False,
            use_popart=args.popart,
            hidden_size=args.hidden_size,
        )

        adv_estimator = AdvantageEstimator(
            actor=agent.actor,
            critics=agent.critics,
            popart=agent.popart,
            method="mean",
            ensembling="mean",
        )

        if args.filter == "classifier":
            adv_filter = AdvClassifierFilter(
                adv_estimator,
                model_t=dc.nets.BigCritic,
                model_kwargs={
                    "state_space_size": env.observation_space.shape[0],
                    "act_space_size": env.action_space.shape[0],
                    "hidden_size": args.hidden_size,
                },
                ensemble_size=5,
                max_unct_anneal=args.classifier_anneal,
                min_conf_anneal=args.classifier_anneal,
            )
        else:
            adv_filter = AdvEstimatorFilter(
                adv_estimator, filter_type=args.filter, beta=args.filter_beta
            )

        # run afbc
        afbc(
            agent=agent,
            adv_filter=adv_filter,
            test_env=env,
            buffer=buffer,
            num_steps=args.steps,
            verbosity=1,
            actor_per=args.actor_per,
            actor_per_type=args.per_type,
            backup_weight_type=args.backup_weight_type,
            backup_weight_temp=args.backup_weight_temp,
            bc_warmup_steps=args.bc_warmup_steps,
            critic_updates_per_step=args.critic_updates_per_step,
            actor_updates_per_step=args.actor_updates_per_step,
            target_critic_ensemble_n=args.target_critic_ensemble_n,
            name=args.name,
            max_episode_steps=args.max_episode_steps,
            actor_lr=args.actor_lr,
            critic_lr=1e-4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="walker_walk")
    parser.add_argument("--noisy_dmc_base_path", type=str)
    parser.add_argument("--d4rl", action="store_true")
    parser.add_argument("--neorl", action="store_true")
    parser.add_argument("--noisy_dmc", action="store_true")
    parser.add_argument("--quality", type=str, default="high")
    parser.add_argument("--size", type=int, choices=[99, 999, 9999], default=9999)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--filter",
        type=str,
        choices=[
            "exp",
            "binary",
            "binary_t",
            "softmax",
            "exp_clamp",
            "exp_norm",
            "classifier",
            "identity",
        ],
        default="binary",
    )
    parser.add_argument("--name", type=str, default="afbc")
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--actor_per", action="store_true")
    parser.add_argument(
        "--backup_weight_type", type=str, choices=[None, "custom", "softmax", "sunrise"]
    )
    parser.add_argument("--backup_weight_temp", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--classifier_anneal", type=int, default=375_000)
    parser.add_argument("--filter_beta", type=float, default=1.0)
    parser.add_argument("--critics", type=int, default=2)
    parser.add_argument("--target_critic_ensemble_n", type=int, default=2)
    parser.add_argument("--critic_updates_per_step", type=int, default=1)
    parser.add_argument("--actor_updates_per_step", type=int, default=1)
    parser.add_argument("--vb", type=int, default=0)
    parser.add_argument("--b", type=int, default=0)
    parser.add_argument("--ok", type=int, default=0)
    parser.add_argument("--g", type=int, default=200_000)
    parser.add_argument("--e", type=int, default=200_000)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--bc_warmup_steps", type=int, default=0)
    parser.add_argument(
        "--per_type", type=str, choices=["binary", "adv"], default="adv",
    )
    args = parser.parse_args()
    main(args)
