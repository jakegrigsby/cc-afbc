import argparse
import copy
import math
import os
from itertools import chain
import random

import numpy as np
import tensorboardX
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as pyd
import tqdm

from rl_utils import nets, run, utils, device, replay

import filters


def afbc(
    agent,
    adv_filter,
    buffer,
    test_env,
    num_steps=750_000,
    actor_updates_per_step=1,
    critic_updates_per_step=1,
    target_critic_ensemble_n=2,
    bc_warmup_steps=0,
    adv_updates_per_step=5,
    actor_per=True,
    actor_per_type="adv",
    batch_size=512,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    gamma=0.99,
    backup_weight_type=None,
    backup_weight_temp=10.0,
    max_episode_steps=1000,
    eval_interval=5000,
    eval_episodes=10,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=2,
    save_interval=50_000,
    name="afbc_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    log_interval=1000,
    verbosity=0,
    **kwargs,
):

    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})
        writer.add_hparams(agent.get_hparams(), {})
        writer.add_hparams(adv_filter.get_hparams(), {})
        writer.add_hparams(adv_filter.adv_estimator.get_hparams(), {})

    qprint = lambda x: print(x) if verbosity else None
    qprint(" ----- AFBC -----")
    for key, val in agent.get_hparams().items():
        qprint(f"\t{key} : {val}")
    for key, val in adv_filter.get_hparams().items():
        qprint(f"\t{key} : {val}")
    qprint(f"\tactor PER: {actor_per}")
    qprint(f"\tactor PER type: {actor_per_type}")
    qprint(f"\tcritic updates per step: {critic_updates_per_step}")
    qprint(" -----       -----")

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
        utils.hard_update(target_critic, agent_critic)
    target_agent.train()
    critic_optimizer = torch.optim.Adam(
        chain(*(critic.parameters() for critic in agent.critics)),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )

    if isinstance(adv_filter, filters.AdvEstimatorFilter):
        using_classifier = False
    elif isinstance(adv_filter, filters.AdvClassifierFilter):
        using_classifier = True
    else:
        raise ValueError(f"Unrecognized filter type {adv_filter}")

    if using_classifier:
        adv_optimizer = torch.optim.Adam(
            chain(*(a.parameters() for a in adv_filter.classifiers)),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )

    ###################
    ## TRAINING LOOP ##
    ###################
    progress_bar = lambda x: tqdm.tqdm(range(x)) if verbosity else range(x)
    done = True

    for step in progress_bar(num_steps):
        for critic_update in range(critic_updates_per_step):
            # critic update
            critic_logs = learn_critics(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                adv_filter=adv_filter,
                critic_clip=critic_clip,
                target_critic_ensemble_n=target_critic_ensemble_n,
                weighted_bellman_temp=backup_weight_temp,
                weight_type=backup_weight_type,
                log=(step % log_interval == 0),
                update_priorities=actor_per,
                priorities_type=actor_per_type,
            )
        if step % log_interval == 0:
            for key, val in critic_logs.items():
                writer.add_scalar(key, val, step)

        # move target model towards training model
        if step % target_delay == 0:
            for agent_critic, target_critic in zip(agent.critics, target_agent.critics):
                utils.soft_update(target_critic, agent_critic, tau)

        # actor update
        for actor_update in range(actor_updates_per_step):
            actor_logs = learn_actor(
                step_num=step,
                buffer=buffer,
                agent=agent,
                adv_filter=adv_filter,
                filter_=(step > bc_warmup_steps),
                actor_optimizer=actor_optimizer,
                batch_size=batch_size,
                actor_clip=actor_clip,
                per=actor_per,
                priorities_type=actor_per_type,
            )

        if step % log_interval == 0:
            for key, val in actor_logs.items():
                writer.add_scalar(key, val, step)

        # adv update
        if using_classifier:
            for adv_update in range(adv_updates_per_step):
                adv_logs = learn_adv(buffer, adv_filter, adv_optimizer, batch_size,)

            if step % log_interval == 0:
                for key, val in adv_logs.items():
                    writer.add_scalar(key, val, step)

        if (step % eval_interval == 0) or (step == num_steps - 1):
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def learn_critics(
    buffer,
    target_agent,
    agent,
    critic_optimizer,
    batch_size,
    gamma,
    adv_filter,
    critic_clip,
    target_critic_ensemble_n,
    weighted_bellman_temp,
    weight_type,
    log=False,
    update_priorities=True,
    priorities_type="binary",
):

    assert isinstance(buffer, replay.PrioritizedReplayBuffer)
    batch, idxs = buffer.sample_uniform(batch_size)
    s, a, r, s1, d = batch
    state_batch = s.to(device)
    action_batch = a.to(device)
    reward_batch = r.to(device)
    next_state_batch = s1.to(device)
    done_batch = d.to(device)

    agent.train()

    critic_logs = {}

    ###################
    ## CRITIC UPDATE ##
    ###################

    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.sample()
        # target critic ensembling
        target_critic_ensemble = random.sample(
            target_agent.critics, target_critic_ensemble_n
        )
        target_critic_ensemble_preds = torch.stack(
            [critic(next_state_batch, action_s1) for critic in target_critic_ensemble],
            dim=0,
        )
        s1_q_pred = target_critic_ensemble_preds.min(0).values
        if agent.popart:
            # denormalize q pred
            s1_q_pred = agent.popart(s1_q_pred, normalized=False)
        td_target = reward_batch + gamma * (1.0 - done_batch) * s1_q_pred

        if agent.popart:
            # update popart stats
            agent.popart.update_stats(td_target)
            # normalize TD target
            td_target = agent.popart.normalize_values(td_target)

        if weight_type is not None:
            with torch.no_grad():
                if weight_type == "custom":
                    target_q_std = target_critic_ensemble_preds.std(0)
                    weights = torch.sigmoid(-target_q_std * weighted_bellman_temp) + 0.5
                    critic_logs["bellman_weight"] = weights.mean().item()
                elif weight_type == "softmax":
                    target_q_std = target_critic_ensemble_preds.std(0)
                    weights = batch_size * F.softmax(
                        -target_q_std * weighted_bellman_temp, dim=0
                    )
                    critic_logs["bellman_weight_max"] = weights.max(0).values.item()
                    critic_logs["bellman_weight_min"] = weights.min(0).values.item()
                elif weight_type == "sunrise":
                    # compute weighted bellman coeffs using SUNRISE Eq 5
                    target_q_std = torch.stack(
                        [q(state_batch, action_batch) for q in target_agent.critics],
                        dim=0,
                    ).std(0)
                    weights = torch.sigmoid(-target_q_std * weighted_bellman_temp) + 0.5
                    critic_logs["bellman_weight"] = weights.mean().item()

    critic_loss = 0.0
    for i, critic in enumerate(agent.critics):
        agent_critic_pred = critic(state_batch, action_batch)
        if agent.popart:
            agent_critic_pred = agent.popart(agent_critic_pred)
        td_error = td_target - agent_critic_pred
        if weight_type is not None:
            critic_loss += 0.5 * weights * (td_error ** 2)
        else:
            critic_loss += 0.5 * (td_error ** 2)
    critic_logs["td_target"] = td_target.mean().item()

    critic_loss = critic_loss.mean() / len(agent.critics)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(critic.parameters() for critic in agent.critics)), critic_clip,
        )
    critic_optimizer.step()

    if update_priorities:
        new_adv = adv_filter.adv_estimator(state_batch, action_batch)
        if priorities_type == "binary":
            new_priorities = (new_adv >= 0.0).float()
        elif priorities_type == "adv":
            new_priorities = F.relu(new_adv)
        new_priorities = (new_priorities + 1e-5).squeeze(1).cpu().detach().numpy()
        buffer.update_priorities(idxs, new_priorities)

    return critic_logs


def learn_actor(
    step_num,
    buffer,
    agent,
    adv_filter,
    actor_optimizer,
    batch_size,
    actor_clip,
    filter_=True,
    per=True,
    priorities_type="binary",
):
    agent.train()
    assert isinstance(buffer, replay.PrioritizedReplayBuffer)
    actor_logs = {}

    if per:
        batch, priority_weights, priority_idxs = buffer.sample(batch_size)
        priority_weights = priority_weights.unsqueeze(1).to(device)
    else:
        batch, _ = buffer.sample_uniform(batch_size)
    s, a, *_ = batch
    s = s.to(device)
    a = a.to(device)

    dist = agent.actor(s)
    logp_a = dist.log_prob(a).sum(-1, keepdim=True).clamp(-1000.0, 1000.0)
    if filter_:
        with torch.no_grad():
            filter_vals = adv_filter(s, a, step_num)
    else:
        filter_vals = torch.ones_like(logp_a)
    actor_logs["filter_vals_max"] = filter_vals.max().item()
    actor_logs["filter_vals_mean"] = filter_vals.mean().item()
    actor_logs["filter_vals_min"] = filter_vals.min().item()
    actor_loss = -(filter_vals * logp_a).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
    actor_optimizer.step()

    if per:
        new_adv = adv_filter.adv_estimator(s, a)
        if priorities_type == "binary":
            new_priorities = (new_adv >= 0.0).float()
        elif priorities_type == "adv":
            new_priorities = F.relu(new_adv)
        new_priorities = (new_priorities + 1e-5).squeeze(1).cpu().detach().numpy()
        buffer.update_priorities(priority_idxs, new_priorities)
        actor_logs.update(
            {
                "max_priority": priority_weights.max().item(),
                "min_priority": priority_weights.min().item(),
                "mean_priority": priority_weights.mean().item(),
            }
        )
    actor_logs["actor_loss"] = actor_loss.item()
    return actor_logs


def learn_adv(
    buffer, adv_filter, adv_optimizer, batch_size,
):
    batch, _ = buffer.sample_uniform(batch_size)
    s, a, *_ = batch
    s = s.to(device)
    a = a.to(device)
    with torch.no_grad():
        # labels are binary 1 = positive advantage, 0 = negative,
        # based on standard CRR/AWAC estimates.
        labels = adv_filter.generate_labels(s, a)
    loss = 0.0
    for classifier in adv_filter.classifiers:
        # binary classification loss. learn to identify positive
        # advantages, but don't worry about predicting the magnitude.
        # this makes it easier to judge the confidence of these
        # classifications, which is what we really care about.
        preds = torch.sigmoid(classifier(s, a))
        loss += F.binary_cross_entropy(preds, labels)
    adv_optimizer.zero_grad()
    loss /= len(adv_filter.classifiers)
    loss.backward()
    adv_optimizer.step()
    return {"adv_loss": loss.item()}
