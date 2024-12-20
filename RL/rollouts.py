"""
DO NOT MODIFY BESIDES HYPERPARAMETERS
"""
import copy

import torch
import numpy as np
from networks import PGPolicy
from utils import device, eval_mode

import gymnasium as gym

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)


def rollout(
        env,
        agent,
        replay_buffer=None,
        render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []
    done = False
    o, _ = env.reset()
    episode_step = 0
    if render:
        env.render()

    while not done:
        o_for_agent = o.copy()
        if isinstance(agent, PGPolicy):
            o_for_agent = torch.from_numpy(o_for_agent[None]).to(device).float()
            action, _, _ = agent(o_for_agent)
            action = action.cpu().detach().numpy()[0]
        else:
            with eval_mode(agent):
                action = agent.act(o_for_agent, sample=False)

        # Step the simulation forward
        next_o, r, terminated, truncated, _ = env.step(copy.deepcopy(action))
        # print("reward: ", r)
        done = terminated or truncated
        done_no_max = 0 if episode_step + 1 == 300 else done  # changed for this!!
        if replay_buffer is not None:
            replay_buffer.add(o, action, r, next_o, done, done_no_max)

        # Render the environment
        if render:
            env.render()
        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        if terminated:
            print("DONE")
        if done:
            break
        o = next_o
        episode_step += 1

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images=np.array(images)
    )


def evaluate(env, policy, num_validation_runs=10, render=False):
    import random
    import metaworld
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    ml1 = metaworld.ML1('reach-v2')
    for k in range(num_validation_runs):

        random.seed(0)
        task = random.choice(ml1.train_tasks)
        env_evaluation = env
        env_evaluation.set_task(task)  # Set task
        path = rollout(
            env_evaluation,
            policy,
            render=render)
        if env.spec.id == 'Ant-v5':
            success = np.sum(path['rewards']) > 0
        elif env.spec.id == 'InvertedPendulum-v5':
            success = len(path['dones']) == env.spec.max_episode_steps

        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count / num_validation_runs)
    print("Average reward (success only): ", rewards_suc / max(success_count, 1))
    print("Average reward (all): ", rewards_all / num_validation_runs)


def evaluate_agent(eval_env, agent, step, verbose=False, num_episodes=10):
    import random
    import metaworld
    average_episode_reward = 0
    av_ep_ln = 0
    for _ in range(num_episodes):

        max_episode_steps = 300
        random.seed(0)

        result = rollout(eval_env, agent)
        ep_ln = len(result['rewards'])
        episode_reward = np.sum(result['rewards'])
        average_episode_reward += episode_reward
        av_ep_ln += ep_ln
        if verbose:
            print(f"eval episode reward {episode_reward}, episode length {ep_ln}")
    frames_eval = rollout_frames(eval_env, agent)
    average_episode_reward /= num_episodes
    av_ep_ln /= num_episodes
    print(f"eval step {step}, average episode reward {average_episode_reward}, average episode length {av_ep_ln}")
    return frames_eval


def rollout_frames(
        env,
        agent,
):
    # Collect the following data
    frames = []
    done = False
    o, _ = env.reset()
    frames.append(env.render())

    while not done:
        o_for_agent = o.copy()
        if isinstance(agent, PGPolicy):
            o_for_agent = torch.from_numpy(o_for_agent[None]).to(device).float()
            action, _, _ = agent(o_for_agent)
            action = action.cpu().detach().numpy()[0]
        else:
            with eval_mode(agent):
                action = agent.act(o_for_agent, sample=False)
        # Step the simulation forward
        next_o, r, terminated, truncated, _ = env.step(copy.deepcopy(action))
        done = terminated or truncated

        # Render the environment
        frames.append(env.render())
        if len(frames) >= 2000:
            frames.pop(0)  # Remove the oldest frame

        if done:
            break
        o = next_o

    return frames
