import metaworld
from PIL import Image

import random

import numpy as np
import gym

# rom garage.envs import PointMaze
from garage.envs import GymEnv

import torch
import sac
from agents import train_agent
from utils import ReplayBuffer
from rollouts import evaluate, evaluate_agent, rollout, rollout_frames

from video import create_video

# random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

train = False  # if train the model or load from trained
video = True  # if video is displayed

print("Metaworld...")

print(metaworld.ML1.ENV_NAMES)

env_name = 'reach-v2'  # Pick an environment name

SEED = 0  # random seed
benchmark = metaworld.ML1(env_name, seed=SEED)

render_mode = 'rgb_array'  # set a render mode

ml1 = metaworld.ML1(env_name)  # construct the benchmark, sampling tasks

camera_name = 'corner3'  # one of: ['corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV']

env = ml1.train_classes[env_name](render_mode=render_mode, camera_name=camera_name)

task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())]

num_train_steps = 250_000
num_seed_steps = 5000
eval_frequency = 5_000
num_eval_episodes = 5
replay_buffer = ReplayBuffer(obs_size, ac_size, num_train_steps, device)
hidden_dim = 256
hidden_depth = 2
batch_size = 500
discount_factor = 0.99

agent = sac.SACAgent(
                obs_dim=obs_size,
                action_dim=ac_size,
                action_range=action_range,
                device=device,
                discount=discount_factor,
                init_temperature=0.1,
                alpha_lr=3e-4,
                actor_lr=3e-4,
                critic_lr=3e-4,
                critic_tau=0.005,
                batch_size=batch_size,
                target_entropy=-ac_size,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                double_critic=True,
                temperature=True
            )
if train:
    policy_loss, critic_loss, batch_reward, frames = (
                    train_agent(agent,
                                env,
                                num_train_steps=num_train_steps,
                                num_seed_steps=num_seed_steps,
                                eval_frequency=eval_frequency,
                                num_eval_episodes=num_eval_episodes,
                                replay_buffer=replay_buffer))

    agent.save(f'model_final.pth')
    create_video(frames, 'output_video_train.mp4')


# Load the saved state dictionary
agent.actor.load_state_dict(torch.load('model_final.pth')["actor"])
evaluate_agent(env, agent, "final", num_episodes=1, verbose=True)

obs = env.reset()

if video:
    max_episode_steps = 500
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    frames = rollout_frames(env, agent)

    create_video(frames)