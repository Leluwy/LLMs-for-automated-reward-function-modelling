import metaworld
from PIL import Image

import random

import torch
import sac
from agents import train_agent
from utils import ReplayBuffer

print("Metaworld...")

print(metaworld.ML1.ENV_NAMES)

env_name = 'assembly-v2'  # Pick an environment name

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

num_train_steps = 10_000
num_seed_steps = 5000
eval_frequency = 10_000
num_eval_episodes = 10
replay_buffer = ReplayBuffer(obs_size, ac_size, num_train_steps, device)
hidden_dim = 256
hidden_depth = 2
batch_size = 256
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

policy_loss, critic_loss, batch_reward = (
                train_agent(agent,
                            env,
                            num_train_steps=num_train_steps,
                            num_seed_steps=num_seed_steps,
                            eval_frequency=eval_frequency,
                            num_eval_episodes=num_eval_episodes,
                            replay_buffer=replay_buffer))

agent.save(f'model_final.pth')

for i in range(100):
    a = env.action_space.sample()  # Sample an action
    a = [0, 0, 10, 0]
    obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action

# Render and save the image
img = env.render()  # Capture the rendered image as an RGB array
img_pil = Image.fromarray(img)  # Convert the array to a PIL image
img_pil.save("assembly_v2_sample_image.png")  # Save the image as a PNG file

print("Image saved as 'assembly_v2_sample_image.png'")

img_pil.show()
