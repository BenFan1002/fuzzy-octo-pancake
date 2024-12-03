import os
import time
from datetime import datetime

import cv2  # Import OpenCV
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, Logger
import gym_donkeycar
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# Callback to save the model at regular intervals
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Check if a new episode has started
        if self.locals.get('infos') is not None:
            info = self.locals['infos'][0]
            if 'episode' in info.keys():
                self.episode_count += 1
                if self.episode_count % self.save_freq == 0:
                    model_path = os.path.join(self.save_path, f'model_baseline_{self.episode_count}_episodes')
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Model saved to {model_path} after {self.episode_count} episodes")
        return True


# Callback to log episode rewards and lengths
class DetailedLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(DetailedLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.monitor_activations = []
        self.actions = []
        self.episode_count = 0
        self.data = []

    def _on_step(self):
        # Log actions
        action = self.locals['actions'][0]
        self.actions.append(action)

        # Check if a new episode has started
        if self.locals.get('infos') is not None:
            info = self.locals['infos'][0]
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1

                # Extract monitor activation frequency for the episode
                monitor_actions = [1 for a in self.actions]
                monitor_activation_freq = sum(1 for m in monitor_actions if m > 0.5) / len(monitor_actions)

                self.monitor_activations.append(monitor_activation_freq)

                # Reset actions for the next episode
                self.actions = []

                if self.verbose > 0:
                    print(
                        f"Episode {self.episode_count}: Reward = {episode_reward}, Length = {episode_length}, Monitor Activation Frequency = {monitor_activation_freq}")

                # Save episode data
                self.data.append({
                    'episode': self.episode_count,
                    'reward': episode_reward,
                    'length': episode_length,
                    'monitor_activation_freq': monitor_activation_freq
                })
        return True

    def _on_training_end(self):
        # Save the data to a CSV file
        df = pd.DataFrame(self.data)
        df.to_csv('training_log.csv', index=False)
        if self.verbose > 0:
            print("Training data saved to training_log.csv")


# Baseline environment without monitoring and sensor faults
class DonkeyCarBaselineEnv(gym.Env):
    def __init__(self, conf, max_episode_length=500):
        # Initialize the base DonkeyCar environment
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        # Action space: [steering, throttle]
        self.action_space = self.env.action_space
        # Observation space: same as DonkeyCar env
        self.observation_space = self.env.observation_space
        # Max episode length
        self.max_episode_length = max_episode_length
        # Initialize step counter
        self.current_step = 0
        # Initialize variables for rendering
        self.last_obs = None
        self.current_reward = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset()
        # Store the initial observation
        self.current_step = 0

        self.last_obs = obs.copy()
        self.current_reward = 0.0
        return obs

    def step(self, action):
        # Step the base environment
        obs, reward, done, info = self.env.step(action)
        # Store the current observation
        # Increment the step counter
        self.current_step += 1
        # Check if max episode length is reached
        if self.current_step >= self.max_episode_length:
            done = True
        self.last_obs = obs.copy()
        self.current_reward = reward
        # print(f"Steering: {action[0]}, Throttle: {action[1]}, Reward: {reward}", flush=True)
        return obs, reward, done, info

    def close(self):
        self.env.close()

from stable_baselines3.common.logger import KVWriter, Logger

class NoOpWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0):
        pass  # Do nothing

    def close(self):
        pass  # Do nothing


# SET UP ENVIRONMENT
start_time = datetime.now()
exe_path = "C:\\Users\\10944\\Code\\CSCE642-Project\\DonkeySimWin\\donkey_sim.exe"
port = 9091

conf = {"exe_path": exe_path, "port": port}

# Create the baseline environment
env = DonkeyCarBaselineEnv(conf)
env.env.render(mode='rgb_array')

# Create the RL model using PPO with a CNN policy for image observations
if os.path.exists("donkeycar_baseline_ppo.zip"):
    print("Loading existing baseline model")
    model = PPO.load("donkeycar_baseline_ppo.zip", env, verbose=1, tensorboard_log="./ppo_donkeycar_baseline_tensorboard/")
else:
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_donkeycar_baseline_tensorboard/")

# Set up callbacks and logging
save_freq = 10
save_path = "./saved_models/"
log_path = "./logs/"
save_model_callback = SaveModelCallback(save_freq=save_freq, save_path=save_path)
detailed_logger_callback = DetailedLoggerCallback()
callback = CallbackList([save_model_callback, detailed_logger_callback])

# Configure the logger
# new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
# model.set_logger(new_logger)
# Create a no-op logger
noop_logger = Logger(folder=None, output_formats=[NoOpWriter()])
model.set_logger(noop_logger)

# Train the baseline model
model.learn(total_timesteps=30000, callback=callback)

# Save the trained model
model.save("donkeycar_baseline_ppo")

# Close the environment
env.close()

print(f"Training completed in {datetime.now() - start_time}")
