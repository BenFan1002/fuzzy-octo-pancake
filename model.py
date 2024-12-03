import os
from datetime import datetime

import cv2  # Import OpenCV
import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import gym_donkeycar


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
                    model_path = os.path.join(self.save_path, f'model_monitor_{self.episode_count}_episodes')
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Model saved to {model_path} after {self.episode_count} episodes")
        return True


# Callback to log detailed data per episode
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
                monitor_actions = [a[2] for a in self.actions]
                monitor_activation_freq = sum(1 for m in monitor_actions if m > 0.5) / len(monitor_actions)

                self.monitor_activations.append(monitor_activation_freq)

                # Reset actions for the next episode
                self.actions = []

                if self.verbose > 0:
                    print(
                        f"Episode {self.episode_count}: Reward = {episode_reward}, Length = {episode_length}, Monitor Activation Frequency = {monitor_activation_freq}"
                    )

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
        df.to_csv('training_log_monitor.csv', index=False)
        if self.verbose > 0:
            print("Training data saved to training_log_monitor.csv")


class DonkeyCarMMDPEnv(gym.Env):
    def __init__(self, conf):
        # Initialize the base DonkeyCar environment
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        # Define action space: [steering, throttle, monitor_action]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        # Observation space: same as DonkeyCar env
        self.observation_space = self.env.observation_space
        # Variables to simulate monitor and faulty sensors
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        self.max_monitor_delay = 10  # Steps before monitor fails
        # Initialize variables for rendering
        self.last_obs = None
        self.current_reward = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        # Store the initial observation
        self.last_obs = obs.copy()
        self.current_reward = 0.0
        return obs

    def step(self, action):
        # Extract actions
        steering = action[0]
        throttle = action[1]
        monitor_action = action[2]

        # Apply monitor action
        if monitor_action > 0.5:
            self.monitor_active = True
            self.time_since_last_monitor_activation = 0
        else:
            self.time_since_last_monitor_activation += 1
            if self.time_since_last_monitor_activation > self.max_monitor_delay:
                self.monitor_active = False

        # Combine steering and throttle into an action for the base env
        base_action = np.array([steering, throttle])

        # Step the base environment
        obs, reward, done, info = self.env.step(base_action)

        # Store the current observation
        self.last_obs = obs.copy()

        # If monitor is inactive, simulate faulty sensors by modifying the observation
        if not self.monitor_active:
            obs = np.zeros_like(obs)  # Black image or replace with noise if preferred

        # Monitor Reward
        monitor_reward = -0.5 if monitor_action > 0.5 else 0.0  # Cost for activating monitor

        # Proxy Reward
        if self.monitor_active:
            proxy_reward = reward + monitor_reward
        else:
            proxy_reward = 0.0  # Reward is unobservable when monitor is inactive

        # Total reward
        total_reward = proxy_reward + monitor_reward

        # Store the current total reward for rendering
        self.current_reward = total_reward

        # Optional: Print action and reward for debugging
        # print(
        #     f"Steering: {steering}, Throttle: {throttle}, Monitor Action: {monitor_action}, Total Reward: {total_reward}",
        #     flush=True)

        return obs, total_reward, done, info

    def close(self):
        self.env.close()


from stable_baselines3.common.logger import KVWriter, Logger


class NoOpWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0):
        pass  # Do nothing

    def close(self):
        pass  # Do nothing

start_time = datetime.now()

# SET UP ENVIRONMENT
exe_path = "C:\\Users\\10944\\Code\\CSCE642-Project\\DonkeySimWin\\donkey_sim.exe"
port = 9091

conf = {"exe_path": exe_path, "port": port}

# Create the environment
env = DonkeyCarMMDPEnv(conf)
env.env.render(mode='rgb_array')

# Create the RL model using PPO with a CNN policy for image observations
if os.path.exists("donkeycar_mmdp_ppo.zip"):
    print("Loading existing model")
    model = PPO.load("donkeycar_mmdp_ppo.zip", env, verbose=1, tensorboard_log="./ppo_donkeycar_tensorboard/")
else:
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_donkeycar_tensorboard/")

# Set up callbacks and logging
save_freq = 10  # Save the model every 10 episodes
save_path = "./saved_models/"
log_path = "./logs/"
save_model_callback = SaveModelCallback(save_freq=save_freq, save_path=save_path)
detailed_logger_callback = DetailedLoggerCallback()
callback = CallbackList([save_model_callback, detailed_logger_callback])

# Configure the logger
# Create a no-op logger to suppress default logging
noop_logger = Logger(folder=None, output_formats=[NoOpWriter()])
model.set_logger(noop_logger)

# Train the model
model.learn(total_timesteps=30000, callback=callback)

# Save the trained model
model.save("donkeycar_mmdp_ppo.zip")

# Close the environment
env.close()
print(f"Training completed in {datetime.now() - start_time}")
