import os

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO


class LunarLanderMMDPEnv(gym.Env):
    def __init__(self):
        super(LunarLanderMMDPEnv, self).__init__()
        # Initialize the base LunarLander environment
        self.env = gym.make('LunarLander-v2')

        # Define action space: Discrete(8) for all combinations of base_action and monitor_action
        self.action_space = spaces.Discrete(8)  # Actions 0 to 7

        # Observation space: same as LunarLander env
        self.observation_space = self.env.observation_space

        # Variables to simulate monitor and faulty sensors
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        self.max_monitor_delay = 10  # Steps before monitor fails

    def reset(self):
        obs = self.env.reset()
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        return obs

    def step(self, action):
        # Ensure the action is an integer
        action = int(action)

        # Decode the action into base_action and monitor_action
        base_action = action % 4  # Values from 0 to 3
        monitor_action = action // 4  # Values 0 or 1

        # Apply monitor action
        if monitor_action == 1:
            self.monitor_active = True
            self.time_since_last_monitor_activation = 0
        else:
            self.time_since_last_monitor_activation += 1
            if self.time_since_last_monitor_activation > self.max_monitor_delay:
                self.monitor_active = False

        # Step the base environment with the base action
        obs, reward, done, info = self.env.step(base_action)

        # If monitor is inactive, simulate faulty sensors by modifying the observation
        if not self.monitor_active:
            obs = np.zeros_like(obs)  # Replace with zeros or add noise if preferred

        env_reward = reward  # Use the original reward from the environment

        # Monitor Rewards (r_m)
        monitor_reward = -0.1 if monitor_action == 1 else 0.0  # Cost for activating monitor

        # Proxy Rewards (r̂_e)
        if self.monitor_active:
            proxy_reward = env_reward
        else:
            proxy_reward = 0.0  # Reward is unobservable when monitor is inactive

        # Total reward
        total_reward = proxy_reward + monitor_reward
        if total_reward > 0:
            # print(f"Action: {action} (Base: {base_action}, Monitor: {monitor_action}) Step reward: {total_reward}"
            #       f" Proxy reward: {proxy_reward} Monitor reward: {monitor_reward}")
            pass
        return obs, total_reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()


if __name__ == '__main__':
    # Create the environment
    env = LunarLanderMMDPEnv()

    # Define the path where the model will be saved/loaded
    model_path = "lunarlander_mmdp_ppo.zip"

    # Check if a saved model exists
    if os.path.exists(model_path):
        # Load the model
        model = PPO.load(model_path, env=env)
        print("Loaded saved model.")

        # Suppress logs for the loaded model
        from stable_baselines3.common.logger import configure
        logger = configure(folder=None, format_strings=[])
        model.set_logger(logger)
    else:
        # Create a new model
        model = PPO('MlpPolicy', env, verbose=0)
        print("Created new model.")

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the updated model
    model.save("lunarlander_mmdp_ppo")

    # Close the environment
    env.close()
