import os
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.wrappers import RecordVideo
import gym_donkeycar

# Base DonkeyCar Environment (Normal)
class DonkeyCarNormalEnv(gym.Env):
    def __init__(self, conf):
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    # Add render method
    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

# DonkeyCar Environment with Sensor Faults (No Monitoring)
class DonkeyCarFaultyEnv(gym.Env):
    def __init__(self, conf, faulty_sensor_probability=0.3):
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.faulty_sensor_probability = faulty_sensor_probability

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if np.random.rand() < self.faulty_sensor_probability:
            obs = self.simulate_faulty_sensor(obs)
        return obs, reward, done, info

    def simulate_faulty_sensor(self, obs):
        faulty_obs = np.zeros_like(obs)  # Black image
        return faulty_obs

    # Add render method
    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

# DonkeyCar Environment with Monitoring Mechanism
class DonkeyCarMMDPEnv(gym.Env):
    def __init__(self, conf):
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self.observation_space = self.env.observation_space
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        self.max_monitor_delay = 10

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.monitor_active = False
        self.time_since_last_monitor_activation = 0
        return obs

    def step(self, action):
        steering = action[0]
        throttle = action[1]
        monitor_action = action[2]

        if monitor_action > 0.5:
            self.monitor_active = True
            self.time_since_last_monitor_activation = 0
        else:
            self.time_since_last_monitor_activation += 1
            if self.time_since_last_monitor_activation > self.max_monitor_delay:
                self.monitor_active = False

        base_action = np.array([steering, throttle])
        obs, reward, done, info = self.env.step(base_action)

        if not self.monitor_active:
            obs = np.zeros_like(obs)  # Simulate faulty sensor

        # Monitor Reward
        monitor_reward = -0.5 if monitor_action > 0.5 else 0.0

        # Proxy Reward
        if self.monitor_active:
            proxy_reward = reward + monitor_reward
        else:
            proxy_reward = 0.0

        total_reward = proxy_reward + monitor_reward
        return obs, total_reward, done, info

    # Add render method
    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

from stable_baselines3 import PPO

# Paths to trained models
baseline_model_path = "donkeycar_baseline_ppo.zip"
faulty_model_path = "donkeycar_faulty_ppo.zip"
monitor_model_path = "donkeycar_mmdp_ppo_good.zip"

# Load models
baseline_model = PPO.load(baseline_model_path)
faulty_model = PPO.load(faulty_model_path)
monitor_model = PPO.load(monitor_model_path)

def evaluate_agent(env, model, num_episodes=10, is_monitor_model=False, max_episode_length=500, stuck_threshold=50):
    episode_rewards = []
    episode_lengths = []
    monitor_activations = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        monitor_actions = []
        throttle_history = []
        stuck_steps = 0  # Counter for stuck steps

        while not done and length < max_episode_length:
            action, _states = model.predict(obs, deterministic=True)
            if is_monitor_model:
                monitor_actions.append(action[2])
            else:
                pass  # No monitor action

            obs, reward, done, info = env.step(action)

            # Render the environment
            env.render(mode='rgb_array')

            # Check if throttle is near zero
            throttle = action[1]
            throttle_history.append(throttle)
            if abs(throttle) < 0.01:
                stuck_steps += 1
            else:
                stuck_steps = 0  # Reset if throttle is non-zero

            # If stuck for too many steps, end the episode
            if stuck_steps >= stuck_threshold:
                print(f"Agent is stuck for {stuck_steps} steps. Ending episode early.")
                done = True

            total_reward += reward
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        if is_monitor_model:
            monitor_activation_freq = sum(1 for m in monitor_actions if m > 0.5) / len(monitor_actions)
            monitor_activations.append(monitor_activation_freq)
        else:
            monitor_activations.append(0.0)  # No monitoring

        print(f"Episode {episode + 1}: Reward = {total_reward}, Length = {length}")

    # Create DataFrame to store results
    results = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'reward': episode_rewards,
        'length': episode_lengths,
        'monitor_activation_freq': monitor_activations
    })
    return results

# Configuration for the simulator
exe_path = "C:\\Users\\10944\\Code\\CSCE642-Project\\DonkeySimWin\\donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}

# Number of evaluation episodes
num_episodes = 1  # Adjust as needed for video recording

# Evaluate Baseline Agent in Normal Environment
print("Evaluating Baseline Agent in Normal Environment...")
env_normal = DonkeyCarNormalEnv(conf)

# Wrap the environment with RecordVideo
env_normal = RecordVideo(
    env_normal,
    video_folder='videos/baseline_normal',
    episode_trigger=lambda episode_id: True,
    name_prefix='baseline_normal'
)

baseline_results_normal = evaluate_agent(env_normal, baseline_model, num_episodes=num_episodes, is_monitor_model=False)
baseline_results_normal.to_csv('evaluation_baseline_normal.csv', index=False)
env_normal.close()

# Evaluate Agent with Monitoring in Faulty Environment
print("Evaluating Agent with Monitoring in Faulty Environment...")
env_monitor = DonkeyCarMMDPEnv(conf)

# Wrap the environment with RecordVideo
env_monitor = RecordVideo(
    env_monitor,
    video_folder='videos/monitor_faulty',
    episode_trigger=lambda episode_id: True,
    name_prefix='monitor_faulty'
)

monitor_results_faulty = evaluate_agent(env_monitor, monitor_model, num_episodes=num_episodes, is_monitor_model=True)
monitor_results_faulty.to_csv('evaluation_monitor_faulty.csv', index=False)
env_monitor.close()
