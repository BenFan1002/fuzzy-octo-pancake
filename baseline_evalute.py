import os
import gym
import gym_donkeycar
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Set up the environment configuration
exe_path = "C:\\Users\\10944\\Code\\CSCE642-Project\\DonkeySimWin\\donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}


class DonkeyCarBaselineEnv(gym.Env):
    def __init__(self, conf):
        self.env = gym.make("donkey-generated-track-v0", conf=conf)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.last_obs = None
        self.current_reward = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.last_obs = obs.copy()
        self.current_reward = 0.0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs.copy()
        self.current_reward = reward
        return obs, reward, done, info

    def close(self):
        self.env.close()


# Initialize the test environment
test_env = DonkeyCarBaselineEnv(conf)

# Load the trained model or exit if the model file doesn't exist
model_path = "donkeycar_baseline_ppo.zip"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")

print("Loading the trained model...")
model = PPO.load(model_path, test_env)


# Evaluate the model
def evaluate_model(env, model, num_episodes=10):
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            print(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward} ± {std_reward}")
    return avg_reward, std_reward


# Run evaluation
try:
    print("Evaluating the model...")
    average_reward, reward_std = evaluate_model(test_env, model, num_episodes=5)
    print(f"Evaluation Complete: Average Reward = {average_reward}, Reward Std Dev = {reward_std}")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
finally:
    # Close the environment
    test_env.close()
    print("Environment closed.")
