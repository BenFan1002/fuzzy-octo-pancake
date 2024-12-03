import gym
import gym_donkeycar
from stable_baselines3 import A2C  # Replace with the algorithm you used
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make('CartPole-v1')

# Load the trained model
model = A2C.load("donkeycar_baseline_ppo.zip")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Optionally, run the model and visualize its behavior
for episode in range(5):  # Run 5 episodes
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
