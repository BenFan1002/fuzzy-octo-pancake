import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data from all agents
df_baseline = pd.read_csv('training_log.csv')
df_faulty = pd.read_csv('training_log_faulty.csv')
df_monitor = pd.read_csv('training_log_monitor.csv')

# Set the style for the plot
sns.set(style="whitegrid")

# Plot average reward per episode for all agents
plt.figure(figsize=(14, 8))  # Increase figure size
sns.lineplot(x='episode', y='reward', data=df_baseline, label='Baseline (No Faults, No Monitor)', marker='o', markersize=6, alpha=0.5)
sns.lineplot(x='episode', y='reward', data=df_faulty, label='Faulty Env (No Monitor)', marker='s', markersize=6, alpha=0.5)
sns.lineplot(x='episode', y='reward', data=df_monitor, label='Faulty Env (With Monitor)', marker='^', markersize=6, alpha=0.5)

# Add prediction line (example using a simple linear regression)
for df, label in zip([df_baseline, df_faulty, df_monitor],
                     ['Baseline (No Faults, No Monitor)', 'Faulty Env (No Monitor)', 'Faulty Env (With Monitor)']):
    x = df['episode']
    y = df['reward']
    # Fit a linear regression model
    coeffs = np.polyfit(x, y, 1)
    p = np.poly1d(coeffs)
    plt.plot(x, p(x), linestyle='--', label=f'{label} Prediction', alpha=0.9, linewidth=2.5)

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Average Reward per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_per_episode.png', dpi=300)
plt.show()
