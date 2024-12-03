import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# Load evaluation data
baseline_normal = pd.read_csv('evaluation_baseline_normal.csv')
baseline_faulty = pd.read_csv('evaluation_baseline_faulty.csv')
faulty_faulty = pd.read_csv('evaluation_faulty_faulty.csv')
faulty_normal = pd.read_csv('evaluation_faulty_normal.csv')
monitor_faulty = pd.read_csv('evaluation_monitor_faulty.csv')
monitor_normal = pd.read_csv('evaluation_monitor_normal.csv')

# Prepare data for average reward comparison with standard deviation
avg_rewards = pd.DataFrame({
    'Environment': ['Normal', 'Normal', 'Normal', 'Faulty', 'Faulty', 'Faulty'],
    'Agent': ['Baseline', 'Faulty', 'Monitor', 'Baseline', 'Faulty', 'Monitor'],
    'Average Reward': [
        baseline_normal['reward'].mean(),
        faulty_normal['reward'].mean(),
        monitor_normal['reward'].mean(),
        baseline_faulty['reward'].mean(),
        faulty_faulty['reward'].mean(),
        monitor_faulty['reward'].mean()
    ],
    'Reward Std': [
        baseline_normal['reward'].std(),
        faulty_normal['reward'].std(),
        monitor_normal['reward'].std(),
        baseline_faulty['reward'].std(),
        faulty_faulty['reward'].std(),
        monitor_faulty['reward'].std()
    ]
})

# Create bar plot grouped by environment and agent
fig, ax = plt.subplots(figsize=(5, 6))

# Define positions for grouped bars
x = np.arange(len(avg_rewards['Environment'].unique()))  # Number of unique environments
bar_width = 0.2  # Width of each bar (skinnier)
offsets = [-bar_width, 0, bar_width]  # Offset positions for agents (space between bars)
colors = ['skyblue', 'lightgreen', 'salmon']

for i, agent in enumerate(avg_rewards['Agent'].unique()):
    agent_data = avg_rewards[avg_rewards['Agent'] == agent]
    ax.bar(
        x + offsets[i],
        agent_data['Average Reward'],
        bar_width,
        yerr=agent_data['Reward Std'],
        capsize=5,
        label=agent,
        color=colors[i],
        edgecolor='black'  # Add black outline to bars
    )

# Add labels and title
ax.set_xticks(x)
ax.set_xticklabels(avg_rewards['Environment'].unique())
ax.set_ylabel('Average Reward')
ax.set_xlabel('Environment')
ax.set_title('Average Reward by Environment and Agent Type')
ax.legend(title='Agent', loc='upper right')  # Place legend in the top right corner
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.savefig('avg_reward_comparison.png', dpi=300)
plt.show()

baseline_normal['Environment'] = 'Normal'
baseline_faulty['Environment'] = 'Faulty'
faulty_faulty['Environment'] = 'Faulty'
faulty_normal['Environment'] = 'Normal'
monitor_faulty['Environment'] = 'Faulty'
monitor_normal['Environment'] = 'Normal'
baseline_normal['Agent'] = 'Baseline in Normal Env'
baseline_faulty['Agent'] = 'Baseline in Faulty Env'
faulty_faulty['Agent'] = 'Faulty Agent in Faulty Env'
faulty_normal['Agent'] = 'Faulty Agent in Normal Env'
monitor_faulty['Agent'] = 'Monitor Agent in Faulty Env'
monitor_normal['Agent'] = 'Monitor Agent in Normal Env'

all_data = pd.concat([baseline_normal, baseline_faulty, faulty_faulty, faulty_normal, monitor_faulty, monitor_normal],
                     ignore_index=True)

# Plot reward distribution in two collections

# Plot reward distribution
plt.figure(figsize=(14, 8))
sns.boxplot(x='Environment', y='reward', hue='Agent', data=all_data, palette="muted")
plt.ylabel('Episode Reward')
plt.title('Episode Reward Distribution Across Agents and Environments')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('reward_distribution.png', dpi=300)
plt.show()

# Plot episode length comparison with standard deviation
avg_lengths = pd.DataFrame({
    'Agent and Environment': ['Baseline in Normal Env', 'Baseline in Faulty Env', 'Faulty Agent in Faulty Env',
                              'Faulty Agent in Normal Env', 'Monitor Agent in Faulty Env',
                              'Monitor Agent in Normal Env'],
    'Average Episode Length': [baseline_normal['length'].mean(), baseline_faulty['length'].mean(),
                               faulty_faulty['length'].mean(), faulty_normal['length'].mean(),
                               monitor_faulty['length'].mean(), monitor_normal['length'].mean()],
    'Length Std': [baseline_normal['length'].std(), baseline_faulty['length'].std(), faulty_faulty['length'].std(),
                   faulty_normal['length'].std(), monitor_faulty['length'].std(), monitor_normal['length'].std()]})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Agent and Environment', y='Average Episode Length', data=avg_lengths,
                 yerr=avg_lengths['Length Std'], capsize=0.1, palette="muted")
plt.ylabel('Average Episode Length')
plt.title('Average Episode Length of Agents in Different Environments')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('episode_length_comparison.png', dpi=300)
plt.show()

# Plot monitor activation frequency over episodes for the monitoring agent
plt.figure(figsize=(10, 6))
plt.plot(monitor_faulty['episode'], monitor_faulty['monitor_activation_freq'], label='Faulty Environment', marker='o')
plt.plot(monitor_normal['episode'], monitor_normal['monitor_activation_freq'], label='Normal Environment', marker='s')
plt.xlabel('Episode')
plt.ylabel('Monitor Activation Frequency')
plt.title('Monitor Activation Frequency Over Episodes')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monitor_activation_frequency.png', dpi=300)
plt.show()

# Since the scatter plot doesn't provide helpful data due to limited episodes,
# we can consider plotting the average reward vs. average monitor activation frequency.

# Calculate average monitor activation frequency and rewards
avg_monitor_data = pd.DataFrame({'Environment': ['Faulty Environment', 'Normal Environment'],
                                 'Average Monitor Activation Frequency': [
                                     monitor_faulty['monitor_activation_freq'].mean(),
                                     monitor_normal['monitor_activation_freq'].mean()],
                                 'Average Reward': [monitor_faulty['reward'].mean(), monitor_normal['reward'].mean()]})

# Bar plot of average monitor activation frequency and average reward
fig, ax1 = plt.subplots(figsize=(8, 6))

ax2 = ax1.twinx()
sns.barplot(x='Environment', y='Average Monitor Activation Frequency', data=avg_monitor_data, palette="Blues_d", ax=ax1)
sns.lineplot(x='Environment', y='Average Reward', data=avg_monitor_data, marker='o', color='red', ax=ax2, sort=False,
             linewidth=2.5)

ax1.set_xlabel('Environment')
ax1.set_ylabel('Average Monitor Activation Frequency', color='blue')
ax2.set_ylabel('Average Reward', color='red')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
plt.title('Average Reward vs. Monitor Activation Frequency')
plt.tight_layout()
plt.savefig('avg_reward_vs_monitor_activation.png', dpi=300)
plt.show()
