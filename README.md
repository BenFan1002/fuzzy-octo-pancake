# Enhancing Fault Tolerance in Autonomous Driving Agents through Monitoring Mechanisms in Reinforcement Learning

## Project Description

This project explores the development of a reinforcement learning (RL) agent capable of handling sensor faults in autonomous driving scenarios. By integrating a monitoring mechanism into the agent's action space, the agent can detect and respond to sensor faults during operation. The agent is trained and evaluated in the DonkeyCar Simulator environment, modified to simulate sensor faults and incorporate the monitoring mechanism.

## Repository Contents

- `donkeycar_mmdp_ppo.py`: Main script for training the agent with the monitoring mechanism.
- `donkeycar_faulty_ppo.py`: Script for training the agent in a faulty environment without monitoring.
- `donkeycar_baseline_ppo.py`: Script for training the baseline agent in a normal environment.
- `generate_evaluate_data.py`: Script for evaluating the trained agents and collecting performance data.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, providing instructions on how to run the code.
- `images/`: Directory containing graphs generated from the evaluation data.
- `logs/training/`: Directory containing training logs and CSV files.
- `models/`: Directory where trained models are saved.

## Getting Started

### Prerequisites

- **Operating System**: Windows 10/11 (recommended), Linux, or macOS.
- **Python Version**: Python 3.8 or higher.

### Software and Libraries

Ensure you have the following software installed:

1. **DonkeyCar Simulator**: Download and install the DonkeyCar Simulator appropriate for your operating system.

   - **Download Link**: [DonkeyCar Simulator Releases](https://github.com/tawnkramer/gym-donkeycar/releases)
   - Extract the simulator to a known directory. You'll need the path to the simulator executable (`donkey_sim.exe` or equivalent) for configuration.

2. **Python Packages**: Install the required Python packages using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

### Additional Dependencies

- **OpenAI Gym**
- **Stable Baselines3**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

## Running the Code

### 1. Configure the DonkeyCar Simulator Path

In the main scripts (`donkeycar_mmdp_ppo.py`, `donkeycar_faulty_ppo.py`, `donkeycar_baseline_ppo.py`), set the `exe_path` variable to the path of your DonkeyCar Simulator executable.

```python
# Example for Windows
exe_path = "C:\\Path\\To\\DonkeySimWin\\donkey_sim.exe"
```

### 2. Training the Agents

#### a. Train the Baseline Agent

```bash
python donkeycar_baseline_ppo.py
```

- This script trains the agent in a normal environment without sensor faults or monitoring.
- The trained model will be saved as `donkeycar_baseline_ppo.zip`.

#### b. Train the Faulty Agent without Monitoring

```bash
python donkeycar_faulty_ppo.py
```

- This script trains the agent in an environment with sensor faults but without a monitoring mechanism.
- The trained model will be saved as `donkeycar_faulty_ppo.zip`.

#### c. Train the Agent with Monitoring Mechanism

```bash
python donkeycar_mmdp_ppo.py
```

- This script trains the agent in a faulty environment with the monitoring mechanism.
- The trained model will be saved as `donkeycar_mmdp_ppo.zip`.

**Note**: Training may take some time depending on your hardware capabilities. Each script is configured to train for 30,000 timesteps.

### 3. Evaluating the Agents

After training, you can evaluate the agents using the `generate_evaluate_data.py` script.

```bash
python generate_evaluate_data.py
```

- This script loads the trained models and evaluates each agent in both normal and faulty environments.
- Evaluation data is saved as CSV files in the `logs/testing` directory.
- Graphs are generated and saved in the `images` directory.

### 4. Viewing Results

- **Training Logs**: Check the `logs/training/` directory for CSV files containing training data.
- **Evaluation Data**: Evaluation results are saved as CSV files in the `logs/testing` directory.
- **Graphs**: Generated graphs are saved in the `images/` directory. You can view these to analyze the performance of the agents.

## Customization

### Adjusting Training Parameters

You can adjust the training parameters such as `total_timesteps`, `learning_rate`, and others by modifying the respective script files.

### Changing the Faulty Sensor Probability

In the `DonkeyCarFaultyEnv` class, you can change the `faulty_sensor_probability` to simulate different levels of sensor unreliability.

```python
self.faulty_sensor_probability = 0.3  # Adjust the probability as needed
```

### Modifying the Monitoring Mechanism

The monitoring mechanism parameters, such as the monitor activation penalty and maximum monitor delay, can be adjusted in the `DonkeyCarMMDPEnv` class within `environments.py`.

```python
self.max_monitor_delay = 10  # Steps before monitor fails
```

## Project Structure

- **Main Scripts**:
  - `donkeycar_baseline_ppo.py`
  - `donkeycar_faulty_ppo.py`
  - `donkeycar_mmdp_ppo.py`
  - `generate_evaluate_data.py`

- **Custom Environments**:
  - `DonkeyCarNormalEnv`
  - `DonkeyCarFaultyEnv`
  - `DonkeyCarMMDPEnv`

- **Callbacks**:
  - `SaveModelCallback`
  - `DetailedLoggerCallback`

- **Utilities**:
  - `requirements.txt`
  - `README.md`

- **Data and Outputs**:
  - `logs/training/`: Training logs and CSV files.
  - `logs/testing`: Evaluation results and CSV files.
  - `images/`: Generated graphs from evaluation data.
  - `models/`: Saved trained models.

## Troubleshooting

- **Simulator Connection Issues**: Ensure that the `exe_path` and `port` in the configuration are correctly set and that no other processes are using the same port.
- **Package Installation Errors**: Verify that all packages in `requirements.txt` are installed. Use a virtual environment to manage dependencies if necessary.
- **Performance Issues**: Training can be resource-intensive. Close unnecessary applications to free up system resources.

## References

- **DonkeyCar Simulator**: [https://github.com/tawnkramer/gym-donkeycar/releases/tag/v22.11.06](https://github.com/tawnkramer/gym-donkeycar/releases/tag/v22.11.06)
- **gym-donkeycar**: [https://github.com/tawnkramer/gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)
- **Stable Baselines3**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **OpenAI Gym**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
