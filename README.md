# Cable-Driven Parallel Robot Control using Reinforcement Learning

This project focuses on controlling a Cable-Driven Parallel Robot (CDPR) using Reinforcement Learning (RL) techniques. Specifically, we implement the Deep Deterministic Policy Gradient (DDPG) algorithm to achieve high precision in trajectory tracking and point-to-point control.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Details](#environment-details)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project demonstrates the application of reinforcement learning for controlling a CDPR. RL techniques are used to manage the complexity and non-linearity of the system, allowing the robot to learn optimal policies for performing specific tasks.

## Features
- Implementation of DDPG algorithm for controlling CDPR.
- Simulation environment built using MuJoCo.
- High precision in trajectory tracking and point-to-point control.
- Statistical analysis of errors and performance metrics.
- Visualizations of trajectory tracking, position errors, and actuator actions.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CDPR_RL.git
    cd CDPR_RL
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have MuJoCo installed and properly set up. For installation instructions, refer to the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/).

## Usage
1. Train the model:
    ```bash
    python learn_CDPR.py
    ```

2. Evaluate the model:
    ```bash
    python test_CDPR.py
    ```



## Environment Details
The environment for the CDPR simulation is built using MuJoCo. Key features of the environment include:
- Definition of the CDPR using an XML file.
- Observation space includes positions, velocities, and errors.
- Action space corresponds to the control inputs to the robot's motors.
- Reward function designed to minimize tracking errors and control efforts.

## Results
The project provides comprehensive results including:
- Trajectory tracking performance.
- Position and velocity errors.
- Actuator actions over time.

![Trajectory Tracking](results/trajectory_tracking.png)
![Position Errors](results/position_errors.png)
![Velocity Errors](results/velocity_errors.png)

## Future Work
Future improvements to this project include:
- Implementing a meta-learner for dynamic weight adjustment in the reward function.
- Adding constraints related to acceleration for smoother trajectory tracking.
- Experimenting with various RL models and comparing their performance.
- Ensuring stability and safety in the control policies.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [MuJoCo](https://mujoco.readthedocs.io/en/latest/)
