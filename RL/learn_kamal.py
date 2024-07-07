from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from kamal_env import CableControlEnv
# Initialize your environment
env = CableControlEnv(render_mode="rgb_array")

# Check your custom environment
check_env(env)

# Define and train the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")
model.learn(total_timesteps=200000, log_interval=4)
model.save("ppo_cable_control")

# %tensorboard --logdir ./ppo_cable_control_tensorboard/ --port 6007