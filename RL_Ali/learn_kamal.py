from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from kamal_env import CableControlEnv
from callback import RenderCallback

# def make_env():
#     def _init():
#         env = CableControlEnv(render_mode="human")
#         return env
#     return _init

# # Create the vectorized environment
# n_envs = 6  # Define the number of environments
# env = make_vec_env(make_env, n_envs=n_envs)
# Initialize your environment
env = CableControlEnv(render_mode="rgb_array")

# Check your custom environment
check_env(env)

# render_callback = RenderCallback()

# # Create the vectorized environment
# env = make_vec_env(lambda: env, n_envs=6)
# Define and train the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")
# model.learn(total_timesteps=200000, log_interval=4, callback=render_callback)
model.learn(total_timesteps=200000, log_interval=4)
model.save("ppo_cable_control")

# %tensorboard --logdir ./ppo_cable_control_tensorboard/ --port 6007