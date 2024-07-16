from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from kamal_env3 import CableControlEnv
# from kamal_env_end_to_end import CableControlEnv
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
model_save_path = '/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Models/DDPG_cable_control_circle50test'
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")
model.learn(total_timesteps=300000, log_interval=4) 
            # callback=render_callback)
model.save(model_save_path)
# model.save("ppo_cable_control_end_to_end")
# %tensorboard --logdir ./ppo_cable_control_tensorboard/ --port 6007