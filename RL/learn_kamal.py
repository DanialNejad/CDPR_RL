from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from gymnasium import utils
from stable_baselines3.common.env_checker import check_env
# from kamal_env3 import CableControlEnv
from kamal_env_ppo import CableControlEnv
# from kamal_env_end_to_end import CableControlEnv
from callback import RenderCallback

# Initialize your environment
env = CableControlEnv(render_mode="rgb_array")
check_env(env)
# if __name__ == '__main__':
#     # Define a helper function to create the environment
#     def make_env(rank):
#         def _init():
#             env = CableControlEnv(render_mode="rgb_array")
#             env.seed(rank)
#             return env
#         return _init

#     # Create the vectorized environment
#     num_envs = 8
#     env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
#     env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

#     # Check the environment
#     check_env(env)

#     # Define the model and training parameters
#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")

#     # Train the model
#     model.learn(total_timesteps=1000000)

#     # Save the trained model
#     model.save("ppo_cable_control_parallel")

# render_callback = RenderCallback()

# # Create the vectorized environment
# env = make_vec_env(lambda: env, n_envs=6)
# Define and train the model
model_save_path = '/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Models/PPO_cable_control_circle50new'
# model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")
model.learn(total_timesteps=1500000, log_interval=4) 
#             # callback=render_callback)
model.save(model_save_path)
# model.save("ppo_cable_control_end_to_end")
# %tensorboard --logdir ./ppo_cable_control_tensorboard/ --port 6007