# from stable_baselines3 import PPO
# from kamal_env import CableControlEnv
# import cv2
# import imageio

# env = CableControlEnv(render_mode="rgb_array")
# model = PPO.load("ppo_cable_control.zip")

# obs, info = env.reset()
# frames = []
# for _ in range(500):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)

#     image = env.render()
#     if _ % 5 == 0:
#         frames.append(image)
#     if done or truncated:
#         obs, info = env.reset()

# print('observe: ' , obs)
# imageio.mimsave('ppo_cable_control.gif', frames, fps=120)

from stable_baselines3 import PPO
from kamal_env3 import CableControlEnv
import cv2
import imageio

# Initialize the environment and the model
env = CableControlEnv(render_mode="human")
model = PPO.load("ppo_cable_control6.zip")

# Reset the environment
obs, info = env.reset()
frames = []

# Run the model for a number of steps and collect frames
for _ in range(3000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Render the environment and store the frame
    image = env.render()
    frames.append(image)
    if done or truncated:
        obs, info = env.reset()

# Save the frames as a video
video_filename = 'ppo_cable_control5.mp4'
height, width, layers = frames[0].shape
video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print(f'Video saved as {video_filename}')


# from stable_baselines3 import PPO
# from kamal_env1 import CableControlEnv
# import cv2
# import imageio

# # Initialize the environment and the model
# env = CableControlEnv(render_mode="rgb_array")
# model = PPO.load("ppo_cable_control1.zip")

# # Define new targets
# def generate_test_targets(env, radius=0.5, center=(0, -0.03, 0.8), num_targets=6):
#     return env._generate_circular_targets(radius, center, num_targets)

# new_targets = generate_test_targets(env)

# # Reset the environment with new targets
# obs, info = env.reset(options={'new_targets': new_targets})
# frames = []

# # Run the model for a number of steps and collect frames
# for _ in range(500):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)

#     # Render the environment and store the frame
#     image = env.render()
#     frames.append(image)
#     if done or truncated:
#         obs, info = env.reset(options={'new_targets': new_targets})

# # Save the frames as a video
# video_filename = 'ppo_cable_control5.mp4'
# height, width, layers = frames[0].shape
# video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# for frame in frames:
#     video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# video.release()
# print(f'Video saved as {video_filename}')
