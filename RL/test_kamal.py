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

# from stable_baselines3 import PPO
# from kamal_env3 import CableControlEnv
# import cv2
# import imageio

# # Initialize the environment and the model
# env = CableControlEnv(render_mode="human")
# model = PPO.load("ppo_cable_control_circle200.zip")

# # Reset the environment
# obs, info = env.reset()
# frames = []

# # Run the model for a number of steps and collect frames
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)

#     # Render the environment and store the frame
#     image = env.render()
#     frames.append(image)
#     if done or truncated:
#         obs, info = env.reset()

# # Save the frames as a video
# video_filename = 'ppo_cable_control5.mp4'
# height, width, layers = frames[0].shape
# video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# for frame in frames:
#     video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# video.release()
# print(f'Video saved as {video_filename}')


from stable_baselines3 import PPO
from kamal_env3 import CableControlEnv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Initialize the environment and the model
model_save_path = '/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Models/ppo_cable_control_circle50test.zip'
env = CableControlEnv(render_mode="human")
model = PPO.load(model_save_path)

# Reset the environment
obs, info = env.reset()

# Lists to store data
frames = []
desired_trajectory = []
actual_trajectory = []
position_errors = []
velocity_errors = []
actuator_actions = []

# Run the model for a number of steps and collect data
for _ in range(195):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Collect data
    end_effector_pos = obs[:3]
    end_effector_vel = obs[3:6]
    position_error = obs[6]
    velocity_error = obs[7]
    desired_pos = env.target

    desired_trajectory.append(desired_pos)
    actual_trajectory.append(end_effector_pos)
    position_errors.append(position_error)
    velocity_errors.append(velocity_error)
    actuator_actions.append(action)

    # Render the environment and store the frame
    image = env.render()
    frames.append(image)
    if done or truncated:
        obs, info = env.reset()


# Convert lists to numpy arrays
desired_trajectory = np.array(desired_trajectory)
actual_trajectory = np.array(actual_trajectory)
position_errors = np.array(position_errors)
velocity_errors = np.array(velocity_errors)
actuator_actions = np.array(actuator_actions)


# Define the directory to save the plots
plot_save_path = '/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Results'
os.makedirs(plot_save_path, exist_ok=True)

# Plotting Trajectory Tracking
plt.figure(figsize=(10, 6))
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 2], 'r-', label='Desired Trajectory')
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 2], 'b--', label='Actual Trajectory')
plt.title('Trajectory Tracking')
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.gca().set_aspect('equal', adjustable='box') 
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path, 'trajectory_tracking1.png'))
plt.show()

# Plotting Position Errors
time_steps = np.arange(len(position_errors))

plt.figure(figsize=(10, 6))
plt.plot(time_steps, position_errors, 'g-', label='Position Error')
plt.title('Position Errors')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'position_errors1.png'))
plt.show()

# Plotting Velocity Errors
time_steps = np.arange(len(velocity_errors))

plt.figure(figsize=(10, 6))
plt.plot(time_steps, velocity_errors, 'm-', label='Velocity Error')
plt.title('Velocity Errors')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'velocity_errors1.png'))
plt.show()


# Plotting Actuator Actions
plt.figure(figsize=(10, 6))
for i in range(actuator_actions.shape[1]):
    plt.plot(time_steps, actuator_actions[:, i], label=f'Actuator {i+1} Action')
plt.title('Actuator Actions')
plt.xlabel('Time Steps')
plt.ylabel('Action')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'actuator_actions1.png'))
plt.show()

# Plotting Individual End-Effector Positions with Desired Positions
plt.figure(figsize=(10, 6))
plt.plot(time_steps, actual_trajectory[:, 0], label='Actual X Position')
plt.plot(time_steps, actual_trajectory[:, 1], label='Actual Y Position')
plt.plot(time_steps, actual_trajectory[:, 2], label='Actual Z Position')
plt.plot(time_steps, desired_trajectory[:, 0], '--', label='Desired X Position')
plt.plot(time_steps, desired_trajectory[:, 1], '--', label='Desired Y Position')
plt.plot(time_steps, desired_trajectory[:, 2], '--', label='Desired Z Position')
plt.title('End-Effector Positions Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'end_effector_positions1.png'))
plt.show()

