from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from kamal_env3 import CableControlEnv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import mean_squared_error

# Initialize the environment and the model
model_save_path = '/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Models/DDPG_cable_control_circle100_shanbe_1.zip'
env = CableControlEnv(render_mode="rgb_array")
model = DDPG.load(model_save_path)
# model = PPO.load(model_save_path)
# Reset the environment
obs, info = env.reset()

# Lists to store data
frames = []
desired_trajectory = []
actual_trajectory = []
position_errors = []
velocity_errors = []
actuator_actions = []
tendon_lengths = []
desired_velocities = []
actual_velocities = []

# Run the model for a number of steps and collect data
for _ in range(195):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Collect data
    end_effector_pos = obs[:3]
    end_effector_vel = obs[3:6]
    position_error = obs[6]
    velocity_error = obs[7]
    tendon_len = obs[8:11]
    desired_pos = env.target

    desired_trajectory.append(desired_pos)
    actual_trajectory.append(end_effector_pos)
    position_errors.append(position_error)
    velocity_errors.append(velocity_error)
    actuator_actions.append(action)
    tendon_lengths.append(tendon_len)
    desired_velocities.append(env.target_vel)
    actual_velocities.append(end_effector_vel)

    # Render the environment and store the frame
    image = env.render()
    frames.append(image)
    if done or truncated:
        obs, info = env.reset()

# Save the frames as a video
video_filename = 'DDPG_cable_control_test3.mp4'
height, width, layers = frames[0].shape
video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print(f'Video saved as {video_filename}')

# Convert lists to numpy arrays
desired_trajectory = np.array(desired_trajectory)
actual_trajectory = np.array(actual_trajectory)
position_errors = np.array(position_errors)
velocity_errors = np.array(velocity_errors)
actuator_actions = np.array(actuator_actions)
tendon_lengths = np.array(tendon_lengths)
desired_velocities = np.array(desired_velocities)
actual_velocities = np.array(actual_velocities)

# Calculate RMSE for position and velocity
position_rmse = np.sqrt(mean_squared_error(desired_trajectory[:, :2], actual_trajectory[:, :2]))
velocity_rmse = np.sqrt(mean_squared_error(desired_velocities[:, :2], actual_velocities[:, :2]))
print(f'Position RMSE: {position_rmse}')
print(f'Velocity RMSE: {velocity_rmse}')

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
plt.savefig(os.path.join(plot_save_path, 'trajectory_tracking7.png'))
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
plt.savefig(os.path.join(plot_save_path,'position_errors7.png'))
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
plt.savefig(os.path.join(plot_save_path,'velocity_errors7.png'))
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
plt.savefig(os.path.join(plot_save_path,'actuator_actions7.png'))
plt.show()

# Plotting Tendon Lengths
plt.figure(figsize=(10, 6))
plt.plot(time_steps, tendon_lengths[:, 0], label='Tendon1')
plt.plot(time_steps, tendon_lengths[:, 1], label='Tendon2')
plt.plot(time_steps, tendon_lengths[:, 2], label='Tendon3')
plt.title('Tendons length')
plt.xlabel('Time Steps')
plt.ylabel('Length (m)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'tendon_length7.png'))
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
plt.savefig(os.path.join(plot_save_path,'end_effector_positions7.png'))
plt.show()

# Plotting Desired and Actual Velocities
plt.figure(figsize=(10, 6))
plt.plot(time_steps, desired_velocities[:, 0], '--', label='Desired X Velocity')
plt.plot(time_steps, desired_velocities[:, 1], '--', label='Desired Y Velocity')
plt.plot(time_steps, desired_velocities[:, 2], '--', label='Desired Z Velocity')
plt.plot(time_steps, actual_velocities[:, 0], label='Actual X Velocity')
plt.plot(time_steps, actual_velocities[:, 1], label='Actual Y Velocity')
plt.plot(time_steps, actual_velocities[:, 2], label='Actual Z Velocity')
plt.title('Desired and Actual Velocities Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_save_path,'desired_actual_velocities2.png'))
plt.show()

position_errors = np.array(position_errors)
velocity_errors = np.array(velocity_errors)

# Calculate statistical metrics for position errors
mean_position_error = np.mean(position_errors)
variance_position_error = np.var(position_errors)
std_dev_position_error = np.std(position_errors)

# Calculate statistical metrics for velocity errors
mean_velocity_error = np.mean(velocity_errors)
variance_velocity_error = np.var(velocity_errors)
std_dev_velocity_error = np.std(velocity_errors)

print("Mean Position Error:", mean_position_error)
print("Variance Position Error:", variance_position_error)
print("Standard Deviation Position Error:", std_dev_position_error)

print("Mean Velocity Error:", mean_velocity_error)
print("Variance Velocity Error:", variance_velocity_error)
print("Standard Deviation Velocity Error:", std_dev_velocity_error)