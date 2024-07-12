from stable_baselines3 import PPO
from kamal_env_end_to_end import CableControlEnv
import cv2
import imageio
import numpy as np
# Initialize the environment and the model
env = CableControlEnv(render_mode="human")
model = PPO.load("ppo_cable_control_end_to_end.zip")

# Set specific initial and target points for testing
def test_env(initial_point, target_point):
    env.initial_point = initial_point
    env.target = target_point

    # Reset the environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation from the tuple

    frames = []

    # Run the model for a number of steps and collect frames
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the observation from the tuple

        # Render the environment and store the frame
        image = env.render()
        frames.append(image)
        if done or truncated:
            break

    # Save the frames as a video
    video_filename = 'ppo_cable_control_test.mp4'
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f'Video saved as {video_filename}')

# Example test with specific initial and target points
initial_point = np.array([0.3, -0.03, 1.5])
target_point = np.array([0.5, -0.03, 1.2])
test_env(initial_point, target_point)