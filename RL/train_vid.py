
from kamal_env1 import CableControlEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import cv2
import numpy as np
import os

class VideoEvalCallback(EvalCallback):
    def __init__(self, *args, video_folder="./logs/videos", render_freq=5000, **kwargs):
        super(VideoEvalCallback, self).__init__(*args, **kwargs)
        self.video_folder = video_folder
        self.render_freq = render_freq

        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

    def _on_step(self) -> bool:
        result = super(VideoEvalCallback, self)._on_step()
        if self.n_calls % self.render_freq == 0:
            self._render_and_save_video()
        return result

    def _render_and_save_video(self):
        frames = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done, truncated = False, False
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]
                frame = self.eval_env.render()
                frames.append(frame)

        video_filename = f'{self.video_folder}/ppo_cable_control_step_{self.num_timesteps}.mp4'
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()
        print(f'Video saved as {video_filename}')

# Initialize your environment
train_env = CableControlEnv(render_mode="rgb_array")

# Separate evaluation env
eval_env = CableControlEnv(render_mode="rgb_array")

# Directory to save the video
video_folder = "./logs/videos"

# Use deterministic actions for evaluation
eval_callback = VideoEvalCallback(eval_env, best_model_save_path="./logs/",
                                  log_path="./logs/", eval_freq=500,
                                  deterministic=True, render=False,
                                  video_folder=video_folder, render_freq=5000)

# Define the model
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_cable_control_tensorboard/")

# Train the model
model.learn(total_timesteps=800000, callback=eval_callback)
model.save("ppo_cable_control")
