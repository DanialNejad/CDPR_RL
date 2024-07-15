import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import matplotlib.pyplot as plt

class CableControlEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, frame_skip=5, max_timesteps=1000, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        xml_path = os.path.abspath("/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/assets/Kamal_final_ver2.xml")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        
        self.frame_skip = frame_skip
        self.max_timesteps = max_timesteps
        self.w1 = 1.0  
        self.w2 = 0.01
        
        # Parameters for circular trajectory
        self.C = [0, 1.15]
        self.r = 0.2
        self.phi = 0
        self.current_timesteps = 0
        self.const = 5/self.max_timesteps
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            **kwargs
        )

    def _sample_target(self):
        return self._target_trajectory()

    def _target_trajectory(self):
        target_x = self.C[0] + self.r * np.sin(self.phi)
        target_y = -0.03
        target_z = self.C[1] + self.r * np.cos(self.phi)
        return np.array([target_x, target_y, target_z])

    def _target_trajectory_velocity(self):
        dphi = 0.7 * (1 - np.tanh(0.6 * self.const * self.current_timesteps))
        target_vel_x = self.r * np.cos(self.phi) * dphi
        target_vel_y = 0
        target_vel_z = -self.r * np.sin(self.phi) * dphi
        return np.array([target_vel_x, target_vel_y, target_vel_z])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        
        # Update phi for the next step to move along the trajectory
        self.phi += 0.7 * (1 - np.tanh(0.6 * self.const * self.current_timesteps))
        self.current_timesteps += 1

        # Check for truncation conditions
        truncated = self._is_truncated()

        # If the task is done, give an additional reward
        if done:
            reward += 100

        return obs, reward, bool(done), bool(truncated), {}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        print('initial position: ', qpos)
        # Reset the end effector position and velocity
        qpos[0] = self.init_qpos[0]  + self.np_random.uniform(low=-0.7, high=0.7, size=1)
        qpos[1] = self.init_qpos[1]  + self.np_random.uniform(low=0.3, high=1.3, size=1)

        self.set_state(qpos, qvel)
        self.phi = 0  # Reset phi to start the trajectory from the beginning
        self.current_timesteps = 0  # Reset the timestep counter
        print('position end effector: ', qpos)        
        return self._get_obs()

    def _get_obs(self):
        # Get the position of the end effector from the 'framepos' sensor
        end_effector_pos_x = np.array([self.data.sensordata[3]])
        end_effector_pos_y = np.array([self.data.sensordata[4]])
        end_effector_pos_z = np.array([self.data.sensordata[5]])
        # Get the lengths of the tendons from the 'tendonpos' sensors
        tendon1_length = np.array([self.data.sensordata[0]])
        tendon2_length = np.array([self.data.sensordata[1]])
        tendon3_length = np.array([self.data.sensordata[2]])

        end_effector_vel_x = np.array([self.data.sensordata[6]])
        end_effector_vel_y = np.array([self.data.sensordata[7]])
        end_effector_vel_z = np.array([self.data.sensordata[8]])
        
        end_effector_pos = np.array([end_effector_pos_x, end_effector_pos_y, end_effector_pos_z]).flatten()
        end_effector_vel = np.array([end_effector_vel_x, end_effector_vel_y, end_effector_vel_z]).flatten()
        
        self.target = self._target_trajectory()
        self.target_vel = self._target_trajectory_velocity()
        
        position_error = self.target - end_effector_pos
        velocity_error = self.target_vel - end_effector_vel 
        
        position_error_norm = np.linalg.norm(position_error, 2)
        velocity_error_norm = np.linalg.norm(velocity_error, 2)

        position_error_norm = np.array([position_error_norm])
        velocity_error_norm = np.array([velocity_error_norm])
        
        observation = np.concatenate([end_effector_pos, end_effector_vel, position_error_norm, velocity_error_norm])
        return observation

    def _compute_reward(self, obs):
        X_e = obs[6]  # position_error_norm
        Xdot_e = obs[7]  # velocity_error_norm
        reward = -self.w1 * X_e - self.w2 * Xdot_e
        return reward

    def _is_done(self, obs):
        distance = np.linalg.norm(obs[:3] - self.target)
        if distance < 0.005:  # Threshold for reaching a target point
            return True
        return False

    def _is_truncated(self):
        # Check if the agent has completed the circle based on the time steps
        return self.current_timesteps >= self.max_timesteps

    # def render(self, mode='human'):
    #     if mode == 'rgb_array':
    #         self.viewer.cam.lookat[0] = 0  
    #         self.viewer.cam.lookat[1] = -2
    #         self.viewer.cam.lookat[2] = 0
    #         self.viewer.cam.distance = 6
    #         self.viewer.cam.elevation = -25
    #         self.viewer.cam.azimuth = -90
    #     return super().render(mode=mode)
