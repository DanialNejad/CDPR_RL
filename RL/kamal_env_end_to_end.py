import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

class CableControlEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, max_timesteps=1000, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        xml_path = os.path.abspath("/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/assets/Kamal_final_ver2.xml")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        
        self.max_timesteps = max_timesteps
        self.current_timesteps = 0
        self.w1 = 1.0  
        self.w2 = 0.01
        self.initial_point = None
        self.target = None

    def _sample_point(self):
        x = np.random.uniform(-0.5, 0.5)
        y = -0.03
        z = np.random.uniform(0.3, 1.3)
        return np.array([x, y, z])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_timesteps += 1 

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        
        truncated = self.current_timesteps >= self.max_timesteps

        # If the task is done, give an additional reward
        if done:
            reward += 100
            self.target = self._sample_point()  # Sample a new target point

        return obs, reward, bool(done), bool(truncated), {}

    def reset_model(self, initial_pos=None, target_pos=None):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        if initial_pos is not None:
            qpos[:3] = initial_pos  # Ensure the correct shape assignment
        else:
            qpos[0] = self.init_qpos[0] + self.np_random.uniform(low=-0.7, high=0.7, size=1)
            qpos[1] = self.init_qpos[1] + self.np_random.uniform(low=0.3, high=1.3, size=1)

        self.set_state(qpos, qvel)
        self.current_timesteps = 0
        self.target = target_pos if target_pos is not None else self._sample_point()
        print('Initial Point:', initial_pos)
        print('Target Point:', self.target)
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
        
        position_error = self.target - end_effector_pos
        velocity_error = -end_effector_vel 
        
        position_error_norm = np.linalg.norm(position_error, 2)
        velocity_error_norm = np.linalg.norm(velocity_error, 2)

        position_error_norm = np.array([position_error_norm])
        velocity_error_norm = np.array([velocity_error_norm])

        observation = np.concatenate([end_effector_pos, end_effector_vel, position_error_norm, velocity_error_norm, tendon1_length, tendon2_length, tendon3_length, self.target])

        return observation

    def _compute_reward(self, obs):
        X_e = obs[6]
        Xdot_e = obs[7]
        reward = -self.w1 * X_e - self.w2 * Xdot_e
        return reward

    def _is_done(self, obs):
        distance = np.linalg.norm(obs[:3] - self.target)
        return distance < 0.005