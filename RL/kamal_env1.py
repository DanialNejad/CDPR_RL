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
        
        # Define the observation space with the correct shape
        observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        
        self.max_timesteps = max_timesteps
        self.current_timesteps = 0
        self.targets = self._generate_circular_targets()
        self.current_target_index = 0

    def _generate_circular_targets(self, radius=0.5, center=(0, -0.03, 0.8), num_targets=6):
        targets = []
        angles = np.linspace(0, 2 * np.pi, num_targets, endpoint=False)
        for angle in angles:
            target_x = center[0] + radius * np.cos(angle)
            target_y = center[1]
            target_z = center[2] + radius * np.sin(angle)
            targets.append(np.array([target_x, target_y, target_z]))
        return targets

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_timesteps += 1 

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = False
        truncated = self.current_timesteps >= self.max_timesteps

        # Check if the target is reached
        if self._is_target_reached(obs):
            reward += 100
            self.current_target_index += 1
            if self.current_target_index >= len(self.targets):
                reward += 500  # Additional reward for completing all targets
                done = True  # End episode after completing the sequence

        return obs, reward, bool(done), bool(truncated), {}

    def reset_model(self, new_targets=None):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[0] = self.init_qpos[0] + self.np_random.uniform(low=-0.7, high=0.7, size=1)
        qpos[1] = self.init_qpos[1] + self.np_random.uniform(low=0.3, high=1.3, size=1)

        self.set_state(qpos, qvel)
        self.current_timesteps = 0
        if new_targets is not None:
            self.targets = new_targets
        else:
            self.targets = self._generate_circular_targets()
        self.current_target_index = 0
        return self._get_obs()

    def reset(self, seed=None, options=None):
        self.current_timesteps = 0
        self.np_random, seed = utils.seeding.np_random(seed)
        if options is not None and 'new_targets' in options:
            self.targets = options['new_targets']
        else:
            self.targets = self._generate_circular_targets()
        self.current_target_index = 0
        obs = self.reset_model()
        return obs, {}

    def _get_obs(self):
        end_effector_pos_x = np.array([self.data.sensordata[3]])
        end_effector_pos_y = np.array([self.data.sensordata[4]])
        end_effector_pos_z = np.array([self.data.sensordata[5]])
        tendon1_length = np.array([self.data.sensordata[0]])
        tendon2_length = np.array([self.data.sensordata[1]])
        tendon3_length = np.array([self.data.sensordata[2]])

        # Current target should be a part of the observation
        current_target = self.targets[self.current_target_index]
        observation = np.concatenate([end_effector_pos_x, end_effector_pos_y, end_effector_pos_z, current_target, tendon1_length, tendon2_length, tendon3_length])
        return observation

    def _compute_reward(self, obs):
        end_effector_pos = obs[:3]
        target_pos = obs[3:6]
        tracking_error = np.linalg.norm(end_effector_pos - target_pos)

        R1 = np.exp(-3 * tracking_error)
        distance = R1

        return distance

    def _is_target_reached(self, obs):
        end_effector_pos = obs[:3]
        target_pos = obs[3:6]
        distance = np.linalg.norm(end_effector_pos - target_pos)
        return distance < 0.005
