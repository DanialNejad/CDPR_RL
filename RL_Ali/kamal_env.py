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
        xml_path = os.path.abspath("./assets/Kamal_final_ver2.xml")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip = 5,
            observation_space=observation_space,
            **kwargs
        )
        
        self.max_timesteps = max_timesteps
        self.current_timesteps = 0
        self.phi = 0
        self.target = self._sample_target()

    def _sample_target(self):
        C = [0, 1.15]
        r = 0.2
        target_x = C[0] + r*np.sin(self.phi)
        # target_x = 0.4       
        target_y = -0.03
        # target_z = 0.5
        target_z = C[1] + r*np.cos(self.phi)
        print('target: ',[target_x, target_y, target_z])
        return np.array([target_x, target_y, target_z])

    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        # print('actions: ',action)
        self.current_timesteps += 1 
        # print('time step: ',self.current_timesteps)
        self.phi = self.phi + 0.7*(1-np.tanh(0.6*self.data.time))
        print('time step: ', self.data.time)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        

        truncated = self.current_timesteps == self.max_timesteps


        # If the task is done, give an additional reward
        if done:
            reward += 100

        return obs, reward, bool(done), bool(truncated), {}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        # print('initial position: ', qpos)
        # Reset the end effector position and velocity
        qpos[0] = self.init_qpos[0]
        qpos[1] = self.init_qpos[1]
        # qvel[-2:] = self.init_qvel[-2:] + self.np_random.uniform(low=-0.01, high=0.01, size=2)

        self.set_state(qpos, qvel)
        # self.set_state(qpos)
        self.current_timesteps = 0
        self.target = self._sample_target()
        # print('position end effector: ', qpos)        
        return self._get_obs()

    def _get_obs(self):
        # Get the lengths of the tendons from the 'tendonpos' sensors
        tendon1_length = np.array([self.data.sensordata[0]])
        tendon2_length = np.array([self.data.sensordata[1]])
        tendon3_length = np.array([self.data.sensordata[2]])
        # Get the position of the end effector from the 'framepos' sensor
        end_effector_pos_x = np.array([self.data.sensordata[3]])
        end_effector_pos_y = np.array([self.data.sensordata[4]])
        end_effector_pos_z = np.array([self.data.sensordata[5]])
        # Get the velocity of the end effector from the 'framepos' sensor
        end_effector_vel_x = np.array([self.data.sensordata[6]])
        end_effector_vel_y = np.array([self.data.sensordata[7]])
        end_effector_vel_z = np.array([self.data.sensordata[8]])
        # Get the acceleration of the end effector from the 'framepos' sensor
        end_effector_acc_x = np.array([self.data.sensordata[9]])
        end_effector_acc_y = np.array([self.data.sensordata[10]])
        end_effector_acc_z = np.array([self.data.sensordata[11]])

        end_effector_pos = np.array([end_effector_pos_x, end_effector_pos_y, end_effector_pos_z]).flatten()
        end_effector_vel = np.array([end_effector_vel_x, end_effector_vel_y, end_effector_vel_z]).flatten()
        end_effector_acc = np.array([end_effector_acc_x, end_effector_acc_y, end_effector_acc_z]).flatten()

        # Concatenate the necessary components into the observation
        observation = np.concatenate([end_effector_pos,  end_effector_vel, end_effector_acc, self.target, tendon1_length, tendon2_length, tendon3_length])
        # print(observation)

        return observation

    # def _compute_reward(self, obs):
    #     end_effector_pos = obs[:3]
    #     target_pos = obs[3:6]
    #     distance = np.linalg.norm(end_effector_pos - target_pos)

    #     return -distance
    
    def _compute_reward(self, obs):
        end_effector_pos = obs[:3]
        target_pos = obs[3:6]
        tracking_error = np.linalg.norm(end_effector_pos - target_pos)

        # R1: Reward for minimizing the tracking error
        R1 = np.exp(-3 * tracking_error)
        distance = R1
        # # R2: Penalize the action values to avoid high cable tensions
        # action = self.last_action  # Assuming you store the last action applied to the environment
        # R2 = -0.05 * np.sum(np.square(action))

        # # R3: Reward for reducing the error derivative
        # error_derivative = np.linalg.norm(self.last_error - tracking_error)
        # R3 = -error_derivative
        # if error_derivative <= -10:
        #     R3 = -10

        # total_reward = R1 + R2 + R3
        # self.last_error = tracking_error  # Update last error for next step

        return distance

    def _is_done(self, obs):
        distance = np.linalg.norm(obs[:3] - obs[3:6])
        # print(distance)
        return distance < 0.005

    # def render(self, mode='human'):
    #     if mode == 'rgb_array':
    #         self.viewer.cam.lookat[0] = 0  
    #         self.viewer.cam.lookat[1] = -2
    #         self.viewer.cam.lookat[2] = 0
    #         self.viewer.cam.distance = 6
    #         self.viewer.cam.elevation = -25
    #         self.viewer.cam.azimuth = -90
    #     return super().render(mode=mode)
    # cam = mj.MjvCamera()     
    # cam.azimuth = -90
    # cam.elevation = -25
    # cam.distance = 6
    # cam.lookat = np.array([0.0, -2, 0])
