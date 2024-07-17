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

    def __init__(self, num_points=50, frame_skip=5, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        xml_path = os.path.abspath("./assets/Kamal_final_ver2.xml")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        
        self.num_points = num_points
        self.frame_skip = frame_skip
        self.theta_increment = 2 * np.pi / num_points  # Increment theta to complete circle in num_points steps
        self.max_timesteps = 2*int(2 * np.pi / self.theta_increment)  # Calculate max timesteps to complete one circle
        self.current_timesteps = 0
        self.w1 = 1.0  
        self.w2 = 0.01
        
        # Parameters for circular trajectory
        self.radius = 0.5
        self.center = np.array([0.0, -0.03, 0.8])
        self.theta = 0
        self.points_reached = 0  # Counter for the number of target points reached

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            **kwargs
        )

    def _sample_target(self):
        return self._target_trajectory(self.theta)

    def _target_trajectory(self, theta):
        target_x = self.center[0] + self.radius * np.cos(theta)
        target_y = self.center[1]
        target_z = self.center[2] + self.radius * np.sin(theta)
        return np.array([target_x, target_y, target_z])

    def _target_trajectory_velocity(self, theta):
        target_vel_x = -self.radius * np.sin(theta) * self.theta_increment
        target_vel_y = 0
        target_vel_z = self.radius * np.cos(theta) * self.theta_increment
        return np.array([target_vel_x, target_vel_y, target_vel_z])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_timesteps += 1 

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        
        # Update theta for the next step to move along the circular trajectory
        self.theta += self.theta_increment
        if self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi

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
        self.current_timesteps = 0
        self.theta = 0  # Reset theta to start the circular trajectory from the beginning
        self.points_reached = 0  # Reset the points reached counter
        print('position end effector: ', qpos)        
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
        
        self.target = self._target_trajectory(self.theta)
        self.target_vel = self._target_trajectory_velocity(self.theta)
        
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
            self.points_reached += 1
        # Done if all points in the trajectory have been reached
        return self.points_reached >= self.num_points

    def _is_truncated(self):
        # Check if the agent has completed the circle
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
