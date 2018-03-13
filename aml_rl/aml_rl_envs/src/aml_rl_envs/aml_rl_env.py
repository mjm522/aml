import gym
import numpy as np
import pybullet as pb
from gym import spaces
from gym.utils import seeding
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer


class AMLRlEnv(gym.Env):

    metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
    }

    def __init__(self, config, set_gravity=True):

        self._pb = pb

        self._config = config

        self._observation = []

        self._env_step_counter = 0

        self._terminated = 0

        self._cam_dist = self._config['cam_dist']
        
        self._cam_yaw = self._config['cam_yaw']
        
        self._cam_pitch = self._config['cam_pitch']

        self._time_step = self._config['time_step']

        self._ctrl_type = self._config['ctrl_type'] 

        self._urdf_root_path = self._config['urdf_root_path']

        self._renders = self._config['renders']

        self._max_steps = self._config['max_steps']

        self._large_val_obs = self._config['large_val_observation']
        
        self.viewer = None
         
        setup_bullet_visualizer(self._renders)

        self._set_gravity = set_gravity

    
    def set_space_lims(self, obs_dim, action_dim, action_high, action_low, is_discrete=False):

        obs_high = np.array([self._large_val_obs] * obs_dim) 

        if (is_discrete):
            
            self._action_space = spaces.Discrete(action_dim)

        else:  

            self._action_bound = 1
            
            if action_high is None:
                
                action_high = np.array([self._action_bound] * action_dim)

            if action_low is None:
                
                action_low = -action_high

            self.action_space = spaces.Box(action_low, action_high)
        
        self.observation_space = spaces.Box(-obs_high, obs_high)


    def setup_env(self):

        self._terminated = 0
        
        pb.resetSimulation()
        
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        
        pb.setTimeStep(self._time_step)

        if self._set_gravity:

            pb.setGravity(0,0,-9.8)


    def simple_step(self):

        pb.stepSimulation()

    def __del__(self):
        
        pb.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _step(self, action):

        raise NotImplementedError("Must be implemented in the subclass")

    def step2(self, action, Kp):

        raise NotImplementedError("Must be implemented in the subclass")

    def _render(self, mode="rgb_array", close=False):
        
        if mode != "rgb_array":
            
            return np.array([])
        
        base_pos,orn = self._pb.getBasePositionAndOrientation(self._robot._robot_id)
        
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        
        proj_matrix = self._pb.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._config['render_wigth'])/self._config['render_height'],
            nearVal=0.1, farVal=100.0)
        
        (_, _, px, _, _) = self._pb.getCameraImage(
            width=self._config['render_wigth'], height=self._config['render_height'], viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._pb.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array = np.array(px)
        
        rgb_array = rgb_array[:, :, :3]

        return rgb_array

