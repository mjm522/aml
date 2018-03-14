import os
import copy
import random
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.task.man_object import ManObject
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG
from aml_rl_envs.pisa_hand.pisa_hand import PisaHand
from aml_rl_envs.hand.config import HAND_OBJ_CONFIG, HAND_CONFIG

class PisaHandObjEnv(AMLRlEnv):

    def __init__(self,  action_dim,  demo2follow=None, 
                        action_high=None, action_low=None, 
                        randomize_box_ori=True, keep_obj_fixed = True, 
                        config=HAND_OBJ_CONFIG):

        self._goal_block_pos = np.array([0, 0, 0.75]) #x,y,z
        
        self._goal_obj_ori = np.array([0.0, 0.00, -1.158])

        self._randomize_box_ori = randomize_box_ori

        self._demo2follow = demo2follow

        self._config = config

        AMLRlEnv.__init__(self, config, set_gravity=True)

        self._reset(obj_base_fixed = keep_obj_fixed)

        obs_dim = 50

        self.set_space_lims(obs_dim, action_dim, action_high, action_low)

        self._seed()


    def _reset(self, box_pos=[-0.6, 0., 0.9], obj_base_fixed = True):

        self.setup_env()

        if self._randomize_box_ori:
            
            box_ori = pb.getQuaternionFromEuler([0., 0., np.random.uniform(-0.08*np.pi, 0.08*np.pi)])
        
        else:
            
            box_ori = [0.,0.,0.,1]

        self._world_id = pb.loadURDF(join(self._urdf_root_path,"plane.urdf"))
        
        pb.resetBasePositionAndOrientation(self._world_id, [0., 0., -0.1], [0.,0.,0.,1])
        
        self._object = ManObject(urdf_root_path=self._config['urdf_root_path'], time_step=self._config['time_step'], 
                                  pos=box_pos, ori=box_ori, scale=0.3, 
                                  use_fixed_Base = obj_base_fixed, obj_type='cube')
        
        base_hand_pos  = [0., 0., 0.7]
        
        base_hand_ori  = pb.getQuaternionFromEuler([0., 3*np.pi/2, 0.])

        self._hand = PisaHand(config=HAND_CONFIG, pos=base_hand_pos, ori=base_hand_ori, scale=3., use_fixed_base=True)

        self._num_fingers = self._hand._num_fingers
        
        self._env_step_counter = 0
        
        pb.stepSimulation()
      
        return np.array(self._observation)
