import numpy as np
import pybullet as pb
from os.path import join
from aml_rl_envs.aml_rl_hand import AMLRlHand
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer

class DLRHand(AMLRlHand):

    def __init__(self, config=AML_RL_ROBOT_CONFIG, scale=1., 
                       hand_type='right', use_fixed_base=False, 
                       pos=(0.,0.,0.), ori=(0.,0.,0.,1.), j_pos=None):

        self._config = config

        self._ee_indices = []

        self._hand_type = hand_type

        self._scale = scale

        self._use_fixed_base = use_fixed_base

        self._base_pos = pos

        self._base_ori = ori

        self._defualt_jnts = j_pos

        AMLRlHand.__init__(self, config, num_fingers=5)

        # setup_bullet_visualizer()

        self.reset()
 
    def reset(self):

        if self._hand_type == 'left':

            urdf_file  = join(self._config['urdf_root_path'], "dlr_hand/dlr_hit2_left.urdf")

        elif self._hand_type == 'right':

            urdf_file  = join(self._config['urdf_root_path'], "dlr_hand/dlr_hit2_right.urdf")

        else:

            raise Exception("Unknown hand type")

        self._robot_id = pb.loadURDF(urdf_file, globalScaling=self._scale, useFixedBase=self._use_fixed_base)

        self._finger_jnt_indices = [[1, 2, 3, 4], #thumb finger joints
                                    [5, 6, 7, 8], # index finger joints
                                    [9, 10, 11, 12], # middle finger joints
                                    [13, 14, 15, 16], # ring finger joints
                                    [17, 18, 19, 20] ] # little finger joints

        self.set_base_pose(pos=self._base_pos, ori=self._base_ori)

        self.setup_hand()