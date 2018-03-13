import numpy as np
import pybullet as pb
from os.path import join
from aml_rl_envs.aml_rl_hand import AMLRlHand
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer

class PisaHand(AMLRlHand):

    def __init__(self, config=AML_RL_ROBOT_CONFIG, scale=0.5, hand_type='right', use_fixed_Base=False):

        self._config = config

        self._ee_indices = []

        self._hand_type = hand_type

        self._scale = scale

        self._use_fixed_base = use_fixed_Base

        AMLRlHand.__init__(self, config, num_fingers=5)

        setup_bullet_visualizer()

        self.reset()
 
    def reset(self):

        if self._hand_type == 'left':

            urdf_file  = join(self._config['urdf_root_path'], "pisa_iit_hand/pisa_hand_left.urdf")

        elif self._hand_type == 'right':

            urdf_file  = join(self._config['urdf_root_path'], "pisa_iit_hand/pisa_hand_right.urdf")

        else:

            raise Exception("Unknown hand type")

        self._robot_id = pb.loadURDF(urdf_file, globalScaling=self._scale, useFixedBase=self._use_fixed_base)

        self._finger_jnt_indices = [[5, 6, 7, 8, 9, 10], #thumb finger joints
                                    [11, 12, 13, 14, 15, 16, 17], # index finger joints
                                    [18, 19, 20, 21, 22, 23, 24], # middle finger joints
                                    [25, 26, 27, 28, 29, 30, 31], # ring finger joints
                                    [32, 33, 34, 35, 36, 37, 38] ] # little finger joints

        self.set_base_pose(pos=(0.,0.,0.), ori=(0.,0.,0.,1.))

        self.setup_hand()