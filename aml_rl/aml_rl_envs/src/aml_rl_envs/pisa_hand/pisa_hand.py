import numpy as np
import pybullet as pb
from os.path import join
from aml_rl_envs.aml_rl_robot import AMLRlRobot
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer

class PisaHand(AMLRlRobot):

    def __init__(self, config=AML_RL_ROBOT_CONFIG, scale=0.5, hand_type='right', use_fixed_Base=False):

        self._config = config

        self._ee_indices = []

        self._hand_type = hand_type

        self._scale = scale

        self._use_fixed_base = use_fixed_Base

        AMLRlRobot.__init__(self, config)

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

        self.set_base_pose(pos=(0.,0.,0.), ori=(0.,0.,0.,1.))

        self._movable_jnts = self.get_movable_joints()

        self._jnt_postns=[0. for _ in range(len(self._movable_jnts))]
        
        self._motor_names = []
        
        self._motor_indices = []
        
        for i in self._movable_jnts:
            
            jnt_info = pb.getJointInfo(self._robot_id, i)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1:

                self._motor_names.append(str(jnt_info[1]))
                
                self._motor_indices.append(i)

        self.set_ctrl_mode()
