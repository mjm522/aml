import numpy as np
import pybullet as p
from dmp_gaits import DMPGaits
from dmp_manipulation import DMPManptln
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG


class DMPConPlan():

    def __init__(self):

        HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

        self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False, keep_obj_fixed=True)
        
        self._gaiter = DMPGaits(env=self._env)
        
        self._maniptn =  DMPManptln(env=self._env)


    def run(self):

        man_primitive = self._maniptn.get_man_primitive()

        gain = 10.

        while True:

            for k in range(man_primitive.shape[0]):

                for j in range(self._env._num_fingers): #dg._num_fingers
                    
                    if self._env._ctrl_type == 'pos':
                        cmd = self._env._hand.inv_kin(j, man_primitive[k, 3*j:3*j+3].tolist())
                    else:
                        cmd = gain*self._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=man_primitive[k, 3*j:3*j+3], goal_vel=np.zeros(3), dt=0.01)
                
                    if np.any(np.isnan(cmd)):
                        continue

                    self._env._hand.apply_action(j, cmd)
                    
                self._env.simple_step()

            for m in range(self._env._num_fingers):

                gait_primitive  = self._gaiter.update_dmp_params(finger_idx=m)
                for n in range(gait_primitive.shape[0]):

                    if self._env._ctrl_type == 'pos':
                        cmd = self._env._hand.inv_kin(m, gait_primitive[n, 1:].tolist())
                    else:
                        cmd = gain*self._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=man_primitive[k, 3*j:3*j+3], goal_vel=np.zeros(3), dt=0.01)
                    
                    if np.any(np.isnan(cmd)):
                        continue

                    self._env._hand.apply_action(m, cmd)
                
                    self._env.simple_step()


dcp = DMPConPlan()
dcp.run()


