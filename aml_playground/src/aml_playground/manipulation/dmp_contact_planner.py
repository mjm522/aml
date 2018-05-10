import os
import copy
import numpy as np
from dmp_gaits import DMPGaits
from dmp_manipulation import DMPManptln
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data

class DMPConPlan():

    def __init__(self):

        HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

        self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False, keep_obj_fixed=True)
        
        self._gaiter = DMPGaits(env=self._env)
        
        self._maniptn =  DMPManptln(env=self._env)

        self._policy = []


    def collect_data(self, cmd):

        self._policy.append(copy.deepcopy(cmd))


    def run(self):

        man_primitive = self._maniptn.get_man_primitive()

        gain = 10.

        done = False

        while not done:

            for k in range(man_primitive.shape[0]):

                joint_cmd = np.array([])

                for j in range(self._env._num_fingers): #dg._num_fingers
                    
                    if self._env._ctrl_type == 'pos':
                        cmd = self._env._hand.inv_kin(j, man_primitive[k, 3*j:3*j+3].tolist())
                    else:
                        cmd = gain*self._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=man_primitive[k, 3*j:3*j+3], goal_vel=np.zeros(3), dt=0.01)
                
                    if np.any(np.isnan(cmd)):
                        continue

                    self._env._hand.apply_action(j, cmd)

                    joint_cmd = np.r_[joint_cmd, cmd]

                self.collect_data(joint_cmd)
                    
                self._env.simple_step()

            for m in range(self._env._num_fingers):

                gait_primitive  = self._gaiter.update_dmp_params(finger_idx=m)

                joint_cmd = np.asarray(self._env._hand.get_jnt_state()[0])

                for n in range(gait_primitive.shape[0]):

                    if self._env._ctrl_type == 'pos':
                        cmd = self._env._hand.inv_kin(m, gait_primitive[n, 1:].tolist())
                    else:
                        cmd = gain*self._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=man_primitive[k, 3*j:3*j+3], goal_vel=np.zeros(3), dt=0.01)
                    
                    if np.any(np.isnan(cmd)):
                        continue

                    self._env._hand.apply_action(m, cmd)

                    joint_cmd[3*m:3*m+3]=np.asarray(cmd)

                    self.collect_data(joint_cmd)
                
                    self._env.simple_step()

            done = True

        save_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy.csv', self._policy)


dcp = DMPConPlan()
dcp.run()


