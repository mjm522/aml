import os
import time
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from aml_io.io_tools import load_data
from scipy.optimize import minimize, linprog
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_lfd.promp.discrete_promp import MultiplePROMPs
from aml_rl_envs.hand.hand_obst_env import HandObstacleEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.collect_demo import plot_demo, get_demo

np.random.seed(123)

class PROMPPolicy():

    def __init__(self, env=None):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObstacleEnv(action_dim=18, config=HAND_OBJ_CONFIG, randomize_box_ori=False,  keep_obj_fixed=False)
        
        else:

            self._env = env

        self._promp = None

        self.encode_promps()


    def get_primitives(self):

        self._data_root = os.environ['AML_DATA'] + '/aml_playground/manipulation/'

        j0_list = []
        j1_list = []
        j2_list = []
        
        for k in range(3):

            file_name = self._data_root + 'demo1/' + 'collect_man_data%d.pkl'%(k)

            j0, j1, j2 = self.parse_data_file(file_name)[0].tolist()

            j0_list.append(copy.deepcopy(j0)[1200:4800])
            
            j1_list.append(copy.deepcopy(j1)[1200:4800])
            
            j2_list.append(copy.deepcopy(j2)[1200:4800])


        # for j0, j1, j2 in zip(j0_list, j1_list, j2_list) :
        #     plt.subplot(311)
        #     plt.plot(j0)
        #     plt.subplot(312)
        #     plt.plot(j1)
        #     plt.subplot(313)
        #     plt.plot(j2)

        # plt.show()

        return [j0_list, j1_list, j2_list]


    def parse_data_file(self, file_name):

        data = load_data(file_name)

        demo_js_pos = []
        demo_js_vel = []
        demo_js_acc = []

        old_vel = np.zeros(6)

        for data_point in data:

            js_pos = self._env._hand.convert_fin_jnt_poss_to_list(data_point['robot_state']['pos_js'], only_mov_jnts=True)

            js_vel = self._env._hand.convert_fin_jnt_poss_to_list(data_point['robot_state']['vel_js'], only_mov_jnts=True) 

            js_acc = (np.asarray(js_vel) - old_vel).tolist() #/self._env._time_step

            # inv_dyn = 50*self._env._hand.get_inv_dyn(js_pos, js_vel, js_acc)

            old_vel = np.asarray(js_vel)

            demo_js_pos.append(np.asarray(js_pos[:3]))
            
            demo_js_vel.append(np.asarray(js_vel[:3]))
            
            demo_js_acc.append(np.asarray(js_acc[:3]))
            
        return np.asarray(demo_js_pos).T, np.asarray(demo_js_vel).T, np.asarray(demo_js_acc).T


    def encode_promps(self):

        primitive_list = self.get_primitives()

        self._promp = MultiplePROMPs(multiple_dim_data=primitive_list)
        
        self._promp.train()


    def generate_promp_traj(self):

        if self._promp is None:
            
            return

        new_mu_traj, new_mu_Dtraj = self._promp.generate_trajectory(phase_speed=1., randomness=1e-4)

        return new_mu_traj, new_mu_Dtraj


def main():

    pg = PROMPPolicy()

    promp_primitive = pg.generate_promp_traj()[0]

    promp_primitive = np.flip(promp_primitive.T, axis=1).T

    while True:

        promp_primitive, promp_Dprimitive = pg.generate_promp_traj(finger_idx=j)

        for k in range(promp_primitive.shape[0]):
            
            pg._env._hand.apply_action(0, promp_primitive[k, :])
            
            pg._env.simple_step()

            ee_poss, ee_oris, ee_vels, ee_omgs = pg._env._hand.get_ee_states(as_tuple=True)
            
            jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = pg._env._hand.get_jnt_states()


if __name__ == '__main__':
    main()
