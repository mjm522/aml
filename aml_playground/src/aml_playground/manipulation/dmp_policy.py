import os
import copy
import numpy as np
import scipy.fftpack as ft
import matplotlib.pyplot as plt
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data

discrete_dmp_config['dof'] = 3

class DMPPolicy():

    def __init__(self):

        HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

        self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False, keep_obj_fixed=True)
        
        self._man_policy  = load_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_manipulation.csv')
        self._gait_policy = load_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_gaiting.csv')
        self._policy = load_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_edited.csv')

        self.encode_dmps()

        self._count = 0

        self._next = 'man'


    def run(self):

        while True:

            for cmd in self._policy:

                self._env._hand.apply_jnt_ctrl(cmd + np.random.randn(12)*0.0)

                self._env.simple_step()

        raw_input("Press any key")


    def plot(self):

        policy = copy.deepcopy(self._policy)

        # save_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_manipulation.csv', policy[:701, :])
        # save_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_gaiting.csv', policy[701:, :])

        # policy = np.delete(policy, range(701,2300), axis=0)
        # policy = np.delete(policy, range(1301,3000), axis=0)
        # policy = np.delete(policy, range(2601,3600), axis=0)
        # policy = np.delete(policy, range(2000,2550), axis=0)
        # policy = np.delete(policy, range(2700, 4350), axis=0)
        # policy = np.delete(policy, range(3375, policy.shape[0]), axis=0)

        # save_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/dmp/policy_edited.csv', policy)
        
        # plt.figure("Manipulation")
        # plt.plot(policy[:701, :])

        # plt.figure("Gaiting")
        # plt.plot(policy[701:, :])

        # plt.show()

    def encode_dmps(self):

        self._man_dmp = {}
        self._man_dmp['config'] = discrete_dmp_config
        self._man_dmp['obj'] = DiscreteDMP(config=discrete_dmp_config)

        self._man_dmp['obj'].load_demo_trajectory(self._man_policy[:,0:3])
        self._man_dmp['obj'].train()

        self._gait_dmp = {}
        self._gait_dmp['config'] = discrete_dmp_config
        self._gait_dmp['obj'] = DiscreteDMP(config=discrete_dmp_config)

        self._gait_dmp['obj'].load_demo_trajectory(self._gait_policy[:,0:3])
        self._gait_dmp['obj'].train()


    def update_dmp_params(self, dmp_type, phase_start=1., speed=1., goal_offset=np.array([0., 0., 0.]), start_offset=np.array([0., 0., 0.]), external_force=None):

        if dmp_type == 'man':
            curr_dmp = copy.deepcopy(self._man_dmp)
        elif dmp_type == 'gait':
            curr_dmp = copy.deepcopy(self._gait_dmp)
        
        dmp    = curr_dmp['obj']
        config = curr_dmp['config']

        config['y0'] = dmp._traj_data[0, 1:] + start_offset
        config['dy'] = np.array([0., 0., 0.])
        config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
        config['tau'] = 1./speed
        config['phase_start'] = phase_start

        if external_force is None:
            external_force = np.array([0.,0.,0.,0.])
            config['type'] = 1
        else:
            config['type'] = 3

        config['extForce'] = external_force

        new_dmp_traj = dmp.generate_trajectory(config=config)

        return new_dmp_traj['pos']

    def man_fsm(self):

        policy = [{'fin':0, 'type':'man'},{'fin':1, 'type':'man'},
                  {'fin':2, 'type':'man'},{'fin':3, 'type':'man'}]

        self._next = 'gait'

        return policy

    def gait_fsm(self):

        if self._count == 0:

             policy = [{'fin':0, 'type':'gait'},{'fin':1, 'type':'stop'},
                       {'fin':2, 'type':'stop'},{'fin':3, 'type':'stop'}]

        if self._count == 1:

            policy = [{'fin':0, 'type':'stop'},{'fin':1, 'type':'gait'},
                      {'fin':2, 'type':'stop'},{'fin':3, 'type':'stop'}]
        
        if self._count == 2:

            policy = [{'fin':0, 'type':'stop'},{'fin':1, 'type':'stop'},
                      {'fin':2, 'type':'gait'},{'fin':3, 'type':'stop'}]

        if self._count == 3:

            policy = [{'fin':0, 'type':'stop'},{'fin':1, 'type':'stop'},
                      {'fin':2, 'type':'stop'},{'fin':3, 'type':'gait'}]

        self._count += 1

        self._count = self._count%4

        if self._count == 0:

            self._next = 'man'

        return policy

    def switch(self):

        if self._next == 'man':

            return self.man_fsm()

        elif self._next == 'gait':

            return self.gait_fsm()


    def update_skill(self, policy):

        skills = []

        for fin_pol in policy:

            if fin_pol['type'] != 'stop':

                skills.append(self.update_dmp_params(dmp_type=fin_pol['type']))
            else:
                skills.append(None)

        return skills


def main():

    dm = DMPPolicy()

    while True:

        done = False
        policy = dm.switch()
        k = 0
        skills = dm.update_skill(policy)

        while not done:

            for j in range(4):

                if skills[j] is not None:

                    if k == len(skills[j]) - 1:
                        done = True
                        break
                    
                    # cmd = dm._env._hand.inv_kin(j, skills[j][k, 1:].tolist())
                    cmd = skills[j][k, :]
                    k += 1

                else:

                    cmd = dm._env.get_hand_joint_state()['pos'][j]

                if np.any(np.isnan(cmd)):
                    continue

                dm._env._hand.apply_action(j, cmd)
                
            dm._env.simple_step()

if __name__ == '__main__':
    main()
