import os
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_opt.pi_traj_opt.pi_traj_opt import PITrajOpt
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_playground.manipulation.config import HAND_OBJ_CONFIG

np.random.seed(42)

state_constraints={
'min':[-0.5, -0.1, 1.40],
'max':[0.5, 0.2, 1.43],
}

config = {
    'timesteps':100,
    'no_rollouts':100,
    'h':10,
    'gain':1e-2,
    'max_iter':35,
    'state_constraints':state_constraints,
    'start':np.array([ 0.40237453, -0.01055326,  1.40791974]),
    'goal':np.array([0.0363268 , 0.40509655, 1.39633195]),
    'init_traj':None,
    'cost_fn':None,
    'smooth_traj':True,
}


class TrajCost():
    def __init__(self, env, points_outside):
        
        self._env = env
        self._points_outside = points_outside

    def __call__(self, traj):
        return self.get(traj)

    def get(self, traj):

        num_points = len(traj)

        to_points  = [tuple(self._points_outside[0]) for _ in range(num_points)]
        frm_points = [tuple(traj[k,:]) for k in range(num_points)] 

        raytest_info = pb.rayTestBatch(rayFromPositions=frm_points,
                                       rayToPositions=to_points)

        cost = np.zeros(num_points)

        for k in range(num_points):
            
            if raytest_info[k][0] == self._env._hand._robot_id:
                value = abs(np.random.randn(1))
            else:
                value = -abs(np.random.randn(1))

            cost [k]= value

        cost = cost - np.min(cost)

        cost = 1./ (1. + np.exp(-cost))

        cost = cost - np.min(cost)

        return cost

class PI2Gaits(object):

    def __init__(self):

        HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

        self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=12, randomize_box_ori=False, keep_obj_fixed=True)

        # self._home = self._env._hand.get_ee_states(as_tuple=False)[0]

        self._home = [np.array([ 0.40237453, -0.01055326,  1.40791974]), np.array([-0.40237451,  0.01055314,  1.40791974])]

        self._points_outside = [[ 0.88157195, -0.00703787, 1.17211503], [0.40319226, -0.01055484, 1.40800266]
 ]
        final_jnt_pos = [-0.52689812,  0.14820803, -1.49498498]
        # init_jnt_pos = [0.014906, 0.9435, -2.0707]
        # jnt_pos = [0.014906, 2.0435, -2.0707]

        # print self._env._hand.get_jnt_states()[0][1]

        self._env._hand.set_fin_joint_state(0, final_jnt_pos)

        self._env.simple_step()

        # print self._env._hand.get_ee_states(as_tuple=False)[0][0]

        # raw_input()

        fin_idx = 0

        self._end = [np.array([0.0363268 , 0.40509655, 1.39633195]), np.array([-0.03632783, -0.4050976 ,  1.39633195])]

        self._cost_fn = TrajCost(self._env, self._points_outside)

        # for k in range(3):

        #     tmp = np.linspace(self._end[fin_idx][k], self._home[fin_idx][k], 100)

        #     if k == 0:
        #         traj = tmp
        #     else:
        #         traj = np.vstack([traj, tmp])

        self._opt = PITrajOpt(config, self._cost_fn, self.visualize)

        self._init_traj = self._opt._init_traj

        # plot_demo(trajectory=traj.T, color=[0,1,1], start_idx=0)
        # plot_demo(trajectory=self._init_traj, color=[1,0,0], start_idx=0)

        # raw_input()

    def visualize(self, traj, traj_samples=None):

        plot_demo(trajectory=traj, color=[0,1,1], start_idx=0, life_time=2.)

        # print self._cost_fn(traj)

        # raw_input()


    def run(self):

        traj = self._opt.run()

        for k in range(traj.shape[0]):

            cmd = self._env._hand.inv_kin(0, traj[k, :].tolist())
          
            self._env._hand.apply_action(0, cmd)
                    
            self._env.simple_step()

        raw_input("Done")


def main():

    lcpf = PI2Gaits()
    lcpf.run()

if __name__ == '__main__':
    main()

