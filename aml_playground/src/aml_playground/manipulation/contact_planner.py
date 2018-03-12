import time 
import copy
import numpy as np
import pybullet as p
from dmp_gaits import DMPGaits
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from manipulation_ctrlr2 import ManCntrlr
from aml_rl_envs.utils.math_utils import skew
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from utils import compute_f_cone_approx, unit_normal, poly_area
from aml_playground.manipulation.config import HAND_OBJ_CONFIG


class ContactPlanner():

    def __init__(self):

        HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

        self._env = HandObjEnv(action_dim=18, randomize_box_ori=False, keep_obj_fixed=False, config=HAND_OBJ_CONFIG)

        self._gaiter = DMPGaits(env=self._env)

        self._maniptn =  ManCntrlr(env=self._env)

    def run(self):

        ori_delta  = np.array([0.,0., np.pi])
        pos_delta = np.array([0.,0.,0.])

        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        goal_ori = ori + ori_delta
        goal_omg = np.array([0., 0., 0.])
        goal_acc = np.array([0., 0., 0.])

        pos  += pos_delta

        x_des =  np.r_[pos, goal_ori]
        xd_des = np.r_[np.zeros(3), goal_omg]
        xdd_des =  np.r_[np.zeros(3), goal_acc]

        self._maniptn.run(x_des, xd_des, xdd_des)

        x_des_traj, dx_des_traj, ddx_des_traj, des_fimp_list = self.manipulate_from_obj_traj(x_des, xd_des, xdd_des)

        contact_points_traj = self._maniptn.compute_contact_points_traj(x_des_traj)

        # contact_forces = self.compute_optimal_contact_forces(contact_points_traj, x_des_traj, ddx_des_traj, des_fimp_list)

        ik_solutions = self._maniptn.compute_ik_solutions(contact_points_traj)

        # for k in range(contact_points_traj.shape[0]-2):

        ik_solutions = self._maniptn.compute_ik_solutions(contact_points_traj) #

        quality_manipulability, quality_area = self._maniptn.compute_grasp_quality(contact_points_traj, ik_solutions)

        real_contact_points, real_joint_positions, real_obj_poses, real_obj_vel, real_obj_acc = self._maniptn.manipulate(ik_solutions)
        
        real_quality_manipulability, real_quality_area = self.compute_grasp_quality(real_contact_points, real_joint_positions)

        print real_quality_manipulability

        print real_quality_area

        # for m in range(self._env._num_fingers):

        #     gait_primitive = self._gaiter.update_dmp_params(finger_idx=m)
            
        #     for n in range(gait_primitive.shape[0]):

        #         cmd = self._env._hand.inv_kin(m, gait_primitive[n, 1:].tolist())
        
        #         if np.any(np.isnan(cmd)):
        #             continue

        #         self._env._hand.applyAction(m, cmd)
            
        #         self._env.simple_step()

        raw_input("Press enter to break")


cp = ContactPlanner()
cp.run()