import time
import copy
import numpy as np
import pybullet as pb
from scipy.signal import savgol_filter
from scipy.optimize import minimize, linprog
from aml_lfd.promp.discrete_promp import MultiplePROMPs
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.collect_demo import plot_demo, get_demo

np.random.seed(123)

class PROMPGaits():

    def __init__(self, env=None):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False,  keep_obj_fixed=True)
        
        else:
            
            self._env = env

        hand_info = self._env.get_hand_limits()

        self._num_fingers = self._env._num_fingers

        self._finger_joint_means  = hand_info['mean']
        
        self._finger_joint_ranges = hand_info['range']
        
        self._promp_list = None

        self.encode_promps()


    def get_primitives(self, offset = 0.3):

        def rotz(ang):
            return np.array([ [np.cos(ang), -np.sin(ang), 0.], [np.sin(ang), np.cos(ang), 0.], [0., 0., 1.] ])

        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()


        def points_on_circum(r,z,n=100): # n number of points on circle
            assert n > 9
            return [(np.cos(2*np.pi/n*i)*r,np.sin(2*np.pi/n*i)*r,z) for i in xrange(0,n+1)]

        n = 100
        traj_radius = self._env._object._radius + self._env._hand._finger_radius
        points = points_on_circum(1.2*traj_radius, 1.4, n)

        # start_points = np.array([[0.35991216,-0.00795211,1.4], 
        #                 [0.00795212, 0.35991216, 1.4], 
        #                 [-0.35991216, 0.00795199, 1.4], 
        #                 [-0.00795211, -0.35991216, 1.4]])

        start_points = np.array([points[0],points[int(n/4)],points[int(n/2)],points[int(3*n/4)]])

        # end_points = np.array([[0.03339137, 0.35844807, 1.4], 
        #               [-0.35844807, 0.03339138, 1.4], 
        #               [-0.03339125, -0.35844808, 1.4], 
        #               [0.35844807, -0.03339137, 1.4]])

        idx_tol = max(1,int(0.02*n)) # idx of point to stop (so that ending point does not collide with the starting point of the next)

        end_points = np.array([points[int(n/4)-idx_tol],points[int(n/2)-idx_tol],points[int(3*n/4)-idx_tol],points[n-idx_tol]])


        mid_points = np.asarray([0.5*(end_points[0]+start_points[0])+np.array([offset, offset, 0]),
                      0.5*(end_points[1]+start_points[1])+np.array([-offset,offset, 0]),
                      0.5*(end_points[2]+start_points[2])+np.array([-offset,-offset, 0]),
                      0.5*(end_points[3]+start_points[3])+np.array([offset,-offset, 0])])


        p_bases = []

        number_of_points = 50

        for j in range(4):

            primitive_base =  np.zeros([3, number_of_points])
            traj_samples = []

            for p in range(10):

                for k in range(3):

                    primitive_base[k, :] = savgol_filter(np.hstack([np.linspace(end_points[j][k], mid_points[j][k], number_of_points/2), 
                                                                    np.linspace(mid_points[j][k], start_points[j][k], number_of_points/2)]) + np.random.randn(number_of_points)*0.1, number_of_points-1, 2)

                traj_samples.append(primitive_base)

                # plot_demo(trajectory=primitive_base.T, color=[np.random.uniform(0,1) for _ in range(3)], start_idx=0, life_time=4.)

            p_bases.append(copy.deepcopy(traj_samples))


        #converting to discrete promp class format
        finger_primitives = []

        for finger_base in p_bases:

            x_list = []; y_list = []; z_list=[]
            
            for traj in finger_base:
                
                x,y,z = traj.tolist()
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

            finger_primitives.append([copy.deepcopy(x_list), copy.deepcopy(y_list), copy.deepcopy(z_list)])

        return finger_primitives


    def encode_promps(self):

        self._promp_list = []

        primitive_list = self.get_primitives()

        for k in range(self._num_fingers):

            promp = {}
            promp['obj'] = MultiplePROMPs(multiple_dim_data=primitive_list[k])
            promp['obj'].train()

            self._promp_list.append(copy.deepcopy(promp))


    def generate_promp_traj(self, finger_idx):

        if self._promp_list is None:
            return

        promp    = self._promp_list[finger_idx]['obj']

        new_mu_traj, new_mu_Dtraj = promp.generate_trajectory(phase_speed=1., randomness=1e-4)

        return new_mu_traj, new_mu_Dtraj


def main():

    pg = PROMPGaits()

    while True:

        for j in range(pg._num_fingers): #pg._num_fingers

            promp_primitive, promp_Dprimitive = pg.generate_promp_traj(finger_idx=j)

            promp_primitive = np.flip(promp_primitive.T, axis=1).T

            plot_demo(trajectory=promp_primitive, color=[np.random.uniform(0,1) for _ in range(3)], start_idx=0, life_time=40.)

            for k in range(promp_primitive.shape[0]):

                if pg._env._ctrl_type == 'pos':

                    cmd = pg._env._hand.inv_kin(j, promp_primitive[k, :].tolist())

                    ee_poss, ee_oris, ee_vels, ee_omgs = pg._env._hand.get_ee_states(as_tuple=True)
                    jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = pg._env._hand.get_jnt_states()

                    print "***********************************************************Start *****************************************************************", k
                    print "target \t", promp_primitive[k, :]
                    print "cmd \t", cmd
                    print "start ee\t", ee_poss[j]
                    print "start js\t", jnt_poss[j]
                
                else:

                    print "j \t", j , "k \t", k
                    
                    cmd = 150*pg._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=promp_primitive[k, :], goal_vel=np.zeros(3), dt=0.01)
                
                if np.any(np.isnan(cmd)):
                    continue
                
                pg._env._hand.apply_action(j, cmd)
                
                # for _ in range(10000):
                #     pg._env._hand.applyAction(j, cmd)
                pg._env.simple_step()


                ee_poss, ee_oris, ee_vels, ee_omgs = pg._env._hand.get_ee_states(as_tuple=True)
                jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = pg._env._hand.get_jnt_states()

                print "final ee\t", ee_poss[j]
                print "final js\t", jnt_poss[j]

            raw_input()


if __name__ == '__main__':
    main()
