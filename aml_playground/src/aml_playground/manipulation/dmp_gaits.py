import time
import copy
import numpy as np
import pybullet as pb
from scipy.signal import savgol_filter
from scipy.optimize import minimize, linprog
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.collect_demo import plot_demo, get_demo

np.random.seed(123)
discrete_dmp_config['dof'] = 3

class DMPGaits():

    def __init__(self, env=None):

        if env is None:

            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False, keep_obj_fixed=True)
        
        else:
            
            self._env = env

        hand_info = self._env.get_hand_limits()

        self._num_fingers = self._env._num_fingers

        self._finger_joint_means  = hand_info['mean']
        
        self._finger_joint_ranges = hand_info['range']
        
        self._dmp_list = None

        self.encode_dmps()


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

        for j in range(4):
            primitive_base =  np.zeros([3, 30])
            for k in range(3):
                primitive_base[k, :] = savgol_filter(np.hstack([np.linspace(end_points[j][k], mid_points[j][k], 15), 
                                                                np.linspace(mid_points[j][k], start_points[j][k], 15)]), 29, 2)

            p_bases.append(primitive_base.T)

            # print "finger \t", j
            # print "start \n", primitive_base[:,0]
            # print "end \n", primitive_base[:,-1]

            plot_demo(trajectory=primitive_base.T, color=[0,1,1], start_idx=0)

        return p_bases


    def encode_dmps(self):

        self._dmp_list = []

        primitive_list = self.get_primitives()

        for k in range(self._num_fingers):

            dmp = {}
            dmp['config'] = discrete_dmp_config
            dmp['obj'] = DiscreteDMP(config=discrete_dmp_config)

            dmp['obj'].load_demo_trajectory(primitive_list[k])
            dmp['obj'].train()

            self._dmp_list.append(copy.deepcopy(dmp))


    def update_dmp_params(self, finger_idx, phase_start=1., speed=1., goal_offset=np.array([0., 0., 0.]), start_offset=np.array([0., 0., 0.]), external_force=None):

        if self._dmp_list is None:
            return
        
        dmp    = self._dmp_list[finger_idx]['obj']
        config = self._dmp_list[finger_idx]['config']

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


def main():

    dg = DMPGaits()

    #start offsets are actual goal positions of demos
    # start_offsets = [np.array([0.45418333, -0.03117599, 1.4]), np.array([0.03117599, 0.45418333, 1.4 ]), np.array([-0.45418333, 0.03117599, 1.4]), np.array([-0.03117599, -0.45418333, 1.4])]
    #start offsets are actual goal positions of demos
    # goal_offsets  = [np.array([0.03007016, 0.44988597, 1.4]), np.array([-0.44988597, 0.03007016, 1.4]), np.array([-0.03007016, -0.44988597, 1.4]), np.array([0.44988597, -0.03007016,  1.4])]

    while True:

        dg._env.simple_step()

        for j in range(dg._num_fingers): #dg._num_fingers

            # dmp_primitive = dg.update_dmp_params(finger_idx=j, goal_offset=-goal_offsets[j], start_offset=-start_offsets[j])
            dmp_primitive = dg.update_dmp_params(finger_idx=j)

            for k in range(dmp_primitive.shape[0]):

                if dg._env._ctrl_type == 'pos':
                    
                    cmd = dg._env._hand.inv_kin(j, dmp_primitive[k, 1:].tolist())

                    ee_poss, ee_oris, ee_vels, ee_omgs = dg._env._hand.get_ee_states(as_tuple=True)
                    jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = dg._env._hand.get_jnt_states()

                    # print "target \t", dmp_primitive[k, 1:]
                    # print "cmd \t", cmd
                    # print "start ee\t", ee_poss[j]
                    # print "start js\t", jnt_poss[j]
                
                else:

                    print "j \t", j , "k \t", k
                    
                    cmd = 150*dg._env._hand.compute_impedance_ctrl(finger_idx=j, Kp=np.ones(3), goal_pos=dmp_primitive[k, 1:], goal_vel=np.zeros(3), dt=0.01)
                
                if np.any(np.isnan(cmd)):
                    continue
                
                dg._env._hand.apply_action(j, cmd)
                
                dg._env.simple_step()

                # ee_poss, ee_oris, ee_vels, ee_omgs = dg._env._hand.get_ee_states(as_tuple=True)
                # jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = dg._env._hand.get_jnt_states()
                # print "final ee\t", ee_poss[j]
                # print "final js\t", jnt_poss[j]

                # raw_input()


if __name__ == '__main__':
    main()
