import time
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from utils import compute_f_cone_approx
from scipy.optimize import minimize, linprog
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.collect_demo import plot_demo, get_demo

np.random.seed(123)

discrete_dmp_config['dof'] = 3

class DMPManptln():

    def __init__(self, env=None):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(action_dim=18, randomize_box_ori=False, config=HAND_OBJ_CONFIG)
        
        else:
            
            self._env = env

        self._finger_limits = self._env.get_hand_limits()
        self._num_fingers = self._env._num_fingers
        self._num_joints_finger = len(self._finger_limits['lower'][0])

        self._finger_joint_means  = np.zeros([self._num_fingers, self._num_joints_finger])
        self._finger_joint_ranges = np.zeros([self._num_fingers, self._num_joints_finger])

        for k in range(self._num_fingers):
            for j in range(self._num_joints_finger):
                self._finger_joint_means[k,j]  = 0.5*(self._finger_limits['lower'][k][j] + self._finger_limits['upper'][k][j])
                self._finger_joint_ranges[k,j] = self._finger_limits['upper'][k][j] - self._finger_limits['lower'][k][j]

        self._dmp_list = None

        self.encode_dmps()


    def get_base_primitive(self, contact_point, obj_radius=0.36, radius_base=0.1):

        def rotz(ang):
            return np.array([ [np.cos(ang), -np.sin(ang), 0.], [np.sin(ang), np.cos(ang), 0.], [0., 0., 1.] ])

        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        contact_angle = np.arctan2(contact_point[1], contact_point[0])

        primitive_start_angle = 0*np.pi/180

        primitive_end_angle =  primitive_start_angle + 91*np.pi/180.

        theta_base = np.arange(primitive_start_angle, primitive_end_angle, 0.1)

        x_b = obj_radius*np.cos(theta_base)
        y_b = obj_radius*np.sin(theta_base)
        z_b = np.ones_like(x_b)*pos[2]

        primitive_base = (np.dot(rotz(contact_angle), np.vstack([x_b,y_b,z_b]))).T

        primitive_base[:,0] +=  pos[0] 
        primitive_base[:,1] +=  pos[1] 

        plot_demo(trajectory=primitive_base, color=[0,1,0], start_idx=0)

        return primitive_base


    def get_primitives(self):

        contact_info = self._env.get_contact_points()

        curr_joint_state = self._env.get_hand_joint_state()['pos']

        primitive_list = []

        radius_of_traj = 1.005*(self._env._object._radius + self._env._hand._finger_radius)

        for k in range(self._num_fingers):

            contact_k =  contact_info[k]

            if contact_k['cp_on_finger']:

                contact_point   = contact_k['cp_on_finger'][0]

                primitive_list.append(self.get_base_primitive(contact_point, radius_of_traj))

            else:

                primitive_list.append(curr_joint_state[k])

        return primitive_list

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

        new_dmp_traj, new_dmp_dtraj = dmp.generate_trajectory(config=config)

        return new_dmp_traj, new_dmp_dtraj


    def compute_B_matrix(self, cp_list, rtn_list=False):

        B_list = []

        for k in range(self._num_fingers):

            f_cone_approx = compute_f_cone_approx(cp_list[k])

            B_list.append(f_cone_approx)

        if rtn_list:
            return B_list
        else:
            return block_diag(*B_list)

    def compute_force_primitive(self):

        self._lambda_dim = 16

        dmp_primitive1, Ddmp_primitive1 = self.update_dmp_params(0)
        dmp_primitive2, Ddmp_primitive2 = self.update_dmp_params(1)
        dmp_primitive3, Ddmp_primitive3 = self.update_dmp_params(2)
        dmp_primitive4, Ddmp_primitive4 = self.update_dmp_params(3)


        vel_primitive = np.hstack([Ddmp_primitive1[:, 1:], Ddmp_primitive2[:, 1:], Ddmp_primitive3[:, 1:], Ddmp_primitive4[:, 1:]])
        force_primitive = np.zeros([dmp_primitive1.shape[0], 3*self._env._num_fingers])

        grasp_map = self._env.compute_grasp_map(finger_idx=None)

        contact_forces = self._env.compute_all_contact_forces_vector()

        slack_var = np.ones(6)*0.01

        lower_bound = 0.
        upper_bound = 3.

        a = np.vstack([np.ones(self._lambda_dim)*lower_bound, np.ones(self._lambda_dim)*upper_bound]).T
        bounds = tuple([tuple(a[k]) for k in range(self._lambda_dim)])


        for k in range(force_primitive.shape[0]-1):

            cp_list = [dmp_primitive1[k,1:], dmp_primitive2[k,1:], dmp_primitive3[k,1:], dmp_primitive4[k,1:]]
            vel_ee  = vel_primitive[k,:]

            B = self.compute_B_matrix(cp_list)

            def compute_f(Lambda):

                return np.dot(B, Lambda)

            cost_function = lambda x: np.linalg.norm( compute_f(x) )**2 

            constraints = ({'type': 'eq', 'fun': lambda x: np.dot(vel_ee, compute_f(x) ) - 0.000001 }) 

            #TNC
            res = minimize(cost_function, tuple(np.abs(np.random.randn(self._lambda_dim))), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter':10000,'disp':True})

            force_primitive[k, :] = compute_f(res['x'])

            print "Status: \t", res['success']
            print "Constraint: \t", np.dot(vel_ee, force_primitive[k, :])

            print "Finished: \t", k, "/", force_primitive.shape[0]

        np.savetxt("force.csv", force_primitive, delimiter=",")
        np.savetxt("velocity.csv", vel_primitive,  delimiter=",")

        return force_primitive, vel_primitive

    def get_man_primitive(self):

        dmp_primitive1, Ddmp_primitive1 = self.update_dmp_params(0)
        dmp_primitive2, Ddmp_primitive2 = self.update_dmp_params(1)
        dmp_primitive3, Ddmp_primitive3 = self.update_dmp_params(2)
        dmp_primitive4, Ddmp_primitive4 = self.update_dmp_params(3)

        pos_primitive = np.hstack([dmp_primitive1[:, 1:], dmp_primitive2[:, 1:], dmp_primitive3[:, 1:], dmp_primitive4[:, 1:]])
        # np.savetxt("pos.csv", pos_primitive, delimiter=",")
        return pos_primitive



def main():

    dm = DMPManptln()

    pos_primitive = dm.get_man_primitive()


    # force_primitive, vel_primitive = dm.compute_force_primitive()

    # force_primitive = np.loadtxt("force.csv", delimiter=",")
    # vel_primitive = np.loadtxt("velocity.csv",  delimiter=",")
    # pos_primitive = np.loadtxt("pos.csv",  delimiter=",")

    # plt.subplot(311)
    # plt.plot(force_primitive[:,0], 'r')
    # plt.plot(vel_primitive[:,0], 'g')
    # plt.subplot(312)
    # plt.plot(force_primitive[:,1], 'r')
    # plt.plot(vel_primitive[:,1], 'g')
    # plt.subplot(313)
    # plt.plot(force_primitive[:,2], 'r')
    # plt.plot(vel_primitive[:,2], 'g')
    # plt.show()

    while True:

        for k in range(pos_primitive.shape[0]):

            for j in range(dm._num_fingers): #dg._num_fingers

                if dm._env._ctrl_type == 'pos':
                    
                    cmd = dm._env._hand.inv_kin(j, pos_primitive[k, 3*j:3*j+3].tolist())

                # if dm._env._ctrl_type == 'vel':

                #     jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = dm._env._hand.get_jnt_states()

                #     Jee = dm._env._hand.get_finger_jacobian(j, jnt_poss)

                #     force_error = force_primitive[k, j*3:j*3+3] - np.asarray(jnt_reaction_forces[j][-1])[:3] #

                #     # print "Vel \t", Ddmp_primitive[k, 1:]

                #     cmd =  0.0*np.dot(Jee.T, force_error) + 0.05*np.dot(np.linalg.pinv(Jee, rcond=1e-1), Ddmp_primitive[k, 1:]) 


                    if np.any(np.isnan(cmd)):
                        continue
                    
                dm._env._hand.applyAction(j, cmd)
                
            dm._env.simple_step()

if __name__ == '__main__':
    main()