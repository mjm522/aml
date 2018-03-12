import time 
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from gait_planner import GaitPlanner
from mpl_toolkits.mplot3d import Axes3D
from aml_rl_envs.utils.math_utils import skew
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_rl_envs.hand.hand_obst_env import HandObstacleEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from utils import compute_f_cone_approx, unit_normal, poly_area

np.random.seed(123)

class ManCntrlr():

    def __init__(self, env=None):

        if env is None:

            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            # self._env = HandObjEnv(action_dim=18, randomize_box_ori=False, keep_obj_fixed=True, config=HAND_OBJ_CONFIG)
            self._env = HandObstacleEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False, keep_obj_fixed=True)
        
        else:
            
            self._env = env

        self._finger_limits = self._env.get_hand_limits()
        self._num_fingers = self._env._num_fingers
        self._num_joints_finger = self._env._num_joints_finger

        self._finger_joint_means  = np.zeros([self._num_fingers, self._num_joints_finger])
        self._finger_joint_ranges = np.zeros([self._num_fingers, self._num_joints_finger])

        for k in range(self._num_fingers):
            for j in range(self._num_joints_finger):
                self._finger_joint_means[k,j]  = 0.5*(self._finger_limits['lower'][k][j] + self._finger_limits['upper'][k][j])
                self._finger_joint_ranges[k,j] = (self._finger_limits['upper'][k][j] - self._finger_limits['lower'][k][j])

        #parameters of manipulation controller
        self._Kp = 50.*np.diag([1., 1., 1., 0.2, 0.2, 0.05])
        self._Kd = 5.*np.diag([1, 1, 1,  0.02, 0.02,  0.005])
        self._Ki = 50.*np.diag([1, 1, 1, 0.02, 0.02, 0.005])

        #paramters of velocity-force controller
        self._Kv = np.diag([0.1, 0.1, 0.1])
        self._Kf = np.diag(np.ones(3))*0.5

        self._lambda_dim = self._num_fingers*4

        self._Md = None

        self._integral_error = 0.

        #cost function coeffients
        self._alpha_1 = 0.01
        self._alpha_2 = 0.01
        self._alpha_3 = 10000

        self._fprev = np.zeros(3*self._num_fingers)

        self._gait_planner = GaitPlanner(env=self._env)


        self._qh_ref = None
        self._qa_ref = None

        self._q_man_list = []
        self._itr = 0


    def compute_B_matrix(self, contact_points, obj_ori, obj_pos, rtn_list=False):

        B_list = []

        obj_rot_matrix = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler( (obj_ori).tolist() )  )).reshape(3,3)
        obj_rot_matrix_inv = np.linalg.inv(obj_rot_matrix)

        for k in range(self._num_fingers):

            contact_point_obj_frame = np.dot(obj_rot_matrix_inv, contact_points[3*k:3*k+3]-obj_pos) 

            f_cone_approx = compute_f_cone_approx(contact_point_obj_frame, obj_rot_matrix)

            B_list.append(f_cone_approx)

        if rtn_list:
            return B_list
        else:
            return block_diag(*B_list)


    def compute_contact_points_traj(self, x_des_traj):

        contact_info = self._env.get_contact_points()

        contact_points_traj = np.zeros([len(x_des_traj), 3*self._num_fingers])

        Robj_W = np.asarray(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler( (x_des_traj[0][3:]).tolist() )  )).reshape(3,3)

        done = True

        while done:

            done = False

            initial_cs = []

            for k in range(self._num_fingers):

                if contact_info[k]['cp_on_block']:
                    contact_point = np.asarray(contact_info[k]['cp_on_block'][0])
                    initial_cs.append( contact_point-x_des_traj[0][:3] )
                else:
                    self._gait_planner.move_to_home(k)
                    contact_info = self._env.get_contact_points()
                    print "here.............................................................."
                    done = True

        for j in range(len(x_des_traj)):

            for k in range(self._num_fingers):
                
                Robj_W = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler( (x_des_traj[j][3:]).tolist() )  )).reshape(3,3)

                new_pos = np.dot(Robj_W, initial_cs[k]) + x_des_traj[j][:3]

                contact_points_traj[j, 3*k:3*k+3] = new_pos

        return contact_points_traj



    def manipulate_from_obj_traj(self, x_des, xd_des, xdd_des):

        object_inertia = self._env.get_object_mass_matrix()
        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        if self._Md is None:
            self._Md = object_inertia

        #compute COM trajectory of the object
        x_des_traj = [np.r_[pos, ori]]
        dx_des_traj = [np.r_[vel, omg]]
        ddx_des_traj = [np.r_[lin_acc, ang_acc]]
        fimp_des = []
        error_in_pos = 100.

        error_list = []

        idx = 0

        while error_in_pos > 0.01:

            error_in_pos = np.linalg.norm(x_des_traj[idx]-x_des)
            error_list.append(error_in_pos)

            pos_error = x_des_traj[idx] - x_des
            vel_error = dx_des_traj[idx] - xd_des

            self._integral_error += pos_error*self._env._time_step*0

            fimp = np.dot(self._Md, xdd_des) - np.dot(self._Kp, pos_error) - np.dot(self._Kd, vel_error) - np.dot(self._Ki, self._integral_error)

            acc_of_object = np.dot(np.linalg.inv(self._Md), fimp)

            vel_of_object = dx_des_traj[idx] + 0.5*acc_of_object*self._env._time_step

            pos_of_object = x_des_traj[idx] + vel_of_object*self._env._time_step

            # pos_of_object[3:] = pos_of_object[3:]%(2*np.pi)

            fimp_des.append(fimp)
            ddx_des_traj.append(acc_of_object)
            dx_des_traj.append(vel_of_object)
            x_des_traj.append(pos_of_object)

            idx += 1

        return x_des_traj, dx_des_traj, ddx_des_traj, fimp_des

    
    def compute_ik_solutions(self, contact_points_traj):

        ik_solutions =  np.zeros([contact_points_traj.shape[0], 3*self._num_fingers])

        for j in range(contact_points_traj.shape[0]):

            for k in range(self._num_fingers):

                ik_solutions[j, 3*k:3*k+3] = self._env._hand.inv_kin(k, contact_points_traj[j, 3*k:3*k+3].tolist())
                
      
        return ik_solutions
            

    def compute_grasp_quality(self, contact_points_traj, ik_solutions, disp=False):


        def get_hand_manipulability(joint_pos):
        
            qh = np.zeros(self._num_fingers)

            for k in range(self._num_fingers):

                joint_pos_k = joint_pos[3*k:3*k+3]

                # if k == 0:

                #     print "Base joint \t", joint_pos_k

                for j in range(self._num_joints_finger-1):

                    qh[k] += ( (joint_pos_k[j] - self._finger_joint_means[k,j] ) / self._finger_joint_ranges[k,j] )**2


            return 0.5*np.sum(qh), 10*qh


        def get_grasp_quality(contact_points, disp=disp):
            """
            finger index should be the free index
            this assumes all four fingers are in contact
            """

            if disp:
                print "Contact points \n", contact_points.reshape(self._num_fingers, 3)

            area = poly_area(contact_points.reshape(self._num_fingers, 3))

            if np.isnan(area):
                area = 0.

            return area

        q_manplty = np.zeros(contact_points_traj.shape[0])
        q_manplty_fin = np.zeros([contact_points_traj.shape[0], self._num_fingers])
        q_area = np.zeros(contact_points_traj.shape[0])

        for k in range(contact_points_traj.shape[0]):

            q_manplty[k], q_manplty_fin[k,:] = get_hand_manipulability(ik_solutions[k,:])
            q_area[k] = get_grasp_quality(contact_points_traj[k,:])


        return q_manplty_fin, q_manplty, q_area


    def get_curr_contact_points(self):

        contact_info = self._env.get_contact_points()
        contact_points =  np.zeros(3*self._num_fingers)
        
        for k in range(self._num_fingers):

            if contact_info[k]['cp_on_block']:

                contact_points[3*k:3*k+3] = np.asarray(contact_info[k]['cp_on_block'][0])
            
            else:

                self._gait_planner.move_to_home(k)

        return contact_points

    def get_curr_joint_positions(self):

        joint_pos = np.zeros(3*self._num_fingers)

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self._env._hand.get_jnt_states()

        #this is because the joint position 
        #retured by hand, has 4 joint values (the last static joint included)
        for k in range(self._num_fingers):

            joint_pos[3*k:3*k+3] = jnt_poss[k][:3]

        return joint_pos

    def manipulate(self, ik_solutions, do_gait=True):

        qman_list = []

        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        flag = False
        done = False

        # print "Initial ori \t", ori*180/np.pi
        # print "Initial pos \t", pos

        real_contact_points  = np.zeros([ik_solutions.shape[0], 3*self._num_fingers])
        real_joint_positions = np.zeros([ik_solutions.shape[0], 3*self._num_fingers])
        real_obj_poses = np.zeros([ik_solutions.shape[0], 6])
        real_obj_vel = np.zeros([ik_solutions.shape[0], 6])
        real_obj_acc = np.zeros([ik_solutions.shape[0], 6])

        _, q_manplty_ref, q_area_ref = self.compute_grasp_quality(self.get_curr_contact_points()[None,:], ik_solutions[0, :][None, :])

        if self._qa_ref is None:
            self._qa_ref = q_area_ref
            self._qh_ref = q_manplty_ref

        # plt.plot(ik_solutions[:, 0], 'r')
        # plt.plot(ik_solutions[:, 1], 'g')
        # plt.plot(ik_solutions[:, 2], 'b')
        # plt.show()

        for j in range(ik_solutions.shape[0]):

            pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

            real_obj_poses[j, :]  = np.r_[pos, ori]
            real_obj_vel[j,:] = np.r_[vel, omg]
            real_obj_acc[j,:] = np.r_[lin_acc, ang_acc]
            real_contact_points[j, :]  = self.get_curr_contact_points()
            real_joint_positions[j, :] = self.get_curr_joint_positions()

            q_manplty_fin, q_manplty, q_area = self.compute_grasp_quality(real_contact_points[j, :][None,:], real_joint_positions[j, :][None, :])

            #if this happens, that means slip has occured
            qman_list.append(q_manplty)

            print q_manplty_fin

            if do_gait and self._itr > 0:

                if  q_manplty_fin[0][0] > 4.:

                    fin_idx = np.argmax(q_manplty_fin)

                    print "Finger to be switched \t", fin_idx

                    self._gait_planner.move_to_home(fin_idx)
                    # self._gait_planner.move_to_home(0)
                    # self._gait_planner.move_to_home(1)
                    # self._gait_planner.move_to_home(2) 
                    # self._gait_planner.move_to_home(3)

                    flag = True

                    self._qa_ref = None

                    break

            for k in range(self._num_fingers):

                cmd = ik_solutions[j, 3*k:3*k+3]
                    
                if np.any(np.isnan(cmd)):
                    continue
                
                self._env._hand.applyAction(k, cmd) #, Kp=np.ones(3)

            self._env.simple_step()

            done = self.reached_goal()

            if done:
                break

        self._q_man_list.append(qman_list)

        if flag: print "Slip occured at ... \t", j

        self._itr += 1

        # print "Final ori \t", (ori*180/np.pi)[2]
        # print "Final pos \t", pos

        return real_contact_points, real_joint_positions, real_obj_poses, real_obj_vel, real_obj_acc, done


    def compute_grasp_map(self, contact_points, obj_pos):

        grasp_map = np.zeros([6, 3*self._num_fingers])

        for i in range(self._num_fingers):

            Ree_W = np.eye(3)#np.dot( roty(-np.pi/2), Ree_W )

            skew_rot = np.dot(skew(np.asarray(contact_points[3*i:3*i+3]) - np.asarray(obj_pos)), Ree_W)

            grasp_map[:, 3*i:3*i+3] = np.vstack([Ree_W, skew_rot])

        return grasp_map


    def compute_optimal_contact_forces(self, contact_points_traj, obj_des_traj, obj_acc, fimps_des):

        contact_forces = np.zeros([contact_points_traj.shape[0], 3*self._num_fingers])

        _fprev = np.zeros(3*self._num_fingers)

        for j in range(len(obj_des_traj)):

            print "Current point: \t(", j, "/", contact_points_traj.shape[0], ")"

            #this needs to be recomputed, by passing in the contact points
            grasp_map = self.compute_grasp_map(contact_points_traj[j,:], obj_des_traj[j][:3])

            slack_var = np.ones(6)*0.01

            def compute_f(Lambda):

                B =  self.compute_B_matrix(contact_points=contact_points_traj[j, :], obj_ori=obj_des_traj[j][3:], obj_pos=obj_des_traj[j][:3])

                return np.dot(B, Lambda)

            lower_bound = 0.
            upper_bound = 30.

            a = np.vstack([np.ones(self._lambda_dim)*lower_bound, np.ones(self._lambda_dim)*upper_bound]).T
            bounds = tuple([tuple(a[k]) for k in range(self._lambda_dim)])
            
            cost_function = lambda x: self._alpha_1*np.linalg.norm( compute_f(x) )**2  + self._alpha_3*np.linalg.norm(slack_var)**2 + self._alpha_2*np.linalg.norm( compute_f(x) - _fprev)**2 

            constraints = ({'type': 'eq', 'fun': lambda x: np.dot(self._Md, obj_acc[j] ) -   np.dot( grasp_map, compute_f(x) ) + fimps_des[j] - slack_var }) #slack_var

            res = minimize(cost_function, tuple(np.abs(np.random.randn(self._lambda_dim))), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter':100, 'disp':False})

            contact_forces[j, :] = compute_f(res['x'])

            _fprev = contact_forces[j, :]

 
        return contact_forces

    def compute_hybrid_solutions(self, x_des, f_des):

        tau = np.dot(self._Kv, (des_js_vel-curr_js_vel)) +  np.dot(self._Kf, contact_force_js)


    def find_error(self):

        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        return np.linalg.norm(np.r_[pos, ori]-self._x_des)


    def reached_goal(self, thresh=0.05):
        
        if self.find_error() < thresh:
            return True
        else:
            return False


    def run(self, x_des, xd_des, xdd_des, plot_traj=False):

        self._x_des   = x_des

        horizon = 300

        done  = self.reached_goal()

        while not done:

            x_des_traj, dx_des_traj, ddx_des_traj, des_fimp_list = self.manipulate_from_obj_traj(x_des, xd_des, xdd_des)

            contact_points_traj = self.compute_contact_points_traj(x_des_traj[:horizon]) #[:horizon]

            # plot_demo(contact_points_traj[:, 0:3], color=[np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)], start_idx=0)

            # contact_forces = self.compute_optimal_contact_forces(contact_points_traj, x_des_traj[:horizon], ddx_des_traj[:horizon], des_fimp_list[:horizon])

            ik_solutions = self.compute_ik_solutions(contact_points_traj)

            # tau_commands = self.compute_hybrid_solutions()

            _, q_manplty, q_area = self.compute_grasp_quality(contact_points_traj, ik_solutions)

            real_contact_points, real_joint_positions, real_obj_poses, real_obj_vel, real_obj_acc, done = self.manipulate(ik_solutions)

            _, real_q_manplty, real_q_area = self.compute_grasp_quality(real_contact_points, real_joint_positions, False)

            print "Error *********************************************\t", self.find_error()


            # for k in range(len(self._q_man_list)):
            #     plt.plot(self._q_man_list[k])

            # plt.draw()
            # plt.pause(0.0001)

            # raw_input("continue")



        if plot_traj:

            fig = plt.figure(figsize=(10,10))
            ax1=plt.subplot(211)
            ax1.plot(q_manplty, 'r', label='pre')
            ax1.plot(real_q_manplty, 'g', label='post')
            ax1.set_title("quality manipulability")
            ax1.set_xlabel("number of contacts")
            ax1.set_ylabel("combined magnitude")
            ax1.legend()

            ax2=plt.subplot(212)
            ax2.plot(q_area, 'r', label='pre')
            ax2.plot(real_q_area, 'g', label='post')
            ax2.set_title("quality area")
            ax2.set_xlabel("number of contacts")
            ax2.set_ylabel("polygon area")
            ax2.legend()

            x_des_traj = np.asarray(x_des_traj)

            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(real_obj_poses[:,0], real_obj_poses[:,1], real_obj_poses[:,2], 'o', markersize=5, color='black')
            ax.plot(x_des_traj[:,0], x_des_traj[:,1], x_des_traj[:,2], '*', markersize=5, color='black')

            colors = ['red', 'green', 'blue', 'cyan']

            for k in range(self._num_fingers):

                ax.plot(contact_points_traj[:, 3*k], contact_points_traj[:, 3*k+1], contact_points_traj[:, 3*k+2], '-', markersize=3, color=colors[k], alpha=0.7)
                ax.plot(real_contact_points[:, 3*k], real_contact_points[:, 3*k+1], real_contact_points[:, 3*k+2], '*', markersize=8, color=colors[k], alpha=0.7)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.title('Object-Contacts-Traj')

            plt.show(False)

        else:

            raw_input("Press enter to continue")


        

def main():

    ori_deltas  = [np.array([0.,0., -np.pi/2])]#, np.array([0.,0.,-1.8]), np.array([0.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,0.,np.pi/6])]
    pos_deltas = [np.array([0.,0.,0.])]#, np.array([0.,0.,0.]), np.array([0.5,0.,0.]), np.array([0.5,-0.5,0.]), np.array([0.,0., 0.5]), np.array([0.,0.3,-0.2])]

    # ori_deltas  = [np.array([0.,0.,1.8]), np.array([0.,0.,-1.8]), np.array([0.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,0.,np.pi/6])]
    # pos_deltas = [np.array([0.,0.,0]), np.array([0.,0.,0.]), np.array([0.5,0.,0.]), np.array([0.,0.5,0.]), np.array([0.,0., 0.3]), np.array([0.,0.3,-0.2])]

    mc = ManCntrlr()

    # time.sleep(5.)

    for ori_delta, pos_delta in zip(ori_deltas, pos_deltas):

        # print "****************************************************************************************************************************"

        print "Goal_delta \t", ori_delta
        print "Pos delta \t", pos_delta

        mc._env._reset(obj_base_fixed = False)

        pos, ori, vel, omg, lin_acc, ang_acc = mc._env.get_obj_curr_state()

        goal_ori = ori + ori_delta
        goal_omg = np.array([0., 0., 0.0])
        goal_acc = np.array([0., 0., 0.])

        pos  += pos_delta

        x_des =  np.r_[pos, goal_ori]
        xd_des = np.r_[np.zeros(3), goal_omg]
        xdd_des =  np.r_[np.zeros(3), goal_acc]

        mc.run(x_des, xd_des, xdd_des)


if __name__ == '__main__':
    main()

    