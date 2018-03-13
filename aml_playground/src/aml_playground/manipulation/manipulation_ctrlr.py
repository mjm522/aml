import copy
import numpy as np
import pybullet as pb
from qp_vel_gaits import QPVelGaits
from scipy.optimize import minimize
from scipy.linalg import block_diag
from utils import compute_f_cone_approx
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

            self._env = HandObjEnv(action_dim=18, randomize_box_ori=False, keep_obj_fixed=True, config=HAND_OBJ_CONFIG)
    
        else:
            
            self._env = env

        self._vel_gait = QPVelGaits(env=self._env)

        self._num_fingers = self._env._num_fingers
        
        hand_info = self._env.get_hand_limits()

        self._num_fingers = self._env._num_fingers

        self._finger_joint_means  = hand_info['mean']
        
        self._finger_joint_ranges = hand_info['range']

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


    def compute_B_matrix(self, rtn_list=False):

        contact_info = self._env.get_contact_points()

        B_list = []

        for k in range(self._num_fingers):

            contact_k =  contact_info[k]

            if contact_k['cp_on_finger']:

                contact_point   = contact_k['cp_on_finger'][0]

                f_cone_approx = compute_f_cone_approx(np.asarray(contact_point))
            else:
                f_cone_approx = np.zeros([3,4])

            B_list.append(f_cone_approx)

        if rtn_list:
            return B_list
        else:
            return block_diag(*B_list)

      

    def compute_f_star(self, finger_idx, x_des, xd_des, xdd_des):

        object_inertia = self._env.get_object_mass_matrix()
        pos, ori, vel, omg, lin_acc, ang_acc = self._env.get_obj_curr_state()

        pos_error = np.r_[pos, ori] - x_des
        vel_error = np.r_[vel, omg] - xd_des

        self._integral_error += pos_error*self._env._time_step

        if self._Md is None:
            self._Md = object_inertia

        fimp = np.dot(self._Md, xdd_des) - np.dot(self._Kp, pos_error) - np.dot(self._Kd, vel_error) + np.dot(self._Ki, self._integral_error)

        grasp_map = self._env.compute_grasp_map(finger_idx=None)

        contact_forces = self._env.compute_all_contact_forces_vector()

        slack_var = np.ones(6)*0.01


        def compute_f(Lambda):

            B =  self.compute_B_matrix()

            return np.dot(B, Lambda)


        lower_bound = 0.
        upper_bound = 30.

        a = np.vstack([np.ones(self._lambda_dim)*lower_bound, np.ones(self._lambda_dim)*upper_bound]).T
        bounds = tuple([tuple(a[k]) for k in range(self._lambda_dim)])
        
        cost_function = lambda x: self._alpha_1*np.linalg.norm( compute_f(x) )**2  + self._alpha_3*np.linalg.norm(slack_var)**2 + self._alpha_2*np.linalg.norm( compute_f(x) - self._fprev)**2 

        constraints = ({'type': 'eq', 'fun': lambda x: np.dot(self._Md, np.r_[lin_acc, ang_acc] ) -   np.dot( grasp_map, compute_f(x) ) + fimp - slack_var }) #slack_var

        #TNC
        res = minimize(cost_function, tuple(np.abs(np.random.randn(self._lambda_dim))), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter':10000,'disp':True})

        self._fprev = compute_f(res['x'])

        print "*******************************************Status Qp Man Ctrlr **********************************",res['success']
        # print "Lambda (Coloumns denote friction cone weights) \n", np.round(res['x'].reshape(4, self._num_fingers), 3).T
        print "Fimp \t", np.round(fimp,3)
        print "Fdes \t", np.round(np.dot( grasp_map, self._fprev ), 3)
        print "Constraint \t", np.linalg.norm(np.dot(self._Md, np.r_[lin_acc, ang_acc] ) - np.dot( grasp_map, self._fprev ) + fimp )

        Lambda = res['x'].reshape(4, self._num_fingers)
        B_list = self.compute_B_matrix(rtn_list=True)
        B =  self.compute_B_matrix()
        for k in range(self._num_fingers):
            tmp = np.zeros_like(self._fprev)
            tmp[3*k:3*k+3] = self._fprev[3*k:3*k+3]
            print "Result of finger \t", k," force on object \t", np.round(np.dot(grasp_map, tmp), 2), ", Force:\t", np.round(self._fprev[3*k:3*k+3], 2)
            # print "Lambda \n", Lambda[:, k]

        # raw_input()

        return self._fprev

    def velocity_ctrl_cmd(self, x_des, xd_des, xdd_des, gaiting=False):

        contact_info = self._env.get_contact_points()

        contact_forces = self._env.compute_all_contact_forces_vector()

        if not gaiting:
            f_star = self.compute_f_star(None, x_des, xd_des, xdd_des)

        B_list =  self.compute_B_matrix(rtn_list=True)

        colors = [[0.,1.,0.], [0.,1.,1.], [1.,0.,1.], [1.,1.,0.]]

        # raw_input()

        ee_poss, ee_oris, ee_vels, ee_omgs = self._env._hand.get_ee_states(as_tuple=True)

        # hand_jacobian = self._env._hand.get_hand_jacobian()

        obj_state = p.getLinkState(self._env._object._obj_id, 0)

        obj_pos = obj_state[0]; obj_ori = obj_state[1]

        plot_cones = True

        for k in range(self._num_fingers):

            if contact_info[k]['cp_on_block']:

                contact_point = np.asarray(contact_info[k]['cp_on_block'][0])
                color = colors[k]

                if gaiting:
                    f_star = self.compute_f_star(k, x_des, xd_des, xdd_des)

                if plot_cones:
                    for j in range(4):

                        p.addUserDebugLine(contact_point, contact_point+B_list[k][:,j], lifeTime=0, lineColorRGB=color, lineWidth=2.5)
                    p.addUserDebugLine(contact_point, contact_point+f_star[3*k:3*k+3],  lifeTime=0, lineColorRGB=[1.,0.,0.], lineWidth=2.5)


                jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self._env._hand.get_jnt_states()

                ee_force = np.asarray(jnt_reaction_forces[k][-1][:3])

                
                force_on_k =  f_star[3*k:3*k+3]

                if gaiting:
                    #desired velocity computed from gait planner
                    des_js_vel, curr_js_vel = self._vel_gait.run(finger_idx=k)

                #jacobian 
                J = self._vel_gait.get_jacobian(finger_idx=k, local_point=contact_point.tolist())

                #equation 9 part 2
                contact_force_js =  0.5*(np.asarray(obj_pos) - np.asarray(contact_point)) + force_on_k

                if gaiting:
                    #equation 9
                    tau = np.dot(self._Kv, (des_js_vel-curr_js_vel)) +  np.dot(self._Kf, contact_force_js)
                else:
                    tau = np.dot(self._Kf, np.dot(J.T, contact_force_js )  )

                print "Computed torque \t", k, "\t", np.round(tau,3)

                self._env._hand.applyAction(finger_idx=k, motor_commands=3*tau)

        p.stepSimulation()

        

def main():

    mc = ManCntrlr()

    pos, ori, vel, omg, lin_acc, ang_acc = mc._env.get_obj_curr_state()

    goal_ori = ori+np.array([0., 0., -1.8])
    goal_omg = np.array([0., 0., -0.1])
    goal_acc = np.array([0., 0., 0.])

    x_des =  np.r_[pos, goal_ori]
    xd_des = np.r_[np.zeros(3), goal_omg]
    xdd_des =  np.r_[np.zeros(3), goal_acc]

    while True:

        mc.velocity_ctrl_cmd(x_des, xd_des, xdd_des)
        p.stepSimulation()


if __name__ == '__main__':
    main()

    