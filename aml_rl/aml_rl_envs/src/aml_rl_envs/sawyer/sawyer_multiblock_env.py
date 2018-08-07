import os
import copy
import time
import math
import random
import numpy as np
import pybullet as pb
from aml_io.log_utils import aml_logging
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.sawyer.sawyer import Sawyer
from aml_rl_envs.utils.collect_demo import plot_demo

class SawyerEnv(AMLRlEnv):

    def __init__(self, config):

        self._config = config

        self._logger = aml_logging.get_logger(__name__)

        AMLRlEnv.__init__(self, config, set_gravity=False)

        self._reset()

    def _reset(self, num_boxes = 3, mass_diff = 2, lf=0., sf=0., rf=0., r=0., jnt_pos = None):

        self.setup_env()

        scale = 0.5

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"ridge_table.urdf"), useFixedBase=True, globalScaling=scale, physicsClientId=self._cid)

        self._box_ids = []

        mass = 1

        box_pos_list = np.linspace(-1.5*scale, 1.5*scale, num_boxes + 2)

        for i in range(num_boxes):

            self._box_ids.append(pb.loadURDF(os.path.join(self._urdf_root_path,"box.urdf"), globalScaling=scale, physicsClientId=self._cid))

            pb.resetBasePositionAndOrientation(self._box_ids[i], [0.5, -0.3 + box_pos_list[i+2], -0.3 + (0.05+0.15)*scale], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

            pb.changeDynamics(self._box_ids[i], -1, mass = mass, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)

            mass += mass_diff

        pb.setGravity(0., 0., -9.81)

        pb.resetBasePositionAndOrientation(self._table_id, [0.5, -0.3, -0.3], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        pb.changeDynamics(self._table_id, -1, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)

        self._sawyer = Sawyer(config=self._config['robot_config'], cid=self._cid, jnt_pos = jnt_pos)

        self.simple_step()



    def torque_controller(self, x_des, dx_des=np.zeros(3), ddx_des=np.zeros(3), Kp=None, Kd=None):

        des_ori = np.array([-1.539, 0.121, 2.695])

        if Kp is None:
            Kp = np.ones(3)

        if Kd is None:
            Kd =  np.ones(3)

        state = self._sawyer.state()
        jac = state['jacobian']
        x = state['ee_point']
        dx = state['ee_vel']
        dw = state['ee_omg']

        theta = state['ee_ori']
        theta = np.asarray(pb.getEulerFromQuaternion((theta[1],theta[2],theta[3], theta[0])))

        ee_wrench = self._sawyer.get_ee_wrench(local=False)

        jac_inv = np.linalg.pinv(jac)
        Mee_inv = state['Mee_inv']

        err_ee_pos = np.hstack([x_des,  des_ori]) - np.hstack([x, theta])
        err_ee_vel = np.hstack([dx_des, np.zeros(3)]) - np.hstack([dx, dw])

        des_ee_acc = np.hstack([ddx_des, np.zeros(3)])

        Kp_aug = np.diag(np.hstack([Kp, np.ones(3)*1]))
        Kd_aug = np.diag(np.hstack([Kd, np.ones(3)*1]))

        Kp_term = np.dot(Mee_inv, np.dot( Kp_aug, err_ee_pos ) )

        Kd_term = np.dot(Mee_inv, np.dot( Kd_aug, err_ee_vel ) )

        # Kp_term = np.dot(Mee_inv, np.dot( np.diag(Kp), (x_des-x) ) )
        # Kd_term = 0*np.dot(Mee_inv, np.dot( np.diag(Kd), (dx_des-dx) ) )
        # f_term = 
        # import pdb
        # pdb.set_trace()

        u = np.dot(jac_inv, (des_ee_acc + Kp_term + Kd_term))

        return u

    def reward(self, traj):
        '''
        Computing reward for the given (forward-simulated) trajectory
        '''

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        desired_traj = traj['traj']
        true_traj = traj['ee_traj']
        ee_vel_traj = traj['ee_vel_traj']
        ee_data_traj = traj['other_ee_data']
        force_traj = traj['ee_wrenches'][:,:3]
        torques_traj = traj['ee_wrenches'][:,3:]
        u_traj = traj['u_list']

        num_data = len(desired_traj)
        # penalty_force_traj = np.zeros(num_data)
        penalty_u_traj = np.zeros(num_data)
        closeness_traj = np.zeros(num_data)

        for k in range(num_data-1):

            #it should be sum since des force traj is negative of spring force
            # penalty_force_traj[k] = np.linalg.norm(self._des_force_traj[k,:] + force_traj[k,:])

            penalty_u_traj[k] = np.linalg.norm(u_traj[k])

            closeness_traj[k] =  np.linalg.norm(desired_traj[k]-true_traj[-1]) 

        u_force_penalty = self._config['u_weight']*sigmoid(penalty_u_traj) 

        goal_penalty =  self._config['goal_weight']*sigmoid(closeness_traj)

        reward_traj = u_force_penalty  + goal_penalty

        total_penalty = -np.sum( reward_traj )

        self._logger.debug("\n*****************************************************************")
        self._logger.debug("u penalty \t %f"%(np.sum(penalty_u_traj),))
        self._logger.debug("goal_penalty \t %f"%(np.sum(closeness_traj),))
        self._logger.debug("*******************************************************************")

        self._penalty = {
                 'total':total_penalty,
                 'reward_traj':reward_traj}

        return self._penalty

    def fwd_simulate(self, traj, ee_ori=None, policy=None, inv_kin=True):
        """
        the part of the code to move the robot along a trajectory
        """
        ee_traj = []
        state_traj = []
        ee_vel_traj = []
        # ee_M_traj = []
        full_contacts_list = []
        ee_wrenches = []
        ee_wrenches_local = []
        ee_data_traj = []
        context_list = []
        param_list = []
        u_list = []

        # plot_demo(traj, color=[1,0,0], start_idx=0, life_time=0., cid=self._cid)

        if ee_ori is None:
            goal_ori = (2.73469166e-02, 9.99530233e-01, 3.31521029e-04, 1.38329146e-02) #None#
        else:
            goal_ori = pb.getQuaternionFromEuler(ee_ori)

        for k in range(traj.shape[0]):

            #for variable impedance, we will have to compute
            #parameters for each time
            if policy is not None:
                s = self.context()
                w  = policy.compute_w(context=s)
                # w = np.zeros(6)
                
                # ee_pos, ee_ori = self._sawyer.get_ee_pose()

                # x_tilde = (traj[k, :] - ee_pos)

                # Kp_calc = np.divide(self._spring_force, x_tilde) #np.clip(np.random.randn(3), -0.5, 2.5)#

                # if np.any(np.isnan(Kp_calc)) or np.any(np.isinf(Kp_calc)):
                #     Kp_calc = np.zeros(3)

                # Kp =  np.ones(3) + Kp_calc #w[:3]
                # Kd =  np.ones(3) + np.zeros(3)#w[3:]

                Kp =  np.ones(3) + w[:3]
                Kd =  np.ones(3) + w[3:]

                context_list.append(copy.deepcopy(s))
                param_list.append(copy.deepcopy(w))
            else:
                Kp, Kd = None, None

            # js_Kp = np.ones(7)*1.5#np.dot(lin_jac.T, Kp)
            # js_Kd = None
            # js_Kd = np.ones(7)*1000#np.clip(js_Kp, 0.01, 1)

            if self._sawyer._ctrl_type == 'pos':
                
                # lin_jac = self._sawyer.state()['jacobian']
                lin_jac = self._sawyer.state()['jacobian'][:3,:]
                inertia_mat = self._sawyer.state()['inertia']

                m_inv_j_trans = np.dot(np.linalg.inv(inertia_mat),lin_jac.T)
                # print self._sawyer.state()['jacobian'][:3,:]
                # raw_input()

                if Kp is not None:
                    # js_Kp = np.dot(np.linalg.pinv(lin_jac), Kp)
                    # js_Kp = np.dot(m_inv_j_trans[:,:3],Kp)
                    js_Kp = np.dot(np.dot(lin_jac.T, np.diag(Kp)), lin_jac)
                    js_Kp = np.clip(js_Kp, 0.01, 10)
                    self._logger.debug("\n \n Kp \t {}".format(np.diag(js_Kp)))
                else:
                    js_Kp = None

                if Kd is not None:
                    # js_Kd = np.dot(np.linalg.pinv(lin_jac), Kd)
                    # js_Kd = np.dot(m_inv_j_trans[:,:3],Kd)
                    js_Kd = np.dot(np.dot(lin_jac.T, np.diag(Kd)), lin_jac)
                    js_Kd = np.clip(js_Kd, 0., 10)
                    self._logger.debug("\n \n Kd \t {}".format(np.diag(js_Kd)))
                else:
                    js_Kd = None

                cmd = self._sawyer.inv_kin(ee_pos=traj[k, :].tolist(), ee_ori=goal_ori)

                state = self._sawyer.state(ori_type = 'eul')

                u = np.diag(np.dot(js_Kp, (cmd-state['position'])) + np.dot(js_Kd, (-state['velocity'])))

                u_list.append(u)

                self._sawyer.apply_action(cmd, np.diag(js_Kp), np.diag(js_Kd))
                # self._sawyer.apply_action(cmd, np.ones(7)*2, np.ones(7)*2)
            
            elif self._sawyer._ctrl_type == 'torque':
                # Kp = np.ones(3)*0.0001
                # Kd = np.ones(3)*0.
                cmd = self.torque_controller(x_des=traj[k, :], dx_des=np.zeros(3), ddx_des=np.zeros(3), Kp=Kp, Kd=Kd)
                u_list.append(cmd)
                self._sawyer.apply_action(cmd)

            ee_pos, ee_ori = self._sawyer.get_ee_pose()

            ee_traj.append(ee_pos)
            # state_traj.append(self._spring_mean-ee_pos)
            # spring_force.append(copy.deepcopy(self._spring_force))

            state = self._sawyer.state(ori_type = 'eul')
            state['req_traj'] = traj[k, :]

            ee_data_traj.append(state)

            ee_vel_traj.append(state['ee_vel'])

            #desired Mass traj
            # ee_M_traj.append(self._sawyer.state()['inertia'])

            full_contacts_list.append(self.get_contact_details())
            ee_wrenches.append(self._sawyer.get_ee_wrench(local=False))
            ee_wrenches_local.append(self._sawyer.get_ee_wrench(local=True))

            # time.sleep(0.1)
            self.simple_step()

        return { 'ee_traj':np.asarray(ee_traj),
                 'state_traj':state_traj,
                 'ee_vel_traj':np.asarray(ee_vel_traj),
                 'traj':traj,
                 'contact_details':full_contacts_list,
                 'ee_wrenches':np.asarray(ee_wrenches),
                 'ee_wrenches_local':np.asarray(ee_wrenches_local),
                 'other_ee_data':ee_data_traj,
                 'contexts':context_list,
                 'params':param_list,
                 'u_list':u_list,}
        
    def execute_policy(self, policy=None, show_demo=True, sinusoid=False):
        """
        this function takes in two arguments
        policy function, here stiffness and damping terms
        context of the policy
        """

        if show_demo:
            plot_demo(self._traj2pull, start_idx=0, life_time=0, cid=self._cid)


        if sinusoid:
            ##create sinusoid
            ori = copy.deepcopy(self._traj2pull)
            
            flipped = np.flip(ori,0)

            self._traj2pull = np.vstack([ori, flipped, ori, flipped, ori, flipped, ori, flipped])

        traj_draw = self.fwd_simulate(traj=self._traj2pull, policy=policy)
     
        reward = self.reward(traj_draw)
        
        return traj_draw, reward['total'] #traj_draw['contexts'], traj_draw['params'], traj_draw['state_traj'], traj_draw['ee_wrenches_local'][:,:3], reward['reward_traj'], traj_draw
        

    def context(self):

        state = self._sawyer.state(ori_type = 'eul')

        return np.hstack([ state['ee_point'], state['ee_vel'], self._sawyer.get_ee_wrench(local=True)[:3] ])

        # return self._sawyer.get_ee_wrench(local=True)[:3]
        # return self._spring_force


    def get_contact_details(self):
        '''
        Get contact details of every contact when the peg is in contact with any part of the hole.
        '''

        full_details = pb.getContactPoints(bodyA=self._sawyer._robot_id, linkIndexA=19,
                              bodyB=self._table_id, physicsClientId=self._cid)
        contact_details = []

        if len(full_details) > 0:

            for contact_id in range(len(full_details)):

                details = {}

                details['obj_link'] = full_details[contact_id][4]
                details['contact_pt'] = full_details[contact_id][6]
                details['contact_force'] = full_details[contact_id][9]

                contact_details.append(details)

        return contact_details



def main():
    from aml_io.io_tools import save_data
    from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import sawyer_env_1

    env = SawyerEnv(sawyer_env_1)

    # for k in spring_ks:
    #     env.configure_spring(K=k)
    #     data = env.execute_policy(policy=None, show_demo=False, sinusoid=True)
    #     save_data(data, file_name%(k))
    #     env._reset()

    while True:
        # try:
        env.simple_step()
        # except KeyboardInterrupt:
        #     break
        # raw_input("Press enter to exit")


if __name__ == '__main__':
    main()
