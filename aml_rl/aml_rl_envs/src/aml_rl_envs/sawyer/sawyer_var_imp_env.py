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
from aml_rl_envs.sawyer.config import SAWYER_ENV_CONFIG, SAWYER_CONFIG

class SawyerEnv(AMLRlEnv):
    
    def __init__(self, config=SAWYER_ENV_CONFIG):

        self._config = config

        self._logger = aml_logging.get_logger(__name__)
        
        AMLRlEnv.__init__(self, config, set_gravity=True)

        self._reset()
        
    def _reset(self, lf=0., sf=0., rf=0., r=0., jnt_pos = None):

        self.setup_env()

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True, globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.69028195, -0.08618135, -.08734368], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        pb.changeDynamics(self._table_id, -1, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)
        
        SAWYER_CONFIG['enable_force_torque_sensors'] = True

        self._sawyer = Sawyer(config=SAWYER_CONFIG, cid=self._cid, jnt_pos = jnt_pos)

        self._spring_force = np.zeros(3)

        self.simple_step()

        self.spring_pull_traj()


    def spring_pull_traj(self, offset=np.array([0.,0.,0.80])):
        """
        generate a trajectory to pull the
        spring upwards. The end point is directly above the current point
        20cm above along the z axis
        """

        state = copy.deepcopy(self._sawyer.state())

        start_ee  = state['ee_point']
        start_ori = state['ee_ori']
        start_ori = (start_ori[1],start_ori[2],start_ori[3],start_ori[0])
        end_ee = start_ee + offset

        self._traj2pull = np.vstack([np.ones(100)*start_ee[0],
                                     np.ones(100)*start_ee[1],
                                     np.linspace(start_ee[2], end_ee[2], 100)]).T

    def virtual_spring(self, mean_pos=np.array([0.5261433,0.26867631,-0.05467355]), K=10.): #-0.05467355
        """
        this function creates a virtual spring between 
        the mean position = this position is on the table when the peg touches the table
        and the current end effector point
        the spring constant is K
        This function can be later used to put a non-linear spring as well
        """

        ee_tip = self._sawyer.state()['ee_point']
        
        x = np.linalg.norm(mean_pos-ee_tip)

        force_dir = (mean_pos-ee_tip)/x                                                                                                                                                                                          
        force = K*x*force_dir

        pb.addUserDebugLine(mean_pos, ee_tip, lifeTime=8, 
                            lineColorRGB=[0,0,1], lineWidth=4, 
                            physicsClientId=self._cid)
        
        pb.applyExternalForce(objectUniqueId=self._sawyer._robot_id,
                              linkIndex=self._sawyer._ee_index,
                              forceObj=force,
                              posObj=ee_tip,
                              flags=pb.WORLD_FRAME,
                              physicsClientId=self._cid)

        self._spring_force = force

        

    def reward(self, traj):
        '''
        Computing reward for the given (forward-simulated) trajectory
        '''

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        desired_traj = traj['traj']
        true_traj = traj['ee_traj']
        ee_data_traj = traj['other_ee_data']
        force_traj = traj['ee_wrenches'][:,:3]
        torques_traj = traj['ee_wrenches'][:,3:]

        num_data = len(desired_traj)

        penalty_traj = np.zeros(num_data)

        closeness_traj = np.zeros(num_data)

        for k in range(num_data-1):

            # desired_force = np.dot(ee_M_traj[k][:3,:3], desired_traj[k+1]-desired_traj[k])

            reaction_force = force_traj[k,:] #*-1

            penalty_force = reaction_force # - np.dot(reaction_force, desired_force)*(desired_force/np.linalg.norm(desired_force))

            penalty_traj[k] = np.linalg.norm(force_traj[k,:])#np.linalg.det(ee_data_traj[k]['task_irr'])

            closeness_traj[k] =  np.linalg.norm(desired_traj[k]-true_traj[k])

        force_penalty = 0.5*np.sum(sigmoid(penalty_traj)) + 0.5*np.sum(sigmoid(np.diff(penalty_traj)))

        penalty =  -force_penalty + 1.5*np.sum(sigmoid(closeness_traj))

        self._logger.debug("\n*****************************************************************")
        # self._logger.debug("penalty_force \t %f"%(force_penalty,))
        self._logger.debug("penalty_force \t %f"%(np.sum(penalty_traj),))
        self._logger.debug("*******************************************************************")

        self._penalty = {'force':force_penalty,
                 'total':penalty}

        return penalty

    def fwd_simulate(self, traj, ee_ori=None, policy=None):
        """
        the part of the code to move the robot along a trajectory
        """
        ee_traj = []
        # ee_M_traj = []
        full_contacts_list = []
        ee_wrenches = []
        ee_wrenches_local = []
        ee_data_traj = []
        context_list = []
        param_list = []

        # plot_demo(traj, color=[1,0,0], start_idx=0, life_time=0., cid=self._cid)

        if ee_ori is None:
            goal_ori = (2.73469166e-02, 9.99530233e-01, 3.31521029e-04, 1.38329146e-02) #None#
        else:
            goal_ori = pb.getQuaternionFromEuler(ee_ori)

        for k in range(traj.shape[0]):

            self.virtual_spring()

            #for variable impedance, we will have to compute
            #parameters for each time
            if policy is not None:
                s = self.context()
                w  = policy.compute_w(context=s)
                Kp = np.ones(3)+w[:3]
                Kd = np.ones(3)+w[3:]
                context_list.append(copy.deepcopy(s))
                param_list.append(copy.deepcopy(w))
            else:
                Kp, Kd = None, None

            cmd = self._sawyer.inv_kin(ee_pos=traj[k, :].tolist(), ee_ori=goal_ori)

            if Kp is not None:
                lin_jac = self._sawyer.state()['jacobian']
                js_Kp = np.dot(lin_jac.T, Kp)
                js_Kp = np.clip(js_Kp, 0.01, 1)
                self._logger.debug("\n \n Kp \t {}".format(js_Kp))
                # print "actual \t", self._sawyer.state()['ee_vel']
                # print "computed \t", np.dot(lin_jac, self._sawyer.state()['velocity'])
                # raw_input("press to continue")
            else: 
                js_Kp = None

            if Kd is not None:
                js_Kd = np.dot(lin_jac.T, Kd)
                js_Kd = np.clip(js_Kd, 0.5, 1)
                self._logger.debug("\n \n Kd \t {}".format(js_Kd))
            else:
                js_Kd = None

            # js_Kp = np.ones(7)*0.01

            self._sawyer.apply_action(cmd, js_Kp, js_Kd)

            ee_pos, ee_ori = self._sawyer.get_ee_pose()

            ee_traj.append(ee_pos)

            state = self._sawyer.state()
            state['spring_force'] = self._spring_force

            ee_data_traj.append(state)

            #desired Mass traj
            # ee_M_traj.append(self._sawyer.state()['inertia'])

            full_contacts_list.append(self.get_contact_details())
            ee_wrenches.append(self._sawyer.get_ee_wrench(local=False))
            ee_wrenches_local.append(self._sawyer.get_ee_wrench(local=True))

            # time.sleep(0.1)
            self.simple_step()

        return { 'ee_traj':np.asarray(ee_traj),
                 'traj':traj, 
                 'contact_details':full_contacts_list, 
                 'ee_wrenches':np.asarray(ee_wrenches),
                 'ee_wrenches_local':np.asarray(ee_wrenches_local),
                 'other_ee_data':ee_data_traj,
                 'contexts':context_list,
                 'params':param_list}
        

    def execute_policy(self, policy=None, show_demo=False):
        """
        this function takes in two arguments
        policy function, here stiffness and damping terms
        context of the policy
        """

        if show_demo:
            plot_demo(self._traj2pull, start_idx=0, life_time=4, cid=self._cid)

        traj_draw = self.fwd_simulate(traj=self._traj2pull, policy=policy)
     
        reward = self.reward(traj_draw)
        
        return traj_draw['contexts'], traj_draw['params'], traj_draw, reward

    def context(self):

        return self._spring_force


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