import os
import gym
import time
import math
import random
import numpy as np
import pybullet as pb
from gym import spaces
from gym.utils import seeding
from aml_io.log_utils import aml_logging
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.sawyer.sawyer import Sawyer
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.controller.imp_ctrlr import ImpController
from aml_rl_envs.sawyer.config import SAWYER_ENV_CONFIG, SAWYER_CONFIG


class SawyerEnv(AMLRlEnv):
    
    def __init__(self, demo2follow=None, config=SAWYER_ENV_CONFIG):

        self._config = config

        self._logger = aml_logging.get_logger(__name__)

        self._action_repeat = config['action_repeat']

        self._goal_box = np.array([0.5, 1.,-0.35]) #x,y,z 

        self._demo2follow = demo2follow
        
        AMLRlEnv.__init__(self, config, set_gravity=True)

        self._reset()
        
        self._goal_ori = np.asarray(pb.getEulerFromQuaternion((-0.52021453, -0.49319602,  0.47898476, 0.50666373)))

        #facing sawyer, from left side
        hole1 = np.array([0., -0.725*0.15, 0.])
        hole2 = np.array([0., -0.425*0.15, 0.])
        hole3 = np.array([0., 0., 0.0])
        hole4 = np.array([0., 0.375*0.15, 0.])
        hole5 = np.array([0., 0.775*0.15, 0.])

        self._hole_locs = [hole1, hole2, hole3, hole4, hole5]

    def _reset(self, lf=0., sf=0., rf=0., r=0., jnt_pos = None):

        self.setup_env()

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True, globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.69028195, -0.08618135, -.08734368], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        pb.changeDynamics(self._table_id, -1, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)
        
        SAWYER_CONFIG['enable_force_torque_sensors'] = True

        self._sawyer = Sawyer(config=SAWYER_CONFIG, cid=self._cid, jnt_pos = jnt_pos)

        # self._imp_ctrlr = ImpController(robot_interface=self._sawyer)

        self.simple_step()
        

    def reward(self, traj):
        '''
            Computing reward for the given (forward-simulated) trajectory
        '''

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        desired_traj = traj['dmp']
        true_traj = traj['ee_traj']
        ee_data_traj = traj['other_ee_data']
        force_traj = traj['ee_wrenches'][:,:3]
        torques_traj = traj['ee_wrenches'][:,3:]


        num_data = len(desired_traj)

        penalty_traj = np.zeros(num_data)

        closeness_traj = np.zeros(num_data)

        for k in range(num_data-1):

            # desired_force = np.dot(ee_M_traj[k][:3,:3], desired_traj[k+1]-desired_traj[k])

            # reaction_force = force_traj[k,:] #*-1

            # penalty_force = reaction_force # - np.dot(reaction_force, desired_force)*(desired_force/np.linalg.norm(desired_force))

            penalty_traj[k] = np.linalg.norm(force_traj[k,:])#np.linalg.det(ee_data_traj[k]['task_irr'])

            closeness_traj[k] =  np.linalg.norm(desired_traj[k]-true_traj[k])

            # print "Acceleration dir \t", np.round(desired_traj[k+1]-desired_traj[k], 3)
            # print "Desired force \t", np.round(desired_force,3)
            # print "reaction_force  \t", np.round(reaction_force,3)
            # print "penalty_force  \t", np.round(penalty_force,3)

            # raw_input("Press to continue")

        # penalty_traj -= np.max(penalty_traj)
        # closeness_traj -= np.max(closeness_traj)

        # penalty_traj /= np.sum(penalty_traj)
        # closeness_traj /= np.sum(closeness_traj)

        # print "\n\n\n\n\n\n\n"
        # print np.sum(sigmoid(penalty_traj))
        # print np.sum(sigmoid(closeness_traj))
        # print "\n\n\n\n\n\n\n"

        # raw_input("press")

        penalty = -0.5*np.sum(sigmoid(penalty_traj)) - 0.8*np.sum(sigmoid(closeness_traj))

        self._logger.debug("*******************************************************************")
        self._logger.debug("penalty_force \t %f"%(penalty,))
        self._logger.debug("*******************************************************************")

        return penalty

    def fwd_simulate(self, dmp, ee_ori=None, joint_space=False, Kp=None, Kd=None):
        """
        implement the dmp
        """
        ee_traj = []
        # ee_M_traj = []
        full_contacts_list = []
        ee_wrenches = []
        ee_wrenches_local = []
        ee_data_traj = []

        # plot_demo(dmp, color=[1,0,0], start_idx=0, life_time=0., cid=self._cid)

        if ee_ori is None:
            goal_ori = None#(2.73469166e-02, 9.99530233e-01, 3.31521029e-04, 1.38329146e-02)
        else:
            goal_ori = pb.getQuaternionFromEuler(ee_ori)

        for k in range(dmp.shape[0]):

            if joint_space:

                cmd = dmp[k, :]

            else:
                # goal_ori = (-0.52021453, -0.49319602, 0.47898476, 0.50666373)
                # goal_ori = 
                # #in euler (3.1401728051502205, 0.027638219089217236, 3.0868671400979135)
                # print pb.getEulerFromQuaternion(goal_ori)
                # raw_input()
                # print "Goal pos \t",dmp[k, :].tolist()
                # cmd = self._imp_ctrlr.ik(ee_pos=dmp[k, :].tolist(), ee_ori=goal_ori)

                cmd = self._sawyer.inv_kin(ee_pos=dmp[k, :].tolist(), ee_ori=goal_ori)

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
                    self._logger.debug("\n \n Kp \t {}".format(js_Kd))
                else:
                    js_Kd = None

            self._sawyer.apply_action(cmd, js_Kp, js_Kd)

            ee_pos, ee_ori = self._sawyer.get_ee_pose()

            ee_traj.append(ee_pos)

            ee_data_traj.append(self._sawyer.state())

            #desired Mass traj
            # ee_M_traj.append(self._sawyer.state()['inertia'])

            full_contacts_list.append(self.get_contact_details())

            ee_wrenches.append(self._sawyer.get_ee_wrench(local=False))
            ee_wrenches_local.append(self._sawyer.get_ee_wrench(local=True))

            # time.sleep(0.1)
            self.simple_step()

            # lin_jac = self._sawyer.state()['jacobian']

            # comp_vel = np.hstack([self._sawyer.state()['ee_vel'], self._sawyer.state()['ee_omg']])

            # print "\n\n"

            # print "Computed \t",np.dot(lin_jac, self._sawyer.state()['velocity'])

            # print "Val\t", comp_vel

            # print "Diff \t", np.linalg.norm(np.dot(lin_jac, self._sawyer.state()['velocity']) - comp_vel)

            # print "\n\n"

            # raw_input("\npresss")

        # block_pos, block_ori = pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)
        # print "Block pos \t", np.asarray(block_pos)
        # print "EE pos\t", np.asarray(ee_pos)
            
        return { 'ee_traj':np.asarray(ee_traj),
                 'dmp':dmp, 
                 'contact_details':full_contacts_list, 
                 'ee_wrenches':np.asarray(ee_wrenches),
                 'ee_wrenches_local':np.asarray(ee_wrenches_local),
                 'other_ee_data':ee_data_traj }
        
    def context(self):
        """
        Context is the bottom base of the box.
        """

        s = np.random.uniform(-0.1, 0.1)

        lf = 0.5 + s 

        self._reset(lf=lf)     

        return np.array([s])


    def execute_policy(self, w, s, show_demo=False):

        if w is not None:
            Kp = np.ones(3)+w[:3]
            Kd = np.ones(3)+w[3:]
        else:
            Kp = Kd = None

        # ee_ori = tuple(np.array([3.1401728051502205, 0.027638219089217236, 3.0868671400979135]) + ee_ori_offset)

        dmp_draw  = self._demo2follow(dmp_type='draw')
       
        if show_demo:
            plot_demo(dmp_draw, start_idx=0, life_time=4, cid=self._cid)

        traj_draw= self.fwd_simulate(dmp=dmp_draw, Kp=Kp, Kd=Kd)
     
        reward = self.reward(traj_draw) #+ 4*np.linalg.norm(js_Kp)
        
        return traj_draw, reward


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