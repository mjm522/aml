import os
import copy
import random
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_io.log_utils import aml_logging
from aml_rl_envs.point_mass.point_mass import PointMass
from aml_robot.bullet.bullet_visualizer import setup_bullet_visualizer
from aml_rl_envs.point_mass.config import POINT_MASS_ENV_CONFIG, POINT_MASS_CONFIG

class PointMassEnv():

    def __init__(self, config=POINT_MASS_ENV_CONFIG):

        self._config = config

        self._time_step = self._config['time_step']

        self._num_traj_points = self._config['num_traj_points']

        self._spring_line = None

        self._logger = aml_logging.get_logger(__name__)

        self._reward_gamma = np.asarray([self._config['reward_gamma']**k for k in range(self._num_traj_points)])

        self._cid = setup_bullet_visualizer(self._config['renders'])

        pb.setGravity(0,0,0., physicsClientId=self._cid)

        self._reset()

    def simple_step(self):
        pb.stepSimulation(physicsClientId=self._cid)


    def _reset(self, lf=0., sf=0., rf=0., r=0., jnt_pos = None):

        pb.resetSimulation(physicsClientId=self._cid)

        self._point_mass = PointMass(cid=self._cid, config=POINT_MASS_CONFIG,  scale=0.5)

        self._spring_force = np.zeros(3)

        pb.setTimeStep(self._time_step, physicsClientId=self._cid)

        self.simple_step()

        self.configure_spring(self._config['spring_stiffness'])

        self.spring_pull_traj(ramp_traj=self._config['ramp_traj_flag'])


    def spring_pull_traj(self, offset=np.array([0.,0.,0.5]), ramp_traj=True):
        """
        generate a trajectory to pull the
        spring upwards. The end point is directly above the current point
        20cm above along the z axis
        """

        self._spring_base  = self._point_mass.get_ee_pose()[0]

        end_ee = self._spring_base + offset

        self._target_point = end_ee

        if ramp_traj:
            self._traj2pull = np.vstack([np.ones(self._num_traj_points)*self._spring_base[0],
                                         np.ones(self._num_traj_points)*self._spring_base[1],
                                         np.linspace(self._spring_base[2], end_ee[2], self._num_traj_points)]).T
        else:
            self._traj2pull = np.tile(self._target_point, (self._num_traj_points, 1))

        self._des_force_traj = -self._spring_K*(self._spring_mean-self._traj2pull)

    def configure_spring(self, K, mean_pos=np.array([0.,0.,0.53])):

        self._spring_K = K
        self._spring_mean = mean_pos

    def virtual_spring(self, max_expected_force=300): #-0.05467355
        """
        this function creates a virtual spring between
        the mean position = this position is on the table when the peg touches the table
        and the current end effector point
        the spring constant is K
        This function can be later used to put a non-linear spring as well
        """

        ee_tip = self._point_mass.get_ee_pose()[0]

        # x = np.linalg.norm(self._spring_mean-ee_tip)

        # force_dir = (self._spring_mean-ee_tip)/x
        # force = self._spring_K*x*force_dir

        force = self._spring_K*(self._spring_mean-ee_tip)

        old_line = self._spring_line
        color = np.linalg.norm(force) / float(max_expected_force) # safe to exceed 1.0
        self._spring_line = pb.addUserDebugLine(self._spring_mean, ee_tip, lifeTime=0,
                            lineColorRGB=[color, 1-color, 0], lineWidth=12,
                            physicsClientId=self._cid)
        if old_line is not None:
            pb.removeUserDebugItem(old_line)

        pb.applyExternalForce(objectUniqueId=self._point_mass._robot_id,
                              linkIndex=-1,
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
            return 1. / (1. + np.exp(-x))

        desired_traj = traj['traj']
        true_traj = traj['ee_traj']
        # ee_vel_traj = traj['ee_vel_traj']
        force_traj = traj['ee_wrenches'][:,:3]
        # torques_traj = traj['ee_wrenches'][:,3:]
        u_traj = traj['u_list']


        # import matplotlib.pyplot as plt
        # plt.subplot(3,1,1)
        # plt.plot(desired_traj[:,0],'r')
        # plt.plot(true_traj[:,0],'g')
        # plt.subplot(3,1,2)
        # plt.plot(desired_traj[:,1],'r')
        # plt.plot(true_traj[:,1],'g')
        # plt.subplot(3,1,3)
        # plt.plot(desired_traj[:,2],'r')
        # plt.plot(true_traj[:,2],'g')
        # plt.show()

        penalty_force_traj = np.zeros(self._num_traj_points)
        penalty_u_traj = np.zeros(self._num_traj_points)
        closeness_traj = np.zeros(self._num_traj_points)

        for k in range(self._num_traj_points):

            #it should be sum since des force traj is negative of spring force
            penalty_force_traj[k] = np.linalg.norm(self._des_force_traj[k,:] + force_traj[k,:])

            penalty_u_traj[k] = np.linalg.norm(u_traj[k])

            closeness_traj[k] =  np.linalg.norm(desired_traj[k,:]-true_traj[k,:])
            # closeness_traj[k] =  np.linalg.norm(self._target_point-true_traj[-1])
            if k == self._num_traj_points-1:
                closeness_traj[k] =  np.linalg.norm(desired_traj[k,:]-true_traj[k,:])*self._config['finishing_weight']

            if np.linalg.norm(desired_traj[-1,:]-true_traj[k,:]) < 0.05:
                # print "\n\n\n\n\n\n\nreached goal....\n\n\n\n\n\n\n"
                closeness_traj[k] = 0.0

        u_penalty = self._config['u_weight']*penalty_u_traj

        force_penalty = self._config['f_des_weight']*penalty_force_traj#self._config['f_dot_weight']*sigmoid( np.hstack( [np.diff(penalty_force_traj), 0] ))

        goal_penalty =  self._config['goal_weight']*closeness_traj

        reward_traj = -sigmoid(np.cumsum( np.multiply( (u_penalty + force_penalty  + goal_penalty), self._reward_gamma ) ))

        # import matplotlib.pyplot as plt
        # plt.plot(reward_traj)
        # plt.show()

        total_penalty = np.sum( reward_traj ) 

        self._logger.debug("\n*****************************************************************")
        self._logger.debug("penalty_force \t %f"%(np.sum(force_penalty),))
        self._logger.debug("u penalty \t %f"%(np.sum(u_penalty),))
        self._logger.debug("goal_penalty \t %f"%(np.sum(goal_penalty),))
        self._logger.debug("total_penalty \t %f"%(total_penalty),) 
        self._logger.debug("*******************************************************************")

        self._penalty = {
                 'force':np.sum(penalty_force_traj),
                 'total':total_penalty,
                 'reward_traj':reward_traj}

        return self._penalty

    def fwd_simulate(self, traj, ee_ori=None, policy=None, inv_kin=True, explore=True, jnt_space=False):
        """
        the part of the code to move the robot along a trajectory
        """
        ee_traj = []
        state_traj = []
        ee_vel_traj = []
        full_contacts_list = []
        ee_wrenches = []
        ee_wrenches_local = []
        ee_data_traj = []
        context_list = []
        param_list = []
        spring_force = []
        u_list = []

        if ee_ori is None:
            goal_ori = (2.73469166e-02, 9.99530233e-01, 3.31521029e-04, 1.38329146e-02) #None#
        else:
            goal_ori = pb.getQuaternionFromEuler(ee_ori)

        tmp = np.array([ 1000.,  1000., 1000.,  np.sqrt(1000.),  np.sqrt(1000.),  np.sqrt(1000.)])

        for k in range(self._num_traj_points):

            self.simple_step()

            self.virtual_spring()

            #for variable impedance, we will have to compute
            #parameters for each time

            if policy is not None:
                s = self.context()
                # const = 12
                # w  = np.abs(np.random.randn(6))*100
                #np.hstack([np.ones(3)*const, 0.01*np.ones(3)*np.sqrt(const)])#np.abs(np.random.randn(6))*0.001
                w  = np.multiply(tmp, policy[k].compute_w(context=s, explore=explore))
                # print w

                Kp =  w[:3]#np.ones(3)*1 + w[:3]
                Kd =   w[3:] #0.01*np.ones(3)*np.sqrt(1) + w[3:]

                # js_Kp = np.clip(Kp, 0.0, 100)
                # js_Kd = np.clip(Kd, 0.0, 100)

                js_Kp = np.diag(Kp)
                js_Kd = np.diag(Kd)

                # if not jnt_space:
                #     self._logger.debug("\n \n EE Param Kp \n {}".format(w[:3]))  
                #     self._logger.debug("\n \n EE Kp \n {}".format(Kp))
                #     self._logger.debug("\n \n EE Parm Kd \n {}".format(w[3:]))
                #     self._logger.debug("\n \n EE Kd \n {}".format(Kd))
                #     self._logger.debug("\n \n JS Kd \n {}".format(np.diag(js_Kd)))
                # self._logger.debug("\n \n JS Kp \n {}".format(np.diag(js_Kp)))
                # self._logger.debug("\n#############################################################")

                ee_pos = self._point_mass.get_ee_pose()[0]
                ee_vel = self._point_mass.get_ee_velocity()[0]

                u = np.dot(js_Kp, (traj[k, :]-ee_pos)) + np.dot(js_Kd, -ee_vel)

                # print "\n\n\n\n\n\n\n\n",u, "\n\n\n\n\n\n\n"

                self._point_mass.apply_action(u=u)

            else:
                js_Kp, js_Kd = None, None

            state = self._point_mass.state(ori_type='eul')

            ee_pos, ee_ori = state['ee_point'], state['ee_ori']

            # print "ee_pos \t", ee_pos

            context_list.append(copy.deepcopy(s))
            param_list.append(copy.deepcopy(w))

            u_list.append(u)
            ee_traj.append(copy.deepcopy(ee_pos))
            state_traj.append(self._spring_mean-ee_pos)
            spring_force.append(copy.deepcopy(self._spring_force))
            
            state['spring_force'] = self._spring_force
            state['req_traj'] = traj[k, :]

            ee_data_traj.append(state)

            ee_vel_traj.append(state['ee_vel'])

            ee_wrenches.append(self._point_mass.get_ee_wrench(local=False))
            ee_wrenches_local.append(self._point_mass.get_ee_wrench(local=True))

        # raw_input()

        return { 'ee_traj':np.asarray(ee_traj),
                 'state_traj':state_traj,
                 'ee_vel_traj':np.asarray(ee_vel_traj),
                 'traj':traj,
                 'contact_details':full_contacts_list,
                 'ee_wrenches':np.asarray(ee_wrenches),
                 'spring_force':np.asarray(spring_force),
                 'ee_wrenches_local':np.asarray(ee_wrenches_local),
                 'other_ee_data':ee_data_traj,
                 'contexts':context_list,
                 'params':param_list,
                 'u_list':u_list,}
        
    def execute_policy(self, policy=None, show_demo=False, sinusoid=False, explore=True, jnt_space = False):
        """
        this function takes in two arguments
        policy function, here stiffness and damping terms
        context of the policy
        """

        # if show_demo:
        #     plot_demo(self._traj2pull, start_idx=0, life_time=0, cid=self._cid)


        # if sinusoid:
        #     ##create sinusoid
        #     ori = copy.deepcopy(self._traj2pull)
            
        #     flipped = np.flip(ori,0)

        #     self._traj2pull = np.vstack([ori, flipped, ori, flipped, ori, flipped, ori, flipped])


        traj_draw = self.fwd_simulate(traj=self._traj2pull, policy=policy, explore=explore, jnt_space = jnt_space)
     
        reward = self.reward(traj_draw)
        
        return traj_draw, reward 
        

    def context(self):

        state = self._point_mass.state(ori_type = 'eul')

        ee_pos = self._point_mass.get_ee_pose()[0]

        ee_vel = self._point_mass.get_ee_velocity()[0]

        return np.hstack([ ee_pos, ee_vel, self._spring_force]) #, self._point_mass.get_ee_wrench(local=True)[:3] 


class DummyPolicy():
    """
    this is a dirty hack policy
    when a context is called it
    simply returns from a saved list of 
    values
    """

    def __init__(self, t):

        self._count = 0
        Kp = 999.6
        Kd = np.sqrt(Kp)
        self._w_list = np.hstack([np.ones(3)*Kp, np.ones(3)*Kd])#np.tile( np.hstack([np.ones(3)*Kp, np.ones(3)*Kd]) , (100, 1) )
        print self._w_list
        # self._w_list = data[-1]['params'][t]

    def compute_w(self, context, explore):
        w = self._w_list
        self._count += 1
        return w

def main():
    from aml_io.io_tools import save_data
    from aml_playground.var_imp_reps.policy.spring_init_policy import create_init_policy
    from aml_playground.var_imp_reps.exp_params.experiment_point_mass_params import exp_params

    env = PointMassEnv()

    spring_ks = [1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]

    kp_traj = np.zeros_like(env._des_force_traj)
    kp_traj[:,2] = env._des_force_traj[:,2]

    kd_traj = np.ones_like(env._des_force_traj)*2
    kd_traj[:,2] = np.sqrt(env._des_force_traj[:,2])

    ctrl_traj = np.hstack([kp_traj,kd_traj])

    # policy = create_init_policy(env._traj2pull, env._des_force_traj, ctrl_traj, exp_params)

    policy = [DummyPolicy(t) for t in range(100)]
    
    for k in spring_ks:
        env.configure_spring(K=k)
        
        # for j in range(100):
        #     env._point_mass.apply_action(u=np.array([0.,0.,20]))
        #     env.simple_step()
        #     print "Pos \t", env._point_mass.get_ee_pose()[0]
        data = env.execute_policy(policy=policy, show_demo=False, sinusoid=False)
        # save_data(data, file_name%(k))
        raw_input("Press enter to exit")
        env._reset()
        # 


if __name__ == '__main__':
    main()

