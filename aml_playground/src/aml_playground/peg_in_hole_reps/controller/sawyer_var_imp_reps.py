import os
import copy
import rospy
import numpy as np

from aml_robot.sawyer_robot import SawyerArm

from rl_algos.agents.gpreps import GPREPSOpt
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.utils.collect_demo import plot_demo

from aml_rl_envs.sawyer.sawyer_imp_env import SawyerEnv

from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

#for controllers
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController

#for gpreps
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from rl_algos.forward_models.context_model import ContextModel
from rl_algos.forward_models.traj_rollout_model import TrajRolloutModel

np.random.seed(123)

class SawyerVarImpREPS():

    def __init__(self, joint_space, exp_params):

        self._sawyer = SawyerArm('right')

        self._sawyer.move_to_joint_position([0.58546387,  0.4992666, 
                                            -1.54154004,  2.02193262,  
                                             2.00224219,  1.73852734, 4.50088965])

        self._exp_params = exp_params

        self._joint_space = joint_space

        kwargs = {}
        kwargs['limb_name'] = 'right' 
        path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_spring_exp_1/right_sawyer_spring_exp_1_01.pkl'

        if not os.path.exists(path_to_demo):
            raise Exception("Enter a valid demo path")
        else:
            kwargs['path_to_demo'] = path_to_demo

        if joint_space:
            self._gen_traj = JSTrajGenerator(load_from_demo=True, **kwargs)
            self._ctrlr = JSPositionController(robot_interface=self._sawyer)
        else:
            self._gen_traj = OSTrajGenerator(load_from_demo=True, **kwargs)
            self._ctrlr = JSPositionController(robot_interface=self._sawyer)
            # self._ctrlr = OSPositionController(robot_interface=self._sawyer)
        
        self._demo_traj = self._gen_traj.generate_traj()

        self._time_steps, self._dof = self._demo_traj['pos_traj'].shape

        self._total_timeout = 5.

        self._rate = rospy.Rate(100)

        self.setup_gpreps(exp_params=self._exp_params['gpreps_params'])

        self._ctrlr.set_active(True)

    def setup_gpreps(self, exp_params):

        policy = LinGaussPolicy(w_dim=exp_params['w_dim'], 
                                context_feature_dim=exp_params['context_feature_dim'], 
                                variance=exp_params['policy_variance'], 
                                initial_params=exp_params['initial_params'], 
                                random_state=exp_params['random_state'],
                                bounds=exp_params['w_bounds'])

        context_model = ContextModel(context_dim=exp_params['context_dim'], 
                                    num_data_points=exp_params['num_samples_fwd_data'])

        traj_model = TrajRolloutModel(w_dim=exp_params['w_dim'], 
                                      x_dim=exp_params['x_dim'], 
                                      cost=self.reward, 
                                      context_model=context_model, 
                                      num_data_points=exp_params['num_samples_fwd_data'])

        self._gpreps = GPREPSOpt(entropy_bound=exp_params['entropy_bound'], 
                                  num_policy_updates=exp_params['num_policy_updates'], 
                                  num_samples_per_update=exp_params['num_samples_per_update'], 
                                  num_old_datasets=exp_params['num_old_datasets'],  
                                  env=self,
                                  context_model=context_model, 
                                  traj_rollout_model=traj_model,
                                  policy=policy,
                                  min_eta=exp_params['min_eta'], 
                                  num_data_to_collect=exp_params['num_data_to_collect'], 
                                  num_fake_data=exp_params['num_fake_data'])

    def update_imp_params(self, Kp, Kd=None):

        if Kp is None:
            return
        
        self._ctrlr._kp = Kp

        if Kd is not None:
            self._ctrlr._kd = Kd


    def reward(self, sawyer_data):
        '''
            Computing reward for the given (forward-simulated) trajectory
        '''

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        data = sawyer_data['sawyer_data']
        true_traj = traj['traj']
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

        self._logger.debug("*******************************************************************")
        self._logger.debug("penalty_force \t %f"%(force_penalty,))
        self._logger.debug("*******************************************************************")

        self._penalty = {'force':force_penalty,
                 'total':penalty}

        return penalty

    def fwd_rollout(self, traj_data, ee_ori=None, Kp=None, Kd=None):
        """
        implement the dmp
        """
        traj = traj_data['pos_traj']
        sawyer_data = []

        if ee_ori is None:
            goal_ori = self._sawyer.state()['ee_ori']
        else:
            goal_ori = ee_ori

        k = 0
        finished = False
        start = rospy.get_time()

        while not finished:

            if self._joint_space:
                cmd = traj[k, :]
            else:
                cmd = self._sawyer.inverse_kinematics(pos=traj[k, :], ori=goal_ori, use_service=True)
            
            lin_jac = self._sawyer.jacobian()[:3,:]

            if Kp is not None:
                js_Kp = np.dot(lin_jac.T, Kp)
                js_Kp = np.clip(js_Kp, 0.01, 1)
                self._logger.debug("\n \n Kp \t {}".format(js_Kp))
            else: 
                js_Kp = None

            if Kd is not None:
                js_Kd = np.dot(lin_jac.T, Kd)
                js_Kd = np.clip(js_Kd, 0.5, 1)
                self._logger.debug("\n \n Kd \t {}".format(js_Kd))
            else:
                js_Kd = None

            # self.update_imp_params(js_Kp, js_Kd)

            # if self._joint_space:
            self._ctrlr.set_goal(goal_js_pos=cmd,
                                 goal_js_vel=None,
                                 goal_js_acc=None)

            pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.)

            # else:
            #     self._ctrlr.set_goal(goal_pos=traj[k, :], 
            #                goal_ori=goal_ori, 
            #                goal_vel=np.zeros(3), 
            #                goal_omg=np.zeros(3), 
            #                orientation_ctrl = True)

            #     pos_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)

            k += 1

            print "pos_error:", k, pos_error

            timed_out = self._total_timeout is not None and rospy.get_time()-start > self._total_timeout

            finished = bool(k >= self._time_steps or timed_out)

            print cmd
            print k, timed_out, finished

            self._rate.sleep()

            sawyer_data.append(self._sawyer.state())

        return { 'traj':traj,
                 'sawyer_data':sawyer_data }
        
    def context(self):
        """
        Context is the bottom base of the box.
        """

        s = np.random.uniform(-0.1, 0.1)

        lf = 0.5 + s 

        self._reset(lf=lf)     

        return np.array([s])

    def execute_policy(self, w=None):

        if w is not None:
            Kp = np.ones(3)+w[:3]
            Kd = np.ones(3)+w[3:]
        else:
            Kp = Kd = None

        traj2follow = self.fwd_rollout(traj_data=self._demo_traj, Kp=Kp, Kd=Kd)
     
        reward = 0#self.reward(traj2follow) #+ 4*np.linalg.norm(js_Kp)
        
        return traj2follow, reward


def main():
    from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import exp_params
    rospy.init_node('sawyer_var_imp')
    svi = SawyerVarImpREPS(False, exp_params)
    svi.execute_policy()
    svi._ctrlr.set_active(False)


#test code
if __name__ == '__main__':

    main()
