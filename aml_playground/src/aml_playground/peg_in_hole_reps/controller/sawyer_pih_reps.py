import os
import numpy as np
from rl_algos.agents.gpreps import GPREPSOpt
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.sawyer.sawyer_peg_env import SawyerEnv
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

#for gpreps
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from rl_algos.forward_models.context_model import ContextModel
from rl_algos.forward_models.traj_rollout_model import TrajRolloutModel

np.random.seed(123)

class SawyerPegREPS():

    def __init__(self, joint_space, exp_params):

        self._env = SawyerEnv()

        self._exp_params = exp_params

        kwargs = {}
        kwargs['limb_name'] = 'right' 

        path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/right_sawyer_exp_peg_in_hole_01.pkl'

        if not os.path.exists(path_to_demo):
            raise Exception("Enter a valid demo path")
        else:
            kwargs['path_to_demo'] = path_to_demo

        if joint_space:
            self._gen_traj = JSTrajGenerator(load_from_demo=True, **kwargs)
            self._demo_traj = self._gen_traj.generate_traj()['pos_traj']
        else:
            #in this file, the orientation is in [w,x,y,z] format
            #pos, ori, vel, omg is the sequence in which data is stored
            path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/sawyer_bullet_ee_states.csv'
            self._demo_traj = load_csv_data(path_to_demo)[:, :3]
            # self._gen_traj = OSTrajGenerator(load_from_demo=True, **kwargs)

        self._dof = self._demo_traj.shape[1]

        self.setup_gpreps(exp_params=self._exp_params['gpreps_params'])

        self.encode_dmps()

    def setup_gpreps(self, exp_params):
        
        policy = LinGaussPolicy(w_dim=exp_params['w_dim'], 
                                context_feature_dim=exp_params['context_feature_dim'], 
                                variance=exp_params['policy_variance'], 
                                initial_params=exp_params['initial_params'], 
                                random_state=exp_params['random_state'])

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

    def encode_dmps(self):

        discrete_dmp_config['dof'] = self._dof

        self._man_dmp = {}
        self._man_dmp['config'] = discrete_dmp_config
        self._man_dmp['obj'] = DiscreteDMP(config=discrete_dmp_config)

        self._man_dmp['obj'].load_demo_trajectory(self._demo_traj)
        self._man_dmp['obj'].train()

    def update_dmp_params(self, phase_start=1., speed=1., goal_offset=None, start_offset=None, external_force=None):

        if goal_offset is None: goal_offset = np.zeros(self._dof)

        if start_offset is None: start_offset = np.zeros(self._dof)

        dmp    = self._man_dmp['obj']
        config = self._man_dmp['config']

        config['y0'] = dmp._traj_data[0, 1:] + start_offset
        config['dy'] = np.zeros(self._dof)
        config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
        config['tau'] = 1./speed
        config['phase_start'] = phase_start

        if external_force is None:
            external_force = np.zeros(self._dof)
            config['type'] = 1
        else:
            config['type'] = 3

        config['extForce'] = external_force

        new_dmp_traj = dmp.generate_trajectory(config=config)

        return new_dmp_traj['pos']

    def fwd_simulate(self, dmp, joint_space = False):
        """
        implement the dmp
        """
        # return np.random.randn(220,3)
        ee_traj = []


        for k in range(dmp.shape[0]):

            if joint_space:

                cmd = dmp[k, :]

            else:

                cmd = self._env._sawyer.inv_kin(ee_pos=dmp[k, :].tolist())

            self._env._sawyer.apply_action(cmd)

            ee_pos, ee_ori = self._env._sawyer.get_ee_pose()
            ee_traj.append(ee_pos)
            
            # import time
            # time.sleep(0.01)
            self._env.simple_step()
            


        return np.asarray(ee_traj)
        
    def context(self):
        """
        Context is the top face of the box.

            Top face of box:
                # x : (0.7, 0.7 + 0.15*0.55)
                # y : (0.1, 0.1 - 0.15*1.90)
                # z : (0.62, 0.62 + 0.15*2.40)
        """

        x = np.random.uniform(0.7, 0.7 + 0.15*0.55)
        y = np.random.uniform(0.1, 0.1 - 0.15*1.90)

        context = np.array([x,y])

        return context


    def reward(self, traj):

        return self._env._simulation_reward(traj)

    def execute_policy(self, w, s):

        dmp = self.update_dmp_params(goal_offset=np.r_[w, 0])

        traj = self.fwd_simulate(dmp)

        reward = self.reward(traj)
        
        return None, reward
