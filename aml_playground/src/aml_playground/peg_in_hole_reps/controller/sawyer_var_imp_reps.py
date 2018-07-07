import os
import copy
import numpy as np
from rl_algos.agents.gpreps_new import GPREPSOpt
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data

#for gpreps
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from rl_algos.forward_models.context_model import ContextModel
from rl_algos.forward_models.traj_rollout_model import TrajRolloutModel

np.random.seed(123)

class SawyerVarImpREPS():

    def __init__(self, exp_params):

        eval_config = copy.deepcopy(exp_params['env_params'])
        sim_config = copy.deepcopy(exp_params['env_params'])

        sim_config['renders']  = False
        eval_config['renders'] = False

        self._eval_env = SawyerEnv(config=eval_config)

        self._sim_env = SawyerEnv(config=sim_config)

        self._exp_params = exp_params

        self.setup_gpreps(exp_params=self._exp_params['gpreps_params'])


    def setup_gpreps(self, exp_params):

        #w_bounds[:,0] = [x_lower, y_lower, z0_lower, z1_lower, theta_lower]
        #w_bounds[:,1] = [x_upper, y_upper, z0_upper, z1_upper, theta_upper]

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
                                      cost=self._sim_env.reward, 
                                      context_model=context_model, 
                                      num_data_points=exp_params['num_samples_fwd_data'])

        self._gpreps = GPREPSOpt(entropy_bound=exp_params['entropy_bound'], 
                                  num_policy_updates=exp_params['num_policy_updates'], 
                                  num_samples_per_update=exp_params['num_samples_per_update'], 
                                  num_old_datasets=exp_params['num_old_datasets'],  
                                  env=self._sim_env,
                                  context_model=context_model, 
                                  traj_rollout_model=traj_model,
                                  policy=policy,
                                  min_eta=exp_params['min_eta'], 
                                  num_data_to_collect=exp_params['num_data_to_collect'], 
                                  num_fake_data=exp_params['num_fake_data'])
