import os
import copy
import numpy as np
from rl_algos.agents.gpreps_new import GPREPSOpt
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data

#for gpreps
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy

from aml_playground.var_imp_reps.fwd_models.reward_model import RewardModel
from aml_playground.var_imp_reps.fwd_models.context_model import ContextModel
from aml_playground.var_imp_reps.fwd_models.next_force_model import NextForcePredictModel

np.random.seed(123)

class SawyerVarImpREPS():

    def __init__(self, exp_params):

        eval_config = copy.deepcopy(exp_params['env_params'])

        eval_config['renders'] = False

        self._eval_env = SawyerEnv(config=eval_config)

        self._exp_params = exp_params

        self.setup_gpreps(exp_params=self._exp_params['gpreps_params'])


    def setup_gpreps(self, exp_params, transform=False):

        policy = LinGaussPolicy(w_dim=exp_params['w_dim'], 
                                context_feature_dim=exp_params['context_feature_dim'], 
                                variance=exp_params['policy_variance'], 
                                initial_params=exp_params['initial_params'], 
                                random_state=exp_params['random_state'],
                                bounds=exp_params['w_bounds'],
                                transform=transform)

        context_model = ContextModel(spring_base=self._eval_env._spring_base,
                                     req_traj=self._eval_env._traj2pull,
                                     spring_k=self._exp_params['env_params']['spring_stiffness'],
                                     min_vel=np.zeros(3),
                                     max_vel=np.ones(3),
                                     context_dim=exp_params['context_feature_dim'])

        reward_model = RewardModel(cost_fn=self._eval_env.reward,
                                   target=self._eval_env._traj2pull[-1,:], 
                                   params=self._exp_params['env_params'],
                                   w_dim=exp_params['w_dim'])

        force_model = NextForcePredictModel(spring_k=self._exp_params['env_params']['spring_stiffness'])

        self._gpreps = GPREPSOpt(entropy_bound=exp_params['entropy_bound'], 
                                  num_policy_updates=exp_params['num_policy_updates'], 
                                  num_samples_per_update=exp_params['num_samples_per_update'], 
                                  num_old_datasets=exp_params['num_old_datasets'],  
                                  env=self._eval_env,
                                  context_model=context_model, 
                                  reward_model=reward_model,
                                  force_model=force_model,
                                  policy=policy,
                                  min_eta=exp_params['min_eta'], 
                                  num_data_to_collect=exp_params['num_data_to_collect'], 
                                  num_fake_data=exp_params['num_fake_data'],
                                  transform_context=transform)
