#! /usr/bin/env python
import os
import time
import copy
import numpy as np
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from aml_io.io_tools import save_data

def main():

    env = SawyerEnv()

    policy = LinGaussPolicy(w_dim=6, 
                            context_feature_dim=10, #(featurizing context 3 dim => 3^2+1) 
                            variance=0.03, 
                            initial_params=0. * np.ones(6), 
                            random_state=np.random.RandomState(0),
                            bounds=np.array([[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
                                             [ 0.5, 0.5,  0.5, 0.5, 0.5,  0.5]]))

    while True:
    
        traj_draw, reward = env.execute_policy(policy=None, show_demo=False)
    
        raw_input("\n\n\nPress enter to continue")

        save_dir = os.environ['AML_DATA'] + '/aml_playground/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_data(filename=save_dir+'/sawyer_bullet.pkl', data=traj_draw['other_ee_data'])

        env._reset()
      
if __name__=="__main__":

    main()
