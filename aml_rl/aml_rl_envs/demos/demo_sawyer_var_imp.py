#! /usr/bin/env python
import os
import time
import copy
import numpy as np
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv

def main():

    env = SawyerEnv()

    traj_draw, reward = env.execute_policy(w=None, s=None, show_demo=False)
    
    raw_input("\n\n\nPress enter to exit")
      
if __name__=="__main__":

    main()