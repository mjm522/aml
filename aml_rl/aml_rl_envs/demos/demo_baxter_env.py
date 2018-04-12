import os
import time
import numpy as np
import pybullet as pb
from aml_rl_envs.baxter.baxter import Baxter
from aml_rl_envs.baxter.baxter_env import BaxterEnv
from aml_rl_envs.utils.data_utils import load_csv_data
from aml_rl_envs.baxter.config import BAXTER_ENV_CONFIG, BAXTER_CONFIG

def main():

    env = BaxterEnv()

    data = load_csv_data(os.environ['AML_DATA'] + '/aml_playground/manipulation/baxter_turn_box.csv')

    # env = Baxter(config=BAXTER_CONFIG, call_renderer=True)
    
    done = False

    obj_data = pb.getVisualShapeData(env._object._obj_id)[0]

    while (not done):

        for k in range(len(data)):

            action = np.r_[np.zeros(9), data[k,:]]

            env._baxter.apply_action(action)

            env.simple_step()

if __name__=="__main__":

    main()
