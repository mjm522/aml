import time
import numpy as np
from aml_rl_envs.dlr_hand.dlr_hand_obj_env import DLRHandObjEnv


def main():

    env = DLRHandObjEnv(action_dim=20, keep_obj_fixed=False)

    fin_indices = env._hand._finger_jnt_indices

    while True:

        for k in range(5):

            cmd = np.zeros(len(fin_indices[k]))

            for j in range(200):

                cmd += 0.001

                env._hand.apply_action(k, cmd)

                env.simple_step()

            time.sleep(0.01)


if __name__ == '__main__':
    main()