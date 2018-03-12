import time
import numpy as np
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_rl_envs.hand.hand_obst_env import HandObstacleEnv


def main():

    # env = HandObjEnv(action_dim=6, randomize_box_ori=False, keep_obj_fixed=False)
    env = HandObstacleEnv(action_dim=6, randomize_box_ori=False, keep_obj_fixed=False)

    cmd = np.array([0.,0., -2.35619])

    while True:

        env._hand.set_joint_state(finger_idx=0, jnt_pos=cmd)

        env.simple_step()

        # cmd += np.array([0.0, 0.00, 0.001])

        # print env.get_contact_points()
        # env.get_point_obj_frame()

        # raw_input(cmd)


if __name__ == '__main__':
    main()