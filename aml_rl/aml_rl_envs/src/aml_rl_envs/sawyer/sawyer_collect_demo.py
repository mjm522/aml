import os
from aml_rl_envs.sawyer.sawyer_peg_env import SawyerEnv
from aml_rl_envs.utils.keyboard_control import KeyBoardControl
from aml_rl_envs.utils.manual_control_collect_demo import ManualDemoCollect
from aml_rl_envs.utils.collect_demo import CollectDemo, get_demo

collect_demo = True


def demo_collect(env):

    cd = ManualDemoCollect(manipulator=env._sawyer, demo_path=os.environ["AML_DATA"]+'/data/sawyerDemo')
    cd.collect_demo()
    cd.check_demo()

def keyboard_ctrl(env):

    kc = KeyBoardControl(manipulator=env._sawyer)
    kc.run()


def main():

    env = SawyerEnv()
    keyboard_ctrl(env)




if __name__ == '__main__':
    main()