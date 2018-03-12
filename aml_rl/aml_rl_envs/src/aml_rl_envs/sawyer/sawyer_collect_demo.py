import os
from aml_rl_envs.sawyer.sawyer_env import SawyerEnv
from aml_rl_envs.utils.manual_control_collect_demo import ManualDemoCollect
from aml_rl_envs.utils.collect_demo import CollectDemo, get_demo

collect_demo = True

def main():

    env = SawyerEnv(renders=True, isDiscrete=False, maxSteps = 10000000)

    cd = ManualDemoCollect(manipulator=env._sawyer, demo_path=os.environ["ROOT_DIR"]+'/data/sawyerDemo')
    
    cd.collect_demo()
    cd.check_demo()


if __name__ == '__main__':
    main()