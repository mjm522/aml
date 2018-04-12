import rospy
from aml_robot.sawyer_robot import SawyerArm
from aml_rl_envs.sawyer.sawyer import Sawyer
from aml_rl_envs.sawyer.config import SAWYER_CONFIG
from aml_lfd.utilities.robot_to_bullet_interface import RobotBulletInterface


def main():

    bullet_config = SAWYER_CONFIG
    bullet_config['call_renderer'] = True

    rospy.init_node("js_mirror_demo_node")

    real_robot = SawyerArm()
    bullet_robot = Sawyer(config=bullet_config, cid=None)

    js_mirror = RobotBulletInterface(robot_interface=real_robot, bullet_interface=bullet_robot)
    js_mirror.run()


if __name__ == "__main__":
    main()
