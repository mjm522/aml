import rospy
from aml_robot.sawyer_robot import SawyerArm
from sawyer_keyboard_config import OS_SAWYER_CONFIG
from aml_teleop.keyboard_teleop.os_teleop_ctrl import OSTeleopCtrl


if __name__ == '__main__':
    

    rospy.init_node("test")

    sawyer = SawyerArm('right')
    sawyer.untuck()

    sawyer.set_arm_speed(max(OS_SAWYER_CONFIG['robot_max_speed'],OS_SAWYER_CONFIG['robot_min_speed'])) # WARNING: max 0.2 rad/s for safety reasons
    sawyer.set_sampling_rate(sampling_rate=200)

    teleop = OSTeleopCtrl(sawyer,OS_SAWYER_CONFIG)
    teleop.run()