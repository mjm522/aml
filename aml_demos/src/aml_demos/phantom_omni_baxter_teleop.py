#!/usr/bin/env python

import rospy
from aml_robot.baxter_robot import BaxterArm
from omni_interface.phantom_omni import PhantomOmni
from aml_teleop.haptic_teleop.direct_joint_pos_ctrl import DirectJointPosCtrl

'''

make sure roslaunch omni_common omni_state.launch is running as root

'''

def control_baxter(limb = 'left'):

    config = {

    'robot_joints':[0,1,2,4,5,6], #these joints will be one to one mapped
    'haptic_joints':[0,1,2,3,4,5],
     
     }

    arm = BaxterArm(limb)

    djpc = DirectJointPosCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config)

    djpc.run()



if __name__ == '__main__':

    rospy.init_node('direct_robot_pos_ctrl', anonymous=True)

    control_baxter()


