#!/usr/bin/env python

import rospy
from aml_robot.baxter_robot import BaxterArm
from omni_interface.phantom_omni import PhantomOmni
from aml_teleop.haptic_teleop.direct_joint_pos_ctrl import DirectJointPosCtrl

'''

make sure roslaunch omni_common omni_state.launch is running as root

'''

def control_baxter(limb = 'left'):

    arm = BaxterArm(limb)

    config = {

    'robot_joints':[0,1,2,4,5,6], #these joints will be one to one mapped
    'haptic_joints':[0,1,2,3,4,5],
    'scale_from_home': True,
    'robot_home':arm._untuck, # home position of baxter
    'rate':200, # rate of the controller
     
     }

    djpc = DirectJointPosCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config)

    djpc.run()



if __name__ == '__main__':

    rospy.init_node('direct_robot_pos_ctrl', anonymous=True)

    control_baxter()


