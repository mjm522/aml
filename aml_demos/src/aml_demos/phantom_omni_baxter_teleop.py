#!/usr/bin/env python

import rospy
import argparse
from aml_robot.baxter_robot import BaxterArm
from omni_interface.phantom_omni import PhantomOmni
from aml_teleop.haptic_teleop.js_teleop_ctrl import JSTeleopCtrl
from aml_teleop.haptic_teleop.os_teleop_ctrl import OSTeleopCtrl

'''

make sure roslaunch omni_common omni_state.launch is running as root

'''

def control_baxter(limb_name, task_space):

    arm = BaxterArm(limb_name)

    if task_space:
    
        config_os_teleop = {

        'rate':200, # rate of the controller
        'ctrlr_type':'torq', #other options are 'vel', 'torq'
  
         }

        djpc = OSTeleopCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config_os_teleop)

    else:

        config_js_teleop = {

        'robot_joints':[0,1,2,4,5,6], #these joints will be one to one mapped
        'haptic_joints':[0,1,2,3,4,5],
        'scale_from_home': True,
        'robot_home':arm._untuck, # home position of baxter
        'rate':200, # rate of the controller
        'ctrlr_type':'pos', #other options are 'vel', 'torq'
         
         }

        djpc = JSTeleopCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config_js_teleop)

    djpc.run()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='baxter_teleop_ctrl')

    parser.add_argument('-l', '--limb_name', default='left', type=str, help='limb index-(left/right)')

    parser.add_argument('-t', '--task_space', default=False, type=bool, help='arm_interface (sawyer/baxter)')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('baxter_teleop_ctrl', anonymous=True)

    control_baxter(args.limb_name, args.task_space)


