#!/usr/bin/env python

import rospy
import argparse
from omni_interface.phantom_omni import PhantomOmni
from aml_teleop.haptic_teleop.js_teleop_ctrl import JSTeleopCtrl
from aml_teleop.haptic_teleop.os_teleop_ctrl import OSTeleopCtrl

'''

make sure roslaunch omni_common omni_state.launch is running as root

'''

def control_robot(arm, task_space):

    if task_space:
    
        config_os_teleop = {

        'rate': 200, # rate of the controller
        'ctrlr_type':'torq', #other options are 'vel', 'torq'
  
         }

        djpc = OSTeleopCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config_os_teleop)

    else:

        config_js_teleop = {

        'robot_joints':[0,1,2,4,5,6], #these joints will be one to one mapped
        'haptic_joints':[0,1,2,3,4,5],
        'scale_from_home': True,
        'robot_home': arm._untuck[0:arm.n_cmd()], # home position of baxter
        'rate': 200, # rate of the controller
        'ctrlr_type':'pos', #other options are 'vel', 'torq'
         
         }

        djpc = JSTeleopCtrl(haptic_interface=PhantomOmni(), robot_interface=arm, config=config_js_teleop)

    djpc.run()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='baxter_teleop_ctrl')

    parser.add_argument('-l', '--limb_name', type=str, default='right',  help='limb index-(left/right)')

    parser.add_argument('-t', '--task_space', type=bool, default=False,  help='task space')

    parser.add_argument('-i', '--arm_interface', type=str, default='baxter', help='arm interface, e.g. baxter/sawyer')

    parser.add_argument('-s', '--arm_speed', type=float, default=1.0, help='Arm speed for position control')

    parser.add_argument('-g', '--gripper_speed', type=float, default=1.0, help='Gripper speed for position control')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('teleop_ctrl', anonymous=True)

    max_speed = 0.20
    
    min_speed = 0.01
    
    if args.arm_interface == "baxter":
        
        from aml_robot.baxter_robot import BaxterArm as ArmInterface
        
        max_speed = 10.0

        sampling_rate = 100
    
    elif args.arm_interface == "sawyer":
        
        from aml_robot.sawyer_robot import SawyerArm as ArmInterface

        sampling_rate = 200

    else:

        raise ValueError("Unknown arm type, press -h for help")
    
    arm = ArmInterface(args.limb_name)

    arm.set_arm_speed(max(min(args.arm_speed, max_speed),min_speed)) # WARNING: max 0.2 rad/s for safety reasons
    
    arm.set_sampling_rate(sampling_rate=sampling_rate) # Arm should report its state as fast as possible.
    
    arm.set_gripper_speed(max(min(args.gripper_speed,0.20),0.01))

    arm.untuck_arm()

    control_robot(arm, args.task_space)


