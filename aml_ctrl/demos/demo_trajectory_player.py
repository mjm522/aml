#!/usr/bin/env python

from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller  import OSPositionController
from aml_ctrl.controllers.js_controllers.js_postn_controller  import JSPositionController
from aml_ctrl.controllers.js_controllers.js_torque_controller  import JSTorqueController
from aml_ctrl.controllers.js_controllers.js_velocity_controller  import JSVelocityController
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

from aml_ctrl.traj_player.traj_player import TrajPlayer

Controller = JSVelocityController

import os
import rospy
import numpy as np
import argparse


def main(robot_interface, load_from_demo=False, demo_idx=None, path_to_demo=None, timeout = 1.0, rate = 100, reverse_traj = False, collect_data=False):
    '''
    the JSTrajGenerator expects either a demo index or a direct path to the demo
    do not pass both of them together. This could be fixed when we have multiple demos to be played out
    '''
    kwargs = {}

    if load_from_demo:

        if demo_idx is None and path_to_demo is None:
            raise Exception("Enter a valid demo index or pass a valid demo path")
        else:

            kwargs['limb_name'] = robot_interface._limb
            
            if demo_idx is not None:
                kwargs['demo_idx']  = demo_idx

            if not os.path.exists(path_to_demo):
                raise Exception("Enter a valid demo path")
            else:
                kwargs['path_to_demo'] = path_to_demo

    else:

        kwargs['start_pos'], kwargs['start_ori'] = robot_interface.ee_pose()

        kwargs['goal_pos'] = kwargs['start_pos'] + np.array([0.,0.,0.5])

        kwargs['goal_ori'] = kwargs['start_ori'] 

    # gen_traj = OSTrajGenerator(load_from_demo=load_from_demo, **kwargs)
    gen_traj    = JSTrajGenerator(load_from_demo=load_from_demo, **kwargs)

    traj = gen_traj.generate_traj()

    if reverse_traj:
        print "Reversing trajectory"
        for key in traj.keys():
            print traj[key].shape
            traj[key] = np.flipud(traj[key]) 

    traj_player = TrajPlayer(robot_interface=robot_interface, controller=JSPositionController, 
                             trajectory=traj, timeout=timeout, rate=rate,collect_data=collect_data)

    traj_player.player()



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Play trajectory')

    parser.add_argument('-m','--name', type=str, default="trajectory player")

    parser.add_argument('-r', '--rate', type=int, default=100, help='max loop execution rate')

    parser.add_argument('-l', '--list_trajectories', action='store_true', help='list trajectories at $AML_DATA/aml_lfd/')

    parser.add_argument('-n', '--idx_trajectory', type=int, default=0, help='trajectory index from the list of trajectories at $AML_DATA/aml_lfd')

    parser.add_argument('-t', '--timeout', type=float, default=0.01, help='trajectory player timeout per waypoint')

    parser.add_argument('-s', '--arm_speed', type=float, default=1.0, help='Arm speed for position control')

    parser.add_argument('-g', '--gripper_speed', type=float, default=1.0, help='Gripper speed for position control')

    parser.add_argument('-d', '--demo_folder', type=str, default='', help='demo folder, e.g. sawyer_right or right_grasp_exp')

    parser.add_argument('-i', '--arm_interface', type=str, default='baxter', help='arm interface, e.g. baxter/sawyer')

    parser.add_argument('-b', '--backward_playback', action='store_true', help='Play trajectory backwards.')

    parser.add_argument('-c', '--collect_data', action='store_true', help='Play a trajectory and record the data.')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('trajectory_player')
    
    max_speed = 0.20
    min_speed = 0.01
    if args.arm_interface == "baxter":
        from aml_robot.baxter_robot import BaxterArm as ArmInterface
        max_speed = 10.0
    elif args.arm_interface == "sawyer":
        from aml_robot.sawyer_robot import SawyerArm as ArmInterface
    elif args.arm_interface =="sawyer_bullet":
        from aml_robot.bullet.bullet_sawyer import BulletSawyerArm as ArmInterface
    
    limb = 'right'
    
    arm = ArmInterface(limb)

    arm.set_arm_speed(max(min(args.arm_speed,max_speed),min_speed)) # WARNING: max 0.2 rad/s for safety reasons
    arm.set_sampling_rate(sampling_rate=1000) # Arm should report its state as fast as possible.
    arm.set_gripper_speed(max(min(args.gripper_speed,0.20),0.01))
    # Trajectories have been recorded at 30 hz, we can play faster or slower than what was recorded by choosing a faster or slower execution rate
    demo_idx = None
    demo_folder = limb + '_grasp_exp/'

    if args.demo_folder is not None:
        demo_folder = args.demo_folder + '/'

    base_path = os.environ['AML_DATA'] + '/aml_lfd/' + demo_folder
    
    traj_list = os.listdir(base_path)
    traj_list.sort() # assuming lexicographic order

    if args.list_trajectories or args.idx_trajectory >= len(traj_list):
        print "Listing trajectories (Total number of trajectories %d)"%(len(traj_list),)
        for i, s in enumerate(traj_list):
            print i, " -> ", traj_list[i]
    elif args.idx_trajectory < len(traj_list):

        print "Playing trajectory idx (%d): %s"%(args.idx_trajectory, traj_list[args.idx_trajectory])
        path_to_demo = base_path + traj_list[args.idx_trajectory]
        main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx, path_to_demo=path_to_demo, timeout = args.timeout, rate = args.rate, reverse_traj = args.backward_playback, collect_data=args.collect_data)


