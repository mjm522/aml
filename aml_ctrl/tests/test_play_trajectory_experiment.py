from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller  import OSPositionController
from aml_ctrl.controllers.js_controllers.js_postn_controller  import JSPositionController
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

from aml_ctrl.traj_player.traj_player import TrajPlayer

import os
import rospy
import numpy as np
import argparse


def main(robot_interface, load_from_demo=False, demo_idx=None, path_to_demo=None, timeout = 1.0, rate = 100, reverse_traj = False):
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

        kwargs['start_pos'], kwargs['start_ori'] = robot_interface.get_ee_pose()

        kwargs['goal_pos'] = kwargs['start_pos'] + np.array([0.,0.,0.5])

        kwargs['goal_ori'] = kwargs['start_ori'] 

    # gen_traj = OSTrajGenerator(load_from_demo=load_from_demo, **kwargs)
    gen_traj    = JSTrajGenerator(load_from_demo=load_from_demo, **kwargs)

    traj = gen_traj.generate_traj()
    if reverse_traj is True:
        for key in traj.keys():
            traj[key] = np.flipud(traj[key]) 

    traj_player = TrajPlayer(robot_interface=robot_interface, controller=JSPositionController, trajectory=traj, rate=rate)

    traj_player.player()

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Play trajectory')

    parser.add_argument('-r', '--rate', type=int, default=100, help='max loop execution rate')

    parser.add_argument('-l', '--list_trajectories', action='store_true', help='list trajectories at $AML_DATA/aml_lfd/')

    parser.add_argument('-n', '--idx_trajectory', type=int, default=0, help='trajectory index from the list of trajectories at $AML_DATA/aml_lfd')

    parser.add_argument('-t', '--timeout', type=float, default=1.0, help='trajectory player timeout per waypoint')

    parser.add_argument('-d', '--demo_folder', type=str, default='', help='demo folder, e.g. sawyer_right or right_grasp_exp')

    parser.add_argument('-i', '--arm_interface', type=str, default='baxter', help='arm interface, e.g. baxter/sawyer')

    parser.add_argument('-b', '--backward_playback', type=bool, default=False, help='Play trajectory backwards? (True/False)')

    args = parser.parse_args()

    rospy.init_node('trajectory_player')
    
    if args.arm_interface == "baxter":
        from aml_robot.baxter_robot import BaxterArm as ArmInterface
    else:
        from aml_robot.sawyer_robot import SawyerArm as ArmInterface
    
    limb = 'right'
    
    arm = ArmInterface(limb)

    arm.set_sampling_rate(sampling_rate=700) # Arm should report its state as fast as possible.

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
        main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx, path_to_demo=path_to_demo, timeout = args.timeout, rate = args.rate, reverse_traj = args.backward_playback)


