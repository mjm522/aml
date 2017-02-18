from aml_ctrl.traj_generator.os_traj_generator import JSTrajGenerator
from aml_ctrl.traj_player.traj_player import TrajPlayer

import rospy
import numpy as np

def main(robot_interface):

    kwargs = {}

    kwargs['start_pos'], kwargs['start_ori'] = robot_interface.get_ee_pose()

    kwargs['goal_pos'] = kwargs['start_pos'] + np.array([0.,0.,0.5])

    kwargs['goal_ori'] = kwargs['start_ori'] 

    js_traj = JSTrajGenerator(load_from_demo=False, **kwargs)

    traj_player = TrajPlayer(robot_interface=robot_interface, controller=OSPositionController, trajectory=js_traj.generate_traj())

    traj_player.player()

if __name__ == '__main__':

    rospy.init_node('trajectory_player')
    
    from aml_robot.baxter_robot import BaxterArm
    
    limb = 'left'
    
    arm = BaxterArm(limb)

    arm.untuck_arm()

    demo_idx = 6

    main(robot_interface=arm)