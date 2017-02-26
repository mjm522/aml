from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller  import OSPositionController
from aml_ctrl.controllers.js_controllers.js_postn_controller  import JSPositionController
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

from aml_ctrl.traj_player.traj_player import TrajPlayer

import rospy
import numpy as np

def main(robot_interface, load_from_demo=False, demo_idx=None):

    kwargs = {}

    if load_from_demo:

        if demo_idx is None:

            print "Enter a valid demo index"
            raise ValueError

        else:

            kwargs['limb_name'] = robot_interface._limb
            kwargs['demo_idx']  = demo_idx

    else:

        kwargs['start_pos'], kwargs['start_ori'] = robot_interface.get_ee_pose()

        kwargs['goal_pos'] = kwargs['start_pos'] + np.array([0.,0.,0.5])

        kwargs['goal_ori'] = kwargs['start_ori'] 

    # gen_traj = OSTrajGenerator(load_from_demo=load_from_demo, **kwargs)
    gen_traj    = JSTrajGenerator(load_from_demo=load_from_demo, **kwargs)

    traj_player = TrajPlayer(robot_interface=robot_interface, controller=JSPositionController, trajectory=gen_traj.generate_traj())

    traj_player.player()

if __name__ == '__main__':

    rospy.init_node('trajectory_player')
    
    from aml_robot.baxter_robot import BaxterArm
    
    limb = 'left'
    
    arm = BaxterArm(limb)

    arm.untuck_arm()

    demo_idx = 0

    main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx)


