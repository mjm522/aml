from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller  import OSPositionController
from aml_ctrl.controllers.js_controllers.js_postn_controller  import JSPositionController
from aml_ctrl.traj_generator.os_traj_generator import OSTrajGenerator
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

from aml_ctrl.traj_player.traj_player import TrajPlayer

import os
import rospy
import numpy as np

def main(robot_interface, load_from_demo=False, demo_idx=None, path_to_demo=None):
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

    traj_player = TrajPlayer(robot_interface=robot_interface, controller=JSPositionController, trajectory=gen_traj.generate_traj(), rate=100)

    traj_player.player()

if __name__ == '__main__':

    rospy.init_node('trajectory_player')
    
    from aml_robot.baxter_robot import BaxterArm
    
    limb = 'right'
    
    arm = BaxterArm(limb)

    demo_idx = None
    path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/' + limb + '_grasp_exp/' + limb + '_grasp_exp_03.pkl'

    main(robot_interface=arm, load_from_demo=True, demo_idx=demo_idx, path_to_demo=path_to_demo)


