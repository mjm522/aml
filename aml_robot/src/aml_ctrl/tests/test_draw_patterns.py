import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.utilities import quatdiff, standard_shape_traj
from aml_ctrl.controllers.osc_torque_controller import OSC_TorqueController

def test_maintain_position(robot_interface, start_pos, start_ori):

    ctrlr = OSC_TorqueController(robot_interface)

    while True:
        
        ctrlr.compute_cmd(goal_pos=start_pos,
                          goal_ori=start_ori,  
                          orientation_ctrl=True)
      
        ctrlr.send_cmd()

def test_draw_pattern(robot_interface, no_set_points = 32, shape='circle'):
    
    ctrlr = OSC_TorqueController(robot_interface)

    robot_interface.untuck_arm()

    start_pos, start_ori  =  robot_interface.get_ee_pose()

    traj_to_follow = standard_shape_traj(curr_pos=start_pos, 
                                        no_set_points=no_set_points,
                                        shape=shape)
    idx = 0

    while True:

        curr_pos, curr_ori  =  arm.get_ee_pose()
        
        idx = idx%no_set_points
        
        error = np.linalg.norm(traj_to_follow[idx] - curr_pos)
        
        ctrlr.compute_cmd(goal_pos=traj_to_follow[idx],
                          goal_ori=start_ori, 
                          orientation_ctrl=True)
        #increment the set point only if the arm is within a certain threshold
        if error < 0.12:
            idx += 1
      
        ctrlr.send_cmd()


if __name__ == '__main__':

    rospy.init_node('draw_patterns_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)
    start_pos, start_ori  =  arm.get_ee_pose()

    #test_maintain_position(robot_interface=arm, start_pos=start_pos, start_ori=start_ori)

    test_draw_pattern(robot_interface=arm, no_set_points = 32, shape='circle')
    