import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.utilities import quatdiff, standard_shape_traj
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController

def test_draw_pattern(robot_interface, no_set_points = 32, shape='eight'):
    
    robot_interface.untuck_arm()

    ctrlr = OSTorqueController(robot_interface)

    start_pos, start_ori  =  robot_interface.get_ee_pose()

    # Generate trajectory to follow
    traj_to_follow = standard_shape_traj(curr_pos=start_pos, 
                                        no_set_points=no_set_points,
                                        shape=shape)
    idx = 0

    # Set first goal and active controller
    ctrlr.set_active(True)


    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
            
            # Set new goal for controller
        ctrlr.set_goal(traj_to_follow[idx],start_ori)

        lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10.0)

        print("lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " success: ", success)

        idx = (idx+1)%no_set_points

        rate.sleep()

    ctrlr.set_active(False)


if __name__ == '__main__':

    rospy.init_node('draw_patterns_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)

    test_draw_pattern(robot_interface=arm, no_set_points = 32, shape='circle')
    