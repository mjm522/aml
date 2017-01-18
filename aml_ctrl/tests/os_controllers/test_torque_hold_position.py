import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.utilities import quatdiff, standard_shape_traj
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController 
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController 

def test_maintain_position(robot_interface):

    robot_interface.untuck_arm()

    ctrlr = OSTorqueController(robot_interface)

    # Activating a controller without setting a goal will just hold its current position and orientation
    ctrlr.set_active(True)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached()
        # print("Current error:", ctrlr._error)
        print("lin_error: %0.4f ang_error: %0.4f elapsed_time: %d.%d"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " success: ", success)

        rate.sleep()

if __name__ == '__main__':

    rospy.init_node('test_hold_position_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)

    test_maintain_position(robot_interface=arm)