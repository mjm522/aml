import numpy as np
import quaternion
import rospy
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_lfd.utilities.utilities import get_os_traj, plot_demo_data, get_sampling_rate

def test_position_controller(robot_interface, pos_traj, ori_traj=None):
    #0 is left and 1 is right

    #ctrlr = OSPositionController(robot_interface)
    ctrlr = OSTorqueController(robot_interface)

    print "Starting position controller"

    rate = rospy.Rate(get_sampling_rate())

    reach_thr = 0.12

    finished = False
    t = 0
    ctrlr.set_active(True)

    n_steps = len(pos_traj)

    if ori_traj is None:
    	_, goal_ori = robot_interface.get_ee_pose()

    while not rospy.is_shutdown() and not finished:

        error_lin = np.linalg.norm(ctrlr._error['linear'])

        goal_pos = pos_traj[t,:]
        
        if ori_traj is not None:
        	goal_ori = ori_traj[t]

        print "Sending goal ",t, " goal_pos:",goal_pos.ravel(), "goal_ori:", goal_ori

        if np.any(np.isnan(goal_pos)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            ctrlr.set_goal(goal_pos, goal_ori)

            print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
            print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success


        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()

    lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10)
    ctrlr.set_active(False)

    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['linear'])
    print "ERROR in orientation \t", np.linalg.norm(ctrlr._error['angular'])

if __name__ == '__main__':

    rospy.init_node('classical_postn_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)

    arm.untuck_arm()

    demo_idx = 6

    #plot_demo_data(demo_idx=demo_idx)

    pos_traj, ori_traj,_,_,_,_  = get_os_traj(demo_idx=demo_idx)
    
    test_position_controller(robot_interface=arm, pos_traj=pos_traj, ori_traj=ori_traj)