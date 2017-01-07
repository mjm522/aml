import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController

def test_position_controller(robot_interface, start_pos, start_ori, goal_pos, goal_ori):
    #0 is left and 1 is right

    ctrlr = OSPositionController(robot_interface)

    min_jerk_interp = MinJerkInterp()

    min_jerk_interp.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_ori, goal_qt=goal_ori)

    min_jerk_traj = min_jerk_interp.get_interpolated_trajectory()

    # print "Starting position controller"

    rate = rospy.Rate(100)

    reach_thr = 0.12

    finished = False
    t = 0
    ctrlr.set_active(True)

    n_steps = len(min_jerk_interp.timesteps)

    while not rospy.is_shutdown() and not finished:

        error_lin = np.linalg.norm(ctrlr._error['linear'])

        goal_pos = min_jerk_traj['pos_traj'][t]
        goal_ori = min_jerk_traj['ori_traj'][t]
        goal_vel = min_jerk_traj['vel_traj'][t]
        goal_omg = min_jerk_traj['omg_traj'][t]

        # print "Sending goal ",t, " goal_pos:", goal_pos.ravel(), " goal_vel:", goal_vel.ravel()

        if np.any(np.isnan(goal_pos)) or np.any(np.isnan(goal_vel)) or np.any(np.isnan(goal_vel)) or np.any(np.isnan(goal_omg)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            ctrlr.set_goal(goal_pos=goal_pos, 
                           goal_ori=goal_ori, 
                           goal_vel=goal_vel, 
                           goal_omg=goal_omg, 
                           orientation_ctrl = False)
            # print "\n command calculated \n", ctrlr._cmd
            # print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success
            # print "lin_error: %0.4f ang_error: %0.4f"%(lin_error,ang_error)
            print "lin_error:", lin_error
            
        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()

    lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10)
    ctrlr.set_active(False)

    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    # print "ERROR in position \t", np.linalg.norm(ctrlr._error['linear'])
    # print "ERROR in orientation \t", np.linalg.norm(ctrlr._error['angular'])

if __name__ == '__main__':

    rospy.init_node('classical_postn_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)

    arm.untuck_arm()

    start_pos, start_ori  =  arm.get_ee_pose()
    
    if limb == 'left':
        # goal_pos = start_pos + np.array([0.,0.35, 0.])
        #for baxter, this is same set point as the moveit test
        goal_pos = np.array([0.81576, 0.093893, 0.2496])
        goal_ori = np.quaternion(0.67253, 0.69283, 0.1977, -0.16912)
    else:
        # goal_pos = start_pos - np.array([0.,0.35, 0.])
        #for baxter, this is same set point as the moveit test
        goal_pos = np.array([ 0.72651, -0.041037, 0.19097])
        goal_ori = np.quaternion(-0.33955, 0.56508, -0.5198, -0.54332)

    # angle    = 90.0
    # axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    # goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])
    
    test_position_controller(arm, start_pos, start_ori, goal_pos, goal_ori)