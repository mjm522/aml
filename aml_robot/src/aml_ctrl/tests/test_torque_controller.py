import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.osc_torque_controller import OSCTorqueController

def test_torque_controller(robot_interface, start_pos, start_ori, goal_pos, goal_ori):
    #0 is left and 1 is right

    ctrlr = OSCTorqueController(robot_interface)

    min_jerk_interp = MinJerkInterp()

    

    min_jerk_interp.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_ori, goal_qt=goal_ori)

    min_jerk_traj = min_jerk_interp.get_min_jerk_trajectory()

    print "Starting position controller"

    rate = rospy.Rate(100)

    finished = False
    t = 0
    ctrlr.set_active(True)

    n_steps = len(min_jerk_interp.timesteps)

    while not rospy.is_shutdown() and not finished:

        goal_pos = min_jerk_traj['pos_traj'][t,:]

        print "Sending goal ",t, " goal_pos:",goal_pos.ravel() 

        if np.any(np.isnan(goal_pos)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            # Setting new goal
            ctrlr.set_goal(goal_pos,start_ori)
            
            print "Waiting..."
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=5.0)
            
            print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success


        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()

    
    ctrlr.wait_until_goal_reached(timeout=5.0)
    #ctrlr.set_active(False)


    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['linear'])
    print "ERROR in orientation \t", np.linalg.norm(ctrlr._error['angular'])

if __name__ == '__main__':

    rospy.init_node('classical_postn_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)

    arm.untuck_arm()

    start_pos, start_ori  =  arm.get_ee_pose()

    print("Starting position:", start_pos)
    
    if limb == 'left':
        goal_pos = start_pos + np.array([0.,0.0, 0.3])
    else:
        goal_pos = start_pos + np.array([0.,0.0, 0.3])

    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])
    
    test_torque_controller(arm, start_pos, start_ori, goal_pos, goal_ori)