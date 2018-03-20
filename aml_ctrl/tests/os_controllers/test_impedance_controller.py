import rospy
import numpy as np
import quaternion
from aml_ctrl.controllers.os_controllers.os_impedance_controller import OSImpedanceController

def impedance_controller(robot_interface, start_pos, start_ori, goal_pos, goal_ori):
    #0 is left and 1 is right

    ctrlr = OSImpedanceController(robot_interface)

    print "Starting position controller"

    rate = rospy.Rate(100)

    finished = False
    t = 0
    ctrlr.set_active(True)

    goal_vel = np.zeros(3)
    goal_omg = np.zeros(3)

    while not rospy.is_shutdown() and not finished:

        print "Sending goal ",t, " goal_pos:",goal_pos.ravel() 

        if np.any(np.isnan(goal_pos)) or np.any(np.isnan(goal_vel)) or np.any(np.isnan(goal_vel)) or np.any(np.isnan(goal_omg)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            # Setting new goal
            ctrlr.set_goal(goal_pos=goal_pos, 
                           goal_ori=goal_ori, 
                           goal_vel=goal_vel, 
                           goal_omg=goal_omg, 
                           orientation_ctrl = False)
            
            print "Waiting..."
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=5.0)
            
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)
        # finished = (lin_error < 0.01)

        rate.sleep()
    
    ctrlr.wait_until_goal_reached(timeout=5.0)
    # ctrlr.set_active(False)

    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['linear'])
    print "ERROR in orientation \t", np.linalg.norm(ctrlr._error['angular'])

if __name__ == '__main__':

    rospy.init_node('classical_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)

    # arm.untuck_arm()

    start_pos, start_ori  =  arm.ee_pose()

    print("Starting position:", start_pos)
    
    if limb == 'left':
        # goal_pos = start_pos + np.array([0.,0.35, 0.])
        #for baxter, this is same set point as the moveit test
        goal_pos = start_pos#np.array([0.81576, 0.093893, 0.2496])
        goal_ori = start_ori#np.quaternion(0.67253, 0.69283, 0.1977, -0.16912)
    else:
        # goal_pos = start_pos - np.array([0.,0.35, 0.])
        #for baxter, this is same set point as the moveit test
        goal_pos = np.array([ 0.72651, -0.041037, 0.19097])
        goal_ori = np.quaternion(-0.33955, 0.56508, -0.5198, -0.54332)

    # angle    = 90.0
    # axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    # goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])
    
    impedance_controller(arm, start_pos, start_ori, goal_pos, goal_ori)