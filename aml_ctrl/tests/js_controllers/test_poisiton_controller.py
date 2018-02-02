import numpy as np
import quaternion
import rospy
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController

def test_postn_controller(robot_interface):
    #0 is left and 1 is right

    ctrlr = JSPositionController(robot_interface)

    rate = rospy.Rate(100)

    finished = False
    t = 0
    
    ctrlr.set_active(True)

    start_js_pos = robot_interface.get_state()['position']
    delta_js_pos = np.zeros_like(start_js_pos)
    delta_js_pos[5] -= 5.

    while not rospy.is_shutdown() and not finished:        

        goal_js_pos = start_js_pos + delta_js_pos
        goal_js_vel = robot_interface.get_state()['velocity']
        goal_js_acc = np.zeros_like(goal_js_pos)

        print "Sending goal ",t, " goal_js_pos:", np.round(goal_js_pos.ravel(), 2)

        if np.any(np.isnan(goal_js_pos)) or np.any(np.isnan(goal_js_vel)) or np.any(np.isnan(goal_js_acc)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            # Setting new goal"
            ctrlr.set_goal(goal_js_pos=goal_js_pos, 
                           goal_js_vel=goal_js_vel, 
                           goal_js_acc=goal_js_acc)
            
            print "Waiting..."
            
            js_pos_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=5.0)
            
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)

        rate.sleep()
    
    ctrlr.wait_until_goal_reached(timeout=5.0)
    ctrlr.set_active(False)

    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['js_pos'])

if __name__ == '__main__':

    rospy.init_node('classical_jsc_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)
    
    test_postn_controller(robot_interface=arm)