import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.js_traj_generator import JSTrajGenerator
from aml_ctrl.controllers.jsc_torque_controller import JSCTorqueController

def test_torque_controller(robot_interface, demo_idx=1, start_pos=None, start_ori=None, goal_pos=None, goal_ori=None):
    #0 is left and 1 is right

    ctrlr = JSCTorqueController(robot_interface)

    js_traj_gen = JSTrajGenerator()

    js_traj_gen.configure(demo_idx=demo_idx, 
    	                  start_pos=start_pos, 
    	                  goal_pos=goal_pos, 
    	                  start_qt=start_ori, 
    	                  goal_qt=goal_ori)

    js_traj = js_traj_gen.get_interpolated_trajectory()

    #sending the arm to the initial location of js trajectory
    robot_interface.exec_position_cmd(js_traj['pos_traj'][0])

    print "Starting joint space torque controller"

    rate = rospy.Rate(100)

    finished = False
    t = 0
    ctrlr.set_active(True)

    n_steps = js_traj_gen._timesteps

    while not rospy.is_shutdown() and not finished:

        goal_js_pos = js_traj['pos_traj'][t]
        goal_js_vel = js_traj['vel_traj'][t]
        goal_js_acc = js_traj['acc_traj'][t]

        print "Sending goal ",t, " goal_pos:", goal_js_pos.ravel() 

        if np.any(np.isnan(goal_js_pos)) or np.any(np.isnan(goal_js_vel)) or np.any(np.isnan(goal_js_acc)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            # Setting new goal"
            ctrlr.set_goal(goal_js_pos=goal_js_pos, 
                           goal_js_vel=goal_js_vel, 
                           goal_js_acc=goal_js_acc)
            
            print "Waiting..."
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=5.0, jsc=True)
            
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()
    
    ctrlr.wait_until_goal_reached(timeout=5.0)
    #ctrlr.set_active(False)


    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['linear'])
    print "ERROR in orientation \t", np.linalg.norm(ctrlr._error['angular'])

if __name__ == '__main__':

    rospy.init_node('classical_jsc_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)

    # angle    = 90.0
    # axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    # goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])
    
    test_torque_controller(robot_interface=arm, demo_idx=4)