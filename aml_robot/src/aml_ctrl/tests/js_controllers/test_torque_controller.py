import numpy as np
import quaternion
import rospy
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator
from aml_ctrl.controllers.js_controllers.js_torque_controller import JSTorqueController

def test_torque_controller(robot_interface, demo_idx=1):
    #0 is left and 1 is right

    ctrlr = JSTorqueController(robot_interface)

    kwargs = {}

    kwargs['demo_idx'] = demo_idx

    js_traj_gen = JSTrajGenerator(load_from_demo=True, **kwargs)

    js_traj = js_traj_gen.generate_traj()

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
        finished = (t == n_steps)

        rate.sleep()
    
    ctrlr.wait_until_goal_reached(timeout=5.0)
    ctrlr.set_active(False)

    # Error stored in ctrlr._error is the most recent error w.r.t to the most recent sent goal
    print "ERROR in position \t", np.linalg.norm(ctrlr._error['js_pos'])

if __name__ == '__main__':

    rospy.init_node('classical_jsc_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    
    test_torque_controller(robot_interface=arm, demo_idx=4)