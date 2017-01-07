import numpy as np
import quaternion
import rospy

from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController

from aml_lfd.dmp.discrete_dmp_shell import DiscreteDMPShell
from aml_lfd.utilities.utilities import get_os_traj

def plot_traj(des_path, tau=1.0):

    dmp = DiscreteDMPShell()
    dmp.configure(traj2follow=des_path, start=des_path[0,:], goal=des_path[-1,:])
    new_start = des_path[0,:]
    dmp.reset_state_dmp(new_start)
    new_goal = des_path[-1,:]
    dmp.goal = new_goal

    y_track, dy_track, ddy_track = dmp.rollout_dmp(tau)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(311)
    plt.plot(des_path[:,0], 'r--', lw=2)
    plt.plot(y_track[:,0],  'g--', lw=2)
    plt.subplot(312)
    plt.plot(des_path[:,1], 'r--', lw=2)
    plt.plot(y_track[:,1],  'g--', lw=2)
    plt.subplot(313)
    plt.plot(des_path[:,2], 'r--', lw=2)
    plt.plot(y_track[:,2],  'g--', lw=2)

    plt.tight_layout()
    plt.show()

def test_discrete_dmp(robot_interface, des_path, tau=1.0):

    ctrlr = OSPositionController(robot_interface)
    #ctrlr = OSTorqueController(robot_interface)

    dmp = DiscreteDMPShell()
    dmp.configure(traj2follow=des_path, start=des_path[0,:], goal=des_path[-1,:])

    _, start_ori = robot_interface.get_ee_pose()

    print "Starting position controller"

    rate = rospy.Rate(10)

    reach_thr = 0.12

    finished  = False
    t = 0
    ctrlr.set_active(True)

    n_steps = len(des_path)

    while not rospy.is_shutdown() and not finished:

        error_lin = np.linalg.norm(ctrlr._error['linear'])

        goal_pos, _, _ = dmp.step_dmp(tau=tau)

        print "Sending goal ",t, " goal_pos:",goal_pos.ravel()

        if np.any(np.isnan(goal_pos)):
            print "Goal", t, "is NaN, that is not good, we will skip it!"
        else:
            ctrlr.set_goal(goal_pos, start_ori)

            print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=0.5)
            print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()


    
if __name__ == '__main__':

    rospy.init_node('discrete_dmp_test')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)

    arm.untuck_arm()

    demo_idx = 6 #for 4-6: tau = 0.27; for 1-3: tau = 0.1

    des_path, _, _, _, _, _ = get_os_traj(demo_idx=demo_idx) #tau=0.1 makes it match
    # des_path = get_os_traj(debug=True) #tau=2.0 makes it almost match

    #larger the tau, faster the system would reach the goal
    plot_traj(des_path, tau=0.27)

    # test_discrete_dmp(robot_interface=arm, des_path=des_path, tau=0.27)
