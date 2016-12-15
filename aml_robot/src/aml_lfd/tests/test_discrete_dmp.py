import numpy as np
import quaternion
import rospy

from aml_ctrl.controllers.osc_torque_controller import OSCTorqueController
from aml_ctrl.controllers.osc_postn_controller import OSCPositionController

from aml_lfd.dmp.discrete_dmp_shell import DiscreteDMPShell
from aml_lfd.utilities.utilities import get_ee_traj

def plot_traj(des_path):

    dmp = DiscreteDMPShell()
    dmp.configure(traj2follow=des_path, start=des_path[0,:], goal=des_path[-1,:])
    new_start = des_path[0,:]
    dmp.reset_state_dmp(new_start)
    new_goal = des_path[-1,:]
    dmp.goal = new_goal;
    tau = 1.0
    y_track, dy_track, ddy_track = dmp.rollout_dmp(tau)
    
    import matplotlib.pyplot as plt

    # test imitation of path run
    plt.figure(1)
    plt.subplot(311)
    plt.plot(des_path[:,0], 'r--', lw=2)
    plt.title('X traj')
    plt.subplot(312)
    plt.plot(des_path[:,1], 'g--', lw=2)
    plt.title('Y traj')
    plt.subplot(313)
    plt.plot(des_path[:,2], 'b--', lw=2)
    plt.title('Z traj')

    plt.figure(2)
    plt.subplot(311)
    plt.plot(des_path[:,0], lw=2)
    plt.plot(y_track[:,0],  lw=2)
    plt.subplot(312)
    plt.plot(des_path[:,1], lw=2)
    plt.plot(y_track[:,1],  lw=2)
    plt.subplot(313)
    plt.plot(des_path[:,2], lw=2)
    plt.plot(y_track[:,2],  lw=2)

    plt.tight_layout()
    plt.show()

def test_discrete_dmp(robot_interface, des_path):

    ctrlr = OSCPositionController(robot_interface)

    tau      = 1.0

    dmp = DiscreteDMPShell()
    dmp.configure(traj2follow=des_path, start=des_path[0,:], goal=des_path[-1,:])

    print "Starting position controller"

    rate = rospy.Rate(100)

    reach_thr = 0.12

    finished = False
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
            ctrlr.set_goal(goal_pos,start_ori)

            print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
            print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)
        finished = (t == n_steps)

        rate.sleep()


    
if __name__ == '__main__':

    # rospy.init_node('classical_postn_controller')
    # from aml_robot.baxter_robot import BaxterArm
    # limb = 'right'
    # arm = BaxterArm(limb)

    demo_idx = 1

    des_path = get_ee_traj(demo_idx=demo_idx)

    plot_traj(des_path)

    # test_discrete_dmp(robot_interface=arm, des_path=des_path)
