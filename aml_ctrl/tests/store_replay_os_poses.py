import numpy as np
import quaternion
import sys

from os.path import dirname, abspath
import rospy

from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp

data_folder_path = dirname(dirname(abspath(__file__))) + '/tests/data/'

def save_test_locations(arm, limb_idx, num_points=3):

    arm_test_points = []
    
    #btn_state will be false initially

    for k in range(num_points):
        
        start_flag = False

        print "Recording test location number \t", k+1
        print "Press the cuff button, take the arm to location, press cuff button again."

        raw_input('Press enter to continue...')

        robot_state = arm._state
        robot_state['ee_ori'] = quaternion.as_float_array(robot_state['ee_ori'])[0]

        arm_test_points.append(arm._state)
   

    arm_test_points[0]['limb_idx'] = limb_idx
    np.save(data_folder_path+'test_points_data.npy', arm_test_points)

def get_saved_test_locations():

    try:
        test_points_data = np.load(data_folder_path + 'test_points_data.npy')

    except Exception as e:

        raise e

    return test_points_data


def execute_trajectory(robot_interface, trajectory, rate, orientation_ctrl = True):

    ctrlr  = OSPositionController(robot_interface)
    # ctrlr    = BaxterMoveItController()
    # ctrlr  = OSCTorqueController(robot_interface)
    ctrlr.set_active(True)

    t = 0
    n_steps = len(trajectory['pos_traj'])

    while t < n_steps:
    
        goal_pos  = trajectory['pos_traj'][t]
        goal_ori  = trajectory['ori_traj'][t]
        goal_vel  = trajectory['vel_traj'][t]
        goal_omg  = trajectory['omg_traj'][t]

        print "Sending goal ",t, " goal_pos:", goal_pos.ravel()

        if np.any(np.isnan(goal_pos)):

            print "Goal", t, "is NaN, that is not good, we will skip it!"

        else:
            
            ctrlr.set_goal(goal_pos=goal_pos, 
                           goal_ori=goal_ori, 
                           goal_vel=goal_vel, 
                           goal_omg=goal_omg, 
                           orientation_ctrl = orientation_ctrl)

            # print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)

        rate.sleep()

        finished = (t == n_steps)

    lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10)
    ctrlr.set_active(False)

    return finished

def test_saved_test_locations(arm):

    test_points_data = get_saved_test_locations()

    interp_fn = MinJerkInterp()

    num_points = len(test_points_data)

    rate = rospy.Rate(10)

    for k in range(num_points):

        arm.untuck_arm()

        robot_pos, robot_ori = arm.get_ee_pose()
        
        goal_pos = test_points_data[k]['ee_point']
        
        goal_ori = test_points_data[k]['ee_ori']

        interp_fn.configure(robot_pos, robot_ori, goal_pos, goal_ori)

        interp_traj  = interp_fn.get_interpolated_trajectory()

        execute_trajectory(robot_interface=arm, trajectory=interp_traj, rate=rate)

        print "Send to test location \t", k+1
        raw_input('Press enter to continue...')


def main(args):

    from aml_robot.baxter_robot import BaxterArm

    if 'limb_idx=0' in args:

        limb_idx  = 0
        arm       = BaxterArm('left')       

    elif 'limb_idx=1' in args:

        limb_idx = 1
        arm      = BaxterArm('right')
    
    else:

        print "Enter a valid limb index ...  limb_idx=0 for left and lim_idx=1 for right"

        raise ValueError

    
    test_points_data = get_saved_test_locations()
    
    for k in range(4):

        ee_pos = test_points_data[k]['ee_point']

        ee_ori = test_points_data[k]['ee_ori']

        print [ee_pos[0],ee_pos[1],ee_pos[2], ee_ori[0],ee_ori[1],ee_ori[2],ee_ori[3]]

        # ee_ori = quaternion.as_rotation_matrix(np.quaternion(ee_ori[0],ee_ori[1],ee_ori[2],ee_ori[3]))

        # ee_pose = np.hstack([ee_ori, ee_pos[:,None]])

        # print "ee_pose \n"

        # print ee_pose

        # print "ee_pose one line \n"

        # print ee_pose.ravel()


    # test_saved_test_locations(arm)
    
    # save_test_locations(arm=arm, limb_idx=limb_idx, num_points=4)

if __name__ == '__main__':
    rospy.init_node('lfd_node')
    cmdargs = str(sys.argv)
    main(cmdargs)
