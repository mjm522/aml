
import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.osc_torque_controller import OSC_TorqueController


def test_torque_controller(robot_interface, start_pos, start_ori, goal_pos, goal_ori):
    
    ctrlr = OSC_TorqueController(robot_interface)

    min_jerk_interp = MinJerkInterp()

    robot_interface.untuck_arm()

    min_jerk_interp.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_ori, goal_qt=goal_ori)

    min_jerk_traj = min_jerk_interp.get_min_jerk_trajectory()

    print "Starting torque controller"

    for t in range(len(min_jerk_interp.timesteps)):

        ctrlr.compute_cmd(goal_pos=min_jerk_traj['pos_traj'][t,:],
                          goal_ori=None, 
                          orientation_ctrl=False)
        ctrlr.send_cmd()

    final_pos, final_ori  =  robot_interface.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

if __name__ == '__main__':

    rospy.init_node('classical_torque_controller')
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)
    start_pos, start_ori  =  arm.get_ee_pose()
    
    if limb == 'left':
        goal_pos = start_pos + np.array([0.,0.8, 0.])
    else:
        goal_pos = start_pos - np.array([0.,0.8, 0.])

    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])
    
    test_torque_controller(arm, start_pos, start_ori, goal_pos, goal_ori)