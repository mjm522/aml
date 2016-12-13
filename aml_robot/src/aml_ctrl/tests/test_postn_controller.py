import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.osc_postn_controller import OSC_PostnController

def test_position_controller(robot_interface, start_pos, start_ori, goal_pos, goal_ori):
    #0 is left and 1 is right
    
    ctrlr = OSC_PostnController(robot_interface)

    min_jerk_interp = MinJerkInterp()

    robot_interface.untuck_arm()

    min_jerk_interp.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_ori, goal_qt=goal_ori)

    min_jerk_traj = min_jerk_interp.get_min_jerk_trajectory()

    print "Starting position controller"

    for t in range(len(min_jerk_interp.timesteps)):
        ctrlr.compute_cmd(goal_pos=min_jerk_traj['pos_traj'][t,:],
                          goal_ori=None, 
                          orientation_ctrl=False)
        ctrlr.send_cmd()

    final_pos, final_ori  =  arm.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

if __name__ == '__main__':

    rospy.init_node('classical_postn_controller')
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
    
    test_position_controller(arm, start_pos, start_ori, goal_pos, goal_ori)