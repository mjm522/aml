import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.os_controllers.os_bi_arm_controller import OSBiArmController
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController

def test_coop_position_control():
    limb_idx_l = 0 #0 is left and 1 is right
    limb_idx_r = 1 #0 is left and 1 is right
    
    baxter_ctrlr = MinJerkController()
    arm_l = baxter_ctrlr.get_arm_handle(limb_idx_l)
    arm_r = baxter_ctrlr.get_arm_handle(limb_idx_r)
    baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm_l.get_ee_pose()
    goal_pos = start_pos + np.array([0.,0.8, 0.])
    angle    = 90.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2], axis[3])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    left_pos, left_ori   = arm_l.get_ee_pose()
    right_pos, right_ori = arm_r.get_ee_pose()

    pos_rel = right_pos - left_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    min_jerk_traj   = baxter_ctrlr.get_interpolated_trajectory()

    print "Starting co-operative position controller"

    for t in range(baxter_ctrlr.timesteps):

        jac_master_rel = baxter_ctrlr.relative_jac(pos_rel)

        jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
        
        jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.01))) 

        pos_tmp = np.hstack([min_jerk_traj['vel_traj'][t,:], min_jerk_traj['omg_traj'][t,:], vel_rel, omg_rel])
        
        jnt_vel = np.dot(jac_star, pos_tmp)

        if np.any(np.isnan(jnt_vel)):
            jnt_vel_l = np.zeros(7)
            jnt_vel_r = np.zeros(7)
        else:
            jnt_vel_l = jnt_vel[:7]
            jnt_vel_r = jnt_vel[7:]
        
        jnt_l += jnt_vel_l*baxter_ctrlr.dt
        jnt_r += jnt_vel_r*baxter_ctrlr.dt
        
        arm_l.exec_position_cmd(jnt_l)

        arm_r.exec_position_cmd(jnt_r)

        time.sleep(0.1)

    final_pos, final_ori  =  arm_l.get_ee_pose()
    print "ERROR in position \t", np.linalg.norm(final_pos-goal_pos)
    print "ERROR in orientation \t", np.linalg.norm(final_ori-goal_ori)

def test_coop_torque_control():
    limb_idx_l = 0 #0 is left and 1 is right
    limb_idx_r = 1 #0 is left and 1 is right
    
    baxter_ctrlr = MinJerkController()
    arm_l = baxter_ctrlr.get_arm_handle(limb_idx_l)
    arm_r = baxter_ctrlr.get_arm_handle(limb_idx_r)
    #baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm_l.get_ee_pose()
    goal_pos = start_pos + np.array([0.,-0.0, 0.9])
    angle    = 45.0
    axis     = np.array([1.,0.,0.]); axis = np.sin(0.5*angle*np.pi/180.)*axis/np.linalg.norm(axis)
    goal_ori = np.quaternion(np.cos(0.5*angle*np.pi/180.), axis[0], axis[1], axis[2])

    baxter_ctrlr.configure(start_pos, start_ori, goal_pos, goal_ori)

    left_pos, left_ori   = arm_l.get_ee_pose()
    right_pos, right_ori = arm_r.get_ee_pose()

    pos_rel = right_pos - left_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    min_jerk_traj   = baxter_ctrlr.get_interpolated_trajectory()

    print "Starting co-operative torque controller"

    # rate = rospy.Rate(10)
    mags_r = []
    mags_l = []
    for t in range(baxter_ctrlr.timesteps):

        state_l  = arm_l._state
        state_r  = arm_r._state
        
        state_l['jnt_start'] = jnt_l
        state_r['jnt_start'] = jnt_r

        jac_ee_l = state_l['jacobian']
        jac_ee_r = state_r['jacobian']

        left_pos  = state_l['ee_point'] 
        left_ori  = state_l['ee_ori'] 
        right_pos = state_r['ee_point'] 
        right_ori = state_r['ee_ori']

        jac_master_rel = baxter_ctrlr.relative_jac(pos_rel)

        jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
        
        jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.001))) 

        pos_tmp = np.hstack([min_jerk_traj['vel_traj'][t,:], min_jerk_traj['omg_traj'][t,:], vel_rel, omg_rel])
        
        jnt_vel = np.dot(jac_star, pos_tmp)

        ee_req_vel_l        = np.dot(jac_ee_l, jnt_vel[:7]) 
        ee_req_vel_r        = np.dot(jac_ee_r, jnt_vel[7:]) 

        cmd_l = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_l, 
                                              goal_pos=ee_req_vel_l[0:3]*baxter_ctrlr.dt, 
                                              goal_ori=None, 
                                              orientation_ctrl=False)

        cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, 
                                              goal_pos=ee_req_vel_r[0:3]*baxter_ctrlr.dt, 
                                              goal_ori=None, 
                                              orientation_ctrl=False)

        #cmd = [0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.10]
        arm_l.exec_torque_cmd(cmd_l)
        #print "cmd \n", cmd
        #print "magnitude \t", np.linalg.norm(cmd)
        arm_r.exec_torque_cmd(cmd_r)
        mags_r.append(np.linalg.norm(cmd_r))
        mags_l.append(np.linalg.norm(cmd_l))
        time.sleep(0.1)

        jnt_r = state_r['position']
        jnt_l = state_l['position']

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mags_l)
    plt.figure()
    plt.plot(mags_r)
    #plt.show()

    arm_l.exec_position_cmd2(np.zeros(7))
    arm_r.exec_position_cmd2(np.zeros(7))
    
    final_pos, final_ori  =  arm_l.get_ee_pose()
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