import numpy as np
import quaternion
import rospy
from aml_ctrl.controllers.os_controllers.os_bi_arm_controller import OSBiArmController

def test_follow_gps_policy(vel_ee_master, master_arm, slave_arm, dt):

    ctrlr = OSBiArmController(master_arm=master_arm, slave_arm=slave_arm)
    
    master_pos, master_ori   = master_arm.get_ee_pose()
    # vel_ee_master = arm_l.get_ee_velocity()[0:3]
    slave_pos, slave_ori = slave_arm.get_ee_pose()

    pos_rel = slave_pos - master_pos
        
    vel_rel = np.array([0.,0.,0.])
        
    omg_rel = np.array([0.,0.,0.])

    master_state = master_arm._state
    slave_state  = slave_arm._state

    jnt_l = arm_l.angles()
    jnt_r = arm_r.angles()

    #print "Starting co-operative torque controller"

    # rate = rospy.Rate(10)
    mags_r = []
    mags_l = []


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

    rel_jac = baxter_ctrlr.relative_jac(pos_rel)
    #only for position
    jac_master_rel = np.vstack([rel_jac[0:3,:],rel_jac[6:9,:]])

    jac_tmp = np.dot(jac_master_rel, jac_master_rel.T)
    
    jac_star = np.dot(jac_master_rel.T, (np.linalg.inv(jac_tmp + np.eye(jac_tmp.shape[0])*0.001))) 

    pos_tmp = np.hstack([vel_ee_master, vel_rel])
    
    jnt_vel = np.dot(jac_star, pos_tmp)

    ee_req_vel_l        = np.dot(jac_ee_l, jnt_vel[:7])[0:3]
    ee_req_vel_r        = np.dot(jac_ee_r, jnt_vel[7:])[0:3]

    #print ee_req_vel_l
    #print ee_req_vel_r

    # cmd_l = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_l, goal_pos=ee_req_vel_l[0:3]*baxter_ctrlr.dt, goal_ori=None, orientation_ctrl=False)
    # print "cmd_l \n", cmd_l

    #aux arm in position mode
    #cmd_r = jnt_r + jnt_vel[7:]*0.8

    #aux arm in vel mode
    cmd_r = jnt_vel[7:]

    #aux arm in torque mode
    # cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, goal_pos=ee_req_vel_r[0:3], goal_ori=None, orientation_ctrl=False)
    
    print "cmd_r \n", cmd_r

    return cmd_r

def test_follow_gps_policy2(arm_l, arm_r):

    #left_arm_start =[-0.8, 0.7, 0.1, 7.48607379e-01, 0.5, 1.25333658e+00, -1.81064349e-04]
    #rigt_arm_start = [0.8, 0.7, -0.1, 7.48607379e-01, -0.5, 1.25333658e+00, 1.81064349e-04]

    #arm_l.move_to_joint_position(left_arm_start)
    #arm_r.move_to_joint_position(rigt_arm_start)

    baxter_ctrlr = MinJerkController(extern_call=True, trial_arm=arm_l, aux_arm=arm_r)
    
    left_pos,  left_ori  = arm_l.get_ee_pose()
    
    right_pos, right_ori = arm_r.get_ee_pose() 

    state_r  = arm_r._state
    state_r['jnt_start'] = arm_r.angles()

    #print right_pos - left_pos
    #print quatdiff(quaternion.as_float_array(right_ori)[0], quaternion.as_float_array(left_ori)[0])
    #the imaginary coordinate translation w.rt left ee
    #following is the initiall difference when using left_arm_start and rigt_arm_start
    const_tran = np.array([-0.00507125, -0.85750604, -0.00270199])

    #the imaginary coordinate rotation w.rt left ee
    const_qt = np.quaternion(1.,0.,0.,0.)
    
    #find the rotation to coordinate frame
    req_tansf = left_ori*const_qt*left_ori.conjugate()

    #compute required change in position
    req_pos_diff = left_pos + const_tran - right_pos

    #compute required change in orientation
    #req_ori_diff = quatdiff(quaternion.as_float_array(right_ori)[0], quaternion.as_float_array(req_tansf)[0])
    
    #following is the initiall difference when using left_arm_start and rigt_arm_start
    req_ori_diff = np.array([-0., -0.0, 0. ])#np.array([-0.12093466, -0.00218386, 0.5374029 ])

    #cmd_r = np.dot(arm_r._state['jacobian'].T, np.hstack([req_pos_diff, req_ori_diff]))

    cmd_r = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_r, 
                                          goal_pos=req_pos_diff, 
                                          goal_ori=req_ori_diff, 
                                          orientation_ctrl=True)

    return cmd_r


if __name__ == '__main__':

    rospy.init_node('follow_gps_policy')
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