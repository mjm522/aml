from aml_ctrl.classical_controllers import MinJerkController, quatdiff
from aml_ctrl.classical_controllers import test_follow_gps_policy, test_follow_gps_policy2

import numpy as np
import quaternion

import rospy


def main_coop_gravity_comp_demo():

    rospy.init_node("coop_gravity_comp_demo_node")

    baxter_ctrlr = MinJerkController()

    limb_right_idx = 1
    limb_left_idx = 0

    master_limb_idx = 1
    slave_limb_idx  = 0

    arm_master = baxter_ctrlr.get_arm_handle(master_limb_idx)
    arm_slave  = baxter_ctrlr.get_arm_handle(slave_limb_idx)
    #baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()


    master_start_pos, master_start_ori  =  arm_master.get_ee_pose()
    slave_start_pos,  slave_start_ori   =  arm_slave.get_ee_pose()

    cmd = np.zeros(7)
    rate = 500 #Hz
    rate = rospy.timer.Rate(rate)

    rel_pos = slave_start_pos - master_start_pos#np.array([-0.00507125, -0.2750604, -0.00270199]) #np.array([-0.00507125, -0.85750604, -0.00270199]) 
    rel_ori = slave_start_ori.conjugate()*master_start_ori#np.quaternion(1.,0.,0.,0.)

    while True:

        master_pos, master_ori =  arm_master.get_ee_pose()
        slave_pos,  slave_ori  =  arm_slave.get_ee_pose()

        state_slave  = arm_slave._state
        state_slave['jnt_start'] = arm_slave.angles()


        #find the rotation to coordinate frame
        v = np.quaternion(0,rel_pos[0],rel_pos[1],rel_pos[2])

        # Rotation of a point p = (x,y,z) by a quaternion, defined as q*v*conjugate(q) 
        # where v is a pure imaginary quaternion composed by the coordinates of p, such that v = (0,x,y,z)
        # This is equivalent to p_rotated = R(q)*p, where R(q) is the corresponding rotation matrix for a unit quaternion q
        Rp = master_ori*v*master_ori.conjugate() # Rotation of a point by a quaternion, defined as q*v*conjugate(q) 

        rel_pos_rl = np.array([Rp.x,Rp.y,Rp.z]) # goal position of right arm goal w.r.t. left arm

        #compute required change in position
        goal_pos = (master_pos + rel_pos_rl) # goal position of right arm w.r.t. base
        req_pos_diff = goal_pos - slave_pos # error

        error = np.linalg.norm(req_pos_diff)

        print(error)

    
        #following is the initiall difference when using left_arm_start and rigt_arm_start
        goal_ori = master_ori*rel_ori

        req_ori_diff = quatdiff(quat_curr=slave_ori, quat_des=goal_ori)#np.array([0.0, 0.0, 0.0 ])


        # cmd_right = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_right, 
        #                                       goal_pos=req_pos_diff, 
        #                                       goal_ori=req_ori_diff, 
        #                                       orientation_ctrl=True)
        cmd_slave = baxter_ctrlr.osc_torque_cmd(goal_pos=goal_pos, 
                                                goal_ori=goal_ori, 
                                                limb_idx=slave_limb_idx, 
                                                orientation_ctrl=True)
        
        
        arm_slave.exec_torque_cmd(cmd_slave)
        #print "the command \n", cmd_right
        rate.sleep()


if __name__ == "__main__":
    main_coop_gravity_comp_demo()
