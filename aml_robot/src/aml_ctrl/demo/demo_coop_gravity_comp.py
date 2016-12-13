from aml_ctrl.classical_controllers import MinJerkController
from aml_ctrl.classical_controllers import test_follow_gps_policy, test_follow_gps_policy2

import numpy as np
import quaternion

import rospy


def main_coop_gravity_comp_demo():

    rospy.init_node("coop_gravity_comp_demo_node")

    baxter_ctrlr = MinJerkController()

    limb_right_idx = 1
    limb_left_idx = 0

    arm_right = baxter_ctrlr.get_arm_handle(limb_right_idx)
    arm_left = baxter_ctrlr.get_arm_handle(limb_left_idx)
    #baxter_ctrlr.set_neutral()
    baxter_ctrlr.untuck_arms()

    start_pos, start_ori  =  arm_right.get_ee_pose()

    cmd = np.zeros(7)
    rate = 100 #Hz
    rate = rospy.timer.Rate(rate)

    rel_pos = np.array([-0.00507125, -0.2750604, -0.00270199]) #np.array([-0.00507125, -0.85750604, -0.00270199]) 
    rel_ori = np.quaternion(1.,0.,0.,0.)

    while True:

        left_pos, left_ori  =  arm_left.get_ee_pose()
        right_pos, right_ori  =  arm_right.get_ee_pose()

        state_right  = arm_right._state
        state_right['jnt_start'] = arm_right.angles()


        #find the rotation to coordinate frame
        v = np.quaternion(0,rel_pos[0],rel_pos[1],rel_pos[2])

        # Rotation of a point p = (x,y,z) by a quaternion, defined as q*v*conjugate(q) 
        # where v is a pure imaginary quaternion composed by the coordinates of p, such that v = (0,x,y,z)
        # This is equivalent to p_rotated = R(q)*p, where R(q) is the corresponding rotation matrix for a unit quaternion q
        Rp = left_ori*v*left_ori.conjugate() # Rotation of a point by a quaternion, defined as q*v*conjugate(q) 

        rel_pos_rl = np.array([Rp.x,Rp.y,Rp.z]) # goal position of right arm goal w.r.t. left arm

        #compute required change in position
        goal_pos = (left_pos + rel_pos_rl) # goal position of right arm w.r.t. base
        req_pos_diff = goal_pos - right_pos # error

        error = np.linalg.norm(req_pos_diff)

        print(error)

    
        #following is the initiall difference when using left_arm_start and rigt_arm_start
        req_ori_diff = np.array([0.0, 0.0, 0.0 ])


        cmd_right = baxter_ctrlr.osc_torque_cmd_2(arm_data=state_right, 
                                              goal_pos=req_pos_diff, 
                                              goal_ori=req_ori_diff, 
                                              orientation_ctrl=True)
        
        
        
        #increment the set point only if the arm is within a certain threshold
        if error < 0.12:
            pass
      
        arm_right.exec_torque_cmd(cmd_right)
        #print "the command \n", cmd_right
        rate.sleep()


if __name__ == "__main__":
    main_coop_gravity_comp_demo()
