
import numpy as np
import quaternion
import rospy
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController

import numpy as np
import quaternion

from aml_robot.baxter_robot import BaxterArm


import rospy


def main_coop_gravity_comp_demo():

    rospy.init_node("coop_gravity_comp_demo_node")


    master_limb = 'left'
    slave_limb  = 'right'

    arm_master = BaxterArm(master_limb)
    arm_slave  = BaxterArm(slave_limb)
    #baxter_ctrlr.set_neutral()
    arm_master.untuck_arm()
    arm_slave.untuck_arm()

    master_start_pos, master_start_ori  =  arm_master.get_ee_pose()
    slave_start_pos,  slave_start_ori   =  arm_slave.get_ee_pose()

    ctrlr_slave = OSCTorqueController(arm_slave)

    cmd = np.zeros(7)
    rate = 200 #Hz
    rate = rospy.timer.Rate(rate)

    rel_pos = slave_start_pos - master_start_pos#np.array([-0.00507125, -0.2750604, -0.00270199]) #np.array([-0.00507125, -0.85750604, -0.00270199]) 
    rel_ori = slave_start_ori.conjugate()*master_start_ori

    ctrlr_slave.set_active(True)

    while not rospy.is_shutdown():

        master_pos, master_ori =  arm_master.get_ee_pose()
        slave_pos,  slave_ori  =  arm_slave.get_ee_pose()


        #find the rotation to coordinate frame
        v = np.quaternion(0,rel_pos[0],rel_pos[1],rel_pos[2])

        # Rotation of a point p = (x,y,z) by a quaternion, defined as q*v*conjugate(q) 
        # where v is a pure imaginary quaternion composed by the coordinates of p, such that v = (0,x,y,z)
        # This is equivalent to p_rotated = R(q)*p, where R(q) is the corresponding rotation matrix for a unit quaternion q
        Rp = master_ori*v*master_ori.conjugate() # Rotation of a point by a quaternion, defined as q*v*conjugate(q) 

        rel_pos_rl = np.array([Rp.x,Rp.y,Rp.z]) # goal position of right arm goal w.r.t. left arm

        #compute required change in position
        goal_pos = (master_pos + rel_pos_rl) # goal position of right arm w.r.t. base
    
        #following is the initiall difference when using left_arm_start and rigt_arm_start
        goal_ori = master_ori*rel_ori


        ctrlr_slave.set_goal(goal_pos,goal_ori)
        

        lin_error, ang_error, success, time_elapsed = ctrlr_slave.wait_until_goal_reached(timeout=0.5)
        
        print("lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " success: ", success)


        #print "the command \n", cmd_right
        rate.sleep()


if __name__ == "__main__":
    main_coop_gravity_comp_demo()
