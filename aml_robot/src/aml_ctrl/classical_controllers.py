import numpy as np
import quaternion

from aml_robot.baxter_robot import BaxterArm
import time
import baxter_interface
from baxter_interface import CHECK_VERSION
import rospy
import tf
from tf import TransformListener

import sys
sys.argv

curr_time_in_sec = lambda: int(round(time.time() * 1e6))


class MinJerkController():

    def __init__(self, extern_call=False, trial_arm=None, aux_arm=None, tau=5., dt=0.05):
        
        if extern_call:
            if trial_arm is None or aux_arm is None:
                print "Pass the arm handles, it can't be none"
                raise ValueError
            else:
                self.left_arm  = trial_arm
                self.right_arm = aux_arm
        else:
            self.left_arm 	= BaxterArm('left') #object of type Baxter from baxter_mechanism
            self.right_arm  = BaxterArm('right')
            self.start_pos 	= None
            self.goal_pos 	= None
            self.start_qt 	= None
            self.goal_qt 	= None
            self.tau 		= tau
            self.dt 		= dt
            self.timesteps  = int(self.tau/self.dt)

            baxter = baxter_interface.RobotEnable(CHECK_VERSION)
            baxter.enable()
        self.osc_pos_threshold = 0.01

    
    def osc_torque_cmd_2(self, arm_data, goal_pos, goal_ori=None, orientation_ctrl=False):

        #proportional gain
        kp              = 10.
        #derivative gain
        kd              = np.sqrt(kp)
        #null space control gain
        alpha           = 3.25

        jnt_start = arm_data['jnt_start']
        ee_xyz = arm_data['ee_point']
        jac_ee = arm_data['jacobian']
        q      = arm_data['position']
        dq     = arm_data['velocity']
        ee_ori = arm_data['ee_ori']
        
        #to fix the nan issues that happen
        u_old  = np.zeros_like(jnt_start)

        # calculate the inertia matrix in joint space
        Mq     = arm_data['inertia'] 

        # convert the mass compensation into end effector space
        Mx_inv         = np.dot(jac_ee, np.dot(np.linalg.inv(Mq), jac_ee.T))
        svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)

        # cut off any singular values that could cause control problems
        singularity_thresh  = .00025
        for i in range(len(svd_s)):
            svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                1./float(svd_s[i])

        # numpy returns U,S,V.T, so have to transpose both here
        # convert the mass compensation into end effector space
        Mx   = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

        x_des   = goal_pos #- ee_xyz
 
        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError
            else:
                if type(goal_ori) is np.quaternion:
                    omg_des  = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(ee_ori)[0])
                elif len(goal_ori) == 3:
                    omg_des = goal_ori
                else:
                    print "Wrong dimension"
                    raise ValueError
        else:
            omg_des = np.zeros(3)

        # calculate desired force in (x,y,z) space
        Fx                  = np.dot(Mx, np.hstack([x_des, omg_des]))
        # transform into joint space, add vel and gravity compensation
        u                   = (kp * np.dot(jac_ee.T, Fx) - np.dot(Mq, kd * dq))

        # calculate our secondary control signal
        # calculated desired joint angle acceleration

        prop_val            = ((jnt_start - q) + np.pi) % (np.pi*2) - np.pi

        q_des               = (kp * prop_val - kd * dq).reshape(-1,)

        u_null              = np.dot(Mq, q_des)

        # calculate the null space filter
        Jdyn_inv            = np.dot(Mx, np.dot(jac_ee, np.linalg.inv(Mq)))

        null_filter         = np.eye(len(q)) - np.dot(jac_ee.T, Jdyn_inv)

        u_null_filtered     = np.dot(null_filter, u_null)

        #changing the rest q as the last updated q
        jnt_start           = q 

        u                   += alpha*u_null_filtered

        if np.any(np.isnan(u)):
            u               = u_old
        else:
            u_old           = u

        return u








##==============================
# Test code
#===============================
if __name__ == "__main__":

    rospy.init_node('baxter_classical_controller')

    #arguments that can be passed in
    # python classical_controllers.py calib
    # will self calibrate the arm, default is not to do it
    # python classical_controllers.py torque
    # will play the arm in torque mode

    #get the arguments passed to the script
    cmdargs = str(sys.argv)

    #test_torque_control(0)
    #test_position_control()
    #test_coop_position_control()
    
    #test_coop_torque_control()

    #test_follow_gps_policy()


    #ctrlr = MinJerkController()

    #test_follow_gps_policy2(ctrlr.left_arm, ctrlr.right_arm)

    if 'calib' in cmdargs:
        from camera_calib import Baxter_Eye_Hand_Calib
        calib = Baxter_Eye_Hand_Calib()
        calib.self_caliberate()
    
    if 'torque' in cmdargs:
        torque_mode = True
    else:
        torque_mode = False

    test_maintain_position()

    #test_reach_both_sides_box(torque_mode)
    #test_lift_both_sides_box()
