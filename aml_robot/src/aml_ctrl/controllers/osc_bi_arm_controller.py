import numpy as np
import quaternion
import copy
import rospy
from config import OSC_TORQUE_CNTLR
from aml_ctrl.classical_controller import ClassicalController

class OSC_BiArmController(ClassicalController):

    def __init__(self, right_arm , left_arm, mode='torque'):

        self._right_arm  = right_arm
        self._left_arm   = left_arm

        if mode == 'torque':
            from aml_ctrl.controllers.osc_torque_controller import OSC_TorqueController
            self._right_arm_ctrlr = OSC_TorqueController(self._right_arm)
            self._left_arm_ctrlr  = OSC_TorqueController(self._left_arm)
        elif mode == 'postn':
            from aml_ctrl.controllers.osc_postn_controller import OSC_PostnController
            self._right_arm_ctrlr = OSC_PostnController(self._right_arm)
            self._left_arm_ctrlr  = OSC_PostnController(self._left_arm)
        else:
            print "Unknown mode of controller..."
            raise ValueError

    def relative_jac(self, rel_pos):

        #left_arm is the master arm
        
        jac_left = self._left_arm.get_jacobian_from_joints()
        
        jac_right = self._right_arm.get_jacobian_from_joints()

        def make_skew(v):
            return np.array([[0., -v[2], v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

        tmp1 = np.vstack([np.hstack([np.eye(3),-make_skew(rel_pos)]),np.hstack([np.zeros((3,3)),np.eye(3)])])
        
        pos_ee_l, rot_ee_l = self._left_arm.get_cartesian_pos_from_joints()
        
        pos_ee_r, rot_ee_r = self._right_arm.get_cartesian_pos_from_joints()

        tmp2 = np.vstack([np.hstack([-rot_ee_l,np.zeros((3,3))]),np.hstack([np.zeros((3,3)),-rot_ee_l])])

        tmp3 = np.vstack([np.hstack([rot_ee_r, np.zeros((3,3))]), np.hstack([np.zeros((3,3)), rot_ee_r])])

        jac_rel = np.hstack([np.dot(np.dot(tmp1, tmp2), jac_left), np.dot(tmp3, jac_right)])

        jac_master_rel = np.vstack([np.hstack([jac_left, np.zeros_like(jac_left)]), jac_rel])

        return jac_master_rel

    