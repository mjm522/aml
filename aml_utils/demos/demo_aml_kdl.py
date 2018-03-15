#! /usr/bin/env/python

import numpy as np
from aml_utils.aml_pykdl.aml_pykdl import AMLPyKDL
from aml_utils.aml_pykdl.pykdl_utils import kdl_tree_from_urdf_model


class DemoKDL(AMLPyKDL):

    def __init__(self):

        chain = self.setup_chain()

        base_position  = np.zeros(3)

        AMLPyKDL.__init__(self, chain, base_position)

        self.print_robot_description() 


    def setup_chain(self, tip='right_l4_2'): 

        self._kdl_tree = kdl_tree_from_urdf_model(None)

        return self._kdl_tree.getChain('base',  tip)  


    def get_ee(self, q):
       
        self.update_chain(q=q) 

        ee_pos, ee_ori = self.compute_fwd_kinematics(ori_type='quat')

    def get_jac(self, q=None):

        if q is not None:
            self.update_chain(q=q)

        return self.compute_jacobian()

    def print_robot_description(self):

        print "KDL joints: %d" % self._kdl_tree.getNrOfJoints()
        print "KDL segments: %d" % self._kdl_tree.getNrOfSegments()

    def get_num_joints(self):
        return self._kdl_tree.getNrOfJoints()


if __name__ == '__main__':
    
    kdl_ = DemoKDL()

    print kdl_.get_jac(np.zeros(kdl_.get_num_joints()))