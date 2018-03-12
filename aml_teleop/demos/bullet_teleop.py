#!/usr/bin/env python

import time
import rospy
import numpy as np
import pybullet as pb
from aml_rl_envs.hand.man_object import ManObject
from omni_interface.phantom_omni import PhantomOmni


class BulletTeleop():

    def __init__(self):

        self._object = ManObject(scale=0.5, use_fixed_Base=False, obj_type='cube')

        #phantom omni gives in mm, scale accordingly
        self._ph_om = PhantomOmni(scale=1e-2)

    def run(self):

        while not rospy.is_shutdown():

            ee_pos, ee_ori = self._ph_om.get_ee_pose()

            if (ee_pos is not None) and (ee_ori is not None):

                print "pos \t", np.round(ee_pos, 3)
                print "ori \t", np.round(ee_ori, 3)

                self._object.set_pose(ee_pos, ee_ori)

            self._object.simple_step()


def main():

    rospy.init_node('teleop_bullet', anonymous=True)

    bt = BulletTeleop()
    
    bt.run()



if __name__ == '__main__':
    main()
        

