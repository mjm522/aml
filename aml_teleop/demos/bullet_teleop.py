#!/usr/bin/env python

import rospy
import numpy as np
import pybullet as pb
from aml_rl_envs.task.man_object import ManObject
from omni_interface.phantom_omni import PhantomOmni
from aml_rl_envs.pisa_hand.pisa_hand import PisaHand


class BulletTeleop():

    def __init__(self):

        #phantom omni gives in mm, scale accordingly
        self._ph_om = PhantomOmni(scale=1e-2)

        # self._robot = ManObject(scale=0.5, use_fixed_Base=False, obj_type='cube', render=True)

        self._robot = PisaHand(scale=3.5, use_fixed_base=False, hand_type='right', call_renderer=True)

        
    def run(self):

        while not rospy.is_shutdown():

            ee_pos, ee_ori = self._ph_om.get_ee_pose()

            if (ee_pos is not None) and (ee_ori is not None):

                print "pos \t", np.round(ee_pos, 3)
                print "ori \t", np.round(ee_ori, 3)

                self._robot.set_base_pose(ee_pos, ee_ori)

            self._robot.simple_step()


def main():

    rospy.init_node('teleop_bullet', anonymous=True)

    bt = BulletTeleop()
    
    bt.run()



if __name__ == '__main__':
    main()
        

