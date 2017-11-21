#!/usr/bin/env python

import rospy
from aml_services.srv import PushAction, BoxState

import numpy as np


class BaxterPushWorld(object):

    def __init__(self, transform=False):
        rospy.wait_for_service('get_box_state')
        self._get_state_cli = rospy.ServiceProxy('get_box_state', BoxState)

        rospy.wait_for_service('apply_push')
        self._apply_push_cli = rospy.ServiceProxy('apply_push', PushAction)

        self._transform = transform


    def baxter_2_box_2d_transform(self, x0, angle=-0.5*np.pi):
        if isinstance(x0, tuple):
            x0 = np.asarray(x0)

        if angle == 0.:
            return x0
        tmp = x0.copy()
        rot_matrix = np.array([[np.cos(angle),  -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        tmp = np.dot(rot_matrix, np.asarray([x0[0], x0[1]]))
        x0[0] = tmp[0]
        x0[1] = tmp[1]
        # x0[2] = x0[2] + 0.5*np.pi
        # if isinstance(x0, np.ndarray):
        #     x0 = tuple(x0)
        return x0


    def get_state(self):
        
        try:
            
            response = self._get_state_cli()
            state = np.array(response.state)
            
            if self._transform:
                return self.baxter_2_box_2d_transform(state)
            else:
                return state
        
        except rospy.ServiceException, e:
            print "Service call to get_state failed: %s"%e


    def apply_push(self, push_u):
        
        try:
            
            response = self._apply_push_cli(push_u)
            return response.status
        except rospy.ServiceException, e:
            print "Service call to apply_push failed: %s"%e


def main():

    world = BaxterPushWorld()


    print world.get_state()


    print world.apply_push([0.5])


if __name__ == '__main__':
    main()