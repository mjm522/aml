#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import tf
import os
import rospy
import argparse
import numpy as np
from aml_robot.baxter_robot import BaxterArm
from aml_data_collec_utils.pushing_machine import PushMachine


from aml_services.srv import BoxState, PushAction, BoxStateResponse, PushActionResponse
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files
from aml_data_collec_utils.ros_transform_utils import quat2euler

class MPPIPushingService(object):


    def __init__(self, limb = 'left', sample_start_index = 1):
        rospy.init_node('mppi_push_service_server')


        self._get_box_state_srv  = rospy.Service('get_box_state',  BoxState,  self.get_box_state)
        self._apply_push_srv  = rospy.Service('apply_push',  PushAction,  self.apply_push)

        self._arm = BaxterArm(limb)
        
        self._push_machine = PushMachine(robot_interface=self._arm, sample_start_index=args.sample_start_index)


    def get_box_state(self, req):


        state = self._push_machine._box.get_effect()


        bx = state['box_pos'][0]
        by = state['box_pos'][1]

        euler = quat2euler(state['box_ori'])

        theta = euler[0]

        x = np.array([bx,by,theta,0.0,0.0,0.0])

        box_state_response = BoxStateResponse(x)

        return box_state_response

    def apply_push(self, req):


        push_u = req.action

        print push_u[0]
        success = self._push_machine.apply_push2(push_u[0])


        return PushActionResponse(success)

        


    def run(self):

        while not rospy.is_shutdown():
            pass




if __name__ == "__main__":
    


    parser = argparse.ArgumentParser(description='Data collection for push manipulation')
    
    parser.add_argument('-n', '--sample_start_index', type=int, help='start index of sample collection')
    
    args = parser.parse_args()
    
    mppi_pushing_service = MPPIPushingService(limb='left', sample_start_index = args.sample_start_index)
    
    print "calling run"
    mppi_pushing_service.run()