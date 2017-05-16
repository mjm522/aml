import rospy
import numpy as np
import quaternion
from functools import partial
from aml_visual_tools.visual_tools import show_image
from aml_data_collec_utils.box_object import BoxObject
from aml_services.srv import PredictAction, PredictState
from aml_robot.box2d.push_world.box2d_box_object import Box2DBoxObject

def predict_state_client(state, action):
    rospy.wait_for_service('predict_state')
    try:
        predict_state_service = rospy.ServiceProxy('predict_state', PredictState)
        response = predict_state_service(state, action)
        return response.next_state
    except rospy.ServiceException, e:
        print "Service call to predict_state failed: %s"%e


def predict_action_client(curr_state, tgt_state):
    rospy.wait_for_service('predict_action')
    try:
        predict_action_service = rospy.ServiceProxy('predict_action', PredictAction)
        response = predict_action_service(curr_state, tgt_state)
        return response.action[0], response.action[1], response.sigma
    except rospy.ServiceException, e:
        print "Service call to predict_action failed: %s"%e


class Task(object):
    def __init__(self, task_interface):
        if task_interface == 'sim':
            self._task = Box2DBoxObject()
        elif task_interface == 'real':
            self._task = BoxObject()

class GeneralPushingMachine(Task):
    
    def __init__(self, task_interface):
        Task.__init__(self, task_interface)
        self._curr_state = None

    def get_curr_state(self):
        box_state = self._task.get_pose()
        curr_scene =  self._task.get_curr_image()
        return box_state, curr_scene

    def get_predictions(self, target_state):
        box_state, curr_scene = self.get_curr_state()
        
        action_x, action_y, action_sigma = predict_action_client(curr_state=curr_scene, 
                                                                      tgt_state=target_state)

        next_state =  predict_state_client(state=curr_scene, action=(action_x, action_y))

        return (action_x, action_y), next_state

    def execute_action(self):
        action, next_state = self.get_predictions()
        
        if task_interface == 'sim':
            push = (action[0], action[1], 10., 0.)
            self._task.new_push(push=push)
        elif task_interface == 'real':
            pass
