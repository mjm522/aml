#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Point
from rospy_tutorials.msg import Floats

from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_planners.push_planner.push_worlds.box2d_push_world import Box2DPushWorld
from aml_planners.push_planner.push_worlds.config import push_world_config as config


class Box2DViewerROS(Box2DViewer):

    def __init__(self, config):

        self._world  = Box2DPushWorld(config)
        self._world.set_state([0.,0.,0.,0.,0.,0.])
        self._viewer = Box2DViewer(self._world, config, is_thread_loop=False)

        rospy.init_node('box2d_viewer_ros', anonymous=True)
        rospy.Subscriber("box2d_world_pose", Point, self.sub_world_state_callback)
        rospy.Subscriber("box2d_action", Floats, self.sub_action)
        rospy.Subscriber("box2d_update_goal", Point, self.sub_update_goal_location)
        rospy.Subscriber("box2d_update_obstacle", Floats, self.sub_update_obs_location)

        self._state = np.array([0., 0., 0.,0.,0.,0.])
        self._action = [[0.,0.,0.,0.,0.,0.,0.] for _ in range(self._world._num_fingers)]

        self.visualize()

    def sub_update_obs_location(self, data):
        #expects a round object, floats typs, [x, y, radiusprsddffr]
        self._world.update_obs_location(list(data.data))

    def sub_update_goal_location(self, data):
        self._world.update_goal_location([data.x, data.y, data.z])

    def sub_world_state_callback(self, data):
        self._pose = data
        print "Received pose info \n", data
        self._state = np.array([data.x, data.y, data.z, 0.,0.,0.])

    def sub_action(self, data):
        self._action = list(data.data)
        print "Received action \n", self._action

    def visualize(self):

        while not rospy.is_shutdown():

            self._viewer.handle_events()
            self._viewer.clear_screen(color=(255,255,255,255))
            self._world.set_state(self._state)
            self._world.update(self._action)
            self._viewer.draw()


def main():

    box2d_viewer = Box2DViewerROS(config)



if __name__ == '__main__':
    main()