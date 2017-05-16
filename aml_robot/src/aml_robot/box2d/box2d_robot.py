import cv2
import Box2D
import rospy
import numpy as np
from config import config
from box2d_viewer import Box2DViewer
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

class Box2DRobot(object):

    def __init__(self, update_rate=100, config=config):

        self._config = config

        self._ppm = self._config['pixels_per_meter']
        self._dt = self._config['dt']

        # --- pybox2d world setup ---
        # Create the world
        self._world = world(gravity=(0, 0), doSleep=True)

        self._colors = {
            staticBody: (255, 255, 255, 255),
            dynamicBody: (127, 127, 127, 255),
        }

        self._state = None

        self.configure_camera()

        _update_period = rospy.Duration(1.0/update_rate)

        rospy.Timer(_update_period, self._update_state)

        
    def configure_default_pos(self, pos, ori):

        pass


    def configure_camera(self):

        pass


    def get_image(self, display_image=False):

        pass


    def _update_state(self, event):

        state = {}
        now                      = rospy.Time.now()
        state = {}
        state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        self._state = state

    def get_ee_pose(self):

        pass

    def get_pos_ori(self):

        pass

    def get_contact_points(self):

        pass

    def set_default_pos_ori(self):

        pass

    def set_pos_ori(self, pos, ori):

        pass


    def get_jnt_state(self):

        pass


    def set_jnt_state(self, jnt_state):

        pass

    def apply_external_force(self, body, force_mag, point, theta, local=False):

        px_world, py_world = self.get_point(body, point)
        ix, iy = self.to_vec(theta)

        if local:
            body.ApplyLinearImpulse(impulse=(ix*force_mag,iy*force_mag), point=point, wake=True)
        else:
            body.ApplyLinearImpulse(impulse=(ix*force_mag,iy*force_mag), point=(px_world, py_world), wake=True)


    def apply_external_torque(self, link_idx, torque, point, local=True):

        pass


    def get_point(self,body,local_point):
        p = body.transform*local_point

        return p

    def get_screen_point(self,world_point):
        px = world_point[0]*self._ppm
        py = self._config['image_height'] - world_point[1]*self._ppm 
        
        p = (px,py)

        return p

    def get_screen_point2(self,body,local_point):
        px, py = body.transform*local_point*self._ppm
        py = self._config['image_height'] - py
        p = (px,py)

        return p