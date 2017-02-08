import mujoco_py
from mujoco_py import mjviewer
from mujoco_py import glfw
import numpy as np
import rospy
import random


import time

class MujocoViewer():      
    def __init__(self, mujoco_robot, width=640, height=480, update_rate=100):  
        
        self._robot = mujoco_robot
        self._update_rate = update_rate
        self._inited = False
        self._width  = width
        self._height = height

    def viewer_setup(self, cam_pos=None):
        
        self._viewer = mjviewer.MjViewer(visible=True, init_width=self._width, init_height=self._height)
        self._viewer.start()
        self._viewer.set_model(self._robot._model)

        if cam_pos is not None:
            for i in range(3):
                self._viewer.cam.lookat[i] = cam_pos[i]
            self._viewer.cam.distance = cam_pos[3]
            self._viewer.cam.elevation = cam_pos[4]
            self._viewer.cam.azimuth = cam_pos[5]
            self._viewer.cam.trackbodyid = -1
    

    def viewer_end(self):
        if self._viewer is not None:
            self._viewer.finish()
            self._viewer = None

    def viewer_start(self):
        if self._viewer is None:
            self.viewerSetup()
        return self._viewer   

    def get_image(self):
        data, width, height = self.viewer_start().get_image()
        return data

    def loop(self):
        self.viewer_start().loop_once()
                                               
    def reset_model(self):
        self._robot.resetData()
        ob = self.reset_model()
        if self._viewer is not None:
            self._viewer.autoscale()
            self.viewer_setup()
        return ob

    def configure(self, cam_pos=None):

        self.viewer_setup(cam_pos=cam_pos)
        
        self._viewer = self.viewer_start()

        update_period = rospy.Duration(1.0/self._update_rate)
        
        rospy.Timer(update_period, self.update)

    def update(self,event):

        self._robot._model.step()

