import mujoco_py
from mujoco_py import mjviewer
from mujoco_py import glfw
import numpy as np
import rospy
import random

import time

class MujocoSimulator():      
    def __init__(self, mujoco_robot, update_rate=500):  
        
        self._robot = mujoco_robot
        self._update_rate = update_rate

    def viewer_setup(self):
        self._width = 640
        self._height = 480
        self._viewer = mjviewer.MjViewer(visible=True, init_width=self._width, init_height=self._height)
        self._viewer.cam.distance = self._robot._model.stat.extent * 0.75
        self._viewer.cam.lookat[0] = 0 #0.8
        self._viewer.cam.lookat[1] = 0.5 #0.8
        self._viewer.cam.lookat[2] = 0.1 #0.8
        self._viewer.cam.elevation = 160
        self._viewer.cam.azimuth = 100
        self._viewer.cam.camid = -3
        self._viewer.start()
        self._viewer.set_model(self._robot._model)
        #(data, width, height) = self.viewer.get_image()

    def viewer_end(self):
        self._viewer.finish()
        self._viewer = None

    def viewer_start(self):
        if self._viewer is None:
            self.viewerSetup()
        return self._viewer       

    def viewer_render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self.viewer_start().finish()
                self._viewer = None
            return
        if mode == 'rgb_array':
            self.viewer_start().render()
            self.viewer_start().set_model(self._robot._model)
            data, width, height = self.viewer_start().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        elif mode == 'human':
            self.viewer_start().loop_once()
                                               
    def reset_model(self):
        mjlib.mj_resetData(self._robot._model.ptr, self._robot._model.data.ptr)
        ob = self.reset_model()
        if self._viewer is not None:
            self._viewer.autoscale()
            self.viewer_setup()
        return ob

    def configure(self):

        self.viewer_setup()
        
        self._viewer = self.viewer_start()

        update_period = rospy.Duration(1.0/self._update_rate)
        
        rospy.Timer(update_period, self.update)

        
    def update(self, event):
        
        self.viewer_render()
    