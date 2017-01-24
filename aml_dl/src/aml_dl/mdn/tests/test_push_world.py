from aml_robot.box2d.push_world import PushWorld
from aml_robot.box2d.pygame_viewer import PyGameViewer
from aml_robot.box2d.config import config

from aml_robot.box2d.data_manager import DataManager

from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_dl.mdn.training.config import network_params

import tensorflow as tf
import numpy as np

import pygame

config['record_training_data'] = True


class TestModelPushWorld(PushWorld):

    def __init__(self, config):
        PushWorld.__init__(self,config)
        
        self._box_state = self.get_box_state(self._dynamic_body)

        self._sess = tf.Session()

        self._inverse_model = MDNPushInverseModel(sess=self._sess, network_params=network_params)

        self._inverse_model.init_model()

        self._change_after = 100


    def draw(self, screen):
        PushWorld.draw(self,screen)


        body = self._dynamic_body

        # tgt = np.random.randn(2)
        state = self._box_state['linear_velocity']
        # px, py, tgt_x, tgt_y, theta = self._last_push
        input_x = np.expand_dims(np.r_[np.zeros(2), state],0)


        # theta = self._inverse_model.expected_out(input_x,1)

        # ix = np.cos(theta)
        # iy = np.sin(theta)

        ix, iy = self._inverse_model.expected_out2(input_x,5)

        # print "IXIY:", ix, iy

        px, py = self._box_state['position']

        p = self.get_screen_point((px,py))
        p = (int(p[0]),int(p[1]))

        endpoint = (int(p[0]+ix*25),int(p[1]-iy*25))
        # print "DIR: ", direction
        pygame.draw.line(screen,(255,127,127,255),p,endpoint)
        
    def update(self, viewer):

        next_state = PushWorld.update(self, viewer)

        self._box_state = self.get_box_state(self._dynamic_body)

        return next_state



push_world = TestModelPushWorld(config = config)
viewer = PyGameViewer(push_world, config = config)

viewer.loop()